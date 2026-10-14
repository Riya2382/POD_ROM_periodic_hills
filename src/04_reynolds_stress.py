"""
Parametric ROM for the turbulence quantities.

Same leak-free POD + surrogate pipeline as the mean-flow ROM, applied to

    TKE = 0.5 (u'^2 + v'^2 + w'^2)
    uv  = <u'v'>   Reynolds shear stress

These are the terms a RANS model has to close. They are noisier and more
localized (shear layer, downstream hill crest) than the mean flow, so the
errors are expected to be larger.

    figures/10_reynolds_prediction.png
    results/metrics.txt   (appended)

Usage:  python src/04_reynolds_stress.py
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import data_loader as dl
import rom_core as rc
from rom_core import R_MODES, EXAMPLE_CASE


def main():
    d = rc.load_fields_common(["urms", "vrms", "wrms", "uv"])
    f = d["fields"]
    tke = 0.5 * (f["urms"] ** 2 + f["vrms"] ** 2 + f["wrms"] ** 2)
    field_arrays = [tke, f["uv"]]
    params, names, n_pts = d["params"], d["names"], d["n_pts"]
    shp, xi, eta = d["shape"], d["xi"], d["eta"]

    results, floor = {}, None
    for name, factory in [("Gaussian Process", rc.make_gp),
                          ("RBF (thin-plate)", rc.make_rbf)]:
        err, floor = rc.loo_eval(field_arrays, params, R_MODES, factory, n_pts)
        results[name] = err
        print(f"{name:20s} mean LOO rel-L2 = {err.mean() * 100:5.2f}%  "
              f"(median {np.median(err) * 100:5.2f}%)")
    best_name = min(results, key=lambda k: results[k].mean())

    # how compressible is the turbulence field?
    X, _ = rc.scale_blocks(field_arrays, np.arange(len(names)))
    cum = np.cumsum(rc.pod_fit(X)[4])

    # example: the unseen alpha = 1.25 hill
    i = names.index(EXAMPLE_CASE)
    res = rc.predict_case(field_arrays, params, i, R_MODES, rc.make_gp, n_pts)
    tke_t, uv_t = (b.reshape(shp) for b in res["true"])
    tke_p, uv_p = (b.reshape(shp) for b in res["pred"])
    e_tke, e_uv = res["errs"][0] * 100, res["errs"][1] * 100

    XX, YY = np.meshgrid(xi, eta)
    fig, axes = plt.subplots(2, 2, figsize=(13, 6))
    tmax = tke_t.max()
    tmin = min(0.0, tke_p.min())   # the ROM can dip slightly below zero
    for ax, fld, title in [
        (axes[0, 0], tke_t, "DNS truth: TKE"),
        (axes[0, 1], tke_p, f"ROM prediction: TKE  (rel-L2 = {e_tke:.1f}%)"),
    ]:
        pcm = ax.contourf(XX, YY, fld, levels=np.linspace(tmin, tmax, 41),
                          cmap="magma", extend="both")
        ax.set_title(title)
        fig.colorbar(pcm, ax=ax, pad=0.01)

    lim = np.abs(uv_t).max()
    for ax, fld, title in [
        (axes[1, 0], uv_t, "DNS truth: Reynolds shear stress  <u'v'>"),
        (axes[1, 1], uv_p, f"ROM prediction: <u'v'>  (rel-L2 = {e_uv:.1f}%)"),
    ]:
        pcm = ax.contourf(XX, YY, fld, levels=np.linspace(-lim, lim, 41),
                          cmap="RdBu_r", extend="both")
        ax.set_title(title)
        fig.colorbar(pcm, ax=ax, pad=0.01)

    for ax in axes.ravel():
        ax.set_xlabel(r"$\xi=x/L_x$")
        ax.set_ylabel(r"$\eta=y/L_y$")
    fig.suptitle(f"Turbulence quantities on unseen hill {EXAMPLE_CASE}  "
                 fr"($\alpha$={params[i][0]})", y=1.01)
    fig.tight_layout()
    out = os.path.join(dl.FIG_DIR, "10_reynolds_prediction.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)

    lines = ["",
             "Turbulence ROM (TKE + Reynolds shear stress uv) - leak-free",
             "=" * 56,
             f"first {R_MODES} modes capture {cum[R_MODES - 1] * 100:.1f}% "
             f"of turbulence-field energy",
             f"POD basis floor (per fold): {floor.mean() * 100:.2f}% mean"]
    for name, err in sorted(results.items(), key=lambda kv: kv[1].mean()):
        lines.append(f"  {name:20s} mean {err.mean() * 100:5.2f}%  "
                     f"median {np.median(err) * 100:5.2f}%")
    lines += [f"best surrogate: {best_name}",
              f"example unseen case {EXAMPLE_CASE}: "
              f"TKE rel-L2 = {e_tke:.2f}%, uv rel-L2 = {e_uv:.2f}%"]
    txt = "\n".join(lines)
    with open(os.path.join(dl.RES_DIR, "metrics.txt"), "a") as fh:
        fh.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
