"""
Parametric ROM for the mean flow: POD + surrogate, leave-one-case-out validation.

* POD of all 29 cases (descriptive figures only)
* leave-one-case-out error for four surrogates, POD basis rebuilt per fold
* mode-count sweep
* prediction of an unseen geometry with a Gaussian-process uncertainty map

    figures/04_pod_energy.png          cumulative POD energy
    figures/05_pod_modes.png           mean field + leading modes
    figures/06_loo_error.png           per-case error vs. basis floor
    figures/08_prediction_example.png  DNS vs ROM + uncertainty, unseen hill
    figures/09_mode_sweep.png          error vs. number of modes
    results/metrics.txt                (overwritten; scripts 03-05 append to it)

Usage:  python src/02_pod_rom.py
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import data_loader as dl
import rom_core as rc
from rom_core import R_MODES, EXAMPLE_CASE

os.makedirs(dl.FIG_DIR, exist_ok=True)
os.makedirs(dl.RES_DIR, exist_ok=True)

SURROGATES = {
    "Gaussian Process": rc.make_gp,
    "RBF (thin-plate)": rc.make_rbf,
    "Gradient Boosting": rc.make_gbm,
    "Random Forest": rc.make_rf,
}


def _save(fig, name, **kw):
    out = os.path.join(dl.FIG_DIR, name)
    fig.savefig(out, dpi=150, **kw)
    plt.close(fig)
    print("saved", out)


# ---------------------------------------------------------------------------
# descriptive POD figures (whole database)
# ---------------------------------------------------------------------------
def fig_energy(ds):
    cum = np.cumsum(ds["energy"]) * 100
    k = min(20, len(cum))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(np.arange(1, k + 1), ds["energy"][:k] * 100, alpha=0.4, label="per-mode")
    ax.plot(np.arange(1, k + 1), cum[:k], "o-", color="C3", label="cumulative")
    ax.axhline(95, ls="--", color="gray", lw=1)
    ax.axvline(R_MODES, ls=":", color="k", lw=1,
               label=f"{R_MODES} modes ({cum[R_MODES - 1]:.1f}%)")
    ax.set_xlabel("POD mode index")
    ax.set_ylabel("energy captured  [%]")
    ax.set_title("POD spectrum: 25,600-D flow state collapses to ~10 modes")
    ax.legend()
    fig.tight_layout()
    _save(fig, "04_pod_energy.png")


def fig_modes(ds):
    XX, YY = np.meshgrid(ds["xi"], ds["eta"])
    n, shp = ds["n_pts"], ds["shape"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 5.5))

    mean_u = (ds["mean_X"][:n] * ds["scale_u"]).reshape(shp)
    pcm = axes[0, 0].pcolormesh(XX, YY, mean_u, cmap="RdBu_r", shading="auto")
    axes[0, 0].set_title("Mean u-field (POD base)")
    fig.colorbar(pcm, ax=axes[0, 0], pad=0.01)

    for k, ax in zip(range(3), axes.ravel()[1:]):
        mode_u = ds["modes"][k, :n].reshape(shp)
        lim = np.abs(mode_u).max()
        pcm = ax.pcolormesh(XX, YY, mode_u, cmap="RdBu_r", shading="auto",
                            vmin=-lim, vmax=lim)
        ax.set_title(f"POD mode {k + 1}  ({ds['energy'][k] * 100:.1f}% energy)")
        fig.colorbar(pcm, ax=ax, pad=0.01)

    for ax in axes.ravel():
        ax.set_xlabel(r"$\xi = x/L_x$")
        ax.set_ylabel(r"$\eta = y/L_y$")
    fig.suptitle("POD basis of the u mean-flow (normalized coordinates)")
    fig.tight_layout()
    _save(fig, "05_pod_modes.png")


# ---------------------------------------------------------------------------
# leave-one-case-out
# ---------------------------------------------------------------------------
def fig_loo(names, params, err_best, err_floor, best_name):
    order = np.argsort(params[:, 0])
    labels = [names[i] for i in order]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(x - 0.2, err_floor[order] * 100, width=0.4, color="C0", alpha=0.7,
           label="POD basis floor (rebuilt per fold)")
    ax.bar(x + 0.2, err_best[order] * 100, width=0.4, color="C3", alpha=0.8,
           label=f"surrogate prediction ({best_name})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("relative L2 error  [%]")
    ax.set_title("Leave-one-case-out (leak-free): predicting an unseen geometry's flow field")
    ax.legend()
    fig.tight_layout()
    _save(fig, "06_loo_error.png")


def fig_mode_sweep(field_arrays, params, n_pts):
    rs = [2, 3, 5, 8, 10, 12, 15, 18, 22]
    means, floors = [], []
    for r in rs:
        err, floor = rc.loo_eval(field_arrays, params, r, rc.make_rbf, n_pts)
        means.append(err.mean() * 100)
        floors.append(floor.mean() * 100)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(rs, floors, "o-", label="POD basis floor", color="C0")
    ax.plot(rs, means, "s-", label="RBF surrogate (LOO)", color="C3")
    ax.axvline(R_MODES, ls=":", color="k", lw=1, label=f"chosen r = {R_MODES}")
    ax.set_xlabel("number of retained POD modes  r")
    ax.set_ylabel("mean LOO relative-L2 error  [%]")
    ax.set_title("Why ~10 modes: floor keeps falling but the surrogate error flattens")
    ax.legend()
    fig.tight_layout()
    _save(fig, "09_mode_sweep.png")
    return rs, means, floors


def fig_prediction_example(field_arrays, params, names, n_pts, xi, eta, shp):
    i = names.index(EXAMPLE_CASE)
    res = rc.predict_case(field_arrays, params, i, R_MODES, rc.make_gp, n_pts, want_std=True)
    u_true = res["true"][0].reshape(shp)
    u_pred = res["pred"][0].reshape(shp)
    err = res["errs"][0] * 100
    std = res["std_field"].reshape(shp)

    XX, YY = np.meshgrid(xi, eta)
    fig, axes = plt.subplots(2, 2, figsize=(13, 6))
    for ax, fld, title in [
        (axes[0, 0], u_true, "DNS ground truth  (u)"),
        (axes[0, 1], u_pred, "ROM prediction  (u, geometry never seen)"),
    ]:
        pcm = ax.contourf(XX, YY, fld, levels=np.linspace(-0.3, 1.2, 41),
                          cmap="RdBu_r", extend="both")
        ax.contour(XX, YY, fld, levels=[0.0], colors="lime", linewidths=1.2)
        ax.set_title(title)
        fig.colorbar(pcm, ax=ax, pad=0.01)

    diff = u_pred - u_true
    lim = np.abs(diff).max()
    pcm = axes[1, 0].contourf(XX, YY, diff, levels=np.linspace(-lim, lim, 41),
                              cmap="coolwarm", extend="both")
    axes[1, 0].set_title(f"prediction error (u),  field rel-L2 = {err:.1f}%")
    fig.colorbar(pcm, ax=axes[1, 0], pad=0.01)

    pcm = axes[1, 1].contourf(XX, YY, std, levels=40, cmap="magma")
    axes[1, 1].set_title("GP predictive uncertainty (std of u)")
    fig.colorbar(pcm, ax=axes[1, 1], pad=0.01)

    for ax in axes.ravel():
        ax.set_xlabel(r"$\xi=x/L_x$")
        ax.set_ylabel(r"$\eta=y/L_y$")
    a, lx, ly = params[i]
    fig.suptitle(f"Unseen hill shape {EXAMPLE_CASE}  "
                 fr"($\alpha$={a}, $L_x$={lx}, $L_y$={ly})", y=1.01)
    fig.tight_layout()
    _save(fig, "08_prediction_example.png", bbox_inches="tight")
    return err


def main():
    ds = rc.build_dataset()
    fig_energy(ds)
    fig_modes(ds)

    d = rc.load_fields_common(["umean", "vmean"])
    field_arrays = [d["fields"]["umean"], d["fields"]["vmean"]]
    params, names, n_pts = d["params"], d["names"], d["n_pts"]

    results, floor = {}, None
    for name, factory in SURROGATES.items():
        err, floor = rc.loo_eval(field_arrays, params, R_MODES, factory, n_pts)
        results[name] = err
        print(f"{name:20s} mean LOO rel-L2 = {err.mean() * 100:5.2f}%  "
              f"(median {np.median(err) * 100:5.2f}%, max {err.max() * 100:5.2f}%)")

    best_name = min(results, key=lambda k: results[k].mean())
    fig_loo(names, params, results[best_name], floor, best_name)
    rs, sweep_means, sweep_floors = fig_mode_sweep(field_arrays, params, n_pts)
    ex_err = fig_prediction_example(field_arrays, params, names, n_pts,
                                    d["xi"], d["eta"], d["shape"])

    lines = [
        "Parametric POD-ROM (mean flow) - leak-free leave-one-out",
        "=" * 56,
        f"cases: {len(names)}   state dim: {field_arrays[0].shape[1] * 2:,}   "
        f"retained modes: {R_MODES}",
        f"POD basis floor (per fold): {floor.mean() * 100:.2f}% mean",
        "",
        "Surrogate (geometry -> POD coeffs) LOO relative-L2 error:",
    ]
    for name, err in sorted(results.items(), key=lambda kv: kv[1].mean()):
        lines.append(f"  {name:20s} mean {err.mean() * 100:5.2f}%  "
                     f"median {np.median(err) * 100:5.2f}%  max {err.max() * 100:5.2f}%")
    lines += ["", f"best surrogate: {best_name}",
              f"example unseen case {EXAMPLE_CASE}: u rel-L2 = {ex_err:.2f}%",
              "", "Mode sweep (RBF surrogate):"]
    for r, m, fl in zip(rs, sweep_means, sweep_floors):
        lines.append(f"  r={r:2d}  floor {fl:5.2f}%  surrogate {m:5.2f}%")

    txt = "\n".join(lines)
    with open(os.path.join(dl.RES_DIR, "metrics.txt"), "w") as f:
        f.write(txt + "\n")
    print("\n" + txt)


if __name__ == "__main__":
    main()
