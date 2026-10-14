"""
A priori test of the Boussinesq (linear eddy-viscosity) hypothesis on DNS data.

Take the exact DNS mean field of one strongly separated case, compute the mean
strain-rate tensor S, and check how well the Reynolds-stress anisotropy
a_ij = -2 nu_t S_ij can describe the true DNS stresses. No RANS solver is run.
Three diagnostics:

  1. the eddy viscosity implied by the shear stress, nu_t = -<u'v'> / (2 S12),
     and the fraction of the turbulent region where it comes out negative
  2. alignment cos(a, S) between anisotropy and strain; Boussinesq needs -1
  3. correlation between <u'v'> and S12; a single constant nu_t would put every
     point on one straight line

    figures/11_apriori_closure.png
    results/metrics.txt   (appended)

Usage:  python src/05_apriori_closure.py
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import data_loader as dl
import rom_core as rc
from rom_core import EXAMPLE_CASE as CASE


def main():
    d = rc.load_fields_common(["umean", "vmean", "urms", "vrms", "wrms", "uv"])
    names, params, shp = d["names"], d["params"], d["shape"]
    xi, eta = d["xi"], d["eta"]
    i = names.index(CASE)
    alpha, Lx, Ly = params[i]

    f = d["fields"]
    U = f["umean"][i].reshape(shp)
    V = f["vmean"][i].reshape(shp)
    uu = f["urms"][i].reshape(shp) ** 2
    vv = f["vrms"][i].reshape(shp) ** 2
    ww = f["wrms"][i].reshape(shp) ** 2
    uv = f["uv"][i].reshape(shp)

    # physical coordinates, so the gradients are physical (arrays are [y, x])
    x, y = xi * Lx, eta * Ly
    dUdy, dUdx = np.gradient(U, y, x)
    dVdy, dVdx = np.gradient(V, y, x)

    # in-plane strain rate
    S11, S22, S12 = dUdx, dVdy, 0.5 * (dUdy + dVdx)

    # TKE and Reynolds-stress anisotropy
    k = 0.5 * (uu + vv + ww)
    a11 = uu - 2.0 / 3.0 * k
    a22 = vv - 2.0 / 3.0 * k
    a12 = uv

    # 1. eddy viscosity from the shear stress, a12 = -2 nu_t S12
    active = np.abs(S12) > 0.05 * np.abs(S12).max()   # skip near-zero strain
    turb = k > 0.05 * k.max()                         # region with real turbulence
    reg = active & turb
    nu_t = np.full_like(S12, np.nan)
    nu_t[active] = -a12[active] / (2.0 * S12[active])
    neg_frac = np.mean(nu_t[reg] < 0) * 100.0

    # 2. alignment of anisotropy with strain
    aS = a11 * S11 + a22 * S22 + 2.0 * a12 * S12
    amag = np.sqrt(a11 ** 2 + a22 ** 2 + 2.0 * a12 ** 2)
    Smag = np.sqrt(S11 ** 2 + S22 ** 2 + 2.0 * S12 ** 2)
    cos_align = aS / (amag * Smag + 1e-12)
    mean_align = np.mean(cos_align[turb])

    # 3. how linear is stress vs strain?
    s = S12[reg] - S12[reg].mean()
    t = a12[reg] - a12[reg].mean()
    r_shear = np.sum(s * t) / (np.sqrt(np.sum(s * s) * np.sum(t * t)) + 1e-30)
    r2_shear = r_shear ** 2

    # figure
    XX, YY = np.meshgrid(xi, eta)
    fig, axes = plt.subplots(2, 2, figsize=(13, 6.5))

    nu_plot = np.where(reg, nu_t, np.nan)
    nlim = np.nanpercentile(np.abs(nu_plot), 98)
    pcm = axes[0, 0].contourf(XX, YY, nu_plot, levels=np.linspace(-nlim, nlim, 41),
                              cmap="RdBu_r", extend="both")
    axes[0, 0].set_title("Optimal eddy viscosity from shear stress\n"
                         f"{neg_frac:.0f}% of the turbulent region is NEGATIVE (non-physical)")
    fig.colorbar(pcm, ax=axes[0, 0], pad=0.01)

    pcm = axes[0, 1].contourf(XX, YY, np.where(turb, cos_align, np.nan),
                              levels=np.linspace(-1, 1, 41), cmap="RdBu_r", extend="both")
    axes[0, 1].set_title("Stress-strain alignment cos(a, S)\n"
                         f"Boussinesq needs -1 everywhere; domain mean = {mean_align:.2f}")
    fig.colorbar(pcm, ax=axes[0, 1], pad=0.01)

    axes[1, 0].scatter(S12[reg], a12[reg], s=4, alpha=0.2, color="C0")
    axes[1, 0].axhline(0, color="gray", lw=0.6)
    axes[1, 0].axvline(0, color="gray", lw=0.6)
    axes[1, 0].set_xlabel(r"mean shear strain  $S_{12}$")
    axes[1, 0].set_ylabel(r"true DNS shear stress  $\langle u'v'\rangle$")
    axes[1, 0].set_title("Stress vs strain on the turbulent region\n"
                         f"Boussinesq assumes one line; corr = {r_shear:.2f}, R^2 = {r2_shear:.2f}")

    axes[1, 1].hist(cos_align[turb], bins=40, range=(-1, 1), color="C0", alpha=0.8)
    axes[1, 1].axvline(-1.0, color="k", ls="--", lw=1.5, label="Boussinesq ideal = -1")
    axes[1, 1].axvline(mean_align, color="C3", lw=1.5, label=f"domain mean = {mean_align:.2f}")
    axes[1, 1].set_xlabel(r"stress-strain alignment  $\cos(a, S)$")
    axes[1, 1].set_ylabel("number of grid cells")
    axes[1, 1].set_title("Misalignment spread (Boussinesq needs a spike at -1)")
    axes[1, 1].legend(fontsize=8, loc="upper left")

    for ax in (axes[0, 0], axes[0, 1]):
        ax.set_xlabel(r"$\xi = x/L_x$")
        ax.set_ylabel(r"$\eta = y/L_y$")
    fig.suptitle(f"A priori RANS closure test on DNS case {CASE} "
                 fr"($\alpha$={alpha})  -  no RANS solver run", y=1.02)
    fig.tight_layout()
    out = os.path.join(dl.FIG_DIR, "11_apriori_closure.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)

    lines = ["",
             f"A priori RANS closure test (DNS case {CASE}, no solver run)",
             "=" * 56,
             f"optimal eddy viscosity negative over {neg_frac:.0f}% of the turbulent "
             f"region (Boussinesq non-physical there)",
             f"mean stress-strain alignment cos(a,S) = {mean_align:.2f} "
             f"(Boussinesq ideal = -1.00)",
             f"shear stress vs mean shear strain correlation = {r_shear:.2f} "
             f"(R^2 = {r2_shear:.2f}) on the active-strain turbulent region"]
    txt = "\n".join(lines)
    with open(os.path.join(dl.RES_DIR, "metrics.txt"), "a") as fh:
        fh.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
