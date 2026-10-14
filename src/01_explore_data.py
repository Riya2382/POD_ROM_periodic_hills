"""
Exploratory figures for the periodic-hill database.

    figures/01_geometry_family.png    the five hill shapes
    figures/02_meanflow_examples.png  mean u + streamlines for three steepnesses
    figures/03_dataset_map.png        how the 29 cases fill (alpha, Lx, Ly) space

Usage:  python src/01_explore_data.py
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import data_loader as dl
import hill_geometry

os.makedirs(dl.FIG_DIR, exist_ok=True)

ALPHAS = [0.5, 0.75, 1.0, 1.25, 1.5]
COLORS = plt.cm.viridis(np.linspace(0, 0.9, len(ALPHAS)))


def fig_geometry_family():
    yy = np.arange(0, 9, 0.01)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    for a, c in zip(ALPHAS, COLORS):
        x, h = hill_geometry.para_profile(yy, a)
        ax.plot(x, h, color=c, lw=2, label=fr"$\alpha$ = {a}")
    ax.set_xlabel("x / H")
    ax.set_ylabel("y / H")
    ax.set_title(r"Parameterized periodic-hill geometry (shape factor $\alpha$)")
    ax.set_ylim(0, 1.2)
    ax.set_xlim(0, 6)
    ax.legend(ncol=5, loc="upper center", frameon=False)
    fig.tight_layout()
    out = os.path.join(dl.FIG_DIR, "01_geometry_family.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved", out)


def fig_meanflow_examples():
    cases = ["alph05-7071-3036", "alph10-9-3036", "alph15-10929-3036"]
    titles = [r"$\alpha=0.5$ (gentle)", r"$\alpha=1.0$ (baseline)", r"$\alpha=1.5$ (steep)"]

    xi, eta = dl.make_common_grid(256, 128)
    fig, axes = plt.subplots(3, 1, figsize=(9, 9))
    for ax, name, title in zip(axes, cases, titles):
        c = dl.load_case_common(name, xi, eta, keys=("umean", "vmean"))
        X, Y = c["XI"] * c["Lx"], c["ETA"] * c["Ly"]
        u, v = c["fields"]["umean"], c["fields"]["vmean"]

        pcm = ax.pcolormesh(X, Y, u, shading="auto", cmap="RdBu_r", vmin=-0.3, vmax=1.2)
        ax.streamplot(xi * c["Lx"], eta * c["Ly"], u, v,
                      color="k", density=0.8, linewidth=0.5, arrowsize=0.6)
        ax.contour(X, Y, u, levels=[0.0], colors="lime", linewidths=1.5)  # u = 0
        ax.set_title(title)
        ax.set_ylabel("y / H")
        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(0, c["Ly"])
        fig.colorbar(pcm, ax=ax, label=r"$\langle u\rangle / U_b$", pad=0.01)
    axes[-1].set_xlabel("x / H")
    fig.suptitle("Mean flow over the hill: recirculation bubble grows with steepness\n"
                 "(green line = zero streamwise velocity)", y=1.005)
    fig.tight_layout()
    out = os.path.join(dl.FIG_DIR, "02_meanflow_examples.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def fig_dataset_map():
    sizes = {2.024: 60, 3.036: 140, 4.048: 260}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for r in dl.case_table():
        ax.scatter(r["Lx"], r["alpha"], s=sizes[round(r["Ly"], 3)], alpha=0.6,
                   color=COLORS[ALPHAS.index(r["alpha"])], edgecolor="k", linewidth=0.5)
    ax.set_xlabel(r"streamwise length  $L_x / H$")
    ax.set_ylabel(r"hill-shape factor  $\alpha$")
    ax.set_yticks(ALPHAS)
    ax.set_title(r"29-case design space  (marker size = domain height $L_y/H$: 2.024 / 3.036 / 4.048)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(dl.FIG_DIR, "03_dataset_map.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    fig_geometry_family()
    fig_meanflow_examples()
    fig_dataset_map()
