"""
Small MLP as a surrogate (geometry -> POD coefficients), compared against the
RBF baseline under the same leak-free leave-one-case-out protocol.

With 29 snapshots this is a data-scarce setting, so the comparison is mostly
about whether the network is worth it over a kernel method.

    figures/07_method_comparison.png
    results/metrics.txt   (appended)

Needs PyTorch.   Usage:  python src/03_nn_surrogate.py
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import data_loader as dl
import rom_core as rc
from rom_core import R_MODES

torch.manual_seed(0)
np.random.seed(0)


class MLP(nn.Module):
    def __init__(self, n_in=3, n_out=R_MODES, h=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, h), nn.Tanh(),
            nn.Linear(h, h), nn.Tanh(),
            nn.Linear(h, n_out),
        )

    def forward(self, x):
        return self.net(x)


def _train_mlp(X, y, epochs=3000, lr=5e-3, wd=1e-3):
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    model = MLP(n_out=y.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss_fn(model(X), y).backward()
        opt.step()
    return model


class MLPSurrogate:
    """Standardizes the POD-coefficient targets, fits an MLP, predicts (m, r)."""

    def fit(self, X, y):
        self.mu = y.mean(0)
        self.sd = y.std(0)
        self.sd[self.sd == 0] = 1.0
        self.model = _train_mlp(X, (y - self.mu) / self.sd)
        return self

    def predict(self, X):
        with torch.no_grad():
            p = self.model(torch.tensor(X, dtype=torch.float32)).numpy()
        return p * self.sd + self.mu


def make_mlp():
    return MLPSurrogate()


def main():
    d = rc.load_fields_common(["umean", "vmean"])
    field_arrays = [d["fields"]["umean"], d["fields"]["vmean"]]
    params, n_pts = d["params"], d["n_pts"]

    print("RBF baseline ...")
    err_rbf, floor = rc.loo_eval(field_arrays, params, R_MODES, rc.make_rbf, n_pts)
    print("MLP ...")
    err_nn, _ = rc.loo_eval(field_arrays, params, R_MODES, make_mlp, n_pts)

    print(f"POD basis floor   mean LOO rel-L2 = {floor.mean() * 100:5.2f}%")
    print(f"RBF (thin-plate)  mean LOO rel-L2 = {err_rbf.mean() * 100:5.2f}%")
    print(f"MLP               mean LOO rel-L2 = {err_nn.mean() * 100:5.2f}%")

    means = [err_rbf.mean() * 100, err_nn.mean() * 100]
    meds = [np.median(err_rbf) * 100, np.median(err_nn) * 100]
    x = np.arange(2)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - 0.18, means, width=0.36, label="mean", color="C0")
    ax.bar(x + 0.18, meds, width=0.36, label="median", color="C1")
    ax.axhline(floor.mean() * 100, ls="--", color="gray", lw=1,
               label=f"POD basis floor ({floor.mean() * 100:.1f}%)")
    for xi_, m in zip(x - 0.18, means):
        ax.text(xi_, m + 0.1, f"{m:.1f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(["RBF\n(kernel)", "MLP\n(neural net)"])
    ax.set_ylabel("LOO relative-L2 error  [%]")
    ax.set_title("Surrogate comparison (leak-free 29-case leave-one-out)")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(dl.FIG_DIR, "07_method_comparison.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("saved", out)

    with open(os.path.join(dl.RES_DIR, "metrics.txt"), "a") as f:
        f.write("\nNeural-net add-on (PyTorch MLP, leak-free LOO):\n")
        f.write(f"  RBF  mean {err_rbf.mean() * 100:.2f}%  "
                f"median {np.median(err_rbf) * 100:.2f}%\n")
        f.write(f"  MLP  mean {err_nn.mean() * 100:.2f}%  "
                f"median {np.median(err_nn) * 100:.2f}%\n")


if __name__ == "__main__":
    main()
