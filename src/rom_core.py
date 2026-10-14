"""
Core of the parametric reduced-order model (ROM).

1. Load the DNS cases on a common normalized grid (data_loader).
2. Stack the chosen fields (e.g. [u, v], or [TKE, uv]) into a snapshot matrix,
   each field scaled to unit global std so none dominates the POD.
3. POD via SVD gives a low-dimensional basis.
4. A surrogate maps geometry (alpha, Lx, Ly) -> POD coefficients, so the full
   field of an unseen geometry can be reconstructed.

Validation is leave-one-case-out with the POD basis, mean field and field
scaling all rebuilt from the 28 training cases. The held-out geometry never
touches the basis, so the error is a real out-of-sample number.
"""

from __future__ import annotations

import os

import numpy as np
from scipy.interpolate import RBFInterpolator
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF as GPRBF, ConstantKernel, WhiteKernel
from sklearn.multioutput import MultiOutputRegressor

import data_loader as dl

N_XI, N_ETA = 160, 80                  # common-grid resolution
R_MODES = 10                           # retained POD modes
EXAMPLE_CASE = "alph125-99645-3036"    # alpha = 1.25 occurs once, so a truly unseen shape


def make_common_grid():
    return dl.make_common_grid(N_XI, N_ETA)


# ---------------------------------------------------------------------------
# field loading (interpolating 29 ASCII cases is slow, so cache the result)
# ---------------------------------------------------------------------------
def _cache_path(keys):
    tag = "_".join(sorted(keys))
    return os.path.join(dl.RES_DIR, "cache", f"{tag}_{N_XI}x{N_ETA}.npz")


def load_fields_common(keys):
    """
    Load the requested fields for every case on the common grid.

    Returns a dict: names, params (n_cases, 3) = alpha/Lx/Ly,
    fields {key: (n_cases, n_pts)}, n_pts, xi, eta, shape.
    """
    keys = tuple(keys)
    cache = _cache_path(keys)
    n_pts = N_XI * N_ETA
    xi, eta = make_common_grid()

    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return {"names": list(z["names"]), "params": z["params"],
                "fields": {k: z[f"f_{k}"] for k in keys},
                "n_pts": n_pts, "xi": xi, "eta": eta, "shape": (N_ETA, N_XI)}

    names = dl.list_cases()
    params = np.zeros((len(names), 3))
    fields = {k: np.zeros((len(names), n_pts)) for k in keys}
    for i, name in enumerate(names):
        print(f"  loading {name} ({i + 1}/{len(names)})")
        c = dl.load_case_common(name, xi, eta, keys=keys)
        params[i] = [c["alpha"], c["Lx"], c["Ly"]]
        for k in keys:
            fields[k][i] = c["fields"][k].ravel()

    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, names=np.array(names, dtype=object), params=params,
             **{f"f_{k}": fields[k] for k in keys})
    return {"names": names, "params": params, "fields": fields,
            "n_pts": n_pts, "xi": xi, "eta": eta, "shape": (N_ETA, N_XI)}


def tke_field(fields, i):
    """0.5 (u'^2 + v'^2 + w'^2) for case i."""
    return 0.5 * (fields["urms"][i] ** 2 + fields["vrms"][i] ** 2
                  + fields["wrms"][i] ** 2)


# ---------------------------------------------------------------------------
# POD
# ---------------------------------------------------------------------------
def scale_blocks(field_arrays, idx, scales=None):
    """
    Stack (n_cases, n_pts) arrays into one scaled state matrix for rows idx.
    With scales=None the scales are the std of the rows in idx (training set).
    Returns (X, scales).
    """
    if scales is None:
        scales = [field_arrays[k][idx].std() for k in range(len(field_arrays))]
        scales = [s if s > 0 else 1.0 for s in scales]
    X = np.hstack([field_arrays[k][idx] / scales[k] for k in range(len(field_arrays))])
    return X, scales


def pod_fit(X):
    """POD of a state matrix X (n, D). Returns mean, modes (n, D), coeffs, S, energy."""
    mean = X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
    energy = S ** 2 / np.sum(S ** 2)
    return mean, Vt, U * S, S, energy


def split_blocks(state, scales, n_pts):
    """Split a scaled state vector back into physical-unit field blocks."""
    return [state[k * n_pts:(k + 1) * n_pts] * s for k, s in enumerate(scales)]


def rel_l2(pred, true):
    return np.linalg.norm(pred - true) / np.linalg.norm(true)


# ---------------------------------------------------------------------------
# surrogates: fit(X, y) / predict(X) with y of shape (n, r)
# ---------------------------------------------------------------------------
class _RBF:
    def fit(self, X, y):
        self.m = RBFInterpolator(X, y, kernel="thin_plate_spline", smoothing=1e-3)
        return self

    def predict(self, X):
        return self.m(X)


class GPMulti:
    """One independent Gaussian process per POD mode; gives a per-mode std."""

    def fit(self, X, y):
        self.models = []
        for j in range(y.shape[1]):
            kern = (ConstantKernel(1.0, (1e-2, 1e3))
                    * GPRBF(length_scale=1.0, length_scale_bounds=(1e-1, 1e2))
                    + WhiteKernel(1e-3, (1e-6, 1e1)))
            gp = GaussianProcessRegressor(kernel=kern, normalize_y=True,
                                          n_restarts_optimizer=2, random_state=0)
            self.models.append(gp.fit(X, y[:, j]))
        return self

    def predict(self, X):
        return np.column_stack([m.predict(X) for m in self.models])

    def predict_std(self, X):
        return np.column_stack([m.predict(X, return_std=True)[1] for m in self.models])


def make_rbf():
    return _RBF()


def make_gp():
    return GPMulti()


def make_rf():
    return MultiOutputRegressor(RandomForestRegressor(n_estimators=300, random_state=0))


def make_gbm():
    return MultiOutputRegressor(
        GradientBoostingRegressor(n_estimators=200, max_depth=2, random_state=0))


# ---------------------------------------------------------------------------
# leave-one-case-out
# ---------------------------------------------------------------------------
def _standardize(train, query):
    mu, sd = train.mean(0), train.std(0)
    sd[sd == 0] = 1.0
    return (train - mu) / sd, (query - mu) / sd


def _fit_fold(field_arrays, params, i, r, model_factory):
    """Train everything on all cases except i. Returns the pieces needed to predict i."""
    n = params.shape[0]
    tr = [j for j in range(n) if j != i]
    Xtr, scales = scale_blocks(field_arrays, tr)
    mean, modes_all, coeffs_all, _, _ = pod_fit(Xtr)
    modes, coeffs_tr = modes_all[:r], coeffs_all[:, :r]

    Xs_tr, Xs_te = _standardize(params[tr], params[i:i + 1])
    model = model_factory().fit(Xs_tr, coeffs_tr)
    return mean, modes, scales, model, Xs_te


def loo_eval(field_arrays, params, r, model_factory, n_pts):
    """
    Leave-one-case-out relative-L2 error, POD basis rebuilt for every fold.

    Returns (err, floor), both (n_cases,). `floor` is the error from projecting
    the held-out case onto the training basis, i.e. the best any surrogate
    could do with r modes.
    """
    n = params.shape[0]
    err, floor = np.zeros(n), np.zeros(n)
    for i in range(n):
        mean, modes, scales, model, Xs_te = _fit_fold(
            field_arrays, params, i, r, model_factory)
        true_phys = np.hstack([a[i] for a in field_arrays])

        pred_c = np.asarray(model.predict(Xs_te)).ravel()
        recon = mean + pred_c @ modes
        err[i] = rel_l2(np.hstack(split_blocks(recon, scales, n_pts)), true_phys)

        x_te, _ = scale_blocks(field_arrays, [i], scales=scales)   # training scales
        proj_c = (x_te.ravel() - mean) @ modes.T
        recon_f = mean + proj_c @ modes
        floor[i] = rel_l2(np.hstack(split_blocks(recon_f, scales, n_pts)), true_phys)
    return err, floor


def predict_case(field_arrays, params, i, r, model_factory, n_pts, want_std=False):
    """
    Train on all cases except i and predict case i.

    Returns pred / true (lists of field blocks), errs (rel-L2 per block),
    scales, and, if want_std and the surrogate supports it, std_field: the GP
    uncertainty of the first block propagated through the POD modes.
    """
    mean, modes, scales, model, Xs_te = _fit_fold(
        field_arrays, params, i, r, model_factory)
    pred_c = np.asarray(model.predict(Xs_te)).ravel()
    pred_blocks = split_blocks(mean + pred_c @ modes, scales, n_pts)
    true_blocks = [a[i] for a in field_arrays]
    errs = [rel_l2(p, t) for p, t in zip(pred_blocks, true_blocks)]

    std_field = None
    if want_std and hasattr(model, "predict_std"):
        sd = model.predict_std(Xs_te).ravel()           # (r,)
        var_state = (sd ** 2) @ (modes ** 2)            # modes assumed independent
        std_field = np.sqrt(var_state[:n_pts] * scales[0] ** 2)

    return {"pred": pred_blocks, "true": true_blocks, "errs": errs,
            "scales": scales, "std_field": std_field}


# ---------------------------------------------------------------------------
# global POD over all 29 cases (descriptive figures only, not used for validation)
# ---------------------------------------------------------------------------
def build_dataset():
    d = load_fields_common(["umean", "vmean"])
    field_arrays = [d["fields"]["umean"], d["fields"]["vmean"]]
    X, scales = scale_blocks(field_arrays, np.arange(len(d["names"])))
    mean, modes, coeffs, S, energy = pod_fit(X)
    return {"names": d["names"], "params": d["params"],
            "X": X, "mean_X": mean, "modes": modes, "coeffs": coeffs,
            "S": S, "energy": energy,
            "scale_u": scales[0], "scale_v": scales[1],
            "n_pts": d["n_pts"], "xi": d["xi"], "eta": d["eta"], "shape": d["shape"]}


if __name__ == "__main__":
    ds = build_dataset()
    print(f"cases           : {len(ds['names'])}")
    print(f"state dimension : {ds['X'].shape[1]:,}")
    cum = np.cumsum(ds["energy"])
    for r in (1, 2, 3, 5, 8, 10, 15):
        print(f"  first {r:2d} modes capture {100 * cum[r - 1]:6.2f}% of energy")
