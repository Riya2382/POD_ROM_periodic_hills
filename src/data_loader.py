"""
Loader for the 2D periodic-hill DNS database (Re = 5600, Laizet 2021).

29 DNS cases, hill geometry varied through three parameters:
    alpha : hill steepness factor   0.50, 0.75, 1.00, 1.25, 1.50
    Lx    : streamwise length / H
    Ly    : domain height / H       2.024, 3.036, 4.048

Each case folder holds ASCII fields on a structured (ny, nx) grid:
    mean_files.dat       x, y, umean, vmean, wmean, pmean
    rms_files1.dat       x, y, urms, vrms, wrms, prms
    rms_files2.dat       x, y, uv, uw, vw
    diss_vort_files.dat  x, y, dissmean, vort
    yp.dat               wall-normal mesh (ny values)

The files are written x-fastest (Fortran: do j=1,ny / do i=1,nx), so
reshape(ny, nx) gives arrays indexed [y, x].

Every case lives on its own mesh and domain. To compare them, each field is
interpolated onto a common normalized grid (xi = x/Lx, eta = y/Ly).
"""

from __future__ import annotations

import os

import numpy as np
from scipy.interpolate import RegularGridInterpolator

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
# override with the PEHILL_DATA environment variable if the data lives elsewhere
DB_DIR = os.environ.get("PEHILL_DATA", os.path.join(ROOT, "data", "pehill-29-cases-DNS"))
FIG_DIR = os.path.join(ROOT, "figures")
RES_DIR = os.path.join(ROOT, "results")

# folder names look like  alph{alpha}-{Lx}-{Ly}, e.g. alph05-7071-3036
_ALPHA_MAP = {"05": 0.5, "075": 0.75, "10": 1.0, "125": 1.25, "15": 1.5}
_LX_MAP = {
    "4071": 4.071, "7071": 7.071, "10071": 10.071,    # alpha = 0.5
    "80355": 8.0355,                                   # alpha = 0.75
    "6": 6.0, "9": 9.0, "12": 12.0,                    # alpha = 1.0
    "99645": 9.9645,                                   # alpha = 1.25
    "7929": 7.929, "10929": 10.929, "13929": 13.929,   # alpha = 1.5
}

FIELD_FILES = {
    "mean": ("mean_files.dat", ["x", "y", "umean", "vmean", "wmean", "pmean"]),
    "rms1": ("rms_files1.dat", ["x", "y", "urms", "vrms", "wrms", "prms"]),
    "rms2": ("rms_files2.dat", ["x", "y", "uv", "uw", "vw"]),
    "diss": ("diss_vort_files.dat", ["x", "y", "diss", "vort"]),
}


def parse_case_name(name):
    """'alph05-7071-3036' -> (alpha, Lx, Ly)"""
    parts = name.split("-")
    return _ALPHA_MAP[parts[0][4:]], _LX_MAP[parts[1]], int(parts[2]) / 1000.0


def list_cases():
    """Case folder names, sorted by (alpha, Lx, Ly)."""
    if not os.path.isdir(DB_DIR):
        raise FileNotFoundError(
            f"DNS database not found at {DB_DIR}. See data/README.md, or set "
            "the PEHILL_DATA environment variable."
        )
    names = [d for d in os.listdir(DB_DIR)
             if d.startswith("alph") and os.path.isdir(os.path.join(DB_DIR, d))]
    return sorted(names, key=parse_case_name)


def case_table():
    rows = []
    for name in list_cases():
        a, lx, ly = parse_case_name(name)
        rows.append({"name": name, "alpha": a, "Lx": lx, "Ly": ly})
    return rows


def _read_dat(path, ncols):
    arr = np.loadtxt(path)
    if arr.shape[1] != ncols:
        raise ValueError(f"{path}: expected {ncols} columns, got {arr.shape[1]}")
    return arr


def load_case_raw(name):
    """
    Load one case on its native grid.

    Returns a dict with name, alpha, Lx, Ly, nx, ny, x (nx,), y (ny,) and
    fields[<column name>] -> (ny, nx).
    """
    folder = os.path.join(DB_DIR, name)
    alpha, Lx, Ly = parse_case_name(name)

    ny = np.loadtxt(os.path.join(folder, "yp.dat")).size
    mean = _read_dat(os.path.join(folder, FIELD_FILES["mean"][0]), 6)
    nx = mean.shape[0] // ny

    x = mean[:nx, 0].copy()    # first nx rows are j = 0
    y = mean[::nx, 1].copy()   # one y value per j

    fields = {}
    for key, (fname, cols) in FIELD_FILES.items():
        arr = mean if key == "mean" else _read_dat(os.path.join(folder, fname), len(cols))
        for c in range(2, len(cols)):
            fields[cols[c]] = arr[:, c].reshape(ny, nx)

    return {"name": name, "alpha": alpha, "Lx": Lx, "Ly": Ly,
            "nx": nx, "ny": ny, "x": x, "y": y, "fields": fields}


def make_common_grid(n_xi=192, n_eta=96):
    return np.linspace(0.0, 1.0, n_xi), np.linspace(0.0, 1.0, n_eta)


def load_case_common(name, xi, eta, keys=("umean", "vmean", "pmean")):
    """Interpolate the requested fields of one case onto the (xi, eta) grid."""
    raw = load_case_raw(name)
    xi_native = raw["x"] / raw["Lx"]
    eta_native = raw["y"] / raw["Ly"]

    XI, ETA = np.meshgrid(xi, eta)                    # (n_eta, n_xi)
    query = np.column_stack([ETA.ravel(), XI.ravel()])

    out = {"name": name, "alpha": raw["alpha"], "Lx": raw["Lx"], "Ly": raw["Ly"],
           "XI": XI, "ETA": ETA, "fields": {}}
    for key in keys:
        interp = RegularGridInterpolator((eta_native, xi_native), raw["fields"][key],
                                         bounds_error=False, fill_value=None)
        out["fields"][key] = interp(query).reshape(XI.shape)
    return out


if __name__ == "__main__":
    # quick sanity check that the database is found and parsed correctly
    cases = case_table()
    print(f"found {len(cases)} cases\n")
    print(f"{'name':22s} {'alpha':>6s} {'Lx':>8s} {'Ly':>7s}")
    for r in cases:
        print(f"{r['name']:22s} {r['alpha']:6.2f} {r['Lx']:8.3f} {r['Ly']:7.3f}")

    raw = load_case_raw("alph10-9-3036")
    u = raw["fields"]["umean"]
    print(f"\nalph10-9-3036: nx={raw['nx']} ny={raw['ny']}  "
          f"umean in [{u.min():.3f}, {u.max():.3f}]")
