"""
Hill profile of the parameterized periodic-hill family.

Piecewise cubic fit of the baseline hill and the alpha-stretching of its
slopes, as published in Xiao et al. (2020), Computers & Fluids 200, 104431.
Lengths inside this module are in units of H/28 (the fit's native scaling);
the public functions return coordinates in units of H.
"""

import numpy as np

# (x_lo, x_hi, c0, c1, c2, c3): h = c0 + c1 x + c2 x^2 + c3 x^3
_SEGMENTS = [
    (0, 9, 28.0, 0.0, 6.775070969851e-03, -2.124527775800e-03),
    (9, 14, 2.507355893131e+01, 9.754803562315e-01, -1.016116352781e-01, 1.889794677828e-03),
    (14, 20, 2.579601052357e+01, 8.206693007457e-01, -9.055370274339e-02, 1.626510569859e-03),
    (20, 30, 4.046435022819e+01, -1.379581654948e+00, 1.945884504128e-02, -2.070318932190e-04),
    (30, 40, 1.792461334664e+01, 8.743920332081e-01, -5.567361123058e-02, 6.277731764683e-04),
    (40, 54, 5.639011190988e+01, -2.010520359035e+00, 1.644919857549e-02, 2.674976141766e-05),
]


def _height(x):
    """Hill height for x in [0, 126] (first half period), flat after x = 54."""
    h = np.zeros_like(x)
    for i, (lo, hi, c0, c1, c2, c3) in enumerate(_SEGMENTS):
        last = i == len(_SEGMENTS) - 1
        m = (x >= lo) & ((x <= hi) if last else (x < hi))
        h[m] = c0 + c1 * x[m] + c2 * x[m] ** 2 + c3 * x[m] ** 3
    first, last_seg = _SEGMENTS[0], _SEGMENTS[-1]
    m0 = (x >= first[0]) & (x < first[1])
    h[m0] = np.minimum(28.0, h[m0])          # cap the crest at H
    m1 = (x >= last_seg[0]) & (x <= last_seg[1])
    h[m1] = np.maximum(0.0, h[m1])           # no negative heights at the foot
    return h


def profile(y):
    """Baseline (alpha = 1) hill height h/H at streamwise positions y (units of H)."""
    x = np.asarray(y, dtype=float) * 28.0
    x = np.where(x > 126.0, 252.0 - x, x)    # second half mirrors the first
    return _height(x) / 28.0


def para_profile(y, alpha):
    """
    Hill of steepness alpha. Returns (x_alpha, h/H): the stretched streamwise
    coordinate and the hill height at each input position.
    """
    y = np.asarray(y, dtype=float)
    x = y * 28.0
    xa = y.copy()

    xa[(x >= 0) & (x < 54)] *= alpha         # windward slope stretched by alpha
    shift = 54.0 / 28.0 * (1.0 - alpha)
    xa[(x > 54) & (x <= 198)] -= shift       # flat part and the shifted crest
    m = (x > 198) & (x <= 252)
    xa[m] -= shift + (x[m] - 198.0) * (1.0 - alpha) / 28.0

    return xa, profile(y)
