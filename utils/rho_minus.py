"""
utils/rho_minus.py
------------------
Implements the distortion lower-bound certificate

    rho_-(t; nu) = (1 / (3 * 2^{n+3}))
                   * min_{R_i in supp(nu), R_i > R_min}
                         sqrt(nu([0, R_i))) / (R_i - R_min)
                   * (t - R_min)

for t > R_min, and 0 for t <= R_min, as in equation (3) / Theorem 5.1
of Mitra & Virk (2024) and equation (eq:weights4bdd) of the manuscript.

The scale measure nu is represented as:
    scales : np.ndarray  shape (N,)   — support points R_1 < ... < R_N
    masses : np.ndarray  shape (N,)   — nu({R_k}) = w_k^2 * R_k^2

so that w_k = sqrt(masses[k]) / scales[k].

Normalisation constraint:  sum_k w_k^2 = sum_k masses[k] / scales[k]^2 = 1.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple


# ══════════════════════════════════════════════════════════════════════════
# Core certificate
# ══════════════════════════════════════════════════════════════════════════

def rho_minus(t: float | np.ndarray,
              scales: np.ndarray,
              masses: np.ndarray,
              n: int = 1,
              L: float | None = None) -> float | np.ndarray:
    """
    Evaluate the distortion certificate rho_-(t; nu).

    Parameters
    ----------
    t      : scalar or array of query points
    scales : (N,) array of support points R_1 < ... < R_N  (strictly positive)
    masses : (N,) array nu({R_k}) = w_k^2 * R_k^2         (strictly positive)
    n      : diagram dimension (default 1, i.e. 1-point diagrams)
    L      : frame size; if given, includes the term nu([0,L)) / (L - R_min)

    Returns
    -------
    scalar or array of the same shape as t
    """
    scales = np.asarray(scales, dtype=float)
    masses = np.asarray(masses, dtype=float)
    if scales.ndim != 1 or masses.ndim != 1 or len(scales) != len(masses):
        raise ValueError("scales and masses must be 1-D arrays of equal length")

    idx = np.argsort(scales)
    scales = scales[idx]
    masses = masses[idx]

    R_min = scales[0]
    prefix = 1.0 / (3.0 * 2.0 ** (n + 3))

    # cumulative mass nu([0, R_i)) = sum of masses for R_k < R_i
    # For R_i = scales[i], nu([0, R_i)) = sum_{k < i} masses[k]
    cumulative = np.concatenate([[0.0], np.cumsum(masses[:-1])])
    # cumulative[i] = nu([0, R_i))

    slopes = []
    for i in range(1, len(scales)):   # R_i > R_min
        denom = scales[i] - R_min
        if denom > 0 and cumulative[i] > 0:
            slopes.append(np.sqrt(cumulative[i]) / denom)

    if L is not None and L > R_min:
        nu_full = np.sum(masses[scales < L])
        if nu_full > 0:
            slopes.append(np.sqrt(nu_full) / (L - R_min))

    if not slopes:
        # degenerate: only one scale point
        slope = 0.0
    else:
        slope = min(slopes)

    scalar_input = np.isscalar(t)
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    result = np.where(t_arr > R_min, prefix * slope * (t_arr - R_min), 0.0)
    return float(result[0]) if scalar_input else result


# ══════════════════════════════════════════════════════════════════════════
# Optimal analytic initialisation  (Theorem 5.1 of Mitra & Virk)
# ══════════════════════════════════════════════════════════════════════════

def analytic_optimal_masses(scales: np.ndarray, L: float) -> np.ndarray:
    """
    Closed-form weights that maximise the distortion constant lambda(nu)
    of Paper I, eq. (3.5), subject to sum_k w_k^2 = 1.

    With d_i := R_i - R_1 (d_1 := 0) and d_{N+1} := L - R_1, equalising
    the N active ratios sqrt(sum_{k<i} w_k^2 R_k^2) / d_i at the
    optimum and telescoping gives

        w_k^2  proportional to  (d_{k+1}^2 - d_k^2) / R_k^2
        nu({R_k}) = w_k^2 R_k^2  proportional to  d_{k+1}^2 - d_k^2

    with the compact-support parameter L entering only through the
    final term (d_{N+1} = L - R_1).

    Returns masses nu = w^2 * R^2 (not w_k), normalised so the
    corresponding w satisfies sum_k w_k^2 = 1.
    """
    scales = np.asarray(scales, dtype=float)
    R1 = scales[0]
    d = scales - R1                                  # d_1 = 0, d_2, ..., d_N
    d_next = np.concatenate([d[1:], [L - R1]])       # d_2, ..., d_{N+1}
    w2 = (d_next ** 2 - d ** 2) / scales ** 2        # proportional
    w2 = np.clip(w2, 0.0, None)                      # guard R_N = L edge case
    w2 /= w2.sum()                                   # sum w_k^2 = 1
    masses = w2 * scales ** 2                        # nu({R_k}) = w_k^2 R_k^2
    return masses


# ══════════════════════════════════════════════════════════════════════════
# Wasserstein-1 distance between scale measures  (Theorem 4, stability)
# ══════════════════════════════════════════════════════════════════════════

def w1_scale_measures(scales1: np.ndarray, masses1: np.ndarray,
                      scales2: np.ndarray, masses2: np.ndarray) -> float:
    """
    1-Wasserstein distance between two discrete measures on R^+,
    computed via the CDFs (exact for discrete measures with equal total mass
    after normalisation).

    Both measures are normalised to unit total mass before comparison.
    """
    from scipy.stats import wasserstein_distance
    # represent as weighted samples
    return wasserstein_distance(scales1, scales2,
                                u_weights=masses1 / masses1.sum(),
                                v_weights=masses2 / masses2.sum())


# ══════════════════════════════════════════════════════════════════════════
# Feasibility projection  (Section 4.4 of manuscript)
# ══════════════════════════════════════════════════════════════════════════

def project_to_feasible(scales: np.ndarray,
                         masses: np.ndarray,
                         gamma: float,
                         tau: float,
                         n: int = 1,
                         L: float | None = None,
                         max_iter: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project (scales, masses) onto the feasibility set S_{gamma,tau}:
        rho_-(tau; nu) >= gamma

    When the constraint is violated, resets to the analytic optimal
    masses which maximize rho_- for the given scale locations.
    This is correct because uniform rescaling + renormalization is a
    no-op on rho_- (it depends on cumulative mass ratios, not absolute
    values), so the only effective repair is redistribution.

    Returns (scales, masses) satisfying the constraint (best effort).
    """
    scales = scales.copy()
    masses = masses.copy()

    rho = rho_minus(tau, scales, masses, n=n, L=L)
    if rho >= gamma:
        return scales, masses

    # Constraint violated: reset to analytic optimum which maximizes
    # rho_- for these scale locations, then renormalize.
    if L is not None:
        masses = analytic_optimal_masses(scales, L)
        rho = rho_minus(tau, scales, masses, n=n, L=L)
        if rho >= gamma:
            return scales, masses

    # If analytic optimum still doesn't satisfy, we can't fix it
    # with these scale locations — return analytic optimum as best effort.
    return scales, masses
