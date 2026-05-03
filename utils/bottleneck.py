"""
utils/bottleneck.py
-------------------
Bottleneck distance on single-point persistence diagrams D_1.

Shared by both uniform (embedding.py) and non-uniform (nonuniform.py)
embedding modules.

    d_B^1(p, q) = min( d_inf(p, q),  max( d_inf(p, Delta), d_inf(q, Delta) ) )

where d_inf(p, Delta) = (d - b) / 2  for p = (b, d).
The via-diagonal route matches both p and q to the diagonal; the
bottleneck cost is the max of the two diagonal-distances, not the sum.
"""

from __future__ import annotations
import numpy as np


def d_B1_batch(positions: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Bottleneck distance d_B^1(p_k, a_i) for landmarks p_k and diagram points a_i.

    Parameters
    ----------
    positions : (K, 2) array — landmark positions (b, d) in the half-plane
    pts       : (m, 2) array — diagram points (birth, death)

    Returns
    -------
    (K, m) array of distances
    """
    K = len(positions)
    m = len(pts)
    if m == 0:
        return np.zeros((K, 0))
    # d_inf(p_k, a_i) = max(|b_k - b_i|, |d_k - d_i|)
    d_inf = np.max(
        np.abs(positions[:, np.newaxis, :] - pts[np.newaxis, :, :]),
        axis=2
    )                                                              # (K, m)
    # max( d_inf(p_k, Delta), d_inf(a_i, Delta) )
    pers_lm = (positions[:, 1] - positions[:, 0]) / 2.0           # (K,)
    pers_pts = (pts[:, 1] - pts[:, 0]) / 2.0                      # (m,)
    d_via = np.maximum(pers_lm[:, np.newaxis], pers_pts[np.newaxis, :])  # (K, m)
    return np.minimum(d_inf, d_via)


def d_B1_to_diagonal(pts: np.ndarray) -> np.ndarray:
    """
    Bottleneck distance from each diagram point to the diagonal Delta.

        d_B^1(a_i, Delta) = (d_i - b_i) / 2

    Parameters
    ----------
    pts : (m, 2) array of (birth, death) pairs

    Returns
    -------
    (m,) array of half-persistences
    """
    if len(pts) == 0:
        return np.array([])
    return (pts[:, 1] - pts[:, 0]) / 2.0
