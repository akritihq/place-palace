"""
experiments/grid_common.py
--------------------------
Shared library for the filtration-grid experiments.

Provides:

  • make_diagram(dataset_type, graph_or_cloud, filt_cfg)  — assemble a
    single diagram for one object under a pool-config spec.
  • build_merged_diagrams(dataset_type, objects, pool, …)  — compute
    atomic diagram caches then merge + top-K truncate per pool entry.
  • embed_fold(…)  — fit PLACE (fixed grid, analytic masses) on the
    training fold and embed both splits.
  • compute_delta_hat / compute_r_m — plug-in class-mean separation and
    Pinelis sample-mean concentration radius.
  • fit_linear_svm / fit_rbf_svm / fit_wlk_svm / fit_nc — the four
    classifiers.  WLK follows overnight_graph_classifiers.py; NC emits
    a per-prediction certificate indicator.
  • evaluate_all(X_tr, X_te, y_tr, y_te, sigma_quantile, inner_seed)
    — run all four classifiers and return a flat dict matching the CSV
    schema.

The per-dataset runners  (exp_grid_orbit5k.py, exp_grid_mutag.py,
exp_grid_imdbm.py)  import from here.

CSV schema (FIELDNAMES) is the single source of truth for records.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import chi2, norm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from embedding.embedding import init_from_dataset, adaptive_sigma
from utils.persistence import (
    graphs_to_persistence_cached,
    pointclouds_to_persistence_cached,
)


# ── fixed hyperparameters (paper-aligned, no optimization) ────────────────
N_SCALES       = 10              # paper default — Paper I §5 Protocol fixes N=10
N_SCALES_GRID  = [5, 10, 15, 20] # robustness sweep; runners iterate this
MAX_LANDMARKS  = 200     # cap natural lattice G_R at 200 landmarks
                         # (Paper I §2 eq. 2.1; matches the rest of the
                         # PALACE pipeline). Without the cap, social/
                         # large-graph datasets allocate (n_train, ~32M)
                         # arrays and OOM the node — see the "known
                         # pitfalls" section of the closeout notes.
TOP_K          = 50
C_GRID         = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
INNER_VAL_FRAC = 0.25    # inner split for SVM C-tuning on each training fold
NC_ALPHA       = 0.05    # confidence 1-α for NC Pinelis certificate

# Bandwidth sweep for WLK.  Prior empirical sweet spots in this repo:
# σ_q=0.25 for point clouds, σ_q=0.95 for graphs (see memory).  The grid
# below bridges both regimes; the runner's dataset-specific default seeds
# the sweep but we pick the best q via joint inner-CV with C_GRID.
WLK_Q_GRID = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
SIGMA_QUANTILE_POINTCLOUD = 0.25   # informational only
SIGMA_QUANTILE_GRAPH      = 0.95   # informational only


# CSV record schema — keep in sync with analyze_filtration_grid.py.
#
# The per-class statistics cover the Thm 3 bookkeeping the paper actually
# uses: Gaussian plug-in certificate (strictly tighter than Pinelis), the
# sample-complexity threshold m_c^* (eq. 4.4), the Prop-3 linear
# separability condition D_c < Δ/2 − r_m, and the fraction of test points
# that fall inside the certified region.
FIELDNAMES = [
    'dataset', 'seed', 'fold', 'filt', 'tau_method',
    'n_scales',                         # N from the paper's multiscale embedding
    'ell', 'R_max_tr',

    # Δ̂ and η̂
    'delta_hat', 'eta_hat',

    # Thm 3 bookkeeping
    'r_m',                    # Pinelis / Hoeffding radius (eq. 4.2)
    'r_m_gauss',              # Gaussian plug-in radius (tighter)
    'nc_certified',           # indicator  r_m < Δ̂/2
    'nc_certified_gauss',     # indicator  r̃_m < Δ̂/2
    'nc_test_coverage',       # fraction of test pts with ‖Φ - μ̂_ĉ‖ ≤ r_m
    'm_star',                 # sample-complexity threshold (eq. 4.4)
    'lin_separable',          # Prop 3: D_c_max < Δ̂/2 − r_m
    'm_min',                  # min_c m_c (training per-class count)

    # Per-class summary statistics on training embedding
    'Sigma_op_max', 'Sigma_op_mean',   # max/mean of ‖Σ̂_c‖_op
    'D_c_max',      'D_c_mean',        # max/mean of within-class radius
    'Delta_c_min',  'Delta_c_max',     # min/max of per-class gap

    # Classifier accuracies — 4 classifiers.  RBF (full-vector Gaussian on
    # standardised Φ) is Paper I's Table 3/5 column; WLK is the Φ_p-
    # structured kernel that Paper II introduces and that is arguably more
    # faithful to our embedding.  Reporting both lets the paper text pick.
    'linear_acc',  'linear_C',
    'rbf_acc',     'rbf_C',
    'wlk_acc',     'wlk_C',     'wlk_q',  'wlk_sigma',
    # RKHS class-mean separation under the WLK kernel at the best σ.
    # γ̂ = min_{c ≠ c'} ‖μ̂_c^H − μ̂_{c'}^H‖_H is the kernel analogue of Δ̂;
    # γ̂/√ℓ is the kernel analogue of η̂ used to test whether the Paper I
    # filtration-selection criterion (linear) still ranks filtrations
    # consistently under the non-linear WLK classifier.
    'wlk_gamma_hat', 'wlk_gamma_over_sqrt_ell',
    'nc_acc',

    # Wall-clock
    't_embed_s',   't_classify_s',
]


# ═══════════════════════════════════════════════════════════════════════════
# Diagram assembly
# ═══════════════════════════════════════════════════════════════════════════

def _truncate_top_k(dgm: np.ndarray, k: int = TOP_K) -> np.ndarray:
    if len(dgm) == 0:
        return np.zeros((0, 2))
    dgm = dgm[dgm[:, 1] > dgm[:, 0]]
    if len(dgm) > k:
        pers = dgm[:, 1] - dgm[:, 0]
        dgm = dgm[np.argsort(pers)[::-1][:k]]
    return dgm


def build_pointcloud_diagrams(dataset_name: str,
                              pcs: list,
                              pool: Sequence[dict]) -> Dict[str, List[np.ndarray]]:
    """Compute persistence diagrams for each atomic point-cloud filtration
    in the pool (cached), then merge dims per pool entry with top-K."""
    # compute atomic caches, one per (method, dims-needed) combination
    atomic_dims_needed: Dict[str, set] = {}
    for cfg in pool:
        method = cfg['method']
        atomic_dims_needed.setdefault(method, set()).update(cfg['dims'])

    atomic: Dict[str, list] = {}
    for method, dims_needed in atomic_dims_needed.items():
        atomic[method] = pointclouds_to_persistence_cached(
            pcs, dataset_name=dataset_name,
            dims=sorted(dims_needed), method=method, extended=False,
        )

    diagrams_per_filt: Dict[str, list] = {}
    for cfg in pool:
        method = cfg['method']
        merged = []
        for d in atomic[method]:
            parts = [d[dim] for dim in cfg['dims']
                     if dim in d and len(d[dim]) > 0]
            merged.append(_truncate_top_k(np.vstack(parts))
                          if parts else np.zeros((0, 2)))
        diagrams_per_filt[cfg['name']] = merged
    return diagrams_per_filt


def build_graph_diagrams(dataset_name: str,
                         graphs: list,
                         pool: Sequence[dict],
                         extended: bool,
                         union_h0: bool) -> Dict[str, List[np.ndarray]]:
    """Compute per-atomic-filtration caches for each filt referenced in the
    pool, then merge into pool entries (single filt or multi-filt pooled)."""
    atomic_filts: set = set()
    atomic_dims_needed: Dict[str, set] = {}
    for cfg in pool:
        for filt in cfg['filts']:
            atomic_filts.add(filt)
            atomic_dims_needed.setdefault(filt, set()).update(cfg['dims'])

    atomic: Dict[str, list] = {}
    for filt in sorted(atomic_filts):
        atomic[filt] = graphs_to_persistence_cached(
            graphs, dataset_name=dataset_name,
            filtration=filt,
            dims=sorted(atomic_dims_needed[filt]),
            extended=extended, union_h0=union_h0,
        )

    diagrams_per_filt: Dict[str, list] = {}
    for cfg in pool:
        merged = []
        for gi in range(len(graphs)):
            parts = []
            for filt in cfg['filts']:
                d = atomic[filt][gi]
                for dim in cfg['dims']:
                    if dim in d and len(d[dim]) > 0:
                        parts.append(d[dim])
            merged.append(_truncate_top_k(np.vstack(parts))
                          if parts else np.zeros((0, 2)))
        diagrams_per_filt[cfg['name']] = merged
    return diagrams_per_filt


# ═══════════════════════════════════════════════════════════════════════════
# Embedding
# ═══════════════════════════════════════════════════════════════════════════

def embed_fold(diagrams: List[np.ndarray],
               labels: np.ndarray,
               tr_idx: np.ndarray,
               te_idx: np.ndarray,
               n_classes: int,
               tau_method: str,
               n_scales: int = N_SCALES,
               max_landmarks: int = MAX_LANDMARKS,
               ) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Fit PLACE embedding on training fold, embed both splits."""
    tr_dgms = [diagrams[i] for i in tr_idx]
    te_dgms = [diagrams[i] for i in te_idx]
    tr_labels = labels[tr_idx]

    dbc = [[tr_dgms[j] for j in range(len(tr_idx)) if tr_labels[j] == c]
           for c in range(n_classes)]
    emb = init_from_dataset(
        dbc, N_scales=n_scales, n_diagram=1,
        max_landmarks=max_landmarks, tau_method=tau_method,
    )
    X_tr = emb.embed_dataset(tr_dgms)
    X_te = emb.embed_dataset(te_dgms)
    meta = {
        'ell':       int(emb.embedding_dim),
        'R_max_tr':  float(np.linalg.norm(X_tr, axis=1).max()),
        'L':         float(emb.L),
        'scales':    emb.scales.copy(),
        'masses':    emb.masses.copy(),
    }
    return X_tr, X_te, meta


# ═══════════════════════════════════════════════════════════════════════════
# Δ̂ / η̂ / r_m
# ═══════════════════════════════════════════════════════════════════════════

def _class_cov_op_norm(Y: np.ndarray, m_c: int) -> float:
    """Operator norm of the sample covariance of a centred class matrix
    Y = X_c − μ̂_c  (shape m_c × d).

    When m_c ≪ d (always true here), the (d × d) covariance matrix has
    the same top eigenvalue as the (m_c × m_c) Gram matrix
    Y Yᵀ / (m_c − 1), so we avoid forming the d × d covariance.
    """
    if m_c < 2:
        return 0.0
    G = (Y @ Y.T) / (m_c - 1)
    try:
        return float(np.linalg.eigvalsh(G).max())
    except Exception:
        return float('nan')


def compute_class_stats(X: np.ndarray, y: np.ndarray,
                        n_classes: int) -> dict:
    """Per-class summary statistics on an embedded training set.

    Returns
    -------
    means   : (n_classes, d) array of empirical class means μ̂_c
    counts  : (n_classes,)   per-class sample counts m_c
    sigma_op: (n_classes,)   per-class covariance operator norms ‖Σ̂_c‖_op
    D_c     : (n_classes,)   within-class radius max_{A: y=c} ‖Φ(A) − μ̂_c‖
    Delta_c : (n_classes,)   per-class gap min_{c'≠c} ‖μ̂_c − μ̂_{c'}‖
    """
    d = X.shape[1]
    means    = np.zeros((n_classes, d))
    counts   = np.zeros(n_classes, dtype=int)
    sigma_op = np.zeros(n_classes)
    D_c      = np.zeros(n_classes)

    for c in range(n_classes):
        mask = (y == c)
        X_c = X[mask]
        m_c = int(X_c.shape[0])
        counts[c] = m_c
        if m_c == 0:
            continue
        means[c]   = X_c.mean(axis=0)
        Y_c        = X_c - means[c]
        D_c[c]     = float(np.linalg.norm(Y_c, axis=1).max())
        sigma_op[c] = _class_cov_op_norm(Y_c, m_c)

    Delta_c = np.full(n_classes, np.inf)
    for c in range(n_classes):
        if counts[c] == 0:
            continue
        for cp in range(n_classes):
            if cp == c or counts[cp] == 0:
                continue
            dist = float(np.linalg.norm(means[c] - means[cp]))
            if dist < Delta_c[c]:
                Delta_c[c] = dist
    Delta_c[~np.isfinite(Delta_c)] = 0.0

    return {
        'means':    means,
        'counts':   counts,
        'sigma_op': sigma_op,
        'D_c':      D_c,
        'Delta_c':  Delta_c,
    }


def compute_delta_hat(X: np.ndarray, y: np.ndarray,
                      n_classes: int) -> Tuple[float, np.ndarray]:
    """Δ̂ = min_{c≠c'} ||μ̂_c − μ̂_{c'}||  on the rows of X.
    (Kept as a thin shim over compute_class_stats for back-compat.)"""
    stats = compute_class_stats(X, y, n_classes)
    valid = stats['counts'] > 0
    means = stats['means'][valid]
    if len(means) < 2:
        return 0.0, stats['means']
    d = pdist(means, metric='euclidean')
    nonzero = d[d > 0]
    return (float(nonzero.min()) if len(nonzero) else float(d.min())), stats['means']


def compute_r_m(R_max: float, y_tr: np.ndarray, n_classes: int,
                alpha: float = NC_ALPHA) -> float:
    """Pinelis sample-mean concentration radius, Paper II radius (i) (L3217, eq:rm-pinelis-pii):
        r_m = 2 R √(2 log(2k/δ) / m_min),
    the factor 2 because the centred summands are bounded by 2R.  Until 2026-11-08 this
    returned R √(2L/m) (uncentred); firing was 0/11 under either form."""
    _, counts = np.unique(y_tr, return_counts=True)
    m_min = int(counts.min()) if len(counts) else 1
    return float(2.0 * R_max * math.sqrt(2.0 * math.log(2.0 * n_classes / alpha) / max(1, m_min)))


def compute_r_m_gauss(sigma_op_max: float, m_min: int, n_classes: int,
                      ell: int, alpha: float = NC_ALPHA) -> float:
    """Gaussian plug-in radius (Thm 5.1(ii), Paper I):
        r̃_m = sqrt(‖Σ̂_c‖_op · χ²_{ℓ, α/k} / m_min).

    Bounds the multivariate-Gaussian norm ‖μ̂_c − μ_c‖ by
    ‖Σ_c‖_op · χ²_ℓ (weighted-χ² envelope of the squared norm).
    Conservative when Σ_c is low rank; firing requires
    ‖Σ_c‖_op · ℓ ≲ Δ_c² m_c / (4 χ²_{ℓ,α/k})."""
    if sigma_op_max <= 0 or m_min <= 0 or ell <= 0:
        return float('inf')
    q = float(chi2.ppf(1.0 - alpha / n_classes, df=int(ell)))
    return float(math.sqrt(sigma_op_max * q / m_min))


def compute_m_star(sigma_op_max: float, delta: float, ell: int,
                   n_classes: int = 2, alpha: float = NC_ALPHA) -> int:
    """Sample-complexity threshold for the Gaussian plug-in certificate
    (Thm 5.1(ii), Paper I) under Pramita's chi-squared form:
        m_c^* = ⌈4 ‖Σ_c‖_op · χ²_{ℓ, α/k} / Δ_c^2⌉.
    The certificate r̃_m < Δ̂/2 fires once m_c ≥ m_c^* for every c."""
    if sigma_op_max <= 0 or delta <= 0 or ell <= 0:
        return 0
    q = float(chi2.ppf(1.0 - alpha / n_classes, df=int(ell)))
    return int(math.ceil(4.0 * sigma_op_max * q / (delta ** 2)))


# ═══════════════════════════════════════════════════════════════════════════
# Classifiers
# ═══════════════════════════════════════════════════════════════════════════

def _inner_split(y: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified inner split for C-tuning. Falls back if class counts are too
    small for stratification (returns tr-tr for val if so)."""
    _, counts = np.unique(y, return_counts=True)
    if counts.min() < 2 or len(y) < 6:
        idx = np.arange(len(y))
        return idx, idx
    sss = StratifiedShuffleSplit(n_splits=1, test_size=INNER_VAL_FRAC,
                                 random_state=seed)
    (tr, val), = sss.split(np.zeros(len(y)), y)
    return tr, val


def fit_linear_svm(X_tr: np.ndarray, y_tr: np.ndarray,
                   X_te: np.ndarray, y_te: np.ndarray,
                   inner_seed: int) -> Tuple[float, float]:
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(X_tr)
    Xte_s = scaler.transform(X_te)

    inner_tr, inner_val = _inner_split(y_tr, inner_seed)
    best_C, best_val = C_GRID[0], -1.0
    for C in C_GRID:
        try:
            svm = LinearSVC(C=C, dual='auto', max_iter=10000)
            svm.fit(Xtr_s[inner_tr], y_tr[inner_tr])
            val = svm.score(Xtr_s[inner_val], y_tr[inner_val])
        except Exception:
            continue
        if val > best_val:
            best_val, best_C = val, C

    try:
        svm = LinearSVC(C=best_C, dual='auto', max_iter=10000).fit(Xtr_s, y_tr)
        acc = float(svm.score(Xte_s, y_te))
    except Exception:
        acc = float('nan')
    return acc, float(best_C)


def fit_rbf_svm(X_tr: np.ndarray, y_tr: np.ndarray,
                X_te: np.ndarray, y_te: np.ndarray,
                inner_seed: int) -> Tuple[float, float]:
    """Full-vector Gaussian RBF SVM on the standardised embedding.  This
    is the kernel reported in Paper I Tables 3/5 (e.g. Orbit5k RBF 89.0%),
    distinct from WLK: the RBF treats Φ(A) as an unstructured ℓ-vector,
    whereas WLK sums per-coordinate RBFs on Φ_p values."""
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(X_tr)
    Xte_s = scaler.transform(X_te)

    inner_tr, inner_val = _inner_split(y_tr, inner_seed)
    best_C, best_val = C_GRID[0], -1.0
    for C in C_GRID:
        try:
            svm = SVC(C=C, kernel='rbf', gamma='scale')
            svm.fit(Xtr_s[inner_tr], y_tr[inner_tr])
            val = svm.score(Xtr_s[inner_val], y_tr[inner_val])
        except Exception:
            continue
        if val > best_val:
            best_val, best_C = val, C

    try:
        svm = SVC(C=best_C, kernel='rbf', gamma='scale').fit(Xtr_s, y_tr)
        acc = float(svm.score(Xte_s, y_te))
    except Exception:
        acc = float('nan')
    return acc, float(best_C)


def compute_gamma_hat_from_gram(G: np.ndarray, y: np.ndarray,
                                n_classes: int) -> float:
    """RKHS class-mean separation under a kernel with Gram `G` on training
    points: γ̂ = min_{c≠c'} ‖μ̂_c^H − μ̂_{c'}^H‖_H, where
    μ̂_c^H = m_c^{-1} Σ_{i: y_i=c} Φ^H(x_i).

    Uses the identity
        ‖μ̂_a − μ̂_b‖² = B_{aa} + B_{bb} − 2 B_{ab},
        B_{ab} := (m_a m_b)^{-1} Σ_{i ∈ a, j ∈ b} G_{ij},
    which is O(n²) given the precomputed Gram.  Clipping the inner sum at
    zero handles small negative values from floating-point error.
    """
    present = [c for c in range(n_classes) if np.any(y == c)]
    if len(present) < 2:
        return 0.0
    masks = {c: np.flatnonzero(y == c) for c in present}
    block = np.zeros((len(present), len(present)))
    for ia, a in enumerate(present):
        for ib, b in enumerate(present):
            block[ia, ib] = float(G[np.ix_(masks[a], masks[b])].mean())
    best = math.inf
    for ia in range(len(present)):
        for ib in range(ia + 1, len(present)):
            v = block[ia, ia] + block[ib, ib] - 2.0 * block[ia, ib]
            if v < best:
                best = v
    return float(math.sqrt(max(best, 0.0)))


def wlk_gram(X1: np.ndarray, X2: np.ndarray, sigma: float) -> np.ndarray:
    """Weighted Landmark Kernel: sum over coordinates of an RBF on each
    dimension (matches overnight_graph_classifiers.py and
    exp_orbit5k_filt_sweep.py).

        K(x, y) = Σ_ℓ exp(−(x_ℓ − y_ℓ)² / (2σ²))

    Memory-aware: chunks the coordinate axis."""
    n1, ell = X1.shape
    n2 = X2.shape[0]
    inv2s2 = 1.0 / (2.0 * sigma ** 2)
    K = np.zeros((n1, n2))
    # cap memory at ~4 GB per chunk
    chunk = max(1, min(100, int(4e9 / max(1, n1 * n2 * 8))))
    for start in range(0, ell, chunk):
        end = min(start + chunk, ell)
        diff = X1[:, start:end, None] - X2[:, start:end, None].transpose(2, 1, 0)
        K += np.exp(-diff ** 2 * inv2s2).sum(axis=1)
        del diff
    return K


def fit_wlk_svm(X_tr: np.ndarray, y_tr: np.ndarray,
                X_te: np.ndarray, y_te: np.ndarray,
                sigma_quantiles,
                inner_seed: int,
                n_classes: int = 0,
                ) -> Tuple[float, float, float, float, float]:
    """SVM with precomputed WLK Gram matrix, with (q, C) jointly selected
    by inner CV.

    Parameters
    ----------
    sigma_quantiles : iterable of quantiles to try.  For each q we compute
                      σ_q = q-quantile of pairwise L2 distances on a 200-
                      point subsample of X_tr, build the Gram once, then
                      sweep C via the shared inner split.  Best (q, C) by
                      inner-val accuracy is retrained on the full fold.
    n_classes       : number of label classes — required to compute γ̂ from
                      the best-σ Gram.  Defaults to 0, in which case γ̂ is
                      inferred from the labels present in y_tr.

    Returns  (test_acc, best_C, best_q, best_sigma, gamma_hat).
    gamma_hat is the RKHS class-mean separation under the WLK kernel at
    the selected σ (see `compute_gamma_hat_from_gram`).
    """
    inner_tr, inner_val = _inner_split(y_tr, inner_seed)

    best = {'val': -1.0, 'q': float('nan'),
            'sigma': float('nan'), 'C': float('nan'),
            'G_tr': None, 'G_te': None}

    for q in sigma_quantiles:
        try:
            sigma = adaptive_sigma(X_tr, quantile=float(q))
        except Exception:
            continue
        if not np.isfinite(sigma) or sigma <= 0:
            continue

        G_tr = wlk_gram(X_tr, X_tr, sigma=sigma)
        G_te = wlk_gram(X_te, X_tr, sigma=sigma)

        for C in C_GRID:
            try:
                svm = SVC(C=C, kernel='precomputed')
                svm.fit(G_tr[np.ix_(inner_tr, inner_tr)], y_tr[inner_tr])
                val = svm.score(G_tr[np.ix_(inner_val, inner_tr)],
                                y_tr[inner_val])
            except Exception:
                continue
            if val > best['val']:
                best.update({'val': float(val), 'q': float(q),
                             'sigma': float(sigma), 'C': float(C),
                             'G_tr': G_tr, 'G_te': G_te})

    if best['G_tr'] is None:
        return (float('nan'), float('nan'), float('nan'),
                float('nan'), float('nan'))

    try:
        svm = SVC(C=best['C'], kernel='precomputed').fit(best['G_tr'], y_tr)
        acc = float(svm.score(best['G_te'], y_te))
    except Exception:
        acc = float('nan')

    k_for_gamma = n_classes if n_classes > 0 else int(y_tr.max()) + 1
    gamma_hat = compute_gamma_hat_from_gram(best['G_tr'], y_tr, k_for_gamma)
    return acc, best['C'], best['q'], best['sigma'], gamma_hat


def fit_nc(X_tr: np.ndarray, y_tr: np.ndarray,
           X_te: np.ndarray, y_te: np.ndarray,
           delta_hat: float, r_m: float) -> Tuple[float, bool, float]:
    """Nearest-centroid classifier on the raw embedding.

    Returns (test_acc, nc_certified, nc_test_coverage) where
      • nc_certified  = 1[r_m < Δ̂/2]  (global certificate)
      • nc_test_coverage = fraction of test points whose distance to the
                           predicted centroid is ≤ r_m, i.e. Paper I's
                           "certified fraction" at test time.
    """
    classes = np.array(sorted(np.unique(y_tr).tolist()))
    centroids = np.array([
        X_tr[y_tr == c].mean(axis=0) if (y_tr == c).any()
        else np.zeros(X_tr.shape[1])
        for c in classes
    ])
    # test-time squared distances to every centroid
    dists_sq = np.sum(
        (X_te[:, None, :] - centroids[None, :, :]) ** 2, axis=2,
    )
    argmin = np.argmin(dists_sq, axis=1)
    preds = classes[argmin]
    acc = float(np.mean(preds == y_te))
    certified = bool(np.isfinite(r_m) and np.isfinite(delta_hat)
                     and r_m < 0.5 * delta_hat)

    if np.isfinite(r_m) and len(X_te):
        pred_dist = np.sqrt(dists_sq[np.arange(len(preds)), argmin])
        coverage = float(np.mean(pred_dist <= r_m))
    else:
        coverage = 0.0
    return acc, certified, coverage


# ═══════════════════════════════════════════════════════════════════════════
# Full per-cell evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_all(X_tr: np.ndarray, y_tr: np.ndarray,
                 X_te: np.ndarray, y_te: np.ndarray,
                 n_classes: int,
                 sigma_quantiles=None,
                 inner_seed: int = 0,
                 R_max_tr: float = 0.0,
                 ) -> dict:
    """Run all three classifiers on the embedded training/test splits and
    compute every per-cell statistic the paper references.  Returns a dict
    matching the CSV-schema tail (from `delta_hat` onward)."""
    t0 = time.time()
    ell = X_tr.shape[1]
    if sigma_quantiles is None:
        sigma_quantiles = WLK_Q_GRID

    # ── Per-class panel (means, counts, Σ̂_c op-norms, D_c, Δ_c) ──────
    stats = compute_class_stats(X_tr, y_tr, n_classes)
    valid = stats['counts'] > 0

    # Global Δ̂ and η̂
    means = stats['means'][valid]
    if len(means) < 2:
        delta_hat = 0.0
    else:
        d_pair = pdist(means, metric='euclidean')
        nonzero = d_pair[d_pair > 0]
        delta_hat = float(nonzero.min()) if len(nonzero) else float(d_pair.min())
    eta_hat = delta_hat / math.sqrt(ell) if ell > 0 else 0.0

    # Worst-case / average summaries
    counts_v   = stats['counts'][valid]
    sigma_v    = stats['sigma_op'][valid]
    D_v        = stats['D_c'][valid]
    Delta_c_v  = stats['Delta_c'][valid]
    m_min         = int(counts_v.min())       if len(counts_v) else 0
    sigma_op_max  = float(sigma_v.max())      if len(sigma_v)  else 0.0
    sigma_op_mean = float(sigma_v.mean())     if len(sigma_v)  else 0.0
    D_c_max       = float(D_v.max())          if len(D_v)      else 0.0
    D_c_mean      = float(D_v.mean())         if len(D_v)      else 0.0
    Delta_c_min   = float(Delta_c_v.min())    if len(Delta_c_v) else 0.0
    Delta_c_max   = float(Delta_c_v.max())    if len(Delta_c_v) else 0.0

    # Thm 3 radii + certificates
    r_m       = compute_r_m(R_max_tr, y_tr, n_classes)
    r_m_gauss = compute_r_m_gauss(sigma_op_max, m_min, n_classes, ell)
    nc_cert_gauss = bool(np.isfinite(r_m_gauss) and np.isfinite(delta_hat)
                         and r_m_gauss < 0.5 * delta_hat)

    # Sample-complexity floor and Prop-3 linear separability
    m_star = compute_m_star(sigma_op_max, delta_hat, ell, n_classes)
    lin_separable = bool(np.isfinite(r_m) and np.isfinite(delta_hat)
                         and D_c_max < 0.5 * delta_hat - r_m)

    # ── Classifiers (4: Linear, RBF, WLK on Φ_p, Nearest Centroid) ─────
    lin_acc, lin_C = fit_linear_svm(X_tr, y_tr, X_te, y_te, inner_seed)
    rbf_acc, rbf_C = fit_rbf_svm(X_tr, y_tr, X_te, y_te, inner_seed + 1)
    wlk_acc, wlk_C, wlk_q, wlk_sigma, wlk_gamma_hat = fit_wlk_svm(
        X_tr, y_tr, X_te, y_te,
        sigma_quantiles=sigma_quantiles,
        inner_seed=inner_seed + 2,
        n_classes=n_classes,
    )
    wlk_gamma_over_sqrt_ell = (
        wlk_gamma_hat / math.sqrt(ell)
        if ell > 0 and np.isfinite(wlk_gamma_hat) else float('nan')
    )
    nc_acc, nc_cert, nc_cov = fit_nc(
        X_tr, y_tr, X_te, y_te,
        delta_hat=delta_hat, r_m=r_m,
    )
    t_classify_s = time.time() - t0

    return {
        'ell':                 ell,
        'delta_hat':           delta_hat,
        'eta_hat':             eta_hat,

        # Thm 3 bookkeeping
        'r_m':                 r_m,
        'r_m_gauss':           r_m_gauss,
        'nc_certified':        int(nc_cert),
        'nc_certified_gauss':  int(nc_cert_gauss),
        'nc_test_coverage':    float(nc_cov),
        'm_star':              int(m_star),
        'lin_separable':       int(lin_separable),
        'm_min':               int(m_min),

        # Per-class summary stats
        'Sigma_op_max':        sigma_op_max,
        'Sigma_op_mean':       sigma_op_mean,
        'D_c_max':             D_c_max,
        'D_c_mean':            D_c_mean,
        'Delta_c_min':         Delta_c_min,
        'Delta_c_max':         Delta_c_max,

        # Classifier accuracies
        'linear_acc':    lin_acc,  'linear_C':   lin_C,
        'rbf_acc':       rbf_acc,  'rbf_C':      rbf_C,
        'wlk_acc':       wlk_acc,  'wlk_C':      wlk_C,
        'wlk_q':         wlk_q,    'wlk_sigma':  wlk_sigma,
        'wlk_gamma_hat': wlk_gamma_hat,
        'wlk_gamma_over_sqrt_ell': wlk_gamma_over_sqrt_ell,
        'nc_acc':        nc_acc,

        # Wall-clock
        't_classify_s':  float(t_classify_s),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Checkpointing
# ═══════════════════════════════════════════════════════════════════════════

def cache_path_for(root: Path, dataset: str,
                   seed: int, fold: int, filt: str, tau: str,
                   n_scales: int = N_SCALES) -> Path:
    """Per-(seed, fold, filt, tau, N) embedding checkpoint."""
    d = root / dataset / f'seed{seed}_fold{fold}'
    d.mkdir(parents=True, exist_ok=True)
    return d / f'{filt}__{tau}__N{n_scales}.npz'


def save_embedding_cache(path: Path, X_tr, X_te, y_tr, y_te,
                         tr_idx, te_idx, meta: dict) -> None:
    np.savez_compressed(
        path,
        X_tr=X_tr, X_te=X_te,
        y_tr=y_tr, y_te=y_te,
        tr_idx=tr_idx, te_idx=te_idx,
        ell=meta['ell'], R_max_tr=meta['R_max_tr'],
        L=meta['L'], scales=meta['scales'], masses=meta['masses'],
    )
