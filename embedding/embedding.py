"""
embedding/embedding.py
----------------------
Multiscale persistence landmark embedding (Sections 2.2 and 3.1).

Grid (Mitra & Virk 2024, Definition 3.1):
    G_R = { (mR, nR) | m in {1,3,5,...}, n in {4,6,8,...}, n >= m+3 }
    G_R^+ = G_R ∪ {Delta}

Coordinate function (hat / tent) on D_1:
    phi_{R,p}(x) = max( 3R/2 - d_B^1(p, x), 0 )

where d_B^1 is the bottleneck distance on single-point diagrams:
    d_B^1(p, q) = min( d_inf(p,q), d_inf(p,Delta) + d_inf(q,Delta) )
    d_inf(p, Delta) = (d - b) / 2  for p = (b,d)

Each coordinate of the multiscale embedding is (Section 3.1, eq. 3.2):

    Phi_p(A; nu) = w_k * 2^{-n-1/2} * int_{D_1} phi_{R_k,p} d mu_A

where mu_A = sum_i delta_{a_i} is the empirical measure of diagram A,
and the integral reduces to sum_i phi_{R_k,p}(a_i) by eq. 2.1.

The scale measure nu is parameterised by:
    scales : (N,) array  — support points R_1 < ... < R_N
    masses : (N,) array  — nu({R_k}) = w_k^2 * R_k^2
so that w_k = sqrt(masses[k]) / scales[k].

The embedding lives in R^{sum_k |G_{R_k}^+|}.

Usage
-----
    emb = MultiscaleEmbedding(scales, masses, L)
    vec = emb.embed(diagram_pts)          # (d,) array
    mat = emb.embed_dataset(list_of_dgms) # (N_graphs, d) array
"""

from __future__ import annotations
import numpy as np
from typing import List

from utils.rho_minus import rho_minus, analytic_optimal_masses
from utils.bottleneck import d_B1_batch


# ══════════════════════════════════════════════════════════════════════════
# Grid construction
# ══════════════════════════════════════════════════════════════════════════

def make_landmark_grid(R: float, L: float,
                       max_landmarks: int = 0,
                       seed: int = 0) -> np.ndarray:
    """
    Build G_R restricted to the frame [0, L]^2.

    Parameters
    ----------
    R  : scale
    L  : frame size
    max_landmarks : if > 0, randomly subsample the grid to this size.
                    This preserves scale resolution (R stays small) while
                    controlling embedding dimension.
    seed : random seed for subsampling

    Returns
    -------
    grid : (K, 2) array of landmark positions (b, d).
           Does NOT include Delta (handled separately).
    """
    landmarks = []
    m = 1
    while m * R <= L:
        n = max(4, m + 3)
        while n * R <= L:
            landmarks.append([m * R, n * R])
            n += 2
        m += 2

    if not landmarks:
        return np.zeros((0, 2))

    grid = np.array(landmarks)
    if max_landmarks > 0 and len(grid) > max_landmarks:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(grid), max_landmarks, replace=False)
        grid = grid[np.sort(idx)]

    return grid


# ══════════════════════════════════════════════════════════════════════════
# Landmark coordinates (Section 3.1)
# ══════════════════════════════════════════════════════════════════════════

def landmark_coords(diagram_pts: np.ndarray,
                    grid: np.ndarray,
                    R: float) -> np.ndarray:
    """
    Landmark coordinates for the kernel (Section 3.1, eq. 3.2).

    Computes the un-weighted coordinate vector over G_R^+ = G_R ∪ {Δ},
    whose p-th entry is

        int_{D_1} phi_{R,p} d mu_A  =  sum_i phi_{R,p}(a_i)

    where phi_{R,p}(a_i) = max(3R/2 - d_B^1(p, a_i), 0)  and
    mu_A = sum_i delta_{a_i}  is the empirical measure of diagram A
    (eq. 2.1).  The block weight  w_k * 2^{-n-1/2}  is applied
    externally in MultiscaleEmbedding.embed().

    Parameters
    ----------
    diagram_pts  : (m, 2) array of birth-death pairs
    grid         : (K, 2) landmark positions from make_landmark_grid
    R            : scale

    Returns
    -------
    (K + 1,) array, non-negative.  First entry is the Delta coordinate.
    """
    threshold = 1.5 * R

    # Delta coordinate: sum_i max(3R/2 - pers(a_i), 0)
    # where pers(a_i) = (d_i - b_i) / 2 = d_B^1(Delta, a_i)
    # Empty diagram (no points) → empty sum = 0.
    if len(diagram_pts) == 0:
        return np.zeros(len(grid) + 1)

    pers = (diagram_pts[:, 1] - diagram_pts[:, 0]) / 2.0
    delta_coord = float(np.sum(np.maximum(threshold - pers, 0.0)))

    if len(grid) == 0:
        return np.array([delta_coord])

    # d_B^1(grid[k], a_i) for all k, i  —  shape (K, m)
    d_B1 = d_B1_batch(grid, diagram_pts)                                 # (K, m)

    # sum_i max(3R/2 - d_B^1(p_k, a_i), 0)  for each landmark p_k
    hat_vals = np.maximum(threshold - d_B1, 0.0)                         # (K, m)
    vals = hat_vals.sum(axis=1)                                          # (K,)

    return np.concatenate(([delta_coord], vals))


# ══════════════════════════════════════════════════════════════════════════
# Multiscale embedding
# ══════════════════════════════════════════════════════════════════════════

class MultiscaleEmbedding:
    """
    Persistent landmark embedding with scale measure nu.

    Parameters
    ----------
    scales  : (N,) array of scale values R_1 < ... < R_N
    masses  : (N,) array nu({R_k}) = w_k^2 * R_k^2
    L       : frame size (diagrams clipped to [0,L]^2)
    n_diagram : diagram dimension n (affects Lipschitz normalisation 2^{-n-1/2})
    """

    def __init__(self,
                 scales: np.ndarray,
                 masses: np.ndarray,
                 L: float,
                 n_diagram: int = 1,
                 max_landmarks: int = 200):
        self.scales  = np.asarray(scales, dtype=float)
        self.masses  = np.asarray(masses, dtype=float)
        self.L       = float(L)
        self.n       = n_diagram
        self.max_landmarks = max_landmarks

        # sort by scale
        idx = np.argsort(self.scales)
        self.scales  = self.scales[idx]
        self.masses  = self.masses[idx]

        # weights w_k = sqrt(nu({R_k})) / R_k
        self.w = np.sqrt(self.masses) / self.scales  # shape (N,)

        # normalisation factor per block
        norm = 2.0 ** (-n_diagram - 0.5)
        self.block_weights = self.w * norm           # shape (N,)

        # precompute landmark grids for each scale
        # Grid subsampling controls dimension without shifting R_min up.
        self.grids: List[np.ndarray] = [
            make_landmark_grid(R, L, max_landmarks=max_landmarks)
            for R in self.scales
        ]
        self.block_dims: List[int] = [len(g) + 1 for g in self.grids]
        self.embedding_dim: int = sum(self.block_dims)

    # ── core ──────────────────────────────────────────────────────────────

    def embed(self, diagram_pts: np.ndarray) -> np.ndarray:
        """
        Embed a single persistence diagram.

        Parameters
        ----------
        diagram_pts : (m, 2) array of (birth, death) pairs, or (0, 2).
                      Points with death <= birth are ignored.

        Returns
        -------
        (embedding_dim,) array of kernel coordinates Phi_p(A; nu).
        """
        # clip to frame [0, L]^2
        if len(diagram_pts) > 0:
            pts = diagram_pts[
                (diagram_pts[:, 0] >= 0) &
                (diagram_pts[:, 1] <= self.L) &
                (diagram_pts[:, 1] > diagram_pts[:, 0])
            ]
        else:
            pts = np.zeros((0, 2))

        blocks = []
        for R, grid, bw in zip(self.scales, self.grids, self.block_weights):
            blocks.append(bw * landmark_coords(pts, grid, R))

        return np.concatenate(blocks)

    def embed_dataset(self,
                      diagrams: List[np.ndarray],
                      verbose: bool = False) -> np.ndarray:
        """
        Embed a list of diagrams.

        Parameters
        ----------
        diagrams : list of (m_i, 2) arrays
        verbose  : print progress

        Returns
        -------
        (len(diagrams), embedding_dim) array
        """
        N = len(diagrams)
        X = np.zeros((N, self.embedding_dim))
        for i, dgm in enumerate(diagrams):
            if verbose and (i + 1) % 100 == 0:
                print(f"  Embedding {i+1}/{N}")
            X[i] = self.embed(dgm)
        return X

    # ── distortion certificate ────────────────────────────────────────────

    def rho_minus(self, t: float | np.ndarray) -> float | np.ndarray:
        """Evaluate rho_-(t; nu) for this embedding."""
        from utils.rho_minus import rho_minus as _rho
        return _rho(t, self.scales, self.masses, n=self.n, L=self.L)

    # ── parameter updates (for outer optimisation loop) ───────────────────

    def update_nu(self, new_scales: np.ndarray, new_masses: np.ndarray):
        """In-place update of the scale measure (recomputes grids)."""
        self.__init__(new_scales, new_masses, self.L, self.n)

    def __repr__(self):
        return (f"MultiscaleEmbedding("
                f"N_scales={len(self.scales)}, "
                f"dim={self.embedding_dim}, "
                f"L={self.L})")


# ══════════════════════════════════════════════════════════════════════════
# Kernel bandwidth selection
# ══════════════════════════════════════════════════════════════════════════

def adaptive_sigma(X: np.ndarray, quantile: float = 0.75) -> float:
    """
    Choose kernel bandwidth sigma from embedded training data.

    Uses the median heuristic: a quantile of pairwise Euclidean
    distances.  This is more robust than per-coordinate statistics
    and less sensitive to the choice of quantile.

    Parameters
    ----------
    X        : (n, d) embedding matrix from embed_dataset
    quantile : quantile of pairwise L2 distances (default 0.75)

    Returns
    -------
    sigma : positive scalar
    """
    from scipy.spatial.distance import pdist

    n = X.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 samples to estimate sigma")

    # subsample for speed on large datasets
    rng = np.random.default_rng(0)
    idx = rng.choice(n, min(n, 200), replace=False)
    Xs = X[idx]

    dists = pdist(Xs, 'euclidean')
    nonzero = dists[dists > 0]

    if len(nonzero) == 0:
        raise ValueError("All pairwise distances are zero; "
                         "embedding is degenerate")

    return float(np.quantile(nonzero, quantile))


# ══════════════════════════════════════════════════════════════════════════
# Data-adaptive tau* estimation
# ══════════════════════════════════════════════════════════════════════════

def estimate_tau_star(diagrams_by_class: List[List[np.ndarray]],
                      k_per_class: int = 30,
                      seed: int = 0) -> float:
    """
    Estimate tau* as the crossing point of within-class and between-class
    bottleneck distance distributions, using a stratified subsample.

    Falls back to the median half-persistence proxy if the bottleneck
    computation is too expensive (diagrams with > 2000 points) or if
    no crossing is found.

    Parameters
    ----------
    diagrams_by_class : list of lists; diagrams_by_class[t] = diagrams for class t
    k_per_class       : subsample size per class (default 30)
    seed              : random seed

    Returns
    -------
    tau_star : float, estimated separation scale
    """
    rng = np.random.default_rng(seed)
    all_diagrams = [d for cls_dgms in diagrams_by_class for d in cls_dgms]

    # Median proxy as fallback
    arrays = [(d[:, 1] - d[:, 0]) / 2.0 for d in all_diagrams if len(d) > 0]
    if not arrays:
        return 0.01
    pers_values = np.concatenate(arrays)
    proxy = float(np.median(pers_values))

    # Check if bottleneck is feasible (max diagram size)
    max_pts = max((len(d) for d in all_diagrams), default=0)
    if max_pts > 2000:
        return proxy  # too expensive

    try:
        import gudhi
    except ImportError:
        return proxy

    # Stratified subsample
    n_classes = len(diagrams_by_class)
    sub_dgms, sub_labels = [], []
    for c, cls_dgms in enumerate(diagrams_by_class):
        k = min(k_per_class, len(cls_dgms))
        idx = rng.choice(len(cls_dgms), k, replace=False)
        for i in idx:
            sub_dgms.append(cls_dgms[i])
            sub_labels.append(c)
    sub_labels = np.array(sub_labels)

    # Pairwise bottleneck
    n = len(sub_dgms)
    within, between = [], []
    for i in range(n):
        for j in range(i + 1, n):
            dist = gudhi.bottleneck_distance(sub_dgms[i], sub_dgms[j])
            if sub_labels[i] == sub_labels[j]:
                within.append(dist)
            else:
                between.append(dist)

    within, between = np.array(within), np.array(between)
    if len(within) < 10 or len(between) < 10:
        return proxy

    # Find crossing point of KDEs
    from scipy.stats import gaussian_kde
    grid = np.linspace(0, max(within.max(), between.max()), 500)
    try:
        kde_w = gaussian_kde(within)(grid)
        kde_b = gaussian_kde(between)(grid)
        cross_idx = np.where(np.diff(np.sign(kde_w - kde_b)))[0]
        if len(cross_idx) > 0:
            return float(grid[cross_idx[0]])
    except Exception:
        pass

    return proxy


# ══════════════════════════════════════════════════════════════════════════
# Factory: initialise from dataset statistics
# ══════════════════════════════════════════════════════════════════════════

def init_from_dataset(diagrams_by_class: List[List[np.ndarray]],
                      N_scales: int = 5,
                      L: float | None = None,
                      tau_star: float | None = None,
                      n_diagram: int = 1,
                      max_landmarks: int = 200,
                      tau_method: str = "auto") -> MultiscaleEmbedding:
    """
    Initialise a MultiscaleEmbedding by placing N_scales support points
    at quantiles of the persistence distribution.

    Parameters
    ----------
    diagrams_by_class : list of lists; diagrams_by_class[t] = diagrams for class t
    N_scales          : number of support points
    L                 : frame size (auto-detected if None)
    tau_star          : scale center (if None, estimated from data)
    n_diagram         : diagram dimension (affects 2^{-n-1/2} normalisation)
    max_landmarks     : budget per scale (grid subsampled if exceeded)
    tau_method        : "auto" (subsampled crossing, fallback proxy),
                        "crossing" (subsampled crossing only),
                        "proxy" (median half-persistence)

    Returns
    -------
    MultiscaleEmbedding with analytic-optimal masses
    """
    # flatten
    all_diagrams = [d for cls_dgms in diagrams_by_class for d in cls_dgms]

    # auto-detect L from maximum death value
    if L is None:
        L = max(
            (d[:, 1].max() for d in all_diagrams if len(d) > 0),
            default=1.0
        ) * 1.1

    if tau_star is None:
        if tau_method == "proxy":
            arrays = [(d[:, 1] - d[:, 0]) / 2.0 for d in all_diagrams if len(d) > 0]
            pers_values = np.concatenate(arrays) if arrays else np.array([L / 4])
            tau_star = float(np.median(pers_values))
        else:
            # "auto" or "crossing": use subsampled crossing estimate
            tau_star = estimate_tau_star(diagrams_by_class)

    # Signal-driven R_min: tau*/4 ensures scales resolve the class
    # separation scale.  Grid subsampling (not R_min shift) controls
    # embedding dimension.
    lo = tau_star / 4
    hi = min(tau_star * 2, L * 0.95)
    if hi <= lo:
        hi = lo * 10
    scales = np.geomspace(lo, hi, N_scales)
    scales = np.unique(scales)   # deduplicate

    masses = analytic_optimal_masses(scales, L)

    return MultiscaleEmbedding(scales, masses, L, n_diagram=n_diagram,
                               max_landmarks=max_landmarks)


# ══════════════════════════════════════════════════════════════════════════
# Scale optimizer: learn R_k via RatioCut, analytic mass reset
# ══════════════════════════════════════════════════════════════════════════

class ScaleOptimizer:
    """
    Learn scale positions R_k by minimising empirical RatioCut on the
    WLK distance (omega=1), with masses reset analytically after each
    scale update.

    The gradient of RatioCut w.r.t. R_k is estimated via central finite
    differences (the grid G_{R_k} changes discretely with R, making
    analytic differentiation impractical).

    After each gradient step the masses are reset to the closed-form
    argmax of lambda(nu), w_k^2 ∝ (d_{k+1}^2 - d_k^2)/R_k^2, where
    d_i = R_i - R_1 and d_{N+1} = L - R_1 (see utils/rho_minus.py).

    Parameters
    ----------
    emb         : MultiscaleEmbedding (modified in-place via update_nu)
    sigma       : kernel bandwidth for WLK
    lr          : learning rate for scale gradient descent
    T           : number of gradient steps
    delta       : finite-difference step size (relative to R_k)
    R_min_clip  : lower bound on any R_k (positive, for stability)
    """

    def __init__(self, emb: MultiscaleEmbedding, sigma: float,
                 lr: float = 1e-3, T: int = 30,
                 delta: float = 1e-3, R_min_clip: float = 1e-4):
        self.emb = emb
        self.sigma = sigma
        self.lr = lr
        self.T = T
        self.delta = delta
        self.R_min_clip = R_min_clip

    @staticmethod
    def _wlk_D2(X: np.ndarray, inv2s2: float) -> np.ndarray:
        """WLK squared distance matrix (omega=1)."""
        N, K = X.shape
        D2 = np.zeros((N, N))
        for k in range(K):
            diff = X[:, k:k+1] - X[:, k:k+1].T
            D2 += 2.0 * (1.0 - np.exp(-diff**2 * inv2s2))
        return D2

    @staticmethod
    def _ratiocut(D2: np.ndarray, labels: np.ndarray) -> float:
        """Empirical RatioCut from squared distance matrix."""
        classes = np.unique(labels)
        rcut = 0.0
        for c in classes:
            mask = labels == c
            cut_c = D2[np.ix_(mask, mask)].sum()
            assoc_c = D2[mask, :].sum()
            if assoc_c > 0:
                rcut += cut_c / assoc_c
        return rcut

    def _eval_ratiocut(self, scales: np.ndarray, diagrams: List[np.ndarray],
                       labels: np.ndarray) -> float:
        """Build embedding from scales, embed, compute RatioCut."""
        masses = analytic_optimal_masses(scales, self.emb.L)
        emb_tmp = MultiscaleEmbedding(
            scales, masses, self.emb.L,
            n_diagram=self.emb.n,
            max_landmarks=self.emb.max_landmarks
        )
        X = emb_tmp.embed_dataset(diagrams)
        inv2s2 = 1.0 / (2.0 * self.sigma ** 2)
        D2 = self._wlk_D2(X, inv2s2)
        return self._ratiocut(D2, labels)

    def fit(self, diagrams: List[np.ndarray], labels: np.ndarray,
            verbose: bool = True) -> dict:
        """
        Optimise scales R_k by finite-difference gradient descent on
        RatioCut, resetting masses analytically after each step.

        Returns
        -------
        dict with: 'rcut_history', 'scales_history', 'rho_history'
        """
        import time

        scales = self.emb.scales.copy()
        N = len(scales)
        L = self.emb.L

        rcut_history = []
        scales_history = [scales.copy()]
        rho_history = []

        for t in range(self.T):
            t0 = time.time()

            # Current RatioCut
            rcut_0 = self._eval_ratiocut(scales, diagrams, labels)
            rcut_history.append(rcut_0)

            # Current rho_- certificate
            masses_cur = analytic_optimal_masses(scales, L)
            tau_star = float(np.median(scales))
            rho = rho_minus(tau_star, scales, masses_cur, n=self.emb.n, L=L)
            rho_history.append(rho)

            # Finite-difference gradient: d RCut / d R_k
            grad = np.zeros(N)
            for k in range(N):
                h = max(self.delta * scales[k], 1e-6)

                scales_plus = scales.copy()
                scales_plus[k] += h
                scales_plus[k] = min(scales_plus[k], L * 0.95)

                scales_minus = scales.copy()
                scales_minus[k] = max(scales_minus[k] - h, self.R_min_clip)

                rcut_plus = self._eval_ratiocut(scales_plus, diagrams, labels)
                rcut_minus = self._eval_ratiocut(scales_minus, diagrams, labels)

                denom = scales_plus[k] - scales_minus[k]
                if denom > 0:
                    grad[k] = (rcut_plus - rcut_minus) / denom

            # Gradient step
            scales = scales - self.lr * grad

            # Project: R_k > 0, R_k < L, and maintain ordering
            scales = np.clip(scales, self.R_min_clip, L * 0.95)
            scales = np.sort(scales)

            # Reset masses analytically
            masses_new = analytic_optimal_masses(scales, L)
            self.emb.update_nu(scales, masses_new)

            scales_history.append(scales.copy())

            elapsed = time.time() - t0
            if verbose:
                print(f"  Step {t+1}/{self.T}: RCut={rcut_0:.6f}  "
                      f"rho_-={rho:.6f}  "
                      f"R=[{scales[0]:.4f}, {scales[-1]:.4f}]  "
                      f"|grad|={np.linalg.norm(grad):.4e}  "
                      f"({elapsed:.1f}s)", flush=True)

        # Final evaluation
        rcut_final = self._eval_ratiocut(scales, diagrams, labels)
        rcut_history.append(rcut_final)
        masses_final = analytic_optimal_masses(scales, L)
        rho_final = rho_minus(float(np.median(scales)), scales, masses_final,
                              n=self.emb.n, L=L)
        rho_history.append(rho_final)

        return {
            'rcut_history': rcut_history,
            'scales_history': scales_history,
            'rho_history': rho_history,
        }
