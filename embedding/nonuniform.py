"""
embedding/nonuniform.py
-----------------------
Non-uniform persistence landmark embedding (PALACE, Paper II).

Implements:
  1. NonUniformEmbedding  — learnable positions, radii, weights
  2. Soft coordinate function  — sigmoid-smoothed phi for gradient optimization
  3. Weighted landmark kernel  — k_omega and RKHS distance D_omega^2
  4. ThreeBlockOptimizer  — alternating optimization of (omega, radii, positions)
  5. CertifiedClassifier  — nearest-centroid with per-prediction certificates

Notation (matching Paper II):
  L = {(p_k, r_k, w_k)}_{k=1}^K   landmark configuration
  phi_{p,r}(x) = max(r - d_B(p,x), 0)   coordinate function
  Phi_k(A; L) = w_k * 2^{-n-1/2} * sum_i phi_{p_k,r_k}(a_i)
  rho_nu(tau; L) = c_n * tau * min_{k: r_k >= tau/4} w_k
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional, Tuple

from utils.bottleneck import d_B1_batch


# ══════════════════════════════════════════════════════════════════════════
# Hard and soft coordinate functions
# ══════════════════════════════════════════════════════════════════════════

def hard_coords(positions: np.ndarray, radii: np.ndarray,
                pts: np.ndarray) -> np.ndarray:
    """
    Hard coordinate functions phi_{p_k,r_k}(A) = sum_i max(r_k - d_B(p_k, a_i), 0).

    Parameters
    ----------
    positions : (K, 2) landmark positions
    radii     : (K,) landmark radii
    pts       : (m, 2) diagram points

    Returns
    -------
    (K,) array of coordinate values
    """
    K = len(positions)
    if len(pts) == 0:
        return np.zeros(K)
    d_B = d_B1_batch(positions, pts)                             # (K, m)
    phi = np.maximum(radii[:, np.newaxis] - d_B, 0.0)            # (K, m)
    return phi.sum(axis=1)                                        # (K,)


def soft_coords(positions: np.ndarray, radii: np.ndarray,
                pts: np.ndarray, eps: float) -> np.ndarray:
    """
    Soft coordinate functions (Definition 4.1, Paper II):
        tilde{phi}_{p_k,r_k}^eps(A) = sum_i sigma((r_k - d_B)/eps) * max(r_k - d_B, 0)

    Parameters
    ----------
    positions : (K, 2) landmark positions
    radii     : (K,) landmark radii
    pts       : (m, 2) diagram points
    eps       : temperature (positive)

    Returns
    -------
    (K,) array of soft coordinate values
    """
    K = len(positions)
    if len(pts) == 0:
        return np.zeros(K)
    d_B = d_B1_batch(positions, pts)                             # (K, m)
    hard = np.maximum(radii[:, np.newaxis] - d_B, 0.0)           # (K, m)
    arg = (radii[:, np.newaxis] - d_B) / eps                     # (K, m)
    arg = np.clip(arg, -500, 500)  # prevent overflow
    sigma = 1.0 / (1.0 + np.exp(-arg))                           # (K, m)
    soft = sigma * hard                                           # (K, m)
    return soft.sum(axis=1)                                       # (K,)


# ══════════════════════════════════════════════════════════════════════════
# Analytic gradients for soft coordinates (ported from adaptive-landmark)
# ══════════════════════════════════════════════════════════════════════════

def soft_grad_radii(positions: np.ndarray, radii: np.ndarray,
                    pts: np.ndarray, eps: float) -> np.ndarray:
    """
    Gradient of soft coordinate values w.r.t. radii r_k.

    d/dr_k [sum_i sigma * max(r_k - d, 0)] = sum_i [sigma' * hat + sigma * 1[diff>0]]

    Returns (K,) array.
    """
    K = len(positions)
    if len(pts) == 0:
        return np.zeros(K)
    d_B = d_B1_batch(positions, pts)                                     # (K, m)
    diff = radii[:, np.newaxis] - d_B                                    # (K, m)
    arg = np.clip(diff / eps, -500, 500)
    sigma = 1.0 / (1.0 + np.exp(-arg))                                  # (K, m)
    sigma_prime = sigma * (1.0 - sigma) / eps                            # (K, m)
    hat = np.maximum(diff, 0.0)                                          # (K, m)
    inside = (diff > 0).astype(float)                                    # (K, m)
    return (sigma_prime * hat + sigma * inside).sum(axis=1)              # (K,)


def soft_grad_positions(positions: np.ndarray, radii: np.ndarray,
                        pts: np.ndarray, eps: float) -> np.ndarray:
    """
    Gradient of soft coordinate values w.r.t. positions p_k.

    Uses subgradient of d_B^1(p_k, a_i) w.r.t. p_k.

    Returns (K, 2) array.
    """
    K = len(positions)
    if len(pts) == 0:
        return np.zeros((K, 2))
    m = len(pts)

    # d_B^1 components
    raw_diff = positions[:, np.newaxis, :] - pts[np.newaxis, :, :]       # (K, m, 2)
    abs_diff = np.abs(raw_diff)                                          # (K, m, 2)
    d_inf = abs_diff.max(axis=2)                                         # (K, m)

    pers_lm = (positions[:, 1] - positions[:, 0]) / 2.0                  # (K,)
    pers_pts = (pts[:, 1] - pts[:, 0]) / 2.0                            # (m,)
    d_via = pers_lm[:, np.newaxis] + pers_pts[np.newaxis, :]             # (K, m)

    d_B = np.minimum(d_inf, d_via)                                       # (K, m)
    use_inf = (d_inf <= d_via)                                           # (K, m)

    # subgradient of d_B^1 w.r.t. p_k
    max_axis = np.argmax(abs_diff, axis=2)                               # (K, m)
    grad_d_inf = np.zeros((K, m, 2))
    for ax in range(2):
        mask = (max_axis == ax)
        grad_d_inf[:, :, ax] = np.where(mask, np.sign(raw_diff[:, :, ax]), 0.0)

    grad_d_via = np.zeros((K, m, 2))
    grad_d_via[:, :, 0] = -0.5
    grad_d_via[:, :, 1] = 0.5

    grad_dB = np.where(use_inf[:, :, np.newaxis], grad_d_inf, grad_d_via)

    # Chain rule
    diff = radii[:, np.newaxis] - d_B                                    # (K, m)
    arg = np.clip(diff / eps, -500, 500)
    sigma = 1.0 / (1.0 + np.exp(-arg))                                  # (K, m)
    sigma_prime = sigma * (1.0 - sigma) / eps                            # (K, m)
    hat = np.maximum(diff, 0.0)                                          # (K, m)
    inside = (diff > 0).astype(float)                                    # (K, m)

    scalar = -(sigma_prime * hat + sigma * inside)                       # (K, m)
    return (scalar[:, :, np.newaxis] * grad_dB).sum(axis=1)              # (K, 2)


# ══════════════════════════════════════════════════════════════════════════
# Non-uniform embedding
# ══════════════════════════════════════════════════════════════════════════

class NonUniformEmbedding:
    """
    Non-uniform persistence landmark embedding (Paper II, Section 2).

    Parameters
    ----------
    positions  : (K, 2) array — landmark positions in D_1
    radii      : (K,) array — landmark radii (positive)
    weights    : (K,) array — configuration weights w_k with sum(w_k^2) = 1
    L          : frame size
    n_diagram  : diagram dimension (default 1)
    """

    def __init__(self, positions: np.ndarray, radii: np.ndarray,
                 weights: np.ndarray, L: float, n_diagram: int = 1):
        self.positions = np.asarray(positions, dtype=float)  # (K, 2)
        self.radii = np.asarray(radii, dtype=float)          # (K,)
        self.weights = np.asarray(weights, dtype=float)      # (K,)
        self.L = float(L)
        self.n = n_diagram
        self.K = len(self.positions)
        self.norm = 2.0 ** (-n_diagram - 0.5)

        assert self.positions.shape == (self.K, 2)
        assert self.radii.shape == (self.K,)
        assert self.weights.shape == (self.K,)

    @property
    def embedding_dim(self) -> int:
        return self.K

    def embed(self, pts: np.ndarray, eps: Optional[float] = None) -> np.ndarray:
        """
        Embed a single diagram.

        Parameters
        ----------
        pts : (m, 2) array of (birth, death) pairs
        eps : if given, use soft coordinates; otherwise hard

        Returns
        -------
        (K,) array — embedding coordinates Phi_k(A; L)
        """
        pts = self._clip(pts)
        if eps is None:
            coords = hard_coords(self.positions, self.radii, pts)
        else:
            coords = soft_coords(self.positions, self.radii, pts, eps)
        return self.weights * self.norm * coords

    def embed_dataset(self, diagrams: List[np.ndarray],
                      eps: Optional[float] = None,
                      verbose: bool = False) -> np.ndarray:
        """Embed a list of diagrams. Returns (N, K) array."""
        N = len(diagrams)
        X = np.zeros((N, self.K))
        for i, dgm in enumerate(diagrams):
            if verbose and (i + 1) % 100 == 0:
                print(f"  Embedding {i+1}/{N}")
            X[i] = self.embed(dgm, eps=eps)
        return X

    def rho_nu(self, tau: float) -> float:
        """
        Non-uniform distortion certificate (Theorem 1, Paper II):
            rho_nu(tau; L) = c_n * tau * min_{k: r_k >= tau/4} w_k
        """
        c_n = 2.0 ** (-self.n - 5.0 / 2.0)
        mask = self.radii >= tau / 4.0
        if not np.any(mask):
            return 0.0
        return c_n * tau * np.min(self.weights[mask])

    def lebesgue_number(self, pts_list: List[np.ndarray]) -> float:
        """
        Compute Lebesgue number lambda_0 from training diagrams
        (Lemma 1, Paper II).
        """
        lambda_0 = np.inf
        for pts in pts_list:
            pts = self._clip(pts)
            if len(pts) == 0:
                continue
            d_B = d_B1_batch(self.positions, pts)  # (K, m)
            # depth of each point in its deepest ball
            depths = self.radii[:, np.newaxis] - d_B  # (K, m)
            depths = np.where(depths > 0, depths, -np.inf)
            max_depth_per_pt = np.max(depths, axis=0)  # (m,)
            if len(max_depth_per_pt) > 0:
                lambda_0 = min(lambda_0, np.min(max_depth_per_pt))
        return lambda_0 if np.isfinite(lambda_0) else 0.0

    def is_admissible(self, tau: float, gamma: float = 0.0) -> bool:
        """Check (gamma, tau)-admissibility (Definition 3, Paper II)."""
        if np.min(self.radii) < gamma:
            return False
        if np.max(self.radii) > 4 * tau:
            return False
        if self.rho_nu(tau) <= gamma:
            return False
        return True

    def _clip(self, pts: np.ndarray) -> np.ndarray:
        if len(pts) == 0:
            return np.zeros((0, 2))
        return pts[(pts[:, 0] >= 0) & (pts[:, 1] <= self.L) &
                   (pts[:, 1] > pts[:, 0])]

    def __repr__(self):
        return (f"NonUniformEmbedding(K={self.K}, "
                f"rho_nu_ready={np.any(self.radii >= 0.01)}, "
                f"L={self.L})")


# ══════════════════════════════════════════════════════════════════════════
# Weighted landmark kernel
# ══════════════════════════════════════════════════════════════════════════

class WeightedLandmarkKernel:
    """
    Weighted landmark kernel (Definition 5, Paper II, Section 3.1):

        k_omega(A, B; L) = sum_k omega_k * exp(-(Phi_k(A) - Phi_k(B))^2 / (2*sigma^2))

    Parameters
    ----------
    omega : (K,) array — kernel weights (non-negative, bounded by M)
    sigma : bandwidth parameter
    """

    def __init__(self, omega: np.ndarray, sigma: float):
        self.omega = np.asarray(omega, dtype=float)
        self.sigma = float(sigma)
        self.K = len(self.omega)

    def gram_matrix(self, X: np.ndarray, Y: np.ndarray = None) -> np.ndarray:
        """
        Compute Gram matrix K_{ij} = k_omega(A_i, A_j) or K_{ij} = k_omega(A_i, B_j).

        Parameters
        ----------
        X : (N, K) embedding matrix from NonUniformEmbedding.embed_dataset
        Y : (M, K) optional second embedding matrix. If None, computes X vs X.

        Returns
        -------
        (N, N) or (N, M) Gram matrix
        """
        if Y is None:
            Y = X
        inv2s2 = 1.0 / (2 * self.sigma**2)
        N, M = X.shape[0], Y.shape[0]
        G = np.zeros((N, M))
        for k in range(self.K):
            diff = X[:, k:k+1] - Y[:, k:k+1].T                 # (N, M)
            G += self.omega[k] * np.exp(-diff**2 * inv2s2)
        return G

    def D_omega_sq(self, X: np.ndarray) -> np.ndarray:
        """
        Squared RKHS distance matrix D_omega^2(A_i, A_j).

        Parameters
        ----------
        X : (N, K) embedding matrix

        Returns
        -------
        (N, N) distance matrix
        """
        G = self.gram_matrix(X)
        diag = np.diag(G)
        return diag[:, np.newaxis] + diag[np.newaxis, :] - 2 * G

    def g_components(self, X: np.ndarray) -> np.ndarray:
        """
        Per-landmark distance components g_k(A_i, A_j) = 2(1 - exp(-...)).
        D_omega^2 = sum_k omega_k * g_k.

        Returns (K, N, N) array.
        """
        N = X.shape[0]
        g = np.zeros((self.K, N, N))
        for k in range(self.K):
            diff = X[:, k:k+1] - X[:, k:k+1].T
            g[k] = 2.0 * (1.0 - np.exp(-diff**2 / (2 * self.sigma**2)))
        return g


# ══════════════════════════════════════════════════════════════════════════
# RatioCut objective
# ══════════════════════════════════════════════════════════════════════════

def empirical_ratiocut(D2: np.ndarray, labels: np.ndarray) -> float:
    """
    Empirical RatioCut from squared distance matrix.

    RCut = sum_t cut(C_t) / assoc(C_t)

    Parameters
    ----------
    D2     : (N, N) squared distance matrix D_omega^2
    labels : (N,) integer class labels

    Returns
    -------
    scalar RatioCut value
    """
    classes = np.unique(labels)
    rcut = 0.0
    for c in classes:
        mask_c = labels == c
        cut_c = D2[np.ix_(mask_c, mask_c)].sum()
        assoc_c = D2[mask_c, :].sum()
        if assoc_c > 0:
            rcut += cut_c / assoc_c
    return rcut


def ratiocut_grad_omega(g_components: np.ndarray, labels: np.ndarray,
                        omega: np.ndarray) -> np.ndarray:
    """
    Gradient of empirical RatioCut w.r.t. omega.

    Since D_omega^2 = sum_k omega_k * g_k, the gradient is:
        d RCut / d omega_k = sum_t (d/d omega_k)(cut_t / assoc_t)

    Parameters
    ----------
    g_components : (K, N, N) from WeightedLandmarkKernel.g_components
    labels       : (N,) integer class labels
    omega        : (K,) current kernel weights

    Returns
    -------
    (K,) gradient array
    """
    K, N, _ = g_components.shape
    classes = np.unique(labels)

    # Current D2 = sum_k omega_k * g_k
    D2 = np.tensordot(omega, g_components, axes=([0], [0]))  # (N, N)

    grad = np.zeros(K)
    for c in classes:
        mask_c = labels == c
        cut_c = D2[np.ix_(mask_c, mask_c)].sum()
        assoc_c = D2[mask_c, :].sum()
        if assoc_c <= 0:
            continue
        for k in range(K):
            gk = g_components[k]
            d_cut_k = gk[np.ix_(mask_c, mask_c)].sum()
            d_assoc_k = gk[mask_c, :].sum()
            # quotient rule: d(cut/assoc)/d_omega_k
            grad[k] += (d_cut_k * assoc_c - cut_c * d_assoc_k) / (assoc_c**2)
    return grad


# ══════════════════════════════════════════════════════════════════════════
# Weight optimizer: learn w_k via RatioCut gradient descent
# ══════════════════════════════════════════════════════════════════════════

class WeightOptimizer:
    """
    Learn configuration weights w_k by minimizing empirical RatioCut
    on the WLK distance, with fixed positions, radii, omega=1, and sigma.

    The embedding is Phi_k(A) = w_k * c_n * sum_i phi_{p_k,r_k}(a_i),
    so (Phi_k(A) - Phi_k(B))^2 = w_k^2 * (raw_k(A) - raw_k(B))^2.

    The WLK distance (omega=1) is:
        D^2(A,B) = sum_k 2(1 - exp(-w_k^2 * delta_k^2 / 2sigma^2))
    where delta_k = raw_k(A) - raw_k(B).

    Gradient w.r.t. w_k:
        dD^2/dw_k = sum_k (2 * w_k * delta_k^2 / sigma^2)
                    * exp(-w_k^2 * delta_k^2 / 2sigma^2)
    which feeds into the RatioCut quotient rule.

    Parameters
    ----------
    emb       : NonUniformEmbedding (weights modified in-place)
    sigma     : kernel bandwidth
    lr        : learning rate
    T         : number of gradient steps
    w_min     : lower bound on w_k (positive, for stability)
    """

    def __init__(self, emb: NonUniformEmbedding, sigma: float,
                 lr: float = 1e-3, T: int = 50, w_min: float = 1e-6):
        self.emb = emb
        self.sigma = sigma
        self.lr = lr
        self.T = T
        self.w_min = w_min

    def fit(self, diagrams: List[np.ndarray], labels: np.ndarray,
            verbose: bool = True) -> dict:
        """
        Optimize w_k by gradient descent on RatioCut.

        Parameters
        ----------
        diagrams : list of (m_i, 2) arrays (training diagrams)
        labels   : (N,) integer class labels

        Returns
        -------
        dict with: 'rcut_history', 'weights_history'
        """
        K = self.emb.K
        inv2s2 = 1.0 / (2 * self.sigma ** 2)
        classes = np.unique(labels)

        # Compute raw (unweighted) coordinates: raw_k(A) = c_n * sum_i phi(a_i)
        # Temporarily set weights to 1 to get raw * norm
        saved_weights = self.emb.weights.copy()
        self.emb.weights = np.ones(K)
        X_raw = self.emb.embed_dataset(diagrams)  # (N, K), each col = norm * coords
        self.emb.weights = saved_weights

        w = self.emb.weights.copy()
        rcut_history = []
        weights_history = [w.copy()]

        N = X_raw.shape[0]

        for t in range(self.T):
            # Pass 1: compute full D^2
            D2 = np.zeros((N, N))
            for k in range(K):
                delta_sq = (X_raw[:, k:k+1] - X_raw[:, k:k+1].T) ** 2
                D2 += 2.0 * (1.0 - np.exp(-w[k] ** 2 * delta_sq * inv2s2))

            rcut = empirical_ratiocut(D2, labels)
            rcut_history.append(rcut)

            # Per-class cut/assoc from full D^2
            class_info = []
            for c in classes:
                mask_c = labels == c
                cut_c = D2[np.ix_(mask_c, mask_c)].sum()
                assoc_c = D2[mask_c, :].sum()
                class_info.append((mask_c, cut_c, assoc_c))

            # Pass 2: gradient w.r.t. each w_k (recompute per-k terms)
            grad_w = np.zeros(K)
            for k in range(K):
                delta_sq = (X_raw[:, k:k+1] - X_raw[:, k:k+1].T) ** 2
                exp_term = np.exp(-w[k] ** 2 * delta_sq * inv2s2)
                dg_k = 2.0 * w[k] * delta_sq * inv2s2 * exp_term

                for mask_c, cut_c, assoc_c in class_info:
                    if assoc_c <= 0:
                        continue
                    d_cut = dg_k[np.ix_(mask_c, mask_c)].sum()
                    d_assoc = dg_k[mask_c, :].sum()
                    grad_w[k] += (d_cut * assoc_c - cut_c * d_assoc) / (assoc_c ** 2)

            # Gradient step
            w -= self.lr * grad_w
            w = np.maximum(w, self.w_min)
            weights_history.append(w.copy())

            if verbose and (t + 1) % 10 == 0:
                print(f"  Step {t+1}/{self.T}: RCut={rcut:.6f}, "
                      f"w_min={w.min():.4f}, w_max={w.max():.4f}, "
                      f"w_mean={w.mean():.4f}")

        # Update embedding weights
        self.emb.weights = w.copy()

        return {
            'rcut_history': rcut_history,
            'weights_history': weights_history,
        }


# ══════════════════════════════════════════════════════════════════════════
# Three-block optimizer with analytic gradients
# (ported from adaptive-landmark branch)
# ══════════════════════════════════════════════════════════════════════════

class ThreeBlockOptimizer:
    """
    Three-block alternating optimization for PALACE (Algorithm 1, Paper II).

    Optimizes (omega, radii, positions) to minimize RatioCut using
    analytic soft gradients for radii/positions.

    Parameters
    ----------
    emb         : NonUniformEmbedding (modified in-place)
    sigma       : kernel bandwidth
    tau         : separation scale
    gamma       : certificate lower bound
    T_outer     : number of outer iterations
    T_radii     : radii gradient steps per outer iteration
    T_pos       : position gradient steps per outer iteration
    lr_radii    : learning rate for radii
    lr_pos      : learning rate for positions
    epsilon     : soft coordinate temperature
    """

    def __init__(self, emb: NonUniformEmbedding, sigma: float,
                 tau: float, gamma: float = 1e-4,
                 T_outer: int = 10, T_radii: int = 3, T_pos: int = 3,
                 lr_radii: float = 5e-3, lr_pos: float = 5e-4,
                 epsilon: float = 0.01):
        self.emb = emb
        self.sigma = sigma
        self.tau = tau
        self.gamma = gamma
        self.T_outer = T_outer
        self.T_radii = T_radii
        self.T_pos = T_pos
        self.lr_radii = lr_radii
        self.lr_pos = lr_pos
        self.epsilon = epsilon

    def fit(self, diagrams: List[np.ndarray], labels: np.ndarray,
            verbose: bool = True) -> dict:
        """
        Run three-block optimization (omega fixed at 1).

        Returns dict with 'rcut_history', 'rho_history'.
        """
        import time
        K = self.emb.K
        n_classes = len(np.unique(labels))
        inv2s2 = 1.0 / (2.0 * self.sigma ** 2)
        r_min = max(self.tau / 4.0, 1e-6)
        eps = self.epsilon
        omega = np.ones(K)  # fixed at 1

        # Clip diagrams once
        clipped = [self.emb._clip(dgm) for dgm in diagrams]

        rcut_history = []
        rho_history = []

        for t in range(self.T_outer):
            t0 = time.time()

            # ── Block 1: radii optimization ────────────────────────────
            for _ in range(self.T_radii):
                # Soft embed
                X = np.array([
                    self.emb.weights * self.emb.norm *
                    soft_coords(self.emb.positions, self.emb.radii, pts, eps)
                    for pts in clipped
                ])  # (N, K)

                # WLK distance matrix (omega=1)
                N = X.shape[0]
                D2 = np.zeros((N, N))
                for k in range(K):
                    diff = X[:, k:k+1] - X[:, k:k+1].T
                    D2 += 2.0 * (1.0 - np.exp(-diff**2 * inv2s2))

                # RatioCut gradient components
                masks = [(labels == c) for c in range(n_classes)]
                cut_vals = [D2[np.ix_(m, m)].sum() for m in masks]
                assoc_vals = [D2[m, :].sum() for m in masks]

                # Aggregate gradient over diagrams for each landmark
                gr = np.zeros(K)
                for k in range(K):
                    diff_k = X[:, k:k+1] - X[:, k:k+1].T  # (N, N)
                    exp_k = np.exp(-diff_k**2 * inv2s2)
                    # dg_k/dX_k = 4 * diff_k * inv2s2 * exp_k
                    dg_dX = 4.0 * diff_k * inv2s2 * exp_k  # (N, N)

                    # dX_k/dr_k for each diagram
                    dX_dr = np.array([
                        self.emb.weights[k] * self.emb.norm *
                        soft_grad_radii(self.emb.positions, self.emb.radii, pts, eps)[k]
                        for pts in clipped
                    ])  # (N,)

                    # dg_k/dr_k = dg_dX * (dX_dr_i - dX_dr_j)
                    dg_dr = dg_dX * (dX_dr[:, np.newaxis] - dX_dr[np.newaxis, :])

                    for c_idx, (m, cut_c, assoc_c) in enumerate(
                            zip(masks, cut_vals, assoc_vals)):
                        if assoc_c < 1e-12:
                            continue
                        d_cut = dg_dr[np.ix_(m, m)].sum()
                        d_assoc = dg_dr[m, :].sum()
                        gr[k] += (d_cut * assoc_c - cut_c * d_assoc) / (assoc_c**2)

                new_radii = self.emb.radii - self.lr_radii * gr
                new_radii = np.maximum(new_radii, r_min)
                self.emb.radii = new_radii

            # ── Block 2: position optimization ─────────────────────────
            for _ in range(self.T_pos):
                X = np.array([
                    self.emb.weights * self.emb.norm *
                    soft_coords(self.emb.positions, self.emb.radii, pts, eps)
                    for pts in clipped
                ])

                N = X.shape[0]
                D2 = np.zeros((N, N))
                for k in range(K):
                    diff = X[:, k:k+1] - X[:, k:k+1].T
                    D2 += 2.0 * (1.0 - np.exp(-diff**2 * inv2s2))

                masks = [(labels == c) for c in range(n_classes)]
                cut_vals = [D2[np.ix_(m, m)].sum() for m in masks]
                assoc_vals = [D2[m, :].sum() for m in masks]

                gp = np.zeros((K, 2))
                for k in range(K):
                    diff_k = X[:, k:k+1] - X[:, k:k+1].T
                    exp_k = np.exp(-diff_k**2 * inv2s2)
                    dg_dX = 4.0 * diff_k * inv2s2 * exp_k

                    dX_dp = np.array([
                        self.emb.weights[k] * self.emb.norm *
                        soft_grad_positions(self.emb.positions, self.emb.radii, pts, eps)[k]
                        for pts in clipped
                    ])  # (N, 2)

                    for ax in range(2):
                        dg_dp = dg_dX * (dX_dp[:, ax:ax+1] - dX_dp[:, ax:ax+1].T)
                        for c_idx, (m, cut_c, assoc_c) in enumerate(
                                zip(masks, cut_vals, assoc_vals)):
                            if assoc_c < 1e-12:
                                continue
                            d_cut = dg_dp[np.ix_(m, m)].sum()
                            d_assoc = dg_dp[m, :].sum()
                            gp[k, ax] += (d_cut * assoc_c - cut_c * d_assoc) / (assoc_c**2)

                new_pos = self.emb.positions - self.lr_pos * gp
                new_pos[:, 0] = np.maximum(new_pos[:, 0], 0.0)
                new_pos[:, 1] = np.maximum(new_pos[:, 1], new_pos[:, 0] + 1e-6)
                new_pos = np.clip(new_pos, 0.0, self.emb.L)
                self.emb.positions = new_pos

            # ── Evaluate ───────────────────────────────────────────────
            X = self.emb.embed_dataset(diagrams)
            D2 = np.zeros((len(X), len(X)))
            for k in range(K):
                diff = X[:, k:k+1] - X[:, k:k+1].T
                D2 += 2.0 * (1.0 - np.exp(-diff**2 * inv2s2))
            rcut = empirical_ratiocut(D2, labels)
            rho = self.emb.rho_nu(self.tau)
            rcut_history.append(rcut)
            rho_history.append(rho)

            elapsed = time.time() - t0
            if verbose:
                print(f"  Iter {t+1}/{self.T_outer}: RCut={rcut:.4f}, "
                      f"rho_nu={rho:.6f} ({elapsed:.1f}s)", flush=True)

        return {
            'rcut_history': rcut_history,
            'rho_history': rho_history,
        }


# ══════════════════════════════════════════════════════════════════════════
# Certified nearest-centroid classifier (Algorithm 2, Paper II)
# ══════════════════════════════════════════════════════════════════════════

class CertifiedClassifier:
    """
    Certified nearest-centroid classifier (Section 5.5, Paper II).

    Each prediction is accompanied by a certificate: if
        r_m < kappa * rho_nu / 2,
    the prediction is correct with probability >= 1 - alpha.

    Parameters
    ----------
    emb     : NonUniformEmbedding
    kernel  : WeightedLandmarkKernel
    tau     : separation scale
    alpha   : confidence level (default 0.05)
    """

    def __init__(self, emb: NonUniformEmbedding,
                 kernel: WeightedLandmarkKernel,
                 tau: float, alpha: float = 0.05):
        self.emb = emb
        self.kernel = kernel
        self.tau = tau
        self.alpha = alpha
        self.class_means_ = None
        self.class_covs_ = None
        self.classes_ = None
        self.n_per_class_ = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Compute per-class centroids and covariances in embedding space.

        Parameters
        ----------
        X_train : (N, K) embedding matrix
        y_train : (N,) labels
        """
        self.classes_ = np.unique(y_train)
        k = len(self.classes_)
        K = X_train.shape[1]

        self.class_means_ = np.zeros((k, K))
        self.class_covs_ = np.zeros((k, K, K))
        self.n_per_class_ = np.zeros(k, dtype=int)

        for i, c in enumerate(self.classes_):
            mask = y_train == c
            X_c = X_train[mask]
            self.n_per_class_[i] = len(X_c)
            self.class_means_[i] = X_c.mean(axis=0)
            if len(X_c) > 1:
                self.class_covs_[i] = np.cov(X_c, rowvar=False)

    def predict(self, X_test: np.ndarray,
                kappa: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with certificates.

        Parameters
        ----------
        X_test : (N_test, K) embedding matrix
        kappa  : non-degeneracy constant (Theorem, Section 3.1)

        Returns
        -------
        predictions : (N_test,) predicted labels (-1 = abstain)
        certified   : (N_test,) boolean — True if certificate holds
        margins     : (N_test,) certificate margins (positive = certified)
        """
        from scipy.stats import norm as normal_dist

        N = X_test.shape[0]
        predictions = np.full(N, -1, dtype=int)
        certified = np.zeros(N, dtype=bool)
        margins = np.zeros(N)

        rho = self.emb.rho_nu(self.tau)
        z_alpha = normal_dist.ppf(1 - self.alpha / 2)

        for i in range(N):
            # Nearest centroid
            dists = np.linalg.norm(
                X_test[i:i+1] - self.class_means_, axis=1
            )
            c_hat_idx = np.argmin(dists)
            c_hat = self.classes_[c_hat_idx]

            # Confidence radius
            m_c = self.n_per_class_[c_hat_idx]
            if m_c < 2:
                predictions[i] = c_hat
                margins[i] = -np.inf
                continue

            cov_op_norm = np.linalg.norm(
                self.class_covs_[c_hat_idx], ord=2
            )
            r_m = z_alpha * np.sqrt(cov_op_norm) / np.sqrt(m_c)

            # Certificate check
            threshold = kappa * rho / 2.0
            margin = threshold - r_m
            predictions[i] = c_hat
            certified[i] = margin > 0
            margins[i] = margin

        return predictions, certified, margins

    def coverage_and_accuracy(self, X_test: np.ndarray,
                              y_test: np.ndarray,
                              kappa: float = 1.0) -> dict:
        """
        Evaluate coverage and accuracy.

        Returns
        -------
        dict with: accuracy, certified_accuracy, coverage, abstention_rate
        """
        preds, cert, margins = self.predict(X_test, kappa)

        acc = np.mean(preds == y_test)
        if np.any(cert):
            cert_acc = np.mean(preds[cert] == y_test[cert])
        else:
            cert_acc = float('nan')
        coverage = np.mean(cert)

        return {
            'accuracy': acc,
            'certified_accuracy': cert_acc,
            'coverage': coverage,
            'abstention_rate': 1.0 - coverage,
            'mean_margin': np.mean(margins),
        }


# ══════════════════════════════════════════════════════════════════════════
# Initialization helpers
# ══════════════════════════════════════════════════════════════════════════

def farthest_point_sampling(pts: np.ndarray, K: int,
                            seed: int = 0) -> np.ndarray:
    """
    Greedy farthest-point sampling on diagram points.

    Parameters
    ----------
    pts : (M, 2) array — candidate positions (e.g., all diagram points)
    K   : number of landmarks to select

    Returns
    -------
    (K, 2) array of selected positions
    """
    rng = np.random.default_rng(seed)
    M = len(pts)
    if M <= K:
        return pts.copy()

    selected = [rng.integers(M)]
    min_dists = np.full(M, np.inf)

    for _ in range(K - 1):
        last = pts[selected[-1]]
        d = np.max(np.abs(pts - last[np.newaxis, :]), axis=1)
        pers_last = (last[1] - last[0]) / 2.0
        pers_pts = (pts[:, 1] - pts[:, 0]) / 2.0
        d_via = pers_last + pers_pts
        d_B = np.minimum(d, d_via)
        min_dists = np.minimum(min_dists, d_B)
        selected.append(np.argmax(min_dists))

    return pts[selected]


def init_nonuniform_from_data(diagrams_by_class: List[List[np.ndarray]],
                              K: int, L: float,
                              tau: Optional[float] = None,
                              n_diagram: int = 1,
                              seed: int = 0,
                              alpha: float = 0.75) -> NonUniformEmbedding:
    """
    Initialize a NonUniformEmbedding from training data using
    class-aware farthest-point sampling.

    Parameters
    ----------
    diagrams_by_class : list of lists of (m, 2) arrays
    K                 : landmark budget
    L                 : frame size
    tau               : separation scale (auto-estimated if None)
    n_diagram         : diagram dimension
    seed              : random seed

    Returns
    -------
    NonUniformEmbedding with K landmarks
    """
    # Collect all diagram points
    all_pts = []
    for cls_dgms in diagrams_by_class:
        for dgm in cls_dgms:
            if len(dgm) > 0:
                valid = dgm[(dgm[:, 0] >= 0) & (dgm[:, 1] <= L) &
                            (dgm[:, 1] > dgm[:, 0])]
                if len(valid) > 0:
                    all_pts.append(valid)

    if not all_pts:
        raise ValueError("No valid diagram points found")

    pts_all = np.vstack(all_pts)

    # Estimate tau if not given
    if tau is None:
        pers = (pts_all[:, 1] - pts_all[:, 0]) / 2.0
        tau = float(np.median(pers))

    # Class-aware farthest-point sampling
    J = len(diagrams_by_class)
    K_per_class = max(1, K // J)
    positions_list = []
    for cls_dgms in diagrams_by_class:
        cls_pts = []
        for dgm in cls_dgms:
            if len(dgm) > 0:
                valid = dgm[(dgm[:, 0] >= 0) & (dgm[:, 1] <= L) &
                            (dgm[:, 1] > dgm[:, 0])]
                if len(valid) > 0:
                    cls_pts.append(valid)
        if cls_pts:
            cls_all = np.vstack(cls_pts)
            fps = farthest_point_sampling(cls_all, K_per_class, seed=seed)
            positions_list.append(fps)

    if not positions_list:
        raise ValueError("No valid points for landmark placement")

    positions = np.vstack(positions_list)[:K]

    # Radii: alpha * nearest-neighbour distance, clipped to [tau/2, 4*tau]
    if len(positions) > 1:
        from scipy.spatial.distance import cdist
        D = cdist(positions, positions, metric='chebyshev')
        np.fill_diagonal(D, np.inf)
        nn_dists = D.min(axis=1)
        radii = alpha * nn_dists
    else:
        radii = np.array([tau])

    radii = np.clip(radii, tau / 2.0, 4.0 * tau)

    # Equal weights
    weights = np.full(len(positions), 1.0 / np.sqrt(len(positions)))

    return NonUniformEmbedding(positions, radii, weights, L, n_diagram)


def init_nonuniform_class_weighted(diagrams_by_class: List[List[np.ndarray]],
                                   K: int, L: float,
                                   tau: Optional[float] = None,
                                   n_diagram: int = 1,
                                   seed: int = 0,
                                   alpha: float = 0.75,
                                   beta: float = 1.0,
                                   kde_k: int = 15) -> NonUniformEmbedding:
    """
    Class-weighted variant of `init_nonuniform_from_data`.

    Same class-aware FPS placement and equal weights, but radii are
    scaled by a per-landmark class-discriminability score:

        r_k  =  base_r_k  *  exp(beta * s_k),
                clipped to [tau/2, 4*tau]

        s_k  =  log( own_density(p_k) / mean_other_density(p_k) )

    where `own_density` is leave-one-out k-NN density on landmark p_k's
    own class pool (queried with k+1 neighbours, dropping the first to
    avoid self-inclusion bias), and `mean_other_density` is the mean of
    standard k-NN densities on the OTHER classes' pools.

    Rationale: WKPI (Zhao--Wang 2019) learns a Gaussian mixture weight
    function on diagram space via SGD on classification loss, which on
    label-dominated chemical pools migrates probability mass to
    class-discriminative regions.  PALACE's FPS targets data
    concentration, not label discrimination, so on those datasets the
    standard equal-radius rule under-resolves the discriminative
    region.  Class-weighted radii close part of this gap without
    introducing gradient-based learning: discriminability is computed
    in closed form from per-class k-NN densities.

    `beta = 0` recovers the standard equal-radii rule.

    Parameters
    ----------
    diagrams_by_class : list of lists of (m, 2) arrays
    K, L, tau, n_diagram, seed : as in `init_nonuniform_from_data`
    beta              : scaling on the log-density-ratio score
    kde_k             : k for the k-NN density estimator
    """
    from scipy.spatial import cKDTree
    from scipy.spatial.distance import cdist

    # Per-class point pools (filtering to admissible region)
    pools = []
    for cls_dgms in diagrams_by_class:
        cls_pts = []
        for dgm in cls_dgms:
            if len(dgm) > 0:
                valid = dgm[(dgm[:, 0] >= 0) & (dgm[:, 1] <= L) &
                            (dgm[:, 1] > dgm[:, 0])]
                if len(valid) > 0:
                    cls_pts.append(valid)
        pools.append(np.vstack(cls_pts) if cls_pts
                     else np.zeros((0, 2)))

    if all(len(p) == 0 for p in pools):
        raise ValueError("No valid points for landmark placement")

    if tau is None:
        all_pts = np.vstack([p for p in pools if len(p) > 0])
        pers = (all_pts[:, 1] - all_pts[:, 0]) / 2.0
        tau = float(np.median(pers))

    # Class-aware FPS, tracking class assignment of each landmark
    J = len(diagrams_by_class)
    K_per_class = max(1, K // J)
    positions_list = []
    classes_list = []
    for c, pool_c in enumerate(pools):
        if len(pool_c) == 0:
            continue
        fps = farthest_point_sampling(pool_c, K_per_class, seed=seed)
        positions_list.append(fps)
        classes_list.extend([c] * len(fps))

    if not positions_list:
        raise ValueError("No valid points for landmark placement")

    positions = np.vstack(positions_list)[:K]
    classes = np.array(classes_list[:K])

    # Base radii: alpha * nearest-neighbour distance (Chebyshev)
    if len(positions) > 1:
        D = cdist(positions, positions, metric='chebyshev')
        np.fill_diagonal(D, np.inf)
        nn_dists = D.min(axis=1)
        base_radii = alpha * nn_dists
    else:
        base_radii = np.array([tau])

    # Per-class KD-trees for density queries
    trees = [cKDTree(p) if len(p) > 1 else None for p in pools]

    # Per-landmark class-discriminability score
    n_landmarks = len(positions)
    s = np.zeros(n_landmarks)
    eps = 1e-9
    for i, p in enumerate(positions):
        c_self = int(classes[i])
        tr_own = trees[c_self]
        if tr_own is None:
            continue

        # Own-class density: query with k+1 neighbours, drop the first
        # (self distance, == 0). This is the leave-one-out fix that
        # the prior margin-FPS attempt missed.
        kk_own = min(kde_k + 1, len(tr_own.data))
        d_own, _ = tr_own.query(p, k=kk_own)
        d_own = np.atleast_1d(d_own)
        if len(d_own) > 1:
            d_own = d_own[1:]
        mean_d_own = float(np.mean(d_own))
        own_density = 1.0 / (mean_d_own + eps)

        # Other-class density: standard k-NN query on each other class
        other_densities = []
        for cp, tr_cp in enumerate(trees):
            if cp == c_self or tr_cp is None:
                continue
            kk = min(kde_k, len(tr_cp.data))
            d_cp, _ = tr_cp.query(p, k=kk)
            d_cp = np.atleast_1d(d_cp)
            mean_d_cp = float(np.mean(d_cp))
            other_densities.append(1.0 / (mean_d_cp + eps))

        if other_densities:
            mean_other = float(np.mean(other_densities))
            s[i] = float(np.log((own_density + eps) / (mean_other + eps)))

    # Class-weighted radii: r_k = base_r_k * exp(beta * s_k)
    # Clipped to the standard [tau/2, 4*tau] admissibility range
    factor = np.exp(beta * s)
    radii = base_radii * factor
    radii = np.clip(radii, tau / 2.0, 4.0 * tau)

    weights = np.full(len(positions), 1.0 / np.sqrt(len(positions)))

    return NonUniformEmbedding(positions, radii, weights, L, n_diagram)
