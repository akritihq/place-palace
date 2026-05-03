#!/usr/bin/env python3
"""
§6.3 Domain Inflation: Controlled Validation.

Synthetic 4-class annulus task with controlled L/D inflation.
Reproduces tab:domain_inflation with mean +/- std over 10-fold CV.

Classes:
  0: r_in=0.85, r_out=1.0  (thin ring, large hole)
  1: r_in=0.70, r_out=1.0
  2: r_in=0.50, r_out=1.0
  3: r_in=0.00, r_out=1.0  (solid disk)
n_pts = 60 per cloud, Gaussian noise sigma = 0.08.
n_clouds = 100 per class.

For each outlier offset ell in {1, 2, 3, 4, 5, 8}, append a single
distant off-diagonal pair to every diagram, inflating L while
keeping H_1 features near the origin.

Compare:
  Uniform K=11 grid landmarks
  Non-uniform K=11 FPS landmarks (class-agnostic)

10-fold stratified CV; bandwidth sigma at the 25th-percentile
heuristic per fold; C tuned by inner 3-fold CV.

Outputs results/domain_inflation/results.csv.
"""
import sys; sys.path.insert(0, '.')
import numpy as np
import csv
import time
from pathlib import Path
from scipy.spatial.distance import cdist, pdist
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

import gudhi

from embedding.nonuniform import NonUniformEmbedding, farthest_point_sampling
from utils.datasets import noisy_circle_pointcloud

OUT_DIR = Path('results/domain_inflation')
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED        = 42
N_PTS       = 60
NOISE       = 0.08
N_PER_CLASS = 100
N_CLASSES   = 4
INNER_RADII = [0.85, 0.70, 0.50, 0.00]
OUTER       = 1.0
L_VALUES    = [1, 2, 3, 4, 5, 8]
TOP_K_PERS  = 30
K_LANDMARKS = 11
N_FOLDS     = 10
C_GRID      = [0.01, 0.1, 1, 10, 100, 1000]
Q_BANDWIDTH = 0.25


def log(m):
    print(m, flush=True)


# ── Annulus point cloud ────────────────────────────────────────────────

def annulus(r_in, r_out, n, sigma, rng):
    """n points uniformly in the annulus [r_in, r_out], plus Gaussian noise."""
    angles = rng.uniform(0, 2 * np.pi, n)
    radii  = np.sqrt(rng.uniform(r_in**2, r_out**2, n))   # uniform in disk area
    pts    = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    pts   += rng.normal(0, sigma, pts.shape)
    return pts


def alpha_persistence(pts, top_k):
    """Top-k most persistent H1 points from alpha complex."""
    ac = gudhi.AlphaComplex(points=pts.tolist())
    st = ac.create_simplex_tree()
    st.compute_persistence()
    h1 = st.persistence_intervals_in_dimension(1)
    if len(h1) == 0:
        return np.zeros((0, 2))
    h1 = np.array(h1)
    h1 = h1[np.isfinite(h1[:, 1])]
    if len(h1) == 0:
        return np.zeros((0, 2))
    pers = h1[:, 1] - h1[:, 0]
    idx  = np.argsort(-pers)[:top_k]
    return h1[idx]


def add_outlier(dgm, ell):
    """Append a single off-diagonal pair (0, ell) to the diagram."""
    if dgm.shape[0] == 0:
        return np.array([[0.0, ell]])
    return np.vstack([dgm, [0.0, ell]])


# ── Embedding helpers ──────────────────────────────────────────────────

def uniform_grid_positions(L, K_target):
    """
    Offset grid (m, n) with m odd, n even, n >= m+3, restricted to [0, L]^2.
    Returns up to K_target positions (the smallest grid yielding >= K_target).
    Used here at K_target=11.
    """
    candidates = []
    # Sweep R from very fine (handles small L like the unperturbed L≈0.9)
    # to coarse (handles inflated L up to ~10).
    for R in np.arange(0.02, 2.0, 0.02):
        pts = []
        m = 1
        while m * R <= L + 1e-6:
            for n in range(m + 3, int((L + 1e-6) / R) + 1):
                if n % 2 == 0:
                    pts.append([m * R, n * R])
            m += 2
        if len(pts) >= K_target:
            candidates.append((len(pts), R, np.array(pts)))
    if not candidates:
        # Robust fallback: dense regular grid with y>x triangular filter,
        # densified until we have >= K_target points.
        side = max(int(np.ceil(np.sqrt(2 * K_target))) + 1, 5)
        xs = np.linspace(L * 0.05, L * 0.95, side)
        pts = np.array([(x, y) for x in xs for y in xs if y > x])
        return pts[:K_target]
    best = min(candidates, key=lambda c: c[0])
    return best[2][:K_target]


def fps_positions(diagrams, K, seed=0):
    """All-class farthest-point sampling on union of diagram points."""
    pts = []
    for d in diagrams:
        if len(d) > 0:
            pts.append(d)
    if not pts:
        raise ValueError("no diagram points")
    pool = np.vstack(pts)
    return farthest_point_sampling(pool, K, seed=seed)


def build_embedding(positions, L, tau, n_diagram=1):
    K = len(positions)
    D = cdist(positions, positions, metric='chebyshev')
    np.fill_diagonal(D, np.inf)
    nn = D.min(axis=1)
    radii = np.clip(0.75 * nn, tau / 2.0, 4.0 * tau)
    weights = np.full(K, 1.0 / np.sqrt(K))
    return NonUniformEmbedding(
        positions=positions, radii=radii, weights=weights,
        L=L, n_diagram=n_diagram,
    )


# ── Kernel + SVM ───────────────────────────────────────────────────────

def wlk_gram(X, Y=None, sigma=0.01):
    if Y is None:
        Y = X
    inv2s2 = 1.0 / (2 * sigma ** 2)
    G = np.zeros((X.shape[0], Y.shape[0]))
    for k in range(X.shape[1]):
        d = X[:, k:k+1] - Y[:, k:k+1].T
        G += np.exp(-d ** 2 * inv2s2)
    return G


def quantile_bandwidth(X_tr, q):
    n = X_tr.shape[0]
    if n > 500:
        idx = np.random.default_rng(0).choice(n, 500, replace=False)
        pd = pdist(X_tr[idx], 'euclidean')
    else:
        pd = pdist(X_tr, 'euclidean')
    pd = pd[pd > 0]
    if len(pd) == 0:
        return 1e-3
    return float(np.quantile(pd, q))


def cv_acc(X_tr, X_te, y_tr, y_te, fold_seed):
    """Inner CV picks C; outer test is X_te."""
    sigma = quantile_bandwidth(X_tr, Q_BANDWIDTH)
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=fold_seed)
    G_tr = wlk_gram(X_tr, sigma=sigma)
    best = (-np.inf, C_GRID[0])
    for C in C_GRID:
        accs = []
        for itri, itei in inner.split(G_tr, y_tr):
            svm = SVC(kernel='precomputed', C=C)
            svm.fit(G_tr[np.ix_(itri, itri)], y_tr[itri])
            accs.append(svm.score(G_tr[np.ix_(itei, itri)], y_tr[itei]))
        if np.mean(accs) > best[0]:
            best = (np.mean(accs), C)
    G_te = wlk_gram(X_te, X_tr, sigma=sigma)
    svm  = SVC(kernel='precomputed', C=best[1]).fit(G_tr, y_tr)
    return float(svm.score(G_te, y_te))


# ── Main ──────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    log("Generating annular point clouds …")
    clouds, labels = [], []
    for c, r_in in enumerate(INNER_RADII):
        for _ in range(N_PER_CLASS):
            clouds.append(annulus(r_in, OUTER, N_PTS, NOISE, rng))
            labels.append(c)
    labels = np.array(labels)

    log("Computing alpha persistence (top-30 H_1) …")
    base_dgms = [alpha_persistence(c, TOP_K_PERS) for c in clouds]

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(base_dgms, labels))

    rows = []
    for ell in L_VALUES:
        # Append outlier per cloud
        dgms_ell = [add_outlier(d, float(ell)) for d in base_dgms]

        # Domain L_eff (for embedding scope)
        Lmax = float(max(d[:, 1].max() for d in dgms_ell if len(d) > 0)) * 1.05
        # tau: mean half-persistence of the top-1 feature per UNPERTURBED
        # diagram. The outlier we appended dominates raw max-persistence
        # and alpha-complex noise dominates the all-features median, so
        # we derive tau from the strongest feature per original diagram
        # only -- this captures the discriminative-feature scale (hole
        # radii^2/4, ~0.06-0.18 for classes 0-2) rather than the noise
        # scale. With ~0.05 typical, radii get clipped to [0.025, 0.2],
        # large enough for grid landmarks at x>=0.02 to cover features
        # at (~0.002, 0.06-0.18).
        top1 = []
        for d in base_dgms:
            if len(d) > 0:
                p = (d[:, 1] - d[:, 0]) / 2.0
                top1.append(float(p.max()))
        tau = float(np.mean(top1)) if top1 else 0.05

        for placement in ('uniform', 'nonuniform'):
            fold_accs = []
            for fi, (tri, tei) in enumerate(folds):
                tr_dgms = [dgms_ell[i] for i in tri]
                te_dgms = [dgms_ell[i] for i in tei]

                if placement == 'uniform':
                    positions = uniform_grid_positions(Lmax, K_LANDMARKS)
                else:
                    positions = fps_positions(tr_dgms, K_LANDMARKS,
                                             seed=SEED + fi)

                if len(positions) < K_LANDMARKS:
                    fold_accs.append(np.nan); continue

                emb = build_embedding(positions[:K_LANDMARKS], Lmax, tau)
                X_tr = emb.embed_dataset(tr_dgms)
                X_te = emb.embed_dataset(te_dgms)
                acc  = cv_acc(X_tr, X_te, labels[tri], labels[tei], fi)
                fold_accs.append(acc)

            m, s = float(np.nanmean(fold_accs)*100), float(np.nanstd(fold_accs)*100)
            log(f"  ell={ell}  {placement:>10s}: {m:5.1f} +/- {s:4.1f} %")
            rows.append({
                'ell': ell, 'L_eff': Lmax, 'tau': tau,
                'placement': placement,
                'mean_acc_pct': m, 'std_acc_pct': s,
                'fold_accs': ';'.join(f'{a:.4f}' for a in fold_accs),
            })

    out = OUT_DIR / 'results.csv'
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    log(f"\nWrote {out} ({len(rows)} rows)")
    log(f"Total: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
