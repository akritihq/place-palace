"""
Quick sanity check: does γ̂(k_L)/√K (data-dependent kernel margin / √dim)
correlate with 5-fold CV accuracy across (filtration, fusion, σ_q) configs?

Kernel: TRUE additive WLK, k_L(A,B) = sum_s exp(-(Phi_s(A) - Phi_s(B))^2 / 2σ²).
Placement: non-uniform FPS with K=200 landmarks, equal weights w_k = K^(-1/2).
Filtrations: alpha_H1, dens_H1, and their fusions (concat, pool).

Subsample: 500 diagrams per class × 5 = 2500. Top-50 per diagram.
"""
import sys; sys.path.insert(0, '.')
import numpy as np
import pickle
import time
import os
import csv
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from embedding.nonuniform import init_nonuniform_from_data, WeightedLandmarkKernel

# ── Config ───────────────────────────────────────────────────────────────
SEED = 42
N_PER_CLASS = 500           # subsample to 500/class × 5 = 2500
TOP_K = 50                  # top-50 persistent features per diagram
K_LANDMARKS = 200           # FPS landmarks per filtration
ALPHA = 1.0                 # radius factor (used inside init_nonuniform_from_data; shrink=0.75 baked in)
SIGMA_QS = [0.25, 0.50, 0.75, 0.95]
N_FOLDS = 5
C_GRID = [0.01, 0.1, 1, 10, 100]

ALPHA_PKL   = 'data/diagrams/Orbit5k_1000_alpha_H0H1.pkl'
DENSITY_PKL = 'data/diagrams/Orbit5k_1000_density_H0H1.pkl'
OUT_CSV     = 'results/orbit5k_gamma_sanity.csv'

N_CLASSES = 5
FRAME_L = 1.0   # Orbit5k lives in [0,1]²


def log(msg):
    print(msg, flush=True)


# ── Data loading ─────────────────────────────────────────────────────────

def load_and_subsample():
    """Load alpha and density H0H1 diagrams; subsample, top-50 filter, H1 only."""
    log('Loading diagrams...')
    with open(ALPHA_PKL, 'rb') as f:
        alpha = pickle.load(f)
    with open(DENSITY_PKL, 'rb') as f:
        dens = pickle.load(f)
    assert len(alpha) == 5000 and len(dens) == 5000

    # Orbit5k labels: 1000 per class, 5 classes sequential
    labels_full = np.array([c for c in range(N_CLASSES) for _ in range(1000)])

    # Stratified subsample: first N_PER_CLASS per class
    rng = np.random.default_rng(SEED)
    sel_idx = []
    for c in range(N_CLASSES):
        cls_idx = np.where(labels_full == c)[0]
        pick = rng.choice(cls_idx, size=N_PER_CLASS, replace=False)
        sel_idx.append(pick)
    sel_idx = np.concatenate(sel_idx)
    labels = labels_full[sel_idx]

    alpha_sub = [alpha[i] for i in sel_idx]
    dens_sub = [dens[i] for i in sel_idx]

    # Extract H1 only and top-50 most persistent
    def extract_h1_top50(dgms):
        out = []
        for d in dgms:
            h1 = d.get(1, np.zeros((0, 2)))
            if len(h1) > TOP_K:
                pers = h1[:, 1] - h1[:, 0]
                h1 = h1[np.argsort(pers)[::-1][:TOP_K]]
            out.append(h1)
        return out

    alpha_h1 = extract_h1_top50(alpha_sub)
    dens_h1 = extract_h1_top50(dens_sub)
    log(f'  alpha_H1: avg {np.mean([len(d) for d in alpha_h1]):.1f} pts/diagram')
    log(f'  dens_H1:  avg {np.mean([len(d) for d in dens_h1]):.1f} pts/diagram')
    return alpha_h1, dens_h1, labels


def pool_diagrams(dgms_list):
    """Pooled fusion: concatenate diagram points per sample. Expect dgms_list = [dgms_A, dgms_B]."""
    n = len(dgms_list[0])
    out = []
    for i in range(n):
        parts = [d[i] for d in dgms_list if len(d[i]) > 0]
        if parts:
            out.append(np.vstack(parts))
        else:
            out.append(np.zeros((0, 2)))
    return out


# ── Embedding & gram ─────────────────────────────────────────────────────

def build_fps_embedding(dgms_by_class, K):
    """Build FPS embedding of cardinality K."""
    return init_nonuniform_from_data(
        dgms_by_class, K=K, L=FRAME_L, n_diagram=1, seed=SEED
    )


def dgms_to_class_list(dgms, labels):
    """List[List[dgm]] grouped by class."""
    return [[dgms[i] for i in range(len(dgms)) if labels[i] == c]
            for c in range(N_CLASSES)]


def embed_dgms(emb, dgms):
    """Embed a list of diagrams. Returns (N, K)."""
    return emb.embed_dataset(dgms)


def wlk_gram_matrix(X, sigma):
    """TRUE additive WLK: G[i,j] = sum_s exp(-(X[i,s]-X[j,s])^2 / 2σ²)."""
    K = X.shape[1]
    wlk = WeightedLandmarkKernel(omega=np.ones(K), sigma=sigma)
    return wlk.gram_matrix(X)


def adaptive_sigma_wlk(X, q):
    """σ from quantile of pairwise L2 distances (matches existing pipeline)."""
    n = X.shape[0]
    idx = np.random.default_rng(0).choice(n, size=min(n, 800), replace=False)
    Xs = X[idx]
    sq = np.sum(Xs**2, axis=1)
    D2 = sq[:, None] + sq[None, :] - 2 * Xs @ Xs.T
    iu = np.triu_indices(len(Xs), k=1)
    d2 = D2[iu]
    d2 = d2[d2 > 0]
    return float(np.sqrt(np.quantile(d2, q))) if len(d2) else 1e-6


# ── γ̂ from gram ─────────────────────────────────────────────────────────

def empirical_gamma(G, labels):
    """γ̂ = ½ min_{c≠c'} ||μ̂_c - μ̂_c'||_H via UNBIASED MMD² on gram G.

    For additive WLK, G[i,i] = K (# landmarks), which would badly inflate
    within-class gram means under the biased estimator. We use MMD²_u that
    excludes diagonals from within-class blocks."""
    classes = np.unique(labels)
    diag_G = np.diag(G)
    min_mmd = np.inf
    for i, c in enumerate(classes):
        idx_c = np.where(labels == c)[0]
        n_c = len(idx_c)
        Kcc_sum = G[np.ix_(idx_c, idx_c)].sum() - diag_G[idx_c].sum()
        Kcc_u = Kcc_sum / (n_c * (n_c - 1))
        for c2 in classes[i+1:]:
            idx_c2 = np.where(labels == c2)[0]
            n_c2 = len(idx_c2)
            Kc2c2_sum = G[np.ix_(idx_c2, idx_c2)].sum() - diag_G[idx_c2].sum()
            Kc2c2_u = Kc2c2_sum / (n_c2 * (n_c2 - 1))
            Kcc2 = G[np.ix_(idx_c, idx_c2)].mean()   # no within-class diag
            mmd2 = Kcc_u - 2*Kcc2 + Kc2c2_u
            mmd = np.sqrt(max(mmd2, 0.0))
            if mmd < min_mmd:
                min_mmd = mmd
    return 0.5 * min_mmd


# ── CV accuracy ──────────────────────────────────────────────────────────

def cv_accuracy(G, labels):
    """5-fold stratified CV with precomputed gram. Returns (mean, std) in %."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    accs = []
    for tr, te in skf.split(np.zeros(len(labels)), labels):
        G_tr = G[np.ix_(tr, tr)]
        G_te = G[np.ix_(te, tr)]
        best = 0.0
        for C in C_GRID:
            svm = SVC(kernel='precomputed', C=C)
            svm.fit(G_tr, labels[tr])
            acc = svm.score(G_te, labels[te])
            if acc > best:
                best = acc
        accs.append(best)
    return 100*np.mean(accs), 100*np.std(accs)


# ── Main sweep ───────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    alpha_h1, dens_h1, labels = load_and_subsample()
    log(f'Subsample: n={len(labels)}, {N_CLASSES} classes')

    # Build per-class lists for FPS initialization
    alpha_by_class = dgms_to_class_list(alpha_h1, labels)
    dens_by_class  = dgms_to_class_list(dens_h1,  labels)

    pool_dgms     = pool_diagrams([alpha_h1, dens_h1])
    pool_by_class = dgms_to_class_list(pool_dgms, labels)

    # Build embeddings (one per filt_object × placement). Placement = FPS K=K_LANDMARKS.
    log(f'\nBuilding FPS embeddings (K={K_LANDMARKS} per filtration) …')
    t0 = time.time()
    emb_alpha = build_fps_embedding(alpha_by_class, K_LANDMARKS)
    emb_dens  = build_fps_embedding(dens_by_class,  K_LANDMARKS)
    emb_pool  = build_fps_embedding(pool_by_class,  K_LANDMARKS)
    log(f'  embeddings built in {time.time()-t0:.1f}s')

    # Precompute embeddings for singles (and reuse for concat)
    log('\nComputing embeddings on all diagrams …')
    t0 = time.time()
    X_alpha = embed_dgms(emb_alpha, alpha_h1)     # (n, K)
    X_dens  = embed_dgms(emb_dens,  dens_h1)      # (n, K)
    X_pool  = embed_dgms(emb_pool,  pool_dgms)    # (n, K)
    log(f'  embeddings in {time.time()-t0:.1f}s')

    # Concat is alpha ⊕ dens → (n, 2K); pool uses shared landmarks on pooled diagrams
    X_concat = np.hstack([X_alpha, X_dens])
    log(f'  dims — alpha:{X_alpha.shape[1]}  dens:{X_dens.shape[1]}  '
        f'concat:{X_concat.shape[1]}  pool:{X_pool.shape[1]}')

    configs = [
        ('alpha_H1',            X_alpha),
        ('dens_H1',             X_dens),
        ('alpha_H1+dens_H1 (concat)', X_concat),
        ('alpha_H1+dens_H1 (pool)',   X_pool),
    ]

    rows = []
    header = ['filt_obj', 'K', 'sigma_q', 'sigma', 'gamma_hat',
              'gamma_over_sqrtK', 'acc_mean', 'acc_std']
    Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)

    for filt_name, X in configs:
        K = X.shape[1]
        log(f'\n── {filt_name}  (K={K}) ──')
        for q in SIGMA_QS:
            t0 = time.time()
            sigma = adaptive_sigma_wlk(X, q)
            G = wlk_gram_matrix(X, sigma)           # (n, n)
            gamma = empirical_gamma(G, labels)
            ratio = gamma / np.sqrt(K)
            acc_mean, acc_std = cv_accuracy(G, labels)
            dt = time.time() - t0
            log(f'  σ_q={q:.2f}  σ={sigma:.5f}  γ̂={gamma:.4f}  '
                f'γ̂/√K={ratio:.5f}  acc={acc_mean:.2f}±{acc_std:.2f}  '
                f'[{dt:.1f}s]')
            rows.append([filt_name, K, q, sigma, gamma, ratio, acc_mean, acc_std])

    # Save CSV
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    log(f'\nSaved → {OUT_CSV}')

    # Spearman correlations
    gammas  = np.array([r[4] for r in rows])
    ratios  = np.array([r[5] for r in rows])
    accs    = np.array([r[6] for r in rows])
    r_g,  p_g  = spearmanr(gammas, accs)
    r_gk, p_gk = spearmanr(ratios, accs)
    log(f'\n── Spearman rank correlation (pooled across all {len(rows)} configs) ──')
    log(f'  γ̂        vs acc:  r={r_g:.3f}  p={p_g:.2e}')
    log(f'  γ̂/√K     vs acc:  r={r_gk:.3f}  p={p_gk:.2e}')

    log(f'\nTotal wall time: {time.time()-t_start:.1f}s')


if __name__ == '__main__':
    main()
