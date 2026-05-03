"""
Reproduce Paper II's 90.4% Orbit5k headline: alpha H0+H1 + density k=10 concat,
K=200 per filt, shrink α=1.75, σ=0.001, 10-fold CV.

Also computes γ̂ (unbiased MMD via empirical class means in the WLK RKHS) on each
training fold, so we can see whether γ̂ aligns with the winning config's accuracy.

Builds diagram caches if absent:
  - data/cache/orbit5k_diagrams_dim1_top50.pkl        (alpha H0+H1)
  - data/cache/orbit5k_density_k10_H01_top50.pkl      (density k=10, H0+H1)
"""
import sys; sys.path.insert(0, '.')
import numpy as np
import pickle
import time
import csv
from pathlib import Path
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from embedding.nonuniform import init_nonuniform_from_data
from utils.datasets import load_orbit5k
from utils.persistence import pointcloud_to_persistence, _density_persistence

CACHE_DIR = Path('data/cache')
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Paper's winning config
K = 200
SHRINK = 1.75
SIGMA = 0.001
C_GRID = [0.01, 0.1, 1, 10, 100, 1000]
N_FOLDS = 10
SEED = 42
N_CLASSES = 5


def log(m):
    print(m, flush=True)


# ── Diagram cache builders ───────────────────────────────────────────────

def _alpha_top50(pts):
    dgms = pointcloud_to_persistence(pts, max_dim=1, method='alpha')
    parts = [d for d in dgms if len(d) > 0]
    if not parts:
        return np.zeros((0, 2))
    all_pts = np.vstack(parts)
    if len(all_pts) > 50:
        pers = all_pts[:, 1] - all_pts[:, 0]
        all_pts = all_pts[np.argsort(pers)[::-1][:50]]
    return all_pts


def _density_k10_top50(pts):
    dgms = _density_persistence(pts, max_dim=1, k=10)
    parts = [d for d in dgms if len(d) > 0]
    if not parts:
        return np.zeros((0, 2))
    all_pts = np.vstack(parts)
    # Ensure non-negative (shift if needed)
    if len(all_pts) and all_pts[:, 0].min() < 0:
        shift = -all_pts[:, 0].min() + 0.01
        all_pts = all_pts + shift
    all_pts = all_pts[all_pts[:, 1] > all_pts[:, 0]]
    if len(all_pts) > 50:
        pers = all_pts[:, 1] - all_pts[:, 0]
        all_pts = all_pts[np.argsort(pers)[::-1][:50]]
    return all_pts


def build_cache(cache_path, builder_fn, pcs, label):
    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            dgms = pickle.load(f)
        log(f'  cached: {label} ({len(dgms)} diagrams)')
        return dgms
    log(f'  building {label} (parallel, n_jobs=-1) …')
    t0 = time.time()
    dgms = Parallel(n_jobs=-1, verbose=5)(
        delayed(builder_fn)(pts) for pts in pcs
    )
    with open(cache_path, 'wb') as f:
        pickle.dump(dgms, f)
    log(f'  built {label} in {time.time()-t0:.1f}s, '
        f'avg {np.mean([len(d) for d in dgms]):.1f} pts/diagram')
    return dgms


# ── WLK gram (true additive, σ fixed) ───────────────────────────────────

def wlk_gram(X, Y=None, sigma=SIGMA):
    """G[i,j] = sum_k exp(-(X[i,k] - Y[j,k])^2 / 2σ²)."""
    if Y is None:
        Y = X
    inv2s2 = 1.0 / (2 * sigma ** 2)
    K_dim = X.shape[1]
    G = np.zeros((X.shape[0], Y.shape[0]))
    for k in range(K_dim):
        d = X[:, k:k+1] - Y[:, k:k+1].T
        G += np.exp(-d ** 2 * inv2s2)
    return G


def unbiased_gamma(G, labels):
    """γ̂ = ½ min_{c≠c'} ||μ̂_c - μ̂_c'||_H via unbiased MMD²."""
    classes = np.unique(labels)
    diag_G = np.diag(G)
    min_mmd = np.inf
    for i, c in enumerate(classes):
        idx_c = np.where(labels == c)[0]
        n_c = len(idx_c)
        sum_cc = G[np.ix_(idx_c, idx_c)].sum() - diag_G[idx_c].sum()
        Kcc_u = sum_cc / (n_c * (n_c - 1))
        for c2 in classes[i+1:]:
            idx_c2 = np.where(labels == c2)[0]
            n_c2 = len(idx_c2)
            sum_c2c2 = G[np.ix_(idx_c2, idx_c2)].sum() - diag_G[idx_c2].sum()
            Kc2c2_u = sum_c2c2 / (n_c2 * (n_c2 - 1))
            Kcc2 = G[np.ix_(idx_c, idx_c2)].mean()
            mmd2 = Kcc_u - 2*Kcc2 + Kc2c2_u
            mmd = np.sqrt(max(mmd2, 0.0))
            if mmd < min_mmd:
                min_mmd = mmd
    return 0.5 * min_mmd


# ── Per-fold: build embedding, gram, γ̂, fit SVM ─────────────────────────

def build_emb_with_shrink(dbc, K, L, shrink, tr_dgms):
    emb = init_nonuniform_from_data(dbc, K=K, L=L, n_diagram=1, seed=SEED)
    D = cdist(emb.positions, emb.positions, metric='chebyshev')
    np.fill_diagonal(D, np.inf)
    nn = D.min(axis=1)
    # τ from training diagrams
    pers = []
    for d in tr_dgms:
        if len(d):
            pers.extend((d[:, 1] - d[:, 0]) / 2.0)
    tau = float(np.median(pers)) if pers else 0.01
    emb.radii = np.clip(shrink * nn, tau / 2.0, 4.0 * tau)
    return emb


def run_fold(fi, tri, tei, dgms_a, dgms_d, labels, L_a, L_d):
    trl, tel = labels[tri], labels[tei]
    tr_a = [dgms_a[i] for i in tri]; te_a = [dgms_a[i] for i in tei]
    tr_d = [dgms_d[i] for i in tri]; te_d = [dgms_d[i] for i in tei]

    dbc_a = [[tr_a[j] for j in range(len(tri)) if trl[j] == c]
             for c in range(N_CLASSES)]
    dbc_d = [[tr_d[j] for j in range(len(tri)) if trl[j] == c]
             for c in range(N_CLASSES)]

    emb_a = build_emb_with_shrink(dbc_a, K, L_a, SHRINK, tr_a)
    emb_d = build_emb_with_shrink(dbc_d, K, L_d, SHRINK, tr_d)

    Xtr = np.hstack([emb_a.embed_dataset(tr_a), emb_d.embed_dataset(tr_d)])
    Xte = np.hstack([emb_a.embed_dataset(te_a), emb_d.embed_dataset(te_d)])
    K_dim = Xtr.shape[1]

    G_tr = wlk_gram(Xtr)
    G_te = wlk_gram(Xte, Xtr)

    gamma = unbiased_gamma(G_tr, trl)

    # Inner CV for C
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=fi)
    best_c, best_inner = C_GRID[0], 0.0
    for C in C_GRID:
        accs_in = []
        for itri, itei in inner.split(G_tr, trl):
            svm = SVC(kernel='precomputed', C=C)
            svm.fit(G_tr[np.ix_(itri, itri)], trl[itri])
            accs_in.append(svm.score(G_tr[np.ix_(itei, itri)], trl[itei]))
        if np.mean(accs_in) > best_inner:
            best_inner = np.mean(accs_in); best_c = C

    svm = SVC(kernel='precomputed', C=best_c).fit(G_tr, trl)
    test_acc = svm.score(G_te, tel)
    train_acc = svm.score(G_tr, trl)
    return fi, test_acc, train_acc, best_c, gamma, K_dim


def main():
    t0 = time.time()
    log('Loading Orbit5k point clouds …')
    pcs, labels = load_orbit5k()
    labels = np.array(labels)
    log(f'  {len(pcs)} point clouds, {N_CLASSES} classes')

    # Build/load caches
    log('\nBuilding diagram caches …')
    dgms_a = build_cache(
        CACHE_DIR / 'orbit5k_diagrams_dim1_top50.pkl',
        _alpha_top50, pcs, 'alpha H0+H1 top-50')
    dgms_d = build_cache(
        CACHE_DIR / 'orbit5k_density_k10_H01_top50.pkl',
        _density_k10_top50, pcs, 'density k=10 H0+H1 top-50')

    L_a = max(d[:, 1].max() for d in dgms_a if len(d) > 0) * 1.1
    L_d = max(d[:, 1].max() for d in dgms_d if len(d) > 0) * 1.1
    log(f'  L_alpha={L_a:.4f}  L_dens={L_d:.4f}')

    # Stratified 10-fold
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(labels, labels))

    log(f'\nRunning {N_FOLDS}-fold CV at K={K}, α={SHRINK}, σ={SIGMA} (parallel) …')
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(run_fold)(fi, tri, tei, dgms_a, dgms_d, labels, L_a, L_d)
        for fi, (tri, tei) in enumerate(folds)
    )

    # Aggregate
    log('\n── Per-fold results ──')
    for fi, te_acc, tr_acc, c, gamma, K_dim in sorted(results):
        log(f'  fold {fi}  test={te_acc*100:.2f}%  train={tr_acc*100:.2f}%  '
            f'C={c:<5}  γ̂={gamma:.4f}  (K={K_dim})')

    test_accs = np.array([r[1] for r in results]) * 100
    train_accs = np.array([r[2] for r in results]) * 100
    gammas = np.array([r[4] for r in results])
    K_dim = results[0][5]

    mean_te, std_te = test_accs.mean(), test_accs.std()
    mean_tr = train_accs.mean()
    gamma_mean = gammas.mean()
    gamma_over_sqrtK = gamma_mean / np.sqrt(K_dim)

    log(f'\n── Summary ──')
    log(f'  Test accuracy : {mean_te:.2f} ± {std_te:.2f}% (paper: 90.4 ± 1.1%)')
    log(f'  Train accuracy: {mean_tr:.2f}%')
    log(f'  γ̂ (mean over folds)    : {gamma_mean:.4f}')
    log(f'  γ̂/√K (K={K_dim})       : {gamma_over_sqrtK:.5f}')

    # Save
    out = RESULTS_DIR / 'orbit5k_reproduce_90.csv'
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['fold', 'test_acc', 'train_acc', 'C', 'gamma_hat', 'K_dim'])
        for fi, te, tr, c, g, kd in sorted(results):
            w.writerow([fi, 100*te, 100*tr, c, g, kd])
        w.writerow([])
        w.writerow(['mean', mean_te, mean_tr, '', gamma_mean, K_dim])
        w.writerow(['std', std_te, train_accs.std(), '', gammas.std(), ''])
    log(f'\nSaved → {out}')
    log(f'\nTotal: {(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
