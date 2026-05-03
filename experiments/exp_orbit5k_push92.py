#!/usr/bin/env python3
"""
Push for 92% on Orbit5k — two untested levers:

  Lever A: Triple-filtration concat
    alpha + dens-k10 + dens-k15  (k15/k20 never tested in concat)
    alpha + dens-k10 + dens-k20

  Lever B: top-100 features instead of top-50
    More diagram points → richer signal for FPS

  Lever C: Both combined

All at the best known configs:
  K_per_filt ∈ {200, 300}
  α ∈ {2.5, 4.0}
  σ = 0.01 (best fixed) + q-tuned (q from inner CV)
  Classifiers: WLK-SVM, XGBoost

10-fold CV, seed 42.
Output: results/orbit5k_final_sweep/push92.csv
"""
import sys; sys.path.insert(0, '.')
import numpy as np
import pickle
import time
import csv
from pathlib import Path
from itertools import product
from scipy.spatial.distance import cdist, pdist
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from embedding.nonuniform import init_nonuniform_from_data
from utils.datasets import load_orbit5k

CACHE_DIR = Path('data/cache')
OUT_DIR = Path('results/orbit5k_final_sweep')
OUT_DIR.mkdir(parents=True, exist_ok=True)

C_GRID = [0.01, 0.1, 1, 10, 100, 1000]
Q_GRID = [0.75, 0.85, 0.90, 0.95, 0.99]
N_FOLDS = 10
SEED = 42
N_CLASSES = 5
N_JOBS = 10


def log(m):
    print(m, flush=True)


# ── Experiment configs ─────────────────────────────────────────────────

EXPERIMENTS = [
    # Lever A: triple concat (top-50)
    {
        'name': 'triple_k15_top50',
        'caches': [
            'orbit5k_diagrams_dim1_top50.pkl',
            'orbit5k_density_k10_H01_top50.pkl',
            'orbit5k_density_k15_H01_top50.pkl',
        ],
    },
    {
        'name': 'triple_k20_top50',
        'caches': [
            'orbit5k_diagrams_dim1_top50.pkl',
            'orbit5k_density_k10_H01_top50.pkl',
            'orbit5k_density_k20_H01_top50.pkl',
        ],
    },
    # Lever B: top-100 with baseline double concat
    {
        'name': 'double_k10_top100',
        'caches': [
            'orbit5k_diagrams_dim1_top100.pkl',
            'orbit5k_density_k10_H01_top50.pkl',  # density stays top-50 (only 49.8 avg)
        ],
    },
    # Lever C: triple concat + top-100 alpha
    {
        'name': 'triple_k15_top100',
        'caches': [
            'orbit5k_diagrams_dim1_top100.pkl',
            'orbit5k_density_k10_H01_top50.pkl',
            'orbit5k_density_k15_H01_top50.pkl',
        ],
    },
    {
        'name': 'triple_k20_top100',
        'caches': [
            'orbit5k_diagrams_dim1_top100.pkl',
            'orbit5k_density_k10_H01_top50.pkl',
            'orbit5k_density_k20_H01_top50.pkl',
        ],
    },
    # Baseline (reproduce): double concat top-50
    {
        'name': 'double_k10_top50_baseline',
        'caches': [
            'orbit5k_diagrams_dim1_top50.pkl',
            'orbit5k_density_k10_H01_top50.pkl',
        ],
    },
]

PARAM_GRID = [
    (200, 2.5),
    (200, 4.0),
    (300, 2.5),
    (300, 4.0),
]


# ── Helpers ────────────────────────────────────────────────────────────

def get_tau(dgms):
    p = []
    for d in dgms:
        if len(d) > 0:
            p.extend((d[:, 1] - d[:, 0]) / 2.0)
    return float(np.median(p))


def compute_L(dgms_list):
    return float(max(d[:, 1].max() for d in dgms_list if len(d) > 0) * 1.1)


def build_emb(dbc, K, L, shrink, tr_dgms):
    emb = init_nonuniform_from_data(dbc, K=K, L=L, n_diagram=1, seed=SEED)
    D = cdist(emb.positions, emb.positions, metric='chebyshev')
    np.fill_diagonal(D, np.inf)
    nn = D.min(axis=1)
    tau = get_tau(tr_dgms)
    emb.radii = np.clip(shrink * nn, tau / 2.0, 4.0 * tau)
    return emb


def wlk_gram(X, Y=None, sigma=0.01):
    if Y is None:
        Y = X
    inv2s2 = 1.0 / (2 * sigma ** 2)
    G = np.zeros((X.shape[0], Y.shape[0]))
    for k in range(X.shape[1]):
        d = X[:, k:k+1] - Y[:, k:k+1].T
        G += np.exp(-d ** 2 * inv2s2)
    return G


def adaptive_sigma(X_tr, q):
    n = X_tr.shape[0]
    if n > 1000:
        idx = np.random.default_rng(0).choice(n, size=1000, replace=False)
        pd = pdist(X_tr[idx], 'euclidean')
    else:
        pd = pdist(X_tr, 'euclidean')
    pd = pd[pd > 0]
    if len(pd) == 0:
        return 1e-6
    return float(np.quantile(pd, q))


def eval_xgb(X_tr, X_te, y_tr, y_te):
    if not HAS_XGB:
        return np.nan, np.nan
    scaler = StandardScaler().fit(X_tr)
    clf = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        use_label_encoder=False, eval_metric='mlogloss',
        verbosity=0, random_state=SEED,
    )
    clf.fit(scaler.transform(X_tr), y_tr)
    return float(clf.score(scaler.transform(X_te), y_te)), \
           float(clf.score(scaler.transform(X_tr), y_tr))


# ── Per-fold work ──────────────────────────────────────────────────────

def run_fold(fi, tri, tei, cache_names, dgms_all, labels, K, shrink):
    trl, tel = labels[tri], labels[tei]

    # Embed each filtration
    X_parts_tr, X_parts_te = [], []
    for cache_name in cache_names:
        dgms = dgms_all[cache_name]
        L = compute_L(dgms)
        tr_d = [dgms[i] for i in tri]
        te_d = [dgms[i] for i in tei]
        dbc = [[tr_d[j] for j in range(len(tri)) if trl[j] == c]
               for c in range(N_CLASSES)]
        emb = build_emb(dbc, K, L, shrink, tr_d)
        X_parts_tr.append(emb.embed_dataset(tr_d))
        X_parts_te.append(emb.embed_dataset(te_d))

    X_tr = np.hstack(X_parts_tr)
    X_te = np.hstack(X_parts_te)

    # --- Fixed σ=0.01 WLK ---
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=fi)
    G_fix = wlk_gram(X_tr, sigma=0.01)
    G_fix_te = wlk_gram(X_te, X_tr, sigma=0.01)
    best_c_fix, best_sc = C_GRID[0], 0.0
    for C in C_GRID:
        accs = []
        for itri, itei in inner.split(G_fix, trl):
            svm = SVC(kernel='precomputed', C=C)
            svm.fit(G_fix[np.ix_(itri, itri)], trl[itri])
            accs.append(svm.score(G_fix[np.ix_(itei, itri)], trl[itei]))
        if np.mean(accs) > best_sc:
            best_sc = np.mean(accs)
            best_c_fix = C
    svm_fix = SVC(kernel='precomputed', C=best_c_fix).fit(G_fix, trl)
    wlk_fix_te = svm_fix.score(G_fix_te, tel)
    wlk_fix_tr = svm_fix.score(G_fix, trl)

    # --- q-tuned WLK ---
    sigmas = {q: adaptive_sigma(X_tr, q) for q in Q_GRID}
    best_qt = (-np.inf, Q_GRID[0], C_GRID[0])
    for q in Q_GRID:
        sig = sigmas[q]
        G_q = wlk_gram(X_tr, sigma=sig)
        for C in C_GRID:
            accs = []
            for itri, itei in inner.split(G_q, trl):
                svm = SVC(kernel='precomputed', C=C)
                svm.fit(G_q[np.ix_(itri, itri)], trl[itri])
                accs.append(svm.score(G_q[np.ix_(itei, itri)], trl[itei]))
            if np.mean(accs) > best_qt[0]:
                best_qt = (np.mean(accs), q, C)
    best_q, best_c_q = best_qt[1], best_qt[2]
    best_sig = sigmas[best_q]
    G_qt = wlk_gram(X_tr, sigma=best_sig)
    G_qt_te = wlk_gram(X_te, X_tr, sigma=best_sig)
    svm_qt = SVC(kernel='precomputed', C=best_c_q).fit(G_qt, trl)
    wlk_qt_te = svm_qt.score(G_qt_te, tel)
    wlk_qt_tr = svm_qt.score(G_qt, trl)

    # --- XGBoost ---
    xgb_te, xgb_tr = eval_xgb(X_tr, X_te, trl, tel)

    return {
        'wlk_fix_te': wlk_fix_te, 'wlk_fix_tr': wlk_fix_tr,
        'wlk_qt_te': wlk_qt_te, 'wlk_qt_tr': wlk_qt_tr,
        'wlk_q': best_q, 'wlk_sigma': best_sig,
        'xgb_te': xgb_te, 'xgb_tr': xgb_tr,
    }


# ── Main ──────────────────────────────────────────────────────────────

def main():
    t_total = time.time()

    log("Loading Orbit5k …")
    _, labels = load_orbit5k()
    labels = np.array(labels)

    log("Loading diagram caches …")
    needed = set()
    for exp in EXPERIMENTS:
        needed.update(exp['caches'])
    dgms_all = {}
    for name in needed:
        path = CACHE_DIR / name
        if not path.exists():
            log(f"  MISSING: {path} — skipping experiments that need it")
            continue
        with open(path, 'rb') as f:
            dgms_all[name] = pickle.load(f)
        log(f"  {name}: {len(dgms_all[name])} diagrams, "
            f"avg {np.mean([len(d) for d in dgms_all[name]]):.1f} pts")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(labels, labels))

    total_configs = sum(1 for exp in EXPERIMENTS
                        if all(c in dgms_all for c in exp['caches'])) * len(PARAM_GRID)
    log(f"\n{'█'*72}")
    log(f"  PUSH FOR 92%: {total_configs} configs × {N_FOLDS} folds")
    log(f"{'█'*72}\n")

    rows = []
    for exp in EXPERIMENTS:
        if not all(c in dgms_all for c in exp['caches']):
            log(f"  [skip] {exp['name']}: missing cache")
            continue

        n_filts = len(exp['caches'])
        for K, shrink in PARAM_GRID:
            t0 = time.time()
            dim = K * n_filts
            fold_results = Parallel(n_jobs=N_JOBS, verbose=0)(
                delayed(run_fold)(fi, tri, tei, exp['caches'], dgms_all,
                                  labels, K, shrink)
                for fi, (tri, tei) in enumerate(folds)
            )

            wf = np.array([r['wlk_fix_te'] for r in fold_results]) * 100
            wft = np.array([r['wlk_fix_tr'] for r in fold_results]) * 100
            wq = np.array([r['wlk_qt_te'] for r in fold_results]) * 100
            wqt = np.array([r['wlk_qt_tr'] for r in fold_results]) * 100
            xg = np.array([r['xgb_te'] for r in fold_results]) * 100
            xgt = np.array([r['xgb_tr'] for r in fold_results]) * 100
            qs = [r['wlk_q'] for r in fold_results]

            row = {
                'experiment': exp['name'], 'n_filts': n_filts,
                'K': K, 'shrink': shrink, 'dim': dim,
                'wlk_fix_mean': wf.mean(), 'wlk_fix_std': wf.std(),
                'wlk_fix_train': wft.mean(),
                'wlk_qt_mean': wq.mean(), 'wlk_qt_std': wq.std(),
                'wlk_qt_train': wqt.mean(), 'wlk_qt_q': np.median(qs),
                'xgb_mean': xg.mean(), 'xgb_std': xg.std(),
                'xgb_train': xgt.mean(),
            }
            rows.append(row)

            best = max(wf.mean(), wq.mean(), xg.mean())
            marker = " ***" if best >= 91.5 else (" **" if best >= 91.0 else "")
            log(f"  {exp['name']:<25s} K={K:<3d} α={shrink:<4.2f} dim={dim:4d}  "
                f"WLK={wf.mean():.2f}±{wf.std():.2f}  "
                f"WLK-q={wq.mean():.2f}±{wq.std():.2f}(q̃={np.median(qs):.2f})  "
                f"XGB={xg.mean():.1f}±{xg.std():.1f}  "
                f"({time.time()-t0:.0f}s){marker}")

    # Save
    out = OUT_DIR / 'push92.csv'
    fieldnames = list(rows[0].keys())
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.4f}" if isinstance(r[k], float) else r[k])
                        for k in fieldnames})
    log(f"\nSaved → {out}")

    # Grand summary
    log(f"\n{'█'*72}")
    log(f"  RESULTS RANKED BY BEST CLASSIFIER")
    log(f"{'█'*72}")
    ranked = sorted(rows, key=lambda r: -max(r['wlk_fix_mean'], r['wlk_qt_mean'], r['xgb_mean']))
    for r in ranked[:10]:
        best_val = max(r['wlk_fix_mean'], r['wlk_qt_mean'], r['xgb_mean'])
        best_clf = 'WLK' if r['wlk_fix_mean'] == best_val else \
                   ('WLK-q' if r['wlk_qt_mean'] == best_val else 'XGB')
        log(f"  {best_val:.2f}% ({best_clf})  {r['experiment']:<25s} "
            f"K={r['K']} α={r['shrink']} dim={r['dim']}")

    log(f"\nTotal: {(time.time()-t_total)/60:.1f} min")


if __name__ == '__main__':
    main()
