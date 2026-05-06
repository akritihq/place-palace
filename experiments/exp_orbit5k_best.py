"""
Orbit5k: Fine-grained grid search on alpha_H1 with tau_method='auto'.
Sweep N, sigma_q, and C for best accuracy.
"""
import sys; sys.path.insert(0, '.')
import numpy as np
import pickle
import csv
import time
import os
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from embedding.embedding import init_from_dataset, adaptive_sigma
from utils.datasets import load_orbit5k

CACHE = 'data/cache'
EMB_CACHE = 'data/cache/emb_fisher_auto'
SEED = 42
N_FOLDS = 10
N_CLASSES = 5
N_JOBS = 10
TAU_METHOD = 'auto'

N_GRID = [5, 8, 10, 12, 15, 20]
Q_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
C_GRID = [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000]


def log(msg):
    print(msg, flush=True)


def embed_fold(dgms, labels, tr_idx, te_idx, N_scales, cache_key):
    cache_path = os.path.join(EMB_CACHE, f'{cache_key}.npz')
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        return data['X_tr'], data['X_te']
    tr_dgms = [dgms[i] for i in tr_idx]
    tr_labels = labels[tr_idx]
    dbc = [[tr_dgms[j] for j in range(len(tr_idx)) if tr_labels[j] == c]
           for c in range(N_CLASSES)]
    emb = init_from_dataset(dbc, N_scales=N_scales, n_diagram=1,
                            tau_method=TAU_METHOD)
    X_tr = emb.embed_dataset(tr_dgms)
    X_te = emb.embed_dataset([dgms[i] for i in te_idx])
    np.savez_compressed(cache_path, X_tr=X_tr, X_te=X_te)
    return X_tr, X_te


def wlk_gram(X1, X2, inv2s2):
    """WLK gram: K[i,j] = sum_s exp(-(X1[i,s]-X2[j,s])^2 * inv2s2)."""
    n1, ell = X1.shape
    n2 = X2.shape[0]
    K = np.zeros((n1, n2))
    chunk = max(1, min(100, int(4e9 / (n1 * n2 * 8))))
    for start in range(0, ell, chunk):
        end = min(start + chunk, ell)
        diff = X1[:, start:end, None] - X2[:, start:end, None].transpose(2, 1, 0)
        K += np.exp(-diff**2 * inv2s2).sum(axis=1)
        del diff
    return K


def svm_one_fold(X_tr, X_te, y_tr, y_te, q_grid, c_grid):
    """Run SVM for all (q, C) combos on one fold using WLK kernel."""
    # Sigma from L2 pairwise distances (adaptive_sigma heuristic)
    D2_triu = np.sum(X_tr**2, axis=1)[:, None] + np.sum(X_tr**2, axis=1)[None, :] - 2 * X_tr @ X_tr.T
    triu = np.sqrt(np.maximum(D2_triu[np.triu_indices(D2_triu.shape[0], k=1)], 0))
    del D2_triu

    results = {}
    for q in q_grid:
        sigma = float(np.quantile(triu, q))
        if sigma <= 0:
            sigma = 1e-8
        inv2s2 = 1.0 / (2 * sigma**2)
        K_tr = wlk_gram(X_tr, X_tr, inv2s2)
        K_te = wlk_gram(X_te, X_tr, inv2s2)
        for C in c_grid:
            svm = SVC(kernel='precomputed', C=C)
            svm.fit(K_tr, y_tr)
            results[(q, C)] = svm.score(K_te, y_te)
        del K_tr, K_te
    return results


if __name__ == '__main__':
    t_start = time.time()
    _, labels = load_orbit5k()
    labels = np.array(labels)
    os.makedirs(EMB_CACHE, exist_ok=True)

    log("Loading alpha_H1 diagrams...")
    with open(f'{CACHE}/orbit5k_diagrams_dim1_top50.pkl', 'rb') as f:
        dgms = pickle.load(f)

    rng = np.random.default_rng(SEED)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                          random_state=int(rng.integers(1e6)))
    folds = list(skf.split(range(len(labels)), labels))

    all_results = []

    for N in N_GRID:
        log(f"\nN={N}: embedding folds...")
        t0 = time.time()

        def _embed(fi, tr_idx, te_idx):
            key = f'alpha_H1_N{N}_{TAU_METHOD}_fold{fi}'
            return embed_fold(dgms, labels, tr_idx, te_idx, N, key)

        fold_data = Parallel(n_jobs=N_JOBS)(
            delayed(_embed)(fi, tr_idx, te_idx)
            for fi, (tr_idx, te_idx) in enumerate(folds))
        dim = fold_data[0][0].shape[1]
        log(f"  embedded in {time.time()-t0:.0f}s (dim={dim})")

        log(f"  SVM grid: {len(Q_GRID)} q × {len(C_GRID)} C = {len(Q_GRID)*len(C_GRID)} combos")
        fold_svm = Parallel(n_jobs=N_JOBS)(
            delayed(svm_one_fold)(
                fold_data[fi][0], fold_data[fi][1],
                labels[folds[fi][0]], labels[folds[fi][1]],
                Q_GRID, C_GRID)
            for fi in range(N_FOLDS))

        best_acc, best_q, best_C = 0, 0, 0
        for q in Q_GRID:
            for C in C_GRID:
                accs = [fold_svm[fi][(q, C)] for fi in range(N_FOLDS)]
                acc = np.mean(accs) * 100
                std = np.std(accs) * 100
                all_results.append({'N': N, 'dim': dim, 'sigma_q': q, 'C': C,
                                    'acc_mean': acc, 'acc_std': std})
                if acc > best_acc:
                    best_acc, best_std, best_q, best_C = acc, std, q, C

        log(f"  Best: q={best_q:.2f} C={best_C}  Acc={best_acc:.1f}±{best_std:.1f}%")
        del fold_data

    # Summary
    log(f"\n{'='*70}")
    log("TOP 20 CONFIGURATIONS")
    log(f"{'='*70}")
    rs = sorted(all_results, key=lambda x: x['acc_mean'], reverse=True)
    for r in rs[:20]:
        log(f"  N={r['N']:<3d} dim={r['dim']:<5d} q={r['sigma_q']:.2f} C={r['C']:<8g}  "
            f"Acc={r['acc_mean']:.1f}±{r['acc_std']:.1f}%")

    elapsed = time.time() - t_start
    log(f"\nTotal time: {elapsed/60:.1f} min")

    with open('results/orbit5k_best.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['N', 'dim', 'sigma_q', 'C', 'acc_mean', 'acc_std'])
        w.writeheader()
        w.writerows(all_results)
    log("Saved → results/orbit5k_best.csv")
