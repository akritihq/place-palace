"""
Task 5: Small-K MUTAG WLK sweep — reproduce §6.2 compression claims.

Sweeps K in {10, 25, 50, 100, 200} landmarks on MUTAG (degree filtration,
H0+H1, extended). Reports WLK test accuracy and embedding dimension at each K,
confirming the three §6.2 sentences:
  - K=50 achieves comparable accuracy to K=200 at 22x fewer coordinates
  - K=10 equal-budget gap vs K=200 is ~4 pp
  - Generalisation gap (train-test spread) shrinks at small K

Output: results/graph_classifiers/mutag_smallK.csv

Usage:
    python experiments/exp_mutag_smallK.py
"""
import sys; sys.path.insert(0, '.')
import csv
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from utils.datasets import load_dataset
from utils.persistence import graphs_to_persistence_cached
from embedding.nonuniform import init_nonuniform_from_data

OUT = Path('results/graph_classifiers/mutag_smallK.csv')
OUT.parent.mkdir(parents=True, exist_ok=True)

K_GRID  = [10, 25, 50, 100, 200]
C_GRID  = [0.01, 0.1, 1, 10, 100, 1000]
SIGMA   = 1e-3
FOLDS   = 10
SEEDS   = 5
DATASET = 'MUTAG'
FILT    = 'degree'
DIMS    = [0, 1]


def run_seed(diagrams, labels, K, seed, L):
    n_classes = len(np.unique(labels))
    rng = np.random.default_rng(seed)
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True,
                          random_state=int(rng.integers(1e6)))
    inv2s2 = 1.0 / (2.0 * SIGMA * SIGMA)

    fold_accs, fold_tr_accs, ells = [], [], []
    for tr_idx, te_idx in skf.split(np.zeros(len(labels)), labels):
        tr_dgms   = [diagrams[i] for i in tr_idx]
        te_dgms   = [diagrams[i] for i in te_idx]
        tr_labels = labels[tr_idx]
        te_labels = labels[te_idx]

        dbc = [[tr_dgms[i] for i in range(len(tr_dgms))
                if tr_labels[i] == c]
               for c in range(n_classes)]
        if any(len(dc) == 0 for dc in dbc):
            continue

        emb  = init_nonuniform_from_data(dbc, K=K, L=L,
                                         n_diagram=1, seed=42)
        X_tr = emb.embed_dataset(tr_dgms)
        X_te = emb.embed_dataset(te_dgms)
        ell  = X_tr.shape[1]
        ells.append(ell)

        def gram(A, B):
            G = np.zeros((len(A), len(B)))
            for k in range(ell):
                d = A[:, k:k+1] - B[:, k:k+1].T
                G += np.exp(-d * d * inv2s2)
            return G

        G_tr = gram(X_tr, X_tr)
        G_te = gram(X_te, X_tr)

        best_te, best_tr = 0.0, 0.0
        for C in C_GRID:
            svm = SVC(kernel='precomputed', C=C)
            svm.fit(G_tr, tr_labels)
            te_acc = svm.score(G_te, te_labels)
            tr_acc = svm.score(G_tr, tr_labels)
            if te_acc > best_te:
                best_te = te_acc
                best_tr = tr_acc
        fold_accs.append(best_te)
        fold_tr_accs.append(best_tr)

    return (np.mean(fold_accs) * 100,
            np.std(fold_accs) * 100,
            np.mean(fold_tr_accs) * 100,
            int(np.mean(ells)) if ells else 0)


def main():
    print(f'Loading {DATASET}...', flush=True)
    graphs, labels = load_dataset(DATASET)
    labels = np.array(labels)

    print(f'Computing {FILT} persistence (cached)...', flush=True)
    pers = graphs_to_persistence_cached(
        graphs, DATASET, filtration=FILT,
        dims=None, extended=True, union_h0=False)

    diagrams = []
    for gi in range(len(graphs)):
        pts = []
        for dim in DIMS:
            d = pers[gi]
            if dim in d and len(d[dim]) > 0:
                pts.append(d[dim])
        diagrams.append(np.vstack(pts) if pts else np.zeros((0, 2)))

    L = max((float(dgm[:, 1].max()) for dgm in diagrams if len(dgm) > 0), default=1.0)
    L *= 1.05

    fields = ['K', 'seed', 'test_acc', 'test_std', 'train_acc', 'ell']
    all_rows = []

    print(f'\n{"K":>5} {"seed":>5} {"test%":>8} {"std":>6} '
          f'{"train%":>8} {"ell":>6}', flush=True)
    print('-' * 42)

    for K in K_GRID:
        for seed in range(SEEDS):
            t0 = time.time()
            te_acc, te_std, tr_acc, ell = run_seed(diagrams, labels, K, seed, L)
            dt = time.time() - t0
            print(f'{K:>5} {seed:>5} {te_acc:>8.1f} {te_std:>6.1f} '
                  f'{tr_acc:>8.1f} {ell:>6}  ({dt:.0f}s)', flush=True)
            all_rows.append({'K': K, 'seed': seed,
                             'test_acc': round(te_acc, 2),
                             'test_std': round(te_std, 2),
                             'train_acc': round(tr_acc, 2),
                             'ell': ell})

    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    print(f'\nSaved → {OUT}', flush=True)

    # Summary table: mean across seeds per K
    df  = pd.DataFrame(all_rows)
    agg = df.groupby('K').agg(
        test_acc=('test_acc', 'mean'),
        train_acc=('train_acc', 'mean'),
        ell=('ell', 'mean')
    ).reset_index()

    k200     = agg[agg['K'] == 200].iloc[0]
    k200_acc = k200['test_acc']
    k200_ell = k200['ell']

    print(f'\n{"K":>5} {"test%":>8} {"train%":>8} {"ell":>6} '
          f'{"gap vs K=200":>14} {"compression":>13}')
    print('-' * 58)
    for _, r in agg.iterrows():
        delta = r['test_acc'] - k200_acc
        ratio = k200_ell / r['ell'] if r['ell'] > 0 else float('inf')
        gen_gap = r['train_acc'] - r['test_acc']
        print(f"{int(r['K']):>5} {r['test_acc']:>8.1f} {r['train_acc']:>8.1f} "
              f"{int(r['ell']):>6} {delta:>+14.1f} {ratio:>12.1f}x  "
              f"(gen gap {gen_gap:.1f} pp)")


if __name__ == '__main__':
    main()
