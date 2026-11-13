"""
Outer-fold TEST accuracy of the PALACE landmark-kernel SVM at the
filtration selected on the TRAINING fold, for the chemical benchmarks.

Motivation (review 2026-09-24): the upper block of Paper II's
tab:graph_comparison was populated from `wlk_acc` in
results/filtration_grid/analysis/gamma_filt_*_full46_with_mah.csv, which
analyze_gamma_filt.py computes as the *inner* 5-fold CV accuracy on the
training graphs, and at the accuracy-maximizing filtration.  This script
produces the honest quantity: for every (seed, fold) it takes the
filtration chosen on the training fold by each selector below, fits the
WLK SVM on the training fold (C chosen by inner 5-fold CV on the training
gram) and scores it on the held-out fold.

Selectors (all computed on the training fold only, read from the
analysis CSV so the pick matches the paper's selector tables):
    mah      : argmax rho_mahalanobis            (closed-form, Sec. 4)
    gamma    : argmax gamma_over_sqrt_K          (closed-form, Alg. 2)
    innercv  : argmax wlk_acc                    (nested CV, accuracy-based)

Embedding/kernel configuration is identical to analyze_gamma_filt.py:
FPS with K=200, equal weights, alpha=0.75 nearest-neighbour radii,
sigma = 1e-3, StratifiedKFold(10, shuffle, random_state=42+seed).

Usage:
    python experiments/exp_outer_test_selected_filt.py --dataset MUTAG --workers 8
Output:
    results/filtration_grid/analysis/outer_test_selected_<dataset>.csv
    (one row per seed x fold x selector), plus a printed summary.
"""
from __future__ import annotations
import sys; sys.path.insert(0, '.')

import argparse
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from utils.datasets import load_dataset
from experiments.grid_common import build_graph_diagrams
from experiments import pool_chemical, pool_social
from experiments.analyze_gamma_filt import (
    _N_CLASSES, _POOL_MODULE, _auto_L, inner_cv_accuracy, C_GRID,
    K_LANDMARKS_DEFAULT, SIGMA_DEFAULT,
)
from embedding.nonuniform import init_nonuniform_from_data, WeightedLandmarkKernel

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / 'results' / 'filtration_grid' / 'analysis'

SELECTORS = {
    'mah':     'rho_mahalanobis',
    'gamma':   'gamma_over_sqrt_K',
    'innercv': 'wlk_acc',
}

_ANALYSIS_CSV = {
    'MUTAG': 'gamma_filt_mutag_full46_with_mah.csv',
    'COX2':  'gamma_filt_cox2_full46_with_mah.csv',
    'DHFR':  'gamma_filt_dhfr_full46_with_mah.csv',
    'PTC':   'gamma_filt_ptc_full46_with_mah.csv',
    'NCI1':  'gamma_filt_nci1_with_mah.csv',
    # lower block on the FPS embedding (2026-11-08 re-run; per-seed files merged by merge_seed_files.py)
    'PROTEINS': 'gamma_filt_proteins_with_mah.csv', 'DD': 'gamma_filt_dd_with_mah.csv',
    'IMDB-B': 'gamma_filt_imdb-b_with_mah.csv',    'IMDB-M': 'gamma_filt_imdb-m_with_mah.csv',
    'NCI109': 'gamma_filt_nci109_with_mah.csv',
}

_WSTATE = {}


def _worker_init(dgms_per_filt, labels, L, n_classes, dataset, K_landmarks, sigma):
    _WSTATE.update(dict(dgms_per_filt=dgms_per_filt, labels=labels, L=L,
                        n_classes=n_classes, dataset=dataset,
                        K_landmarks=K_landmarks, sigma=sigma))


def _best_C(G_tr, y_tr, rng_seed):
    """C maximizing inner 5-fold CV accuracy on the training gram."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rng_seed)
    best_C, best_acc = C_GRID[0], -1.0
    for C in C_GRID:
        accs = []
        for tr, te in skf.split(np.zeros(len(y_tr)), y_tr):
            svm = SVC(kernel='precomputed', C=C)
            svm.fit(G_tr[np.ix_(tr, tr)], y_tr[tr])
            accs.append(svm.score(G_tr[np.ix_(te, tr)], y_tr[te]))
        acc = float(np.mean(accs))
        if acc > best_acc:
            best_acc, best_C = acc, C
    return best_C, best_acc


def process_one(job):
    seed, fold, filt, selectors = job
    labels = _WSTATE['labels']; n_classes = _WSTATE['n_classes']
    dgms = _WSTATE['dgms_per_filt'].get(filt)
    if dgms is None:
        return {'dataset': _WSTATE['dataset'], 'seed': seed, 'fold': fold,
                'filt': filt, 'selectors': '|'.join(selectors),
                'test_acc': float('nan'), '_error': 'filt not in pool'}
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42 + seed)
    tr_idx, te_idx = list(skf.split(np.zeros_like(labels), labels))[fold]
    tr_dgms = [dgms[i] for i in tr_idx]; te_dgms = [dgms[i] for i in te_idx]
    y_tr = labels[tr_idx]; y_te = labels[te_idx]
    dbc = [[tr_dgms[i] for i in range(len(tr_dgms)) if y_tr[i] == c]
           for c in range(n_classes)]
    try:
        t0 = time.time()
        emb = init_nonuniform_from_data(dbc, K=_WSTATE['K_landmarks'], L=_WSTATE['L'],
                                        n_diagram=1, seed=42 + seed)
        X_tr = emb.embed_dataset(tr_dgms); X_te = emb.embed_dataset(te_dgms)
        wlk = WeightedLandmarkKernel(omega=np.ones(X_tr.shape[1]), sigma=_WSTATE['sigma'])
        G_tr = wlk.gram_matrix(X_tr); G_te = wlk.gram_matrix(X_te, X_tr)
        C, inner_acc = _best_C(G_tr, y_tr, rng_seed=42 + seed)
        svm = SVC(kernel='precomputed', C=C).fit(G_tr, y_tr)
        test_acc = float(svm.score(G_te, y_te))
        return {'dataset': _WSTATE['dataset'], 'seed': seed, 'fold': fold,
                'filt': filt, 'selectors': '|'.join(selectors),
                'n_train': len(tr_idx), 'n_test': len(te_idx), 'K': X_tr.shape[1],
                'C': C, 'inner_cv_acc': inner_acc, 'test_acc': test_acc,
                't_s': time.time() - t0}
    except Exception as e:  # noqa: BLE001
        return {'dataset': _WSTATE['dataset'], 'seed': seed, 'fold': fold,
                'filt': filt, 'selectors': '|'.join(selectors),
                'test_acc': float('nan'), '_error': repr(e)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=list(_ANALYSIS_CSV))
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--K', type=int, default=K_LANDMARKS_DEFAULT)
    p.add_argument('--sigma', type=float, default=SIGMA_DEFAULT)
    args = p.parse_args()

    ds = args.dataset
    ana = pd.read_csv(ANALYSIS_DIR / _ANALYSIS_CSV[ds])
    ana = ana[ana['dataset'] == ds].dropna(subset=['wlk_acc'])
    n_classes = _N_CLASSES[ds]

    # Per-fold picks from training-fold statistics.
    picks = {}  # (seed, fold, filt) -> [selector names]
    for (seed, fold), g in ana.groupby(['seed', 'fold']):
        for name, col in SELECTORS.items():
            sub = g.dropna(subset=[col])
            if sub.empty:
                continue
            filt = str(sub.loc[sub[col].idxmax(), 'filt'])
            picks.setdefault((int(seed), int(fold), filt), []).append(name)
    print(f'{ds}: {ana.seed.nunique()} seeds x {ana.fold.nunique()} folds; '
          f'{len(picks)} distinct (fold, filt) embeddings to fit', flush=True)

    graphs, labels = load_dataset(ds); labels = np.array(labels)
    pool = _POOL_MODULE[ds].POOL   # full pool, as in the *_full46 analysis runs
    t0 = time.time()
    dgms_per_filt = build_graph_diagrams(dataset_name=ds, graphs=graphs, pool=pool,
                                         extended=True, union_h0=True)
    L = _auto_L([d for dg in dgms_per_filt.values() for d in dg])
    print(f'  diagrams ready in {time.time()-t0:.1f}s, L={L:.3f}', flush=True)

    jobs = [(s, f, filt, sels) for (s, f, filt), sels in sorted(picks.items())]
    out = ANALYSIS_DIR / f'outer_test_selected_{ds.lower()}.csv'
    rows = []
    ctx = mp.get_context('fork')
    with ctx.Pool(args.workers, initializer=_worker_init,
                  initargs=(dgms_per_filt, labels, L, n_classes, ds, args.K, args.sigma)) as pl:
        for i, r in enumerate(pl.imap_unordered(process_one, jobs, chunksize=1), 1):
            rows.append(r)
            if i % max(1, len(jobs) // 10) == 0:
                print(f'  {i}/{len(jobs)}', flush=True)
    df = pd.DataFrame(rows)
    # explode selectors so each (seed, fold, selector) is one row
    df['selector'] = df['selectors'].str.split('|')
    df = df.explode('selector').drop(columns=['selectors'])
    df.to_csv(out, index=False)
    print(f'wrote {out}', flush=True)

    print(f'\n=== {ds}: outer-fold test accuracy (mean ± sd over seed x fold) ===')
    for sel in SELECTORS:
        sub = df[(df['selector'] == sel)].dropna(subset=['test_acc'])
        if sub.empty:
            continue
        print(f'  {sel:<8} n={len(sub):3d}  test {100*sub.test_acc.mean():5.1f} ± {100*sub.test_acc.std(ddof=1):4.1f}'
              f'   (inner-CV at pick {100*sub.inner_cv_acc.mean():5.1f})'
              f'   top filts: {sub.filt.value_counts().head(3).to_dict()}')
    err = df[df['test_acc'].isna()]
    if len(err):
        print(f'  {len(err)} failed rows; first error: {err.iloc[0].get("_error")}')


if __name__ == '__main__':
    main()
