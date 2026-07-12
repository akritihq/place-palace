"""
Filtration grid on PROTEINS (Paper I, Proposition 4 validation + headline
accuracy).

Protocol: 10-fold stratified CV × 5 seeds × 54 filtration configs (shared
social pool — see pool_social.py) × {tau_method = crossing, proxy} ×
4 classifiers (Linear SVM, RBF SVM, WLK, Nearest Centroid).  Records
η̂ = Δ̂/√ℓ alongside every accuracy.

Ordinary persistence, union of sublevel + superlevel H0 (Wang 2019
social/IMDB convention).

Output:
    results/filtration_grid/proteins_records.csv
    results/filtration_grid/proteins_cache/seed{s}_fold{f}/{filt}__{tau}.npz

Run:
    python experiments/exp_grid_proteins.py
    python experiments/exp_grid_proteins.py --seeds 0 --tau-methods crossing
    python experiments/exp_grid_proteins.py --force
"""
from __future__ import annotations
import sys; sys.path.insert(0, '.')

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from utils.datasets import load_dataset
from experiments.grid_common import (
    FIELDNAMES, N_SCALES_GRID, MAX_LANDMARKS, WLK_Q_GRID,
    build_graph_diagrams, embed_fold, evaluate_all,
    cache_path_for, save_embedding_cache,
)
from experiments.pool_social import ATOMIC_SINGLES, ATOMIC_PAIRS, POOL


ROOT      = Path(__file__).resolve().parents[1]
OUT_DIR   = ROOT / 'results' / 'filtration_grid'
CACHE_DIR = OUT_DIR / 'proteins_cache'
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DATASET_NAME = 'PROTEINS'
N_CLASSES    = 2
N_FOLDS      = 10
SEED_BASE    = 42


# Filtration pool is shared across all social / protein datasets (IMDB-B,
# IMDB-M, PROTEINS, DD, REDDIT-5K) — see experiments/pool_social.py for
# spec.  Ordinary persistence, H0 = sublevel ∪ superlevel (Wang §B.2).
_ATOMIC_SINGLES = ATOMIC_SINGLES
_ATOMIC_PAIRS   = ATOMIC_PAIRS
_POOL           = POOL


def log(msg: str) -> None:
    print(msg, flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# Main driver
# ═══════════════════════════════════════════════════════════════════════════

def run(seeds: list, tau_methods: list, n_scales_list: list,
        n_folds: int, force: bool, output: Path,
        filt_only: list[str] | None = None) -> None:
    grand_t0 = time.time()
    pool = _POOL
    if filt_only:
        keep = set(filt_only)
        pool = [cfg for cfg in _POOL if cfg['name'] in keep]
        if not pool:
            raise ValueError(
                f'[PROTEINS] --filt-only {filt_only} matched 0 configs; '
                f'available: {sorted(c["name"] for c in _POOL)[:5]}...'
            )
        log(f'[PROTEINS] --filt-only restriction → {len(pool)}/{len(_POOL)} configs')
    log(f'\n[PROTEINS] filtration grid — {len(_ATOMIC_SINGLES)} atomic + '
        f'{len(_ATOMIC_PAIRS)} pairs = {len(_POOL)} configs')
    log(f'  seeds       : {seeds}')
    log(f'  tau_methods : {tau_methods}')
    log(f'  n_scales    : {n_scales_list}')
    log(f'  n_folds     : {n_folds}')
    log(f'  output      : {output}')

    graphs, labels = load_dataset(DATASET_NAME)
    labels = np.array(labels)
    log(f'  {len(graphs)} graphs, {len(np.unique(labels))} classes')

    # Wang 2019 social convention: union of sublevel + superlevel H0
    log('\n── building / loading atomic diagram caches (union_h0=True) ──')
    diagrams_per_filt = build_graph_diagrams(
        DATASET_NAME, graphs, pool,
        extended=False, union_h0=True,
    )

    header_needed = not output.exists()
    f_out = open(output, 'a', newline='')
    writer = csv.DictWriter(f_out, fieldnames=FIELDNAMES)
    if header_needed:
        writer.writeheader()

    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                              random_state=SEED_BASE + seed)
        folds = list(skf.split(np.arange(len(labels)), labels))
        log(f'\n=== seed {seed}  ({n_folds} folds) ===')

        for fi, (tr_idx, te_idx) in enumerate(folds):
            log(f'\n  -- fold {fi} (train={len(tr_idx)}, test={len(te_idx)}) --')
            for cfg in pool:
                filt_name = cfg['name']
                diagrams = diagrams_per_filt[filt_name]
                for tau in tau_methods:
                    for n_sc in n_scales_list:
                        cache_path = cache_path_for(
                            CACHE_DIR, DATASET_NAME,
                            seed=seed, fold=fi, filt=filt_name, tau=tau,
                            n_scales=n_sc,
                        )
                        t_embed_0 = time.time()
                        if cache_path.exists() and not force:
                            data = np.load(cache_path, allow_pickle=False)
                            X_tr = data['X_tr']; X_te = data['X_te']
                            y_tr = data['y_tr']; y_te = data['y_te']
                            ell = int(data['ell'])
                            R_max_tr = float(data['R_max_tr'])
                        else:
                            try:
                                X_tr, X_te, meta = embed_fold(
                                    diagrams, labels, tr_idx, te_idx,
                                    n_classes=N_CLASSES, tau_method=tau,
                                    n_scales=n_sc,
                                    max_landmarks=MAX_LANDMARKS,
                                )
                                y_tr = labels[tr_idx]; y_te = labels[te_idx]
                                ell = meta['ell']
                                R_max_tr = meta['R_max_tr']
                                save_embedding_cache(
                                    cache_path, X_tr, X_te, y_tr, y_te,
                                    tr_idx, te_idx, meta,
                                )
                            except Exception as e:
                                log(f'    [{filt_name}|{tau}|N={n_sc}] '
                                    f'embed FAILED: {e}')
                                continue
                        t_embed_s = time.time() - t_embed_0

                        inner_seed = (100000 * seed + 1000 * fi
                                      + 10 * n_sc
                                      + hash(filt_name + tau) % 997)
                        try:
                            res = evaluate_all(
                                X_tr, y_tr, X_te, y_te,
                                n_classes=N_CLASSES,
                                sigma_quantiles=WLK_Q_GRID,
                                inner_seed=int(inner_seed),
                                R_max_tr=R_max_tr,
                            )
                        except Exception as e:
                            log(f'    [{filt_name}|{tau}|N={n_sc}] '
                                f'classify FAILED: {e}')
                            continue

                        row = {
                            'dataset':    DATASET_NAME,
                            'seed':       int(seed),
                            'fold':       int(fi),
                            'filt':       filt_name,
                            'tau_method': tau,
                            'n_scales':   int(n_sc),
                            'ell':        ell,
                            'R_max_tr':   float(R_max_tr),
                            't_embed_s':  float(t_embed_s),
                            **res,
                        }
                        writer.writerow(row)
                        f_out.flush()

                        log(f"    [{filt_name:22s}|{tau:8s}|N={n_sc:2d}]  "
                            f"ℓ={ell:4d}  η̂={res['eta_hat']:.5f}  "
                            f"lin={res['linear_acc']*100:5.1f} "
                            f"rbf={res['rbf_acc']*100:5.1f} "
                            f"wlk={res['wlk_acc']*100:5.1f} "
                            f"nc={res['nc_acc']*100:5.1f} "
                            f"cert={res['nc_certified']} "
                            f"cert_g={res['nc_certified_gauss']}  "
                            f"({t_embed_s + res['t_classify_s']:.1f}s)")

    f_out.close()
    log(f'\n[PROTEINS] total wall-clock: {(time.time() - grand_t0)/3600:.2f} h')
    log(f'  → {output}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument('--tau-methods', nargs='+',
                    default=['crossing', 'proxy'],
                    choices=['crossing', 'proxy', 'auto'])
    ap.add_argument('--n-scales', nargs='+', type=int,
                    default=N_SCALES_GRID,
                    help='sweep of N (paper default 10)')
    ap.add_argument('--n-folds', type=int, default=N_FOLDS)
    ap.add_argument('--force', action='store_true',
                    help='recompute embedding caches')
    ap.add_argument('--output', type=str,
                    default=str(OUT_DIR / 'proteins_records.csv'))
    ap.add_argument('--filt-only', nargs='+', default=None,
                    help='Restrict to listed filtration names.')
    args = ap.parse_args()

    run(
        seeds=args.seeds,
        tau_methods=args.tau_methods,
        n_scales_list=args.n_scales,
        n_folds=args.n_folds,
        force=args.force,
        output=Path(args.output),
        filt_only=args.filt_only,
    )


if __name__ == '__main__':
    main()
