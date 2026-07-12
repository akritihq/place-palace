"""
exp_fps_radius_ablation_pii.py
------------------------------

FPS-radius (alpha) x seed x dataset robustness ablation for Paper II.

Closes JMLR-blocker #8 (script-side): produces a per-(dataset, alpha,
seed, fold) accuracy CSV that supports the headline-configuration
robustness claim and identifies the operationally-flat range of alpha.

Sweep grid:
  alpha  in {0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00}
         (radius factor: r_k = alpha * d_NN(p_k); paper-headline alpha = 1.75)
  seed   in {0, 1, 2, 3, 4}
  K      = 200 (paper-headline)
  filt   = paper-headline filtration per dataset
  classifier: RBF-SVM (median-heuristic sigma, C=1.0); 10-fold stratified CV.
  datasets: MUTAG, PTC, COX2, DHFR.

Output:
  results/paper_II/tables/fps_radius_ablation.csv
    columns: dataset, filt, K, alpha, seed, fold, n_train, n_test, acc

Cluster usage (Pegasus@GW):
  python experiments/exp_fps_radius_ablation_pii.py
  python experiments/exp_fps_radius_ablation_pii.py MUTAG DHFR
  python experiments/exp_fps_radius_ablation_pii.py --alpha-grid 0.50 1.00 1.75 3.00

The full sweep is 4 datasets x 9 alphas x 5 seeds = 180 cells (each is
one PALACE init + 10-fold RBF-SVM at K=200).  Per-cell runtime measured
on a 2024 MacBook (single-threaded): MUTAG/PTC ~0.7s, COX2 ~1.7s, DHFR
~2.8s.  Full sweep on the four chemicals is roughly 5 minutes.  No
cluster batch needed; runs comfortably on a laptop.
"""
from __future__ import annotations
import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from embedding.nonuniform import init_nonuniform_from_data
from utils.datasets import load_tu_dataset
from exp_noninterference_audit import load_combined_diagrams, filter_topN, N_MAX

K_LANDMARKS = 200
N_FOLDS = 10
DEFAULT_ALPHA_GRID = [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]

DATASETS = [
    ("MUTAG", ["degree", "hks_t10"]),
    ("PTC",   ["degree", "betweenness"]),
    ("COX2",  ["jaccard", "hks_t10"]),
    ("DHFR",  ["hks_t10"]),
]

OUT_CSV = Path('results/paper_II/tables/fps_radius_ablation.csv')


def auto_detect_L(diagrams) -> float:
    max_d = 0.0
    for d in diagrams:
        if len(d) > 0:
            max_d = max(max_d, float(d[:, 1].max()))
    return 1.1 * max_d


def evaluate_one(name: str, filtrations: list, alpha: float, seed: int) -> list:
    """Run K_LANDMARKS-FPS PALACE at the given alpha + seed; return per-fold rows."""
    diagrams = load_combined_diagrams(name, filtrations)
    diagrams = filter_topN(diagrams, N_MAX)
    label_name = 'PTC_MR' if name == 'PTC' else name
    _, labels = load_tu_dataset(label_name)
    n_g = min(len(diagrams), len(labels))
    diagrams = diagrams[:n_g]
    labels = np.asarray(labels[:n_g])

    L = auto_detect_L(diagrams)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    rows = []
    for fold, (tr, te) in enumerate(skf.split(np.zeros(n_g), labels)):
        diagrams_by_class_tr = [
            [diagrams[i] for i in tr if labels[i] == c]
            for c in np.unique(labels[tr])
        ]
        emb = init_nonuniform_from_data(
            diagrams_by_class_tr, K=K_LANDMARKS, L=L,
            n_diagram=1, seed=seed, alpha=alpha,
        )
        X = np.stack([emb.embed(d) for d in diagrams])
        sigma = max(float(np.median(euclidean_distances(X[tr], X[tr]))), 1e-3)
        Gtr = np.exp(-euclidean_distances(X[tr], X[tr]) ** 2 / (2 * sigma ** 2))
        Gte = np.exp(-euclidean_distances(X[te], X[tr]) ** 2 / (2 * sigma ** 2))
        clf = SVC(kernel='precomputed', C=1.0).fit(Gtr, labels[tr])
        acc = float((clf.predict(Gte) == labels[te]).mean())
        rows.append({
            'dataset': name,
            'filt': '+'.join(filtrations),
            'K': K_LANDMARKS,
            'alpha': alpha,
            'seed': seed,
            'fold': fold,
            'n_train': int(len(tr)),
            'n_test': int(len(te)),
            'acc': acc,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('datasets', nargs='*',
                        help='Subset of datasets (default: all four chemicals)')
    parser.add_argument('--alpha-grid', type=float, nargs='+',
                        default=DEFAULT_ALPHA_GRID)
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    parser.add_argument('--out', type=Path, default=OUT_CSV)
    args = parser.parse_args()

    selected = [(n, f) for n, f in DATASETS
                if not args.datasets or n in args.datasets]
    if not selected:
        sys.exit(f"No datasets matched {args.datasets}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['dataset', 'filt', 'K', 'alpha', 'seed', 'fold',
                  'n_train', 'n_test', 'acc']
    new_file = not args.out.exists()
    with open(args.out, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()

        for (name, filt) in selected:
            for alpha in args.alpha_grid:
                for seed in args.seeds:
                    t0 = time.time()
                    rows = evaluate_one(name, filt, alpha, seed)
                    writer.writerows(rows)
                    f.flush()
                    mean_acc = np.mean([r['acc'] for r in rows])
                    print(f"[{name:>5s}] alpha={alpha:>4.2f} seed={seed} "
                          f"mean_acc={100*mean_acc:5.2f}% "
                          f"({time.time()-t0:5.1f}s)", flush=True)

    print(f"\nWrote {args.out}")


if __name__ == '__main__':
    main()
