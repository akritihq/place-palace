"""
audit_nu_coherence_pii.py
-------------------------

Direct audit of the matching-free $\\nu$-coherence condition
(Definition def:nu_coherence_pII in Paper II) on the four chemical
benchmarks at the paper-headline PALACE configuration.

PII $\\nu$-coherence: a pair $(A, B)$ with $d_B(A,B) \\geq \\tau$
is $\\nu$-coherent w.r.t.\\ $\\LC$ iff there exists an active
landmark $k^\\star$ ($r_{k^\\star} \\geq \\tau/4$) with
  | Phi_{k^\\star}(A; LC) - Phi_{k^\\star}(B; LC) | >= w_{k^\\star} * tau / 4.

This is the matching-free analogue of PI's per-scale block-norm
floor (PI Def 2.1, audited at >= 99.7% hold rate on chemical
benchmarks). PI's audit is in
results/paper_II/pi_coherence_audit.csv (note: filename refers to
PI's results, used for cross-comparison).

For each chemical benchmark at its headline filtration:
  1. Load combined H_0 + H_1 diagrams (top-N_max persistence filter).
  2. Build paper-headline PALACE configuration (class-aware FPS,
     K=200, alpha=1.75, equal weights w_k = 1/sqrt(K)).
  3. Sample N_PAIRS cross-class pairs (A, B) with d_B(A, B) >= tau.
  4. Embed each pair, check the per-coordinate floor at active
     landmarks; record fraction satisfying nu-coherence.

Output: results/paper_II/tables/tab_nu_coherence_pii.csv
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv

import numpy as np
from scipy.spatial.distance import cdist

from embedding.nonuniform import init_nonuniform_from_data
from utils.datasets import load_tu_dataset
from exp_noninterference_audit import (
    load_combined_diagrams, filter_topN, N_MAX,
)

K_LANDMARKS = 200
ALPHA_RADIUS = 1.75
SEED = 42
N_PAIRS = 2000  # cross-class pairs to sample

DATASETS = [
    ("MUTAG", ["degree", "hks_t10"]),
    ("PTC",   ["degree", "betweenness"]),
    ("COX2",  ["jaccard", "hks_t10"]),
    ("DHFR",  ["hks_t10"]),
]

OUT_CSV = Path('results/paper_II/tables/tab_nu_coherence_pii.csv')


def auto_detect_L(diagrams) -> float:
    max_d = 0.0
    for d in diagrams:
        if len(d) > 0:
            max_d = max(max_d, float(d[:, 1].max()))
    return 1.1 * max_d


def db_pair(A: np.ndarray, B: np.ndarray) -> float:
    """Bottleneck distance via binary search on max-cost matching.

    Coarse approximation: for top-N_max diagrams, treat as
    max(min over A of d_inf-to-B-or-diagonal, min over B of
    d_inf-to-A-or-diagonal). This is the L^infty bottleneck distance
    on diagrams paired against the diagonal."""
    if len(A) == 0 and len(B) == 0:
        return 0.0
    if len(A) == 0:
        # all B-points to diagonal
        pers_B = (B[:, 1] - B[:, 0]) / 2.0
        return float(pers_B.max()) if len(pers_B) else 0.0
    if len(B) == 0:
        pers_A = (A[:, 1] - A[:, 0]) / 2.0
        return float(pers_A.max()) if len(pers_A) else 0.0
    # d_inf(A, b) for each (A_i, B_j)
    D = cdist(A, B, metric='chebyshev')
    pers_A = (A[:, 1] - A[:, 0]) / 2.0
    pers_B = (B[:, 1] - B[:, 0]) / 2.0
    # closest-to-diagonal for each pt
    dA = pers_A
    dB = pers_B
    # match A_i to B_j or diagonal
    # cost A_i -> B_j is min(D_ij, dA_i + dB_j)
    # the bottleneck is approximately:
    # d_B = max(min over A of cost to B-or-diag,
    #          min over B of cost from A-or-diag)
    cost_AB = np.minimum(D, dA[:, None] + dB[None, :])
    bottle_A = cost_AB.min(axis=1).min(axis=0)  # closest match cost over A
    bottle_B = cost_AB.min(axis=0).min(axis=0)
    # max-cost match heuristic
    return float(max(cost_AB.min(axis=1).max(),
                     cost_AB.min(axis=0).max()))


def check_nu_coherence(A: np.ndarray, B: np.ndarray,
                       emb, tau: float) -> bool:
    """Check PII nu-coherence: exists active k* with
    |Phi_{k*}(A) - Phi_{k*}(B)| >= w_{k*} * tau / 4."""
    PhiA = emb.embed(A)
    PhiB = emb.embed(B)
    diff = np.abs(PhiA - PhiB)
    # active landmarks: r_k >= tau/4
    active = emb.radii >= (tau / 4.0)
    if not np.any(active):
        return False
    floor = emb.weights * (tau / 4.0)
    # nu-coherence: exists active k* with diff[k] >= floor[k]
    holds = (diff[active] >= floor[active])
    return bool(np.any(holds))


def process(name: str, filtrations: list[str]) -> dict:
    print(f"\n=== {name} ({'+'.join(filtrations)}) ===")

    diagrams = load_combined_diagrams(name, filtrations)
    diagrams = filter_topN(diagrams, N_MAX)
    label_name = 'PTC_MR' if name == 'PTC' else name
    _, labels = load_tu_dataset(label_name)
    n_g = min(len(diagrams), len(labels))
    diagrams = diagrams[:n_g]
    labels = np.asarray(labels[:n_g])
    print(f"  {n_g} diagrams")

    L = auto_detect_L(diagrams)
    diagrams_by_class = [
        [diagrams[i] for i in range(n_g) if labels[i] == c]
        for c in np.unique(labels)
    ]
    emb = init_nonuniform_from_data(
        diagrams_by_class, K=K_LANDMARKS, L=L,
        n_diagram=1, seed=SEED, alpha=ALPHA_RADIUS,
    )
    K = emb.K
    tau = float(np.median(emb.radii))  # use median radius as tau
    print(f"  K = {K}; tau = {tau:.4f}; L = {L:.4f}")

    rng = np.random.default_rng(SEED)
    classes = np.unique(labels)
    if len(classes) < 2:
        return {'dataset': name, 'note': 'single class, skipping'}
    cross_pairs_seen = 0
    qualifying_pairs = 0  # d_B >= tau
    coherent_pairs = 0
    while cross_pairs_seen < 5 * N_PAIRS and qualifying_pairs < N_PAIRS:
        ca, cb = rng.choice(len(classes), size=2, replace=False)
        ia = rng.choice(np.where(labels == classes[ca])[0])
        ib = rng.choice(np.where(labels == classes[cb])[0])
        A = diagrams[ia]; B = diagrams[ib]
        cross_pairs_seen += 1
        d = db_pair(A, B)
        if d < tau:
            continue
        qualifying_pairs += 1
        if check_nu_coherence(A, B, emb, tau):
            coherent_pairs += 1

    rate = coherent_pairs / qualifying_pairs if qualifying_pairs > 0 else float('nan')
    print(f"  qualifying pairs (d_B >= tau): {qualifying_pairs}")
    print(f"  nu-coherent pairs: {coherent_pairs} ({100*rate:.2f}%)")
    return {
        'dataset': name,
        'filt': '+'.join(filtrations),
        'K': K,
        'tau': tau,
        'pairs_examined': cross_pairs_seen,
        'qualifying_pairs': qualifying_pairs,
        'coherent_pairs': coherent_pairs,
        'coherent_pct': 100 * rate,
    }


def main():
    rows = []
    for name, filts in DATASETS:
        try:
            rows.append(process(name, filts))
        except Exception as e:
            print(f"  ERROR on {name}: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()

    if not rows:
        return

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {OUT_CSV}")

    print(f"\n{'Dataset':<8s}  {'K':>3s} {'tau':>8s}  "
          f"{'qualifying':>10s} {'coherent':>9s} {'%':>7s}")
    print('-' * 60)
    for r in rows:
        if 'note' in r:
            continue
        print(f"{r['dataset']:<8s}  {r['K']:>3d} {r['tau']:>8.4f}  "
              f"{r['qualifying_pairs']:>10d} {r['coherent_pairs']:>9d} "
              f"{r['coherent_pct']:>6.2f}%")


if __name__ == '__main__':
    main()
