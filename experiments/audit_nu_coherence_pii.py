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
    """Exact bottleneck distance (gudhi).  Until 2026-11-08 this was a matching-free heuristic
    with a sum-via-diagonal cost: median relative error -54% to -97% against the exact value;
    on MUTAG it accepted 4 pairs with d_B < tau and rejected 3016 of 3121 pairs that qualify."""
    import gudhi
    A = A if len(A) else np.zeros((0, 2)); B = B if len(B) else np.zeros((0, 2))
    return float(gudhi.bottleneck_distance(A, B))


def check_nu_coherence(A: np.ndarray, B: np.ndarray,
                       emb, tau: float) -> bool:
    """Definition nu_coherence_pII on the paper's scale (Phi_k = w_k phi_k): exists active k*
    (r_k >= tau/4) with |Phi_k(A) - Phi_k(B)| >= w_k tau/4, i.e. |phi_k(A) - phi_k(B)| >= tau/4.
    emb.embed carries the implementation's extra 2^{-n-1/2} (nonuniform.py module docstring), so
    until 2026-11-08 this test was 2.83x stricter than the definition (rates were 100% anyway)."""
    from embedding.nonuniform import hard_coords
    phiA = hard_coords(emb.positions, emb.radii, emb._clip(A))
    phiB = hard_coords(emb.positions, emb.radii, emb._clip(B))
    active = emb.radii >= (tau / 4.0)
    if not np.any(active):
        return False
    return bool(np.any(np.abs(phiA - phiB)[active] >= tau / 4.0))


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
    # ONE tau: the median half-persistence of the training points, which is also the clip scale
    # (Paper II protocol table).  The median clipped radius used until 2026-11-08 equals tau/2 on
    # these sets (95-100% of radii sat on the lower clip); reported alongside for comparison.
    tau = float(emb.tau)
    tau_rad = float(np.median(emb.radii))
    print(f"  K = {K}; tau = {tau:.4f} (median radius {tau_rad:.4f}); L = {L:.4f}; cap enforced = {emb.cap_enforced}")

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
        'tau_median_radius': tau_rad,
        'cap_enforced': bool(emb.cap_enforced),
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
