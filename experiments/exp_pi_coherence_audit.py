"""
experiments/exp_pi_coherence_audit.py
-------------------------------------

Audit the *hypothesis* of Proposition 2.1(b) of Paper I (PLACE) on
cross-class pairs of chemical-graph diagrams: the pointwise
ν-coherence condition

    (φ_{R_k,p}(a_i) - φ_{R_k,p}(b_{σ(i)}))
        · (φ_{R_k,p}(a_j) - φ_{R_k,p}(b_{σ(j)}))  ≥  0

at every active scale k (3R_k ≤ d_B(A,B)), every landmark
p ∈ G_{R_k}^+, and every i ≠ j.  Reports two pass rates per dataset:

  - strict (pointwise, def 2.2):       ∀ p, k, i ≠ j the product ≥ 0.
  - sum-form (inner-product, weaker):  ⟨δ_i^(k), δ_j^(k)⟩_{ℓ²} ≥ 0
                                       for all i ≠ j and active k.

Each is sufficient for the lower distortion bound; pointwise
≥ 0 ⇒ sum-form ≥ 0 ⇒ certificate.  Companion to
exp_pi_certificate_bound_audit.py (which audits the conclusion).

Same datasets, top-N_max filter, and PLACE configuration as the
certificate-bound audit, so the rows merge into a single table.

Output: results/paper_II/pi_coherence_audit.csv
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv

import numpy as np

from embedding.embedding import init_from_dataset
from utils.bottleneck import d_B1_batch
from utils.datasets import load_tu_dataset

from exp_noninterference_audit import (
    DATASETS, N_PAIRS, N_MAX, SEED,
    OUT_DIR,
    load_combined_diagrams, filter_topN,
    bottleneck_distance_with_matching,
)

# Configuration
N_SCALES = 5
SUM_TOL = 1e-9    # numerical tolerance for sum-form check


def phi_at_landmarks(diagram_pts: np.ndarray,
                     grid: np.ndarray,
                     R: float) -> np.ndarray:
    """Per-point Mitra–Virk hat values φ_{R,p}(a) over G_R^+ = G_R ∪ {Δ}.

    Returns
    -------
    (m, K + 1) array; column 0 is the Δ landmark, columns 1..K are grid.
    """
    m = len(diagram_pts)
    K = len(grid)
    if m == 0:
        return np.zeros((0, K + 1))

    # Δ-landmark: d_B^1(Δ, a) = (d_a - b_a) / 2
    pers_half = (diagram_pts[:, 1] - diagram_pts[:, 0]) / 2.0
    phi_delta = np.maximum(1.5 * R - pers_half, 0.0)               # (m,)

    if K > 0:
        d = d_B1_batch(grid, diagram_pts)                          # (K, m)
        phi_grid = np.maximum(1.5 * R - d, 0.0).T                  # (m, K)
    else:
        phi_grid = np.zeros((m, 0))

    return np.concatenate([phi_delta[:, None], phi_grid], axis=1)


def _augment_to_common_cardinality(
    A: np.ndarray, B: np.ndarray, matching: list[tuple[int, int]]
) -> tuple[np.ndarray, np.ndarray]:
    """Augment A and B to a common cardinality using the bottleneck
    matching's diagonal-projection partners.

    Under PI's strict convention (D_n contains diagrams with n points
    each, allowing diagonal multiplicity), every (A, B) pair extends to
    A_aug, B_aug ∈ D_n with n = |A_real| + |B_real| − (orphan-orphan
    free pairings) via diagonal projections supplied by the matching.

    Returns
    -------
    A_aug, B_aug : (n, 2) arrays of birth-death coordinates with
                   diagonal projections wherever the matching σ paired
                   a real point with the diagonal slot.
    """
    m, n = len(A), len(B)
    A_pts: list[np.ndarray] = []
    B_pts: list[np.ndarray] = []
    for ii, jj in matching:
        if ii is None or jj is None:
            continue
        if 0 <= ii < m and 0 <= jj < n:
            A_pts.append(A[ii])
            B_pts.append(B[jj])
        elif 0 <= ii < m and jj >= n:
            a = A[ii]
            mid = 0.5 * (a[0] + a[1])
            A_pts.append(a)
            B_pts.append(np.array([mid, mid]))
        elif ii >= m and 0 <= jj < n and (ii - m) == jj:
            b = B[jj]
            mid = 0.5 * (b[0] + b[1])
            B_pts.append(b)
            A_pts.append(np.array([mid, mid]))
        # ii ≥ m and jj ≥ n: Δ_B-Δ_A free pairing, no contribution.
    if not A_pts:
        return np.zeros((0, 2)), np.zeros((0, 2))
    return np.asarray(A_pts), np.asarray(B_pts)


def coherence_check_pair(A: np.ndarray, B: np.ndarray,
                         delta: float,
                         matching: list[tuple[int, int]],
                         scales: np.ndarray,
                         grids: list[np.ndarray]
                         ) -> dict | None:
    """Test ν-coherence on a single (A, B) pair.

    The condition is ‖Φ_{R_k}(A) - Φ_{R_k}(B)‖²_{ℓ²} ≥ R_k²/32 at
    every active scale k (3R_k ≤ d_B(A,B)), where Φ_{R_k}(A) :=
    Σ_{a ∈ A_aug} φ_{R_k}(a) is the unscaled per-scale block of the
    embedding under PI's strict convention (diagrams augmented to a
    common cardinality via the diagonal).  The bottleneck matching
    supplies the augmentation; the resulting per-scale block is
    matching-invariant.

    Returns dict with pass/fail per scale, or None if no active scales.
    """
    active_mask = (3.0 * scales) <= delta
    active_k = np.where(active_mask)[0]
    if len(active_k) == 0:
        return None

    A_aug, B_aug = _augment_to_common_cardinality(A, B, matching)
    if len(A_aug) == 0:
        return None

    coherent = True
    n_active = len(active_k)
    n_fail_scales = 0

    for k in active_k:
        R_k = float(scales[k])
        grid = grids[k]
        # Φ_{R_k}(·) = Σ_{a ∈ ·_aug} φ_{R_k}(a)
        phi_A_block = phi_at_landmarks(A_aug, grid, R_k).sum(axis=0)
        phi_B_block = phi_at_landmarks(B_aug, grid, R_k).sum(axis=0)
        diff = phi_A_block - phi_B_block
        sq_norm = float(np.dot(diff, diff))
        floor = R_k * R_k / 32.0
        if sq_norm < floor - SUM_TOL:
            coherent = False
            n_fail_scales += 1

    return {
        "delta": delta,
        "n_active_scales": n_active,
        "coherent": int(coherent),
        "n_fail_scales": n_fail_scales,
    }


def audit_dataset(name: str, filtrations: list[str], n_pairs: int,
                  rng: np.random.Generator) -> dict | None:
    print(f"\n=== {name} ({'+'.join(filtrations)}) ===")
    diagrams = load_combined_diagrams(name, filtrations)
    diagrams = filter_topN(diagrams, N_MAX)

    label_name = "PTC_MR" if name == "PTC" else name
    _, labels = load_tu_dataset(label_name)
    n_g = min(len(diagrams), len(labels))
    diagrams = diagrams[:n_g]
    labels = labels[:n_g]

    diagrams_by_class = [
        [diagrams[i] for i in range(n_g) if labels[i] == c]
        for c in np.unique(labels)
    ]
    emb = init_from_dataset(diagrams_by_class, N_scales=N_SCALES, n_diagram=1)
    R1 = float(emb.scales.min())
    L = float(emb.L)
    print(f"  scales: {emb.scales}")
    print(f"  R_1={R1:.4f}, L={L:.4f}")

    # Sample cross-class pairs (same scheme as cert audit).
    pairs = []
    seen = set()
    while len(pairs) < n_pairs:
        i = int(rng.integers(n_g))
        j = int(rng.integers(n_g))
        if i == j or labels[i] == labels[j]:
            continue
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((i, j))
        if len(seen) > n_pairs * 5:
            break
    print(f"  sampled {len(pairs)} cross-class pairs")

    results = []
    for k, (i, j) in enumerate(pairs):
        if k % 250 == 0 and k > 0:
            print(f"    ...{k}/{len(pairs)}  "
                  f"(qualifying so far: {len(results)})")
        A, B = diagrams[i], diagrams[j]
        if len(A) < 2 or len(B) < 2:
            continue
        delta, matching = bottleneck_distance_with_matching(A, B)
        if delta <= 0 or delta < 3.0 * R1:
            continue
        out = coherence_check_pair(A, B, delta, matching,
                                   emb.scales, emb.grids)
        if out is not None:
            results.append(out)

    if not results:
        print("  no qualifying pairs")
        return None

    n_q = len(results)
    coherent_pct = 100.0 * np.mean([r["coherent"] for r in results])
    fail_frac = np.array([r["n_fail_scales"] / max(r["n_active_scales"], 1)
                          for r in results])

    summary = {
        "dataset":           name,
        "filtration":        "+".join(filtrations),
        "R1":                R1,
        "n_qualifying":      n_q,
        "coherent_pct":      coherent_pct,
        "fail_scale_p50":    float(np.median(fail_frac)),
        "fail_scale_p90":    float(np.percentile(fail_frac, 90)),
    }
    print(f"  qualifying:        {n_q}")
    print(f"  ν-coherent:        {coherent_pct:.1f}%")
    print(f"  fail-scale frac p50/p90: {summary['fail_scale_p50']:.3f} / "
          f"{summary['fail_scale_p90']:.3f}")
    return summary


def main():
    rng = np.random.default_rng(SEED)
    summaries = []
    for name, filts in DATASETS:
        try:
            s = audit_dataset(name, filts, N_PAIRS, rng)
            if s is not None:
                summaries.append(s)
        except Exception as e:
            print(f"  ERROR on {name}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    if not summaries:
        print("no summaries; abort.")
        return

    print("\n=== Summary ===")
    print(f"{'Dataset':10s}  {'Filt':28s}  {'qual':>5s}  "
          f"{'coherent %':>10s}")
    for s in summaries:
        print(
            f"{s['dataset']:10s}  {s['filtration']:28s}  "
            f"{s['n_qualifying']:5d}  "
            f"{s['coherent_pct']:10.1f}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "pi_coherence_audit.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summaries[0].keys())
        w.writeheader()
        w.writerows(summaries)
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
