"""
experiments/exp_pi_certificate_bound_audit.py
---------------------------------------------

Audit the *conclusion* of Proposition 2.1(b) of Paper I (PLACE) on
cross-class pairs of chemical-graph diagrams, regardless of whether
the proof's non-interference hypothesis is met.

Paper I's bound is multiplicative:
    λ(ν) · d_b(A, B) ≤ ‖Φ(A) - Φ(B)‖_{ℓ²}
on cross-class pairs with d_b(A, B) ≥ 3 R_1 satisfying non-interference.
We measure the ratio
    ‖Φ(A) - Φ(B)‖_{ℓ²} / (λ(ν) · d_b(A, B))
on the same chemical benchmarks, after a top-N_max persistence filter.

For each dataset we use the standard PLACE configuration produced by
embedding.embedding.init_from_dataset (multiscale grid, analytic
optimal masses).

Output: results/paper_II/pi_certificate_bound_audit.csv
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv

import numpy as np

from embedding.embedding import init_from_dataset
from utils.datasets import load_tu_dataset
from utils.rho_minus import rho_minus

from exp_noninterference_audit import (
    DATASETS, N_PAIRS, N_MAX, SEED,
    OUT_DIR,
    load_combined_diagrams, filter_topN,
    bottleneck_distance_with_matching,
)

# Configuration
N_SCALES = 5            # number of support points
TAU_PERCENTILE = 25     # τ for filtering (still requires d_b ≥ 3 R_1)


def lambda_nu(scales: np.ndarray, masses: np.ndarray, L: float,
              n: int = 1) -> float:
    """λ(ν) = ρ_-(t; ν) / (t - R_1) for any t > R_1 (constant on (R_1, ∞))."""
    R1 = float(scales.min())
    t = R1 + 1.0  # any value > R_1
    val = rho_minus(t, scales, masses, n=n, L=L)
    return float(val) / (t - R1)


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

    classes = np.unique(labels)
    diagrams_by_class = [
        [diagrams[i] for i in range(n_g) if labels[i] == c]
        for c in classes
    ]

    # Build PLACE embedding via standard initialization
    emb = init_from_dataset(diagrams_by_class, N_scales=N_SCALES, n_diagram=1)
    R1 = float(emb.scales.min())
    L = float(emb.L)
    lam = lambda_nu(emb.scales, emb.masses, L, n=emb.n)
    print(f"  Scales: {emb.scales}")
    print(f"  Masses: {emb.masses}")
    print(f"  R_1={R1:.4f}, L={L:.4f}, embedding_dim={emb.embedding_dim}")
    print(f"  λ(ν) = {lam:.4f}")

    # Sample cross-class pairs
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
    print(f"  Sampled {len(pairs)} cross-class pairs")

    deltas = []
    phi_norms = []
    for k, (i, j) in enumerate(pairs):
        if k % 250 == 0 and k > 0:
            print(f"    ...{k}/{len(pairs)}")
        if len(diagrams[i]) < 2 or len(diagrams[j]) < 2:
            continue
        delta, _ = bottleneck_distance_with_matching(diagrams[i], diagrams[j])
        if delta <= 0:
            continue
        phi_A = emb.embed(diagrams[i])
        phi_B = emb.embed(diagrams[j])
        phi_norm = float(np.linalg.norm(phi_A - phi_B))
        deltas.append(delta)
        phi_norms.append(phi_norm)

    if not deltas:
        return None

    deltas = np.array(deltas)
    phi_norms = np.array(phi_norms)

    # Apply PI's bound condition: d_b(A, B) ≥ 3 R_1
    qualifying = deltas >= 3.0 * R1
    n_qual = int(qualifying.sum())
    if n_qual == 0:
        print(f"  No pairs satisfy d_b(A,B) ≥ 3R_1 = {3*R1:.4f}")
        return None

    # Ratio against the multiplicative bound λ(ν) · d_b(A, B)
    qual_deltas = deltas[qualifying]
    qual_phi = phi_norms[qualifying]
    ratios = qual_phi / (lam * qual_deltas)

    summary = {
        "dataset":      name,
        "filtration":   "+".join(filtrations),
        "embedding_dim": int(emb.embedding_dim),
        "n_pairs":      len(deltas),
        "n_qualifying": n_qual,
        "R1":           R1,
        "lambda_nu":    lam,
        "bound_holds_pct": 100.0 * float(np.mean(ratios >= 1.0)),
        "ratio_p25":    float(np.percentile(ratios, 25)),
        "ratio_p50":    float(np.percentile(ratios, 50)),
        "ratio_p75":    float(np.percentile(ratios, 75)),
        "ratio_min":    float(ratios.min()),
        "ratio_mean":   float(ratios.mean()),
    }
    print(f"  R_1={R1:.4f}, 3R_1={3*R1:.4f}")
    print(f"  qualifying:    {n_qual}/{len(deltas)}  "
          f"(d_b ≥ 3R_1 = {3*R1:.4f})")
    print(f"  bound holds:   {summary['bound_holds_pct']:.1f}%")
    print(f"  ratio p25/p50/p75: "
          f"{summary['ratio_p25']:.2f} / {summary['ratio_p50']:.2f} / "
          f"{summary['ratio_p75']:.2f}   (min {summary['ratio_min']:.2f}, "
          f"mean {summary['ratio_mean']:.2f})")
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
        print("No summaries; abort.")
        return

    print("\n=== Summary ===")
    print(f"{'Dataset':10s}  {'Filt':28s}  {'dim':>5s}  {'R1':>7s}  "
          f"{'λ(ν)':>8s}  {'qual':>5s}  {'bound %':>7s}  "
          f"{'p25':>5s}  {'p50':>5s}  {'p75':>5s}  {'min':>5s}")
    for s in summaries:
        print(
            f"{s['dataset']:10s}  {s['filtration']:28s}  "
            f"{s['embedding_dim']:5d}  {s['R1']:7.4f}  "
            f"{s['lambda_nu']:8.4f}  {s['n_qualifying']:5d}  "
            f"{s['bound_holds_pct']:7.1f}  "
            f"{s['ratio_p25']:5.2f}  {s['ratio_p50']:5.2f}  "
            f"{s['ratio_p75']:5.2f}  {s['ratio_min']:5.2f}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "pi_certificate_bound_audit.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summaries[0].keys())
        w.writeheader()
        w.writerows(summaries)
    print(f"\nSaved {out_csv}")


if __name__ == "__main__":
    main()
