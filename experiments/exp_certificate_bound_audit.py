"""
experiments/exp_certificate_bound_audit.py
------------------------------------------

Audit the *conclusion* of Theorem 2.6 (the non-uniform distortion
certificate of Paper II / Prop 2.1(b) of Paper I) on cross-class pairs
of chemical-graph diagrams, *regardless* of whether the proof's
non-interference hypothesis is met.

For a fixed dataset, we:
  1. Run FPS on training diagram points to get a configuration LC
     (positions, uniform radii, equal weights w_k = K^{-1/2}).
  2. Set τ from the FPS step distance so LC is τ-admissible.
  3. For each cross-class pair (A, B) with d_b(A, B) ≥ τ, compute
        ‖Φ(A) - Φ(B)‖_2     (sum-pool embedding under LC),
        ρ_ν(τ; LC) = τ/(4√K),
        ratio = ‖Φ(A) - Φ(B)‖_2 / ρ_ν.
  4. Report fraction with ratio ≥ 1 and percentile distribution.

The proof of Theorem 2.6 only goes through under non-interference,
which the §6 audit shows is essentially never met on these datasets.
This script asks the empirical version: does the *conclusion* of the
theorem hold anyway?

Output: results/paper_II/certificate_bound_audit.csv
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv
import pickle

import numpy as np

from utils.datasets import load_tu_dataset

from exp_noninterference_audit import (
    DATASETS, N_PAIRS, N_MAX, SEED,
    DATA_DIR, OUT_DIR,
    load_combined_diagrams, filter_topN,
    bottleneck_distance_with_matching,
    _d_inf,
)

# --- New configuration --------------------------------------------------
K_LANDMARKS = 64           # configuration size
ALPHA_RADIUS = 0.75        # radii = α × nearest-neighbour landmark distance
TAU_PERCENTILE = 25        # τ from this percentile of d_b values

# ----------------------------------------------------------------------
# FPS + admissible configuration
# ----------------------------------------------------------------------
def fps_landmarks(pts: np.ndarray, K: int, seed: int = 0) -> np.ndarray:
    """Greedy farthest-point sampling under the bottleneck-on-D_1 distance."""
    rng = np.random.default_rng(seed)
    M = len(pts)
    if M <= K:
        return pts.copy()

    selected = [int(rng.integers(M))]
    min_dists = np.full(M, np.inf)

    for _ in range(K - 1):
        last = pts[selected[-1]]
        d = np.max(np.abs(pts - last[np.newaxis, :]), axis=1)
        pers_last = (last[1] - last[0]) / 2.0
        pers_pts = (pts[:, 1] - pts[:, 0]) / 2.0
        d_via = pers_last + pers_pts
        d_B = np.minimum(d, d_via)
        min_dists = np.minimum(min_dists, d_B)
        selected.append(int(np.argmax(min_dists)))

    return pts[selected]


def build_configuration(diagrams: list[np.ndarray],
                        K: int = K_LANDMARKS,
                        alpha: float = ALPHA_RADIUS,
                        seed: int = 0):
    """Run FPS on the union of all valid diagram points; uniform-α radii."""
    chunks = [D[D[:, 1] > D[:, 0]] for D in diagrams if len(D) > 0]
    chunks = [c for c in chunks if len(c) > 0]
    if not chunks:
        return None
    pts = np.vstack(chunks)
    positions = fps_landmarks(pts, K, seed=seed)
    K_actual = len(positions)

    if K_actual > 1:
        D = np.max(
            np.abs(positions[:, None, :] - positions[None, :, :]),
            axis=2,
        )
        np.fill_diagonal(D, np.inf)
        nn_dists = D.min(axis=1)
        radii = alpha * nn_dists
    else:
        radii = np.array([1.0])
    return positions, radii, K_actual


def phi_coords(positions: np.ndarray, radii: np.ndarray,
               A: np.ndarray) -> np.ndarray:
    """φ_k(A) = sum_{a ∈ A} max(r_k - d_inf(p_k, a), 0)."""
    if len(A) == 0:
        return np.zeros(len(positions))
    d = np.max(
        np.abs(positions[:, None, :] - A[None, :, :]),
        axis=2,
    )  # (K, m)
    return np.maximum(radii[:, None] - d, 0.0).sum(axis=1)


# ----------------------------------------------------------------------
# Per-dataset audit
# ----------------------------------------------------------------------
def audit_dataset(name: str, filtrations: list[str], n_pairs: int,
                  rng: np.random.Generator) -> dict | None:
    print(f"\n=== {name} ({'+'.join(filtrations)}) ===")
    diagrams = load_combined_diagrams(name, filtrations)
    diagrams = filter_topN(diagrams, N_MAX)

    cfg = build_configuration(diagrams, K=K_LANDMARKS, seed=SEED)
    if cfg is None:
        print(f"  No valid points; skipping")
        return None
    positions, radii, K = cfg
    print(f"  Configuration: K={K} landmarks, "
          f"radii ∈ [{radii.min():.4f}, {radii.max():.4f}], "
          f"median {np.median(radii):.4f}")

    label_name = "PTC_MR" if name == "PTC" else name
    _, labels = load_tu_dataset(label_name)
    n_g = min(len(diagrams), len(labels))
    diagrams = diagrams[:n_g]
    labels = labels[:n_g]

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

    # First pass: compute d_b and Phi-distance for each pair
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
        phi_A = phi_coords(positions, radii, diagrams[i])
        phi_B = phi_coords(positions, radii, diagrams[j])
        # equal weights w_k = K^{-1/2}: ‖Φ(A) - Φ(B)‖ = ‖φ(A) - φ(B)‖ / √K
        phi_norm = float(np.linalg.norm(phi_A - phi_B) / np.sqrt(K))
        deltas.append(delta)
        phi_norms.append(phi_norm)

    if not deltas:
        return None

    deltas = np.array(deltas)
    phi_norms = np.array(phi_norms)

    # τ from a low percentile so most pairs qualify as τ-separated
    tau = float(np.percentile(deltas, TAU_PERCENTILE))
    rho_nu = tau / (4.0 * np.sqrt(K))

    qualifying = deltas >= tau
    n_qual = int(qualifying.sum())
    if n_qual == 0:
        return None

    ratios = phi_norms[qualifying] / rho_nu
    summary = {
        "dataset":      name,
        "filtration":   "+".join(filtrations),
        "K":            K,
        "n_pairs":      len(deltas),
        "n_qualifying": n_qual,
        "tau":          tau,
        "rho_nu":       rho_nu,
        "bound_holds_pct": 100.0 * float(np.mean(ratios >= 1.0)),
        "ratio_p25":    float(np.percentile(ratios, 25)),
        "ratio_p50":    float(np.percentile(ratios, 50)),
        "ratio_p75":    float(np.percentile(ratios, 75)),
        "ratio_min":    float(ratios.min()),
        "ratio_mean":   float(ratios.mean()),
    }
    print(f"  τ (p{TAU_PERCENTILE}):     {tau:.4f}")
    print(f"  ρ_ν = τ/(4√K): {rho_nu:.4f}")
    print(f"  qualifying:    {n_qual}/{len(deltas)}")
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
    print(f"{'Dataset':10s}  {'Filt':28s}  {'K':>3s}  {'τ':>7s}  "
          f"{'ρ_ν':>7s}  {'qual':>5s}  {'bound %':>7s}  "
          f"{'p25':>5s}  {'p50':>5s}  {'p75':>5s}  {'min':>5s}")
    for s in summaries:
        print(
            f"{s['dataset']:10s}  {s['filtration']:28s}  {s['K']:3d}  "
            f"{s['tau']:7.4f}  {s['rho_nu']:7.4f}  {s['n_qualifying']:5d}  "
            f"{s['bound_holds_pct']:7.1f}  "
            f"{s['ratio_p25']:5.2f}  {s['ratio_p50']:5.2f}  "
            f"{s['ratio_p75']:5.2f}  {s['ratio_min']:5.2f}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "certificate_bound_audit.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summaries[0].keys())
        w.writeheader()
        w.writerows(summaries)
    print(f"\nSaved {out_csv}")


if __name__ == "__main__":
    main()
