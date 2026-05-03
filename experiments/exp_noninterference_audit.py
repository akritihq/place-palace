"""
experiments/exp_noninterference_audit.py
---------------------------------------

Audit of the non-interference condition on chemical graph benchmarks.

Two forms tested per pair (A, B):
  - Prior strict (PI Defn 2.1, PII Defn 2.4 pre-revision):
        min_{i != j} d_b(a_i, b_sigma(j)) > 3 * d_b(A, B)
  - Current relaxed (PII Defn 2.4 post-revision):
        exists i* with d_b(a_{i*}, b_sigma(i*)) = d_b(A, B) such that
        min_{j != i*} d_b(a_{i*}, b_sigma(j)) > 2 * d_b(A, B)

Plus the within-class sufficient form (Prop 2.5):
        min_{i != j} d_b(a_i, a_j) > 3 * d_b(A, B)   on both diagrams.

Optimal bottleneck matching computed via binary search over edge-weight
thresholds + scipy bipartite matching (exact, not a sum-optimal proxy).

Output: results/paper_II/noninterference_audit.csv
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv
import pickle

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from utils.datasets import load_tu_dataset


# Configuration
DATASETS = [
    ("MUTAG", ["degree", "hks_t10"]),
    ("PTC",   ["degree", "betweenness"]),
    ("COX2",  ["jaccard", "hks_t10"]),
    ("DHFR",  ["hks_t10"]),
]
N_PAIRS = 2000
N_MAX = 50
SEED = 42
THRESH_STRICT = 3.0       # PI's factor 3 (all i ≠ j)
THRESH_RELAXED_2 = 2.0    # historical: factor 2 / exists-i*
THRESH_RELAXED_1 = 1.0    # PII current: factor > 1 / exists-i*
THRESH_WITHIN_NEW = 2.0   # within-class sufficient for PII (relaxed): > 2 d_b
THRESH_WITHIN_OLD = 4.0   # within-class sufficient for PI: > 4 d_b

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "diagrams"
OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "paper_II"


# ----------------------------------------------------------------------
# Loading + filtering
# ----------------------------------------------------------------------
def load_combined_diagrams(dataset: str, filtrations: list[str]) -> list[np.ndarray]:
    """Load and concatenate H_0+H_1 across filtrations, one diagram per graph."""
    sub_lists = []
    for f in filtrations:
        path = DATA_DIR / f"{dataset}_{f}_H0H1_ext.pkl"
        with open(path, "rb") as fh:
            sub_lists.append(pickle.load(fh))

    n_graphs = min(len(sl) for sl in sub_lists)
    out = []
    for i in range(n_graphs):
        chunks = []
        for sl in sub_lists:
            d = sl[i]
            if isinstance(d, dict):
                for h in (0, 1):
                    if h in d and len(d[h]) > 0:
                        arr = np.asarray(d[h], dtype=np.float64)
                        if arr.ndim == 2 and arr.shape[1] == 2:
                            # drop infinite-death rows defensively
                            arr = arr[np.isfinite(arr).all(axis=1)]
                            chunks.append(arr)
            elif isinstance(d, np.ndarray) and d.ndim == 2 and d.shape[1] == 2:
                chunks.append(d.astype(np.float64))
        out.append(np.vstack(chunks) if chunks else np.zeros((0, 2)))
    return out


def filter_topN(diagrams: list[np.ndarray], n_max: int) -> list[np.ndarray]:
    out = []
    for D in diagrams:
        if len(D) == 0:
            out.append(D)
            continue
        pers = D[:, 1] - D[:, 0]
        idx = np.argsort(pers)[::-1][:n_max]
        out.append(D[idx])
    return out


# ----------------------------------------------------------------------
# Bottleneck matching via binary search + scipy bipartite matching
# ----------------------------------------------------------------------
INF = 1e18


def _build_cost_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Bipartite cost matrix of size (m+n) x (m+n) for bottleneck matching.

    Rows: A (size m) ∪ Δ_B (size n; one diagonal slot per b_j).
    Cols: B (size n) ∪ Δ_A (size m; one diagonal slot per a_i).
    """
    m, n = len(A), len(B)
    N = m + n
    C = np.full((N, N), INF, dtype=np.float64)

    # A x B (sup-norm)
    if m and n:
        diff = np.abs(A[:, None, :] - B[None, :, :])  # (m, n, 2)
        C[:m, :n] = diff.max(axis=2)

    # A x Δ_A (slot i for a_i)
    for i in range(m):
        C[i, n + i] = (A[i, 1] - A[i, 0]) / 2.0

    # Δ_B x B (slot j for b_j)
    for j in range(n):
        C[m + j, j] = (B[j, 1] - B[j, 0]) / 2.0

    # Δ_B x Δ_A: free
    C[m:, n:] = 0.0

    return C


def _has_perfect_matching(C: np.ndarray, threshold: float) -> bool:
    """Check perfect matching exists in {(i,j) : C[i,j] <= threshold}."""
    rows, cols = np.where(C <= threshold + 1e-15)
    if len(rows) == 0:
        return False
    N = C.shape[0]
    sp = csr_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, cols)),
        shape=(N, N),
    )
    match = maximum_bipartite_matching(sp, perm_type="column")
    return bool((match != -1).all())


def bottleneck_distance_with_matching(A: np.ndarray, B: np.ndarray):
    """Return (delta, matching) where matching is a list of (left, right) pairs.

    Edge cases:
      - both empty: delta=0, matching=[]
      - one empty: delta = max diagonal-projection of the non-empty side.
    """
    m, n = len(A), len(B)
    if m == 0 and n == 0:
        return 0.0, []
    if m == 0:
        delta = max((B[j, 1] - B[j, 0]) / 2.0 for j in range(n))
        return float(delta), [(None, j) for j in range(n)]
    if n == 0:
        delta = max((A[i, 1] - A[i, 0]) / 2.0 for i in range(m))
        return float(delta), [(i, None) for i in range(m)]

    C = _build_cost_matrix(A, B)
    finite_mask = C < INF / 2
    candidates = np.unique(C[finite_mask])
    if len(candidates) == 0:
        return 0.0, []

    lo, hi = 0, len(candidates) - 1
    if not _has_perfect_matching(C, candidates[hi]):
        # Should not happen — diagonal augmentation guarantees a matching.
        return float(candidates[hi]), []
    while lo < hi:
        mid = (lo + hi) // 2
        if _has_perfect_matching(C, candidates[mid]):
            hi = mid
        else:
            lo = mid + 1
    delta = float(candidates[lo])

    # Reconstruct matching at delta
    rows, cols = np.where(C <= delta + 1e-15)
    N = C.shape[0]
    sp = csr_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, cols)),
        shape=(N, N),
    )
    match = maximum_bipartite_matching(sp, perm_type="column")
    matching = [(i, int(match[i])) for i in range(N) if match[i] != -1]
    return delta, matching


# ----------------------------------------------------------------------
# Per-pair audit
# ----------------------------------------------------------------------
def _d_inf(p: np.ndarray, q: np.ndarray) -> float:
    return float(max(abs(p[0] - q[0]), abs(p[1] - q[1])))


def audit_pair(A: np.ndarray, B: np.ndarray) -> dict | None:
    m, n = len(A), len(B)
    if m == 0 or n == 0:
        return None

    delta, matching = bottleneck_distance_with_matching(A, B)
    if delta <= 0:
        return None

    # σ map restricted to A↔B matched pairs
    sigma = {i: j for (i, j) in matching
             if i is not None and j is not None and 0 <= i < m and 0 <= j < n}
    if len(sigma) < 2:
        # Need ≥ 2 matched pairs to have any "j ≠ i*" candidate
        return None

    # Strict (prior) form: min over i ≠ k (both A↔B-matched) of d(a_i, b_{σ(k)})
    matched_is = list(sigma.keys())
    strict_dists = []
    for i in matched_is:
        for k in matched_is:
            if i != k:
                strict_dists.append(_d_inf(A[i], B[sigma[k]]))
    strict_min = min(strict_dists)

    # Relaxed (current) form: best over candidate i* of
    #   min_{k ≠ i*, k in σ} d(a_{i*}, b_{σ(k)}).
    # Candidate i* set: any matched i with d(a_i, b_{σ(i)}) = delta.
    candidates = [
        i for i in matched_is
        if abs(_d_inf(A[i], B[sigma[i]]) - delta) <= 1e-12 + 1e-9 * delta
    ]
    relaxed_best_ratio = 0.0
    for i_star in candidates:
        rest = [_d_inf(A[i_star], B[sigma[k]])
                for k in matched_is if k != i_star]
        if rest:
            relaxed_best_ratio = max(relaxed_best_ratio, min(rest) / delta)
    if not candidates:
        # Bottleneck achieved on a diagonal-projection edge only;
        # treat as worst case (no usable i*).
        relaxed_best_ratio = 0.0

    # Within-class min separations
    def _within_min(D: np.ndarray) -> float:
        if len(D) < 2:
            return float("inf")
        diff = np.abs(D[:, None, :] - D[None, :, :]).max(axis=2)
        np.fill_diagonal(diff, np.inf)
        return float(diff.min())

    within_min = min(_within_min(A), _within_min(B))

    return {
        "delta": delta,
        "strict_ratio": strict_min / delta,
        "relaxed_ratio": relaxed_best_ratio,
        "within_ratio": within_min / delta,
    }


# ----------------------------------------------------------------------
# Per-dataset audit
# ----------------------------------------------------------------------
def audit_dataset(name: str, filtrations: list[str], n_pairs: int,
                  rng: np.random.Generator) -> dict | None:
    print(f"\n=== {name} ({'+'.join(filtrations)}) ===")
    diagrams = load_combined_diagrams(name, filtrations)
    diagrams = filter_topN(diagrams, N_MAX)
    print(f"  {len(diagrams)} graphs; sizes after top-{N_MAX}: "
          f"min={min(len(D) for D in diagrams)} "
          f"max={max(len(D) for D in diagrams)} "
          f"median={int(np.median([len(D) for D in diagrams]))}")

    # PTC diagram-files use "PTC_*" but the TU folder is "PTC_MR".
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

    strict_pass, relaxed1_pass, relaxed2_pass = [], [], []
    within_pass_new, within_pass_old = [], []
    relaxed_ratios, strict_ratios = [], []
    skipped = 0
    for k, (i, j) in enumerate(pairs):
        if k % 250 == 0 and k > 0:
            print(f"    ...{k}/{len(pairs)}")
        res = audit_pair(diagrams[i], diagrams[j])
        if res is None:
            skipped += 1
            continue
        strict_pass.append(res["strict_ratio"] > THRESH_STRICT)
        relaxed1_pass.append(res["relaxed_ratio"] > THRESH_RELAXED_1)
        relaxed2_pass.append(res["relaxed_ratio"] > THRESH_RELAXED_2)
        within_pass_new.append(res["within_ratio"] > THRESH_WITHIN_NEW)
        within_pass_old.append(res["within_ratio"] > THRESH_WITHIN_OLD)
        relaxed_ratios.append(res["relaxed_ratio"])
        strict_ratios.append(res["strict_ratio"])

    n_valid = len(strict_pass)
    print(f"  Valid: {n_valid}; skipped: {skipped}")
    if n_valid == 0:
        return None

    summary = {
        "dataset":         name,
        "filtration":      "+".join(filtrations),
        "n":               n_valid,
        "strict_pct":      100.0 * float(np.mean(strict_pass)),
        "relaxed1_pct":    100.0 * float(np.mean(relaxed1_pass)),
        "relaxed2_pct":    100.0 * float(np.mean(relaxed2_pass)),
        "within_new_pct":  100.0 * float(np.mean(within_pass_new)),
        "within_old_pct":  100.0 * float(np.mean(within_pass_old)),
        "strict_p25":      float(np.percentile(strict_ratios, 25)),
        "strict_p50":      float(np.percentile(strict_ratios, 50)),
        "relaxed_p25":     float(np.percentile(relaxed_ratios, 25)),
        "relaxed_p50":     float(np.percentile(relaxed_ratios, 50)),
    }
    print(f"  strict %  (>3, all i≠j):    {summary['strict_pct']:5.1f}   "
          f"strict p25/p50:  {summary['strict_p25']:.3f} / {summary['strict_p50']:.3f}")
    print(f"  relaxed-2 (>2, exists i*):  {summary['relaxed2_pct']:5.1f}")
    print(f"  relaxed-1 (>1, exists i*):  {summary['relaxed1_pct']:5.1f}   "
          f"relaxed p25/p50: {summary['relaxed_p25']:.3f} / {summary['relaxed_p50']:.3f}")
    print(f"  within>2 (new PII):         {summary['within_new_pct']:5.1f}")
    print(f"  within>4 (PI):              {summary['within_old_pct']:5.1f}")
    return summary


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
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
    print(f"{'Dataset':10s}  {'Filtration':28s}  {'n':>5s}  "
          f"{'strict %':>9s}  {'rel-2 %':>7s}  {'rel-1 %':>7s}  "
          f"{'w>2 %':>6s}  {'w>4 %':>6s}  {'rp25':>5s}  {'rp50':>5s}")
    for s in summaries:
        print(
            f"{s['dataset']:10s}  {s['filtration']:28s}  {s['n']:5d}  "
            f"{s['strict_pct']:9.1f}  {s['relaxed2_pct']:7.1f}  "
            f"{s['relaxed1_pct']:7.1f}  "
            f"{s['within_new_pct']:6.1f}  {s['within_old_pct']:6.1f}  "
            f"{s['relaxed_p25']:5.2f}  {s['relaxed_p50']:5.2f}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "noninterference_audit.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summaries[0].keys())
        w.writeheader()
        w.writerows(summaries)
    print(f"\nSaved {out_csv}")


if __name__ == "__main__":
    main()
