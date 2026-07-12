"""
audit_cor32_regularity.py
-------------------------

Per-dataset regularity audit for PII Corollary 3.2 (cor:gamma_via_delta).

Cor 3.2 worst-case hypothesis (current paper):
  K * B^4 <= (1/2) * sigma^2 * Delta^2,
where:
  K     = number of landmarks
  B     = sup_{A, k} |Phi_k(A; LC)|     (per-coordinate envelope)
  sigma = Gaussian kernel bandwidth (q-th quantile of pairwise L2)
  Delta = min_{c != c'} ||mu_c - mu_c'||_2  (raw-coord class-mean separation)

Variance-aware (typical-case) hypothesis (proposed upgrade, parallel to
PI's Pinelis-Bernstein):
  sum_k V_4(k) <= 4 * sigma^2 * Delta^2,    V_4(k) := E_{A,B}[(Phi_k(A) - Phi_k(B))^4]
which replaces the L^4 envelope B^4 with the 4th-moment of cross-pair
embedding-coordinate differences (averaged over coordinates and pairs).

For each chemical benchmark at its headline filtration, this script:
  1. Loads diagrams (combined H0 + H1, top-N_max persistence filter).
  2. Builds a class-aware FPS PALACE configuration with K = 200 landmarks
     (paper-headline settings: alpha = 1.75, sigma_q = 0.75).
  3. Embeds all diagrams: X = (N, K).
  4. Computes B (max coord magnitude), sigma, Delta, and V_4 (sampled).
  5. Reports both the worst-case and the variance-aware regularity ratios.

Output:
  results/paper_II/tables/tab_cor32_regularity.csv
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv
import math

import numpy as np

from embedding.nonuniform import (
    NonUniformEmbedding, init_nonuniform_from_data,
)
from utils.datasets import load_tu_dataset
from exp_noninterference_audit import (
    load_combined_diagrams, filter_topN, N_MAX,
)

# Paper-headline configuration
K_LANDMARKS = 200
ALPHA_RADIUS = 1.75
SIGMA_Q = 0.75
SEED = 42

DATASETS = [
    ("MUTAG", ["degree", "hks_t10"]),
    ("PTC",   ["degree", "betweenness"]),
    ("COX2",  ["jaccard", "hks_t10"]),
    ("DHFR",  ["hks_t10"]),
]

OUT_CSV = Path('results/paper_II/tables/tab_cor32_regularity.csv')


def auto_detect_L(diagrams) -> float:
    """Frame extent: 1.1 x max death across diagrams."""
    max_d = 0.0
    for d in diagrams:
        if len(d) > 0:
            max_d = max(max_d, float(d[:, 1].max()))
    return 1.1 * max_d


def l2_sigma(X: np.ndarray, q: float, seed: int = 0) -> float:
    """sigma = q-th quantile of pairwise L2 distances on a 1000-sample subset."""
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(n, 1000), replace=False)
    Xs = X[idx]
    sq = np.sum(Xs ** 2, axis=1)
    D2 = sq[:, None] + sq[None, :] - 2.0 * Xs @ Xs.T
    iu = np.triu_indices(len(Xs), k=1)
    d2 = D2[iu]
    d2 = np.maximum(d2, 0.0)
    d = np.sqrt(d2)
    return float(np.quantile(d, q))


def class_means(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Stack per-class means."""
    classes = np.unique(y)
    return np.stack([X[y == c].mean(axis=0) for c in classes])


def cross_pair_4th_moments(X: np.ndarray, y: np.ndarray,
                           n_pairs: int = 5000,
                           seed: int = 0) -> np.ndarray:
    """Per-coordinate 4th moment of cross-class differences.

    Returns V_4 in R^K, where V_4[k] = E[(Phi_k(A) - Phi_k(B))^4]
    averaged over n_pairs random cross-class pairs (A, B).
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    K = X.shape[1]
    V4 = np.zeros(K)
    if len(classes) < 2:
        return V4
    counts = 0
    for _ in range(n_pairs):
        ca, cb = rng.choice(len(classes), size=2, replace=False)
        ia = rng.choice(np.where(y == classes[ca])[0])
        ib = rng.choice(np.where(y == classes[cb])[0])
        diff = X[ia] - X[ib]
        V4 += diff ** 4
        counts += 1
    if counts > 0:
        V4 /= counts
    return V4


def delta_min(means: np.ndarray) -> float:
    """min_{c != c'} ||mu_c - mu_c'||_2."""
    n_c = len(means)
    if n_c < 2:
        return 0.0
    d_min = np.inf
    for i in range(n_c):
        for j in range(i + 1, n_c):
            d = float(np.linalg.norm(means[i] - means[j]))
            if d < d_min:
                d_min = d
    return float(d_min)


def process(name: str, filtrations: list[str]) -> dict:
    print(f"\n=== {name} ({'+'.join(filtrations)}) ===")

    diagrams = load_combined_diagrams(name, filtrations)
    diagrams = filter_topN(diagrams, N_MAX)
    label_name = 'PTC_MR' if name == 'PTC' else name
    _, labels = load_tu_dataset(label_name)
    n_g = min(len(diagrams), len(labels))
    diagrams = diagrams[:n_g]
    labels = np.asarray(labels[:n_g])
    print(f"  {n_g} diagrams, {len(np.unique(labels))} classes")

    L = auto_detect_L(diagrams)
    print(f"  L = {L:.4f}")

    # Class-aware FPS configuration
    diagrams_by_class = [
        [diagrams[i] for i in range(n_g) if labels[i] == c]
        for c in np.unique(labels)
    ]
    emb = init_nonuniform_from_data(
        diagrams_by_class, K=K_LANDMARKS, L=L,
        n_diagram=1, seed=SEED, alpha=ALPHA_RADIUS,
    )
    K = emb.K
    print(f"  K = {K} landmarks; max radius = {emb.radii.max():.4f}")

    # Embed all diagrams
    X = emb.embed_dataset(diagrams)  # (N, K)

    # Compute audit quantities
    B = float(np.max(np.abs(X)))
    sigma = l2_sigma(X, SIGMA_Q, seed=SEED)
    means = class_means(X, labels)
    Delta = delta_min(means)

    # Worst-case regularity (current Cor 3.2)
    lhs_wc = K * B ** 4
    rhs_wc = 0.5 * sigma ** 2 * Delta ** 2
    passes_wc = lhs_wc <= rhs_wc

    # Variance-aware regularity (proposed upgrade): sum_k V_4(k)
    V4 = cross_pair_4th_moments(X, labels, n_pairs=5000, seed=SEED)
    sumV4 = float(np.sum(V4))
    rhs_va = 4.0 * sigma ** 2 * Delta ** 2
    passes_va = sumV4 <= rhs_va

    # m_min tightening for Thm 3.1: K -> tr(Sigma_c^H) via Taylor
    # tr(Sigma_c^H) ~ tr(Sigma_c^raw)/sigma^2 in small-sigma regime.
    # Compute per-class raw covariance trace (sum of per-coord variances).
    tr_raw = []
    sigma_op_raw = []  # ||Sigma_c||_op (largest eigenvalue)
    for c in np.unique(labels):
        Xc = X[labels == c]
        if Xc.shape[0] < 2:
            continue
        var_per_coord = Xc.var(axis=0, ddof=1)
        tr_raw.append(float(np.sum(var_per_coord)))
        # Compute op-norm of Sigma_c via top eigenvalue of (m, m) Gram
        Yc = Xc - Xc.mean(axis=0, keepdims=True)
        m_c = Yc.shape[0]
        Gc = (Yc @ Yc.T) / max(m_c - 1, 1)
        if Gc.size > 0:
            top = float(np.linalg.eigvalsh(Gc)[-1])
            sigma_op_raw.append(top)
    tr_Sigma_raw_max = max(tr_raw) if tr_raw else 0.0
    sigma_op_raw_max = max(sigma_op_raw) if sigma_op_raw else 0.0
    tr_Sigma_H_max = tr_Sigma_raw_max / sigma ** 2 if sigma > 0 else float('inf')
    mmin_tighten = tr_Sigma_H_max / K if K > 0 else float('inf')

    # Thm 5.1 radii: (i) Pinelis, (ii) Gaussian plug-in, (iii) Pinelis-Bernstein
    # Use raw-coordinate framework (PII Thm 5.1 is on R^K, not RKHS).
    R_bar = float(np.linalg.norm(X, axis=1).max())  # sup_A ||Phi(A)||
    delta_alpha = 0.05
    log_2k_d = math.log(2 * len(np.unique(labels)) / delta_alpha)
    m_per = min(int((labels == c).sum()) for c in np.unique(labels))
    # (i) Pinelis: r = 2 R_bar sqrt(2 log(2k/delta) / m_min)
    r_pin = 2.0 * R_bar * math.sqrt(2.0 * log_2k_d / m_per)
    # (iii) Pinelis-Bernstein: r = sqrt(2 ||Sigma||_op log(2k/delta)/m_min)
    r_vp = math.sqrt(2.0 * sigma_op_raw_max * log_2k_d / m_per) if sigma_op_raw_max > 0 else float('inf')
    # (ii) Gaussian plug-in: r = sqrt(||Sigma||_op chi2(K, delta/k)/m_min)
    from scipy.stats import chi2
    chi2_q = float(chi2.ppf(1.0 - delta_alpha / len(np.unique(labels)), df=K))
    r_gauss = math.sqrt(sigma_op_raw_max * chi2_q / m_per) if sigma_op_raw_max > 0 else float('inf')
    # Firing condition: r < Delta/2
    half_Delta = Delta / 2.0
    fire_pin = r_pin < half_Delta
    fire_vp = r_vp < half_Delta
    fire_gauss = r_gauss < half_Delta

    # The tightening factor sumV4 / (K * B^4 * 16) compares typical
    # 4th-moment to worst-case (B^4 envelope, * 16 since |u| <= 2B
    # gives u^4 <= 16 B^4)
    tighten = sumV4 / (K * (2.0 * B) ** 4) if B > 0 else float('nan')

    print(f"  B          = {B:.4f}")
    print(f"  sigma      = {sigma:.4f}")
    print(f"  Delta      = {Delta:.4f}")
    print(f"  K*B^4               = {lhs_wc:.4e}    [worst-case LHS]")
    print(f"  (1/2)sigma^2 Delta^2 = {rhs_wc:.4e}    [worst-case RHS]")
    print(f"  ratio (worst-case)   = {lhs_wc/rhs_wc:.3e}  passes={passes_wc}")
    print(f"  sum_k V4            = {sumV4:.4e}    [variance-aware LHS]")
    print(f"  4 sigma^2 Delta^2    = {rhs_va:.4e}    [variance-aware RHS]")
    print(f"  ratio (variance-aw)  = {sumV4/rhs_va:.3e}  passes={passes_va}")
    print(f"  tightening factor (sumV4 / 16KB^4): {tighten:.3e}")
    print(f"  m_min Thm 3.1 lift:")
    print(f"    K (worst-case)              = {K}")
    print(f"    tr(Sigma_c^H) max (var-aware)= {tr_Sigma_H_max:.4e}")
    print(f"    tightening factor (tr/K)    = {mmin_tighten:.3e}")
    print(f"  Thm 5.1 radii (m_min={m_per}, alpha={delta_alpha}):")
    print(f"    Delta/2                     = {half_Delta:.4f}")
    print(f"    R_bar                       = {R_bar:.4f}")
    print(f"    ||Sigma_c||_op max          = {sigma_op_raw_max:.4f}")
    print(f"    (i)   r_Pin                 = {r_pin:.4f}     fires={fire_pin}")
    print(f"    (ii)  r_Gauss               = {r_gauss:.4f}     fires={fire_gauss}")
    print(f"    (iii) r_vP (Pinelis-Bern)   = {r_vp:.4f}     fires={fire_vp}")

    return {
        'dataset': name,
        'filt': '+'.join(filtrations),
        'K': K,
        'B': B,
        'sigma': sigma,
        'Delta': Delta,
        'KB4': lhs_wc,
        'half_sigma2_Delta2': rhs_wc,
        'ratio_wc': lhs_wc / rhs_wc if rhs_wc > 0 else float('inf'),
        'passes_wc': bool(passes_wc),
        'sumV4': sumV4,
        '4_sigma2_Delta2': rhs_va,
        'ratio_va': sumV4 / rhs_va if rhs_va > 0 else float('inf'),
        'passes_va': bool(passes_va),
        'tighten_factor': tighten,
        'tr_Sigma_raw_max': tr_Sigma_raw_max,
        'tr_Sigma_H_max': tr_Sigma_H_max,
        'mmin_tighten_factor': mmin_tighten,
        'R_bar': R_bar,
        'sigma_op_raw_max': sigma_op_raw_max,
        'm_per_class': m_per,
        'half_Delta': half_Delta,
        'r_Pin': r_pin,
        'r_Gauss': r_gauss,
        'r_vP': r_vp,
        'fire_Pin': fire_pin,
        'fire_Gauss': fire_gauss,
        'fire_vP': fire_vp,
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
        print("\nNo results.")
        return

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {OUT_CSV}")

    # Summary table
    print(f"\n{'Dataset':<8s}  {'K':>3s} {'B':>8s} {'sigma':>8s} {'Delta':>8s}  "
          f"{'ratio_wc':>10s} {'pass_wc':>7s}  "
          f"{'ratio_va':>10s} {'pass_va':>7s}  {'tighten':>10s}")
    print('-' * 110)
    for r in rows:
        print(f"{r['dataset']:<8s}  {r['K']:>3d} {r['B']:>8.4f} "
              f"{r['sigma']:>8.4f} {r['Delta']:>8.4f}  "
              f"{r['ratio_wc']:>10.3e} {('YES' if r['passes_wc'] else 'no'):>7s}  "
              f"{r['ratio_va']:>10.3e} {('YES' if r['passes_va'] else 'no'):>7s}  "
              f"{r['tighten_factor']:>10.3e}")


if __name__ == '__main__':
    main()
