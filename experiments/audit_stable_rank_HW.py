"""
audit_stable_rank_HW.py
-----------------------

Validation of the variance-aware Pinelis radius and a Hanson–Wright
upgrade.

For each headline-filt dataset:
1. Load diagrams + labels, build the embedding (init_from_dataset).
2. Embed all training diagrams.
3. Per-class:
     ‖Σ̂_c‖_op  (largest eigenvalue)
     tr(Σ̂_c)
     ‖Σ̂_c‖_F
     stable_rank = tr(Σ)/‖Σ‖_op
4. Compute four radii on the worst class (max σ_op):
     r_Pinelis_R  = R · sqrt(2 L / m)                             (current; uses sup-norm)
     r_vP_op      = sqrt(‖Σ‖_op) · sqrt(2 L / m)                  (my earlier audit; assumes rank≈1)
     r_vP_tr      = sqrt(2 tr(Σ) L / m)                           (TRUE Hilbert-space Pinelis)
     r_HW         = sqrt((tr(Σ) + 2‖Σ‖_F √L + 2‖Σ‖_op L)/m)       (Hanson–Wright)
5. Fire vs Δ̂/2.

Hanson–Wright is tighter than r_vP_tr when ‖Σ‖_F √L < tr(Σ)/2 (typical
in low-rank regime).  Both are tighter than r_Pinelis_R when tr(Σ) ≪ R²·m.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math
import pickle

import numpy as np

from embedding.embedding import init_from_dataset
from utils.datasets import load_tu_dataset
from exp_noninterference_audit import (
    load_combined_diagrams, filter_topN, N_MAX,
)

ALPHA = 0.05

# Reduced grid for speed: closest-to-firing holdouts + a couple firing
# datasets as sanity check
DATASETS = [
    ('MUTAG',     ['degree', 'hks_t10'],     2),  # already fires (sanity)
    ('COX2',      ['jaccard', 'hks_t10'],    2),  # ratio 3.0 (closest holdout)
    ('PTC',       ['degree', 'betweenness'], 2),  # ratio 3.6
    ('DHFR',      ['hks_t10'],               2),  # already fires
]

N_SCALES = 5  # match exp_pi_coherence_audit / cert_firing


def class_cov_stats(X_c: np.ndarray):
    """Returns op-norm, trace, frobenius-norm of Σ̂_c = (1/m) X_c^T X_c
    where X_c is (m, d) centred."""
    m, d = X_c.shape
    if m < 2:
        return 0.0, 0.0, 0.0
    Y = X_c - X_c.mean(axis=0, keepdims=True)
    # Σ = (1/m) Y^T Y is (d, d) but might be huge — work on (m, m) gram instead
    G = (Y @ Y.T) / m  # (m, m); eigenvalues of G match nonzero eigenvalues of Σ
    eig = np.linalg.eigvalsh(G)
    eig = eig[eig > 1e-12]
    op = float(eig[-1]) if len(eig) else 0.0
    tr = float(eig.sum())
    fr = float(math.sqrt((eig ** 2).sum()))
    return op, tr, fr


def main():
    print(f"\n{'Dataset':10s}  {'class':>5s} {'m_c':>5s} "
          f"{'σ²_op':>10s} {'tr(Σ)':>10s} {'‖Σ‖_F':>10s} "
          f"{'sr':>6s}  {'Δ̂/2':>10s}  "
          f"{'r_Pin':>10s} {'r_vP_op':>10s} {'r_vP_tr':>10s} {'r_HW':>10s}  "
          f"{'fPin':>4s} {'fop':>4s} {'ftr':>4s} {'fHW':>4s}")
    print('-' * 165)

    for ds_name, filts, k in DATASETS:
        try:
            diagrams = load_combined_diagrams(ds_name, filts)
        except Exception as e:
            print(f"  {ds_name:10s}: load error: {e}")
            continue
        diagrams = filter_topN(diagrams, N_MAX)

        label_name = "PTC_MR" if ds_name == "PTC" else ds_name
        _, labels = load_tu_dataset(label_name)
        n_g = min(len(diagrams), len(labels))
        diagrams = diagrams[:n_g]
        labels = np.asarray(labels[:n_g])

        diagrams_by_class = [
            [diagrams[i] for i in range(n_g) if labels[i] == cls]
            for cls in np.unique(labels)
        ]
        emb = init_from_dataset(diagrams_by_class, N_scales=N_SCALES, n_diagram=1)
        X = np.array([emb.embed(d) for d in diagrams])

        # Center-class Δ̂
        unique_lbls = np.unique(labels)
        means = np.stack([X[labels == c].mean(axis=0) for c in unique_lbls])
        D = np.linalg.norm(means[:, None] - means[None, :], axis=-1)
        D[D == 0] = np.inf
        delta_min = float(D.min())

        L = math.log(2.0 * k / ALPHA)

        for c in unique_lbls:
            X_c = X[labels == c]
            m_c = X_c.shape[0]
            op, tr, fr = class_cov_stats(X_c)
            sr = tr / max(op, 1e-12)

            R = float(np.linalg.norm(X_c, axis=1).max())  # sup-norm in train
            r_Pin   = R * math.sqrt(2.0 * L / m_c)
            r_vP_op = math.sqrt(op) * math.sqrt(2.0 * L / m_c)
            r_vP_tr = math.sqrt(2.0 * tr * L / m_c)
            r_HW    = math.sqrt((tr + 2.0 * fr * math.sqrt(L) + 2.0 * op * L) / m_c)

            half = 0.5 * delta_min
            fPin = int(r_Pin   < half)
            fop  = int(r_vP_op < half)
            ftr  = int(r_vP_tr < half)
            fHW  = int(r_HW    < half)

            print(
                f"{ds_name:10s}  {int(c):>5d} {m_c:>5d} "
                f"{op:>10.3g} {tr:>10.3g} {fr:>10.3g} "
                f"{sr:>6.2f}  {half:>10.3g}  "
                f"{r_Pin:>10.3g} {r_vP_op:>10.3g} {r_vP_tr:>10.3g} {r_HW:>10.3g}  "
                f"{fPin:>4d} {fop:>4d} {ftr:>4d} {fHW:>4d}"
            )

    print('\nLegend:')
    print('  sr         : stable rank tr(Σ)/‖Σ‖_op (typical for persistence emb: 1-5)')
    print('  r_vP_op    : my earlier audit — VALID only if sr ≈ 1')
    print('  r_vP_tr    : true vector Pinelis (Pinelis 1994 Thm 3.5)')
    print('  r_HW       : Hanson–Wright; tighter than r_vP_tr in low-rank, large-L')


if __name__ == '__main__':
    main()
