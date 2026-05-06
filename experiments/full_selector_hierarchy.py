"""
Print the full selector-hierarchy Spearman correlations against
WLK CV accuracy for all available datasets, mirroring the
structure of tab:selection_statistics in the manuscript.

For each dataset:
  - Group rows by filtration; compute mean of each statistic and
    mean of wlk_acc per filtration.
  - Compute Spearman(stat_mean, acc_mean) across filtrations.

Outputs all 5 selectors of the selector-hierarchy remark plus
the bottleneck data-level rankers:
    gamma_over_sqrt_K    -- Score statistic, Sigma = I
    fisher_kernel        -- per-pair Welch denominator
    fisher_kernel_pooled -- pooled denominator (canonical Mika 1999 form)
    rho_mahalanobis      -- full operator Sigma^-1 with LW shrinkage
    tau_hat            -- 10th-percentile cross-class bottleneck
    rho_nu_hat         -- c_n * tau_hat / sqrt(K)

Reads from results/paper_II/raw/headline_pooled/
gamma_filt_<ds>_full46_with_pooled.csv (the merged per-dataset CSVs).

Usage:
    python experiments/full_selector_hierarchy.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
POOLED_DIR = ROOT / 'results' / 'paper_II' / 'raw' / 'headline_pooled'

DATASETS = ['cox2', 'mutag', 'ptc', 'dhfr', 'nci1',
            'proteins', 'dd', 'imdb-b', 'imdb-m', 'nci109']

STATS = [
    ('gamma_over_sqrt_K',     'gamma/sqrtK'),
    ('fisher_kernel',         'Fisher(pp)'),
    ('fisher_kernel_pooled',  'Fisher(pl)'),
    ('rho_mahalanobis',       'rho_Mah'),
    ('tau_hat',               'tau_hat'),
    ('rho_nu_hat',            'rho_nu_hat'),
]


def per_filt_spearman(df: pd.DataFrame, stat_col: str,
                       acc_col: str = 'wlk_acc') -> tuple[float, int]:
    """Mean over (seed, fold) per filt, then Spearman across filts."""
    if stat_col not in df.columns or acc_col not in df.columns:
        return float('nan'), 0
    grp = df.groupby('filt')[[stat_col, acc_col]].mean().dropna()
    if len(grp) < 3:
        return float('nan'), len(grp)
    rho, _ = spearmanr(grp[stat_col], grp[acc_col])
    return float(rho), len(grp)


def main():
    print(f"\nFull selector-hierarchy Spearman vs WLK CV accuracy")
    print(f"Reading from: {POOLED_DIR}")
    print()
    header = f"{'dataset':10s} {'|F|':>4s}  " + \
             "  ".join(f"{label:>11s}" for _, label in STATS)
    print(header)
    print('-' * len(header))
    for ds in DATASETS:
        csv = POOLED_DIR / f'gamma_filt_{ds}_full46_with_pooled.csv'
        if not csv.exists():
            print(f"{ds:10s}  ----  " + "  ".join("        ----" for _ in STATS))
            continue
        df = pd.read_csv(csv)
        # Number of filtrations available
        n_filts = df['filt'].nunique() if 'filt' in df.columns else 0
        cells = []
        for stat_col, _ in STATS:
            rho, _ = per_filt_spearman(df, stat_col)
            if np.isnan(rho):
                cells.append(f"{'---':>11s}")
            else:
                cells.append(f"{rho:>+11.3f}")
        print(f"{ds:10s} {n_filts:>4d}  " + "  ".join(cells))
    print()
    print("Bold-best-per-row rule: largest positive rho across the "
          "first 4 columns (kernel-margin selectors); tau/rho_nu are "
          "data-level rankers reported separately in tab:selection_statistics.")


if __name__ == '__main__':
    main()
