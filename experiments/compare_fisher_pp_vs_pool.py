"""
Compare Spearman(per-pair Fisher, WLK acc) vs Spearman(pooled Fisher, WLK acc)
across the 5 PII chemical-graph datasets.

Reads:
  results/paper_II/raw/headline/gamma_filt_<ds>_full46[_with_mah].csv
    -- has fisher_kernel (per-pair denom, the published version)
  results/paper_II/raw/headline_pooled/gamma_filt_<ds>_full46_with_pooled.csv
    -- has fisher_kernel (per-pair) AND fisher_kernel_pooled (new)

For each dataset we report:
  - n_filts × n_seeds × n_folds
  - Spearman(fisher_kernel_perpair, wlk_acc) -- existing
  - Spearman(fisher_kernel_pooled, wlk_acc) -- new
  - Whether the two have the same sign and similar magnitude

Run after the four small-dataset reruns finish; NCI1 is allowed to be missing
or partial.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
POOLED_DIR = ROOT / 'results' / 'paper_II' / 'raw' / 'headline_pooled'

DATASETS = ['cox2', 'mutag', 'ptc', 'dhfr', 'nci1',
            'proteins', 'dd', 'imdb-b', 'imdb-m', 'nci109']


def per_filt_means(df: pd.DataFrame, stat_col: str, acc_col: str = 'wlk_acc'):
    """Mean over (seed, fold) per filtration, then Spearman across filtrations.
    Matches how tab:selection_statistics is computed (agg_paper2.py STAT_ORDER)."""
    grp = df.groupby('filt')[[stat_col, acc_col]].mean().dropna()
    if len(grp) < 3:
        return float('nan'), 0
    rho, _ = spearmanr(grp[stat_col], grp[acc_col])
    return float(rho), len(grp)


def main():
    print(f"{'dataset':10s} {'rows':>6s} {'filts':>6s} "
          f"{'rho(perpair,acc)':>17s} {'rho(pooled,acc)':>16s} "
          f"{'sign_match':>10s} {'|delta|':>8s}")
    print('-' * 80)
    for ds in DATASETS:
        csv = POOLED_DIR / f'gamma_filt_{ds}_full46_with_pooled.csv'
        if not csv.exists():
            print(f"{ds:10s} -- file missing --")
            continue
        df = pd.read_csv(csv)
        df = df[df['wlk_acc'].notna() & df['fisher_kernel'].notna()
                & df['fisher_kernel_pooled'].notna()]
        if len(df) == 0:
            print(f"{ds:10s}  empty after dropna")
            continue
        rho_pp, nf_pp = per_filt_means(df, 'fisher_kernel')
        rho_pl, nf_pl = per_filt_means(df, 'fisher_kernel_pooled')
        sign_match = (np.sign(rho_pp) == np.sign(rho_pl)) if (
            np.isfinite(rho_pp) and np.isfinite(rho_pl)) else False
        delta = abs(rho_pp - rho_pl) if (
            np.isfinite(rho_pp) and np.isfinite(rho_pl)) else float('nan')
        print(f"{ds:10s} {len(df):>6d} {nf_pp:>6d} "
              f"{rho_pp:>+17.3f} {rho_pl:>+16.3f} "
              f"{str(sign_match):>10s} {delta:>+8.3f}")


if __name__ == '__main__':
    main()
