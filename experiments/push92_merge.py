"""Merge per-task push92 outputs (PUSH92_CONFIGS / PUSH92_OUT_SUFFIX runs) into one sweep table and one
per-fold table, with mean and sample std over all folds of all seeds per configuration.

    python experiments/push92_merge.py            -> results/orbit5k_final_sweep/push92_rerun.csv, push92_folds_rerun.csv
"""
from pathlib import Path
import pandas as pd

D = Path(__file__).resolve().parents[1] / "results" / "orbit5k_final_sweep"
folds = pd.concat([pd.read_csv(f) for f in sorted(D.glob("push92_folds_seed*_cfg*.csv"))], ignore_index=True)
folds.to_csv(D / "push92_folds_rerun.csv", index=False)
g = folds.groupby(["experiment", "K", "shrink"])
summ = g.agg(n_seeds=("seed", "nunique"), n_folds=("fold", "size"),
             wlk_fix_mean=("wlk_fix_te", "mean"), wlk_fix_std=("wlk_fix_te", "std"),
             wlk_qt_mean=("wlk_qt_te", "mean"), wlk_qt_std=("wlk_qt_te", "std"),
             xgb_mean=("xgb_te", "mean"), xgb_std=("xgb_te", "std")).reset_index()
for c in summ.columns:
    if c.endswith("_mean") or c.endswith("_std"):
        summ[c] = 100 * summ[c]
summ = summ.sort_values("wlk_qt_mean", ascending=False)
summ.to_csv(D / "push92_rerun.csv", index=False)
print(summ.head(12).to_string(index=False))
print(f"\n{len(folds)} fold rows from {folds.seed.nunique()} seeds; wrote push92_rerun.csv, push92_folds_rerun.csv")
