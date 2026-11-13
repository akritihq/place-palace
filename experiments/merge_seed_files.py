"""Concatenate per-seed analysis files written by analyze_gamma_filt.py --out-suffix _with_mah_seed<S>
into the single gamma_filt_<ds>_with_mah.csv that exp_outer_test_selected_filt.py reads.

    python experiments/merge_seed_files.py PROTEINS DD IMDB-B IMDB-M NCI109
    python experiments/merge_seed_files.py --out _full46_with_mah MUTAG COX2 PTC DHFR   (chemical naming)
Input files: gamma_filt_<ds>_with_mah_seed*.csv (any further part suffix allowed).
"""
import sys
from pathlib import Path
import pandas as pd

ANALYSIS = Path(__file__).resolve().parents[1] / "results" / "filtration_grid" / "analysis"
args = sys.argv[1:]
out_suffix = "_with_mah"
if args and args[0] == "--out":
    out_suffix = args[1]; args = args[2:]
for ds in args:
    parts = sorted(ANALYSIS.glob(f"gamma_filt_{ds.lower()}_with_mah_seed*.csv"))
    if not parts:
        print(f"{ds}: no per-seed files"); continue
    df = pd.concat([pd.read_csv(f) for f in parts], ignore_index=True)
    out = ANALYSIS / f"gamma_filt_{ds.lower()}{out_suffix}.csv"
    df.to_csv(out, index=False)
    print(f"{ds}: merged {len(parts)} files, {len(df)} rows, seeds {sorted(df.seed.unique().tolist())} -> {out.name}")
