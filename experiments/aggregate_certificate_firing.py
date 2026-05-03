"""
Task 0: Aggregate per-prediction certificate firing rates across datasets.

Reads *_records_newweights.csv for 6 datasets and computes:
  - Pinelis fire %  : mean(nc_certified)       -- r_m < 0.5 * Delta_hat
  - Gaussian fire % : mean(nc_certified_gauss) -- same with Gaussian bound
  - NC acc          : mean(nc_acc) across all folds
  - NC acc | fired  : mean(nc_acc) on folds where Gaussian fires > 0

Outputs results/paper_II/tables/tab_certificate_firing.tex

Usage:
    python experiments/aggregate_certificate_firing.py
"""
import sys; sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from pathlib import Path
from utils.datasets import load_dataset

RECORDS_DIR = Path('results/filtration_grid')
OUT_PATH    = Path('results/paper_II/tables/tab_certificate_firing.tex')

# (display name, csv stem, dataset name for load_dataset)
DATASETS = [
    ('Orbit5k', 'orbit5k_records_newweights', 'Orbit5k'),
    ('MUTAG',   'mutag_records_newweights',   'MUTAG'),
    ('COX2',    'cox2_records_newweights',    'COX2'),
    ('DHFR',    'dhfr_records_newweights',    'DHFR'),
    ('PTC',     'ptc_records_newweights',     'PTC'),
    ('NCI1',    'nci1_records_newweights',    'NCI1'),
]


def process(name, stem, ds_key):
    path = RECORDS_DIR / f'{stem}.csv'
    if not path.exists():
        print(f'  WARNING: {path} not found — skipping {name}', flush=True)
        return None
    df = pd.read_csv(path)

    _, labels = load_dataset(ds_key)
    n_test = round(len(labels) / 10)
    pinelis_pct  = df['nc_certified'].mean() * 100
    gauss_pct    = df['nc_certified_gauss'].mean() * 100
    nc_acc_all   = df['nc_acc'].mean() * 100

    fired = df[df['nc_certified_gauss'] > 0]
    nc_acc_fired = fired['nc_acc'].mean() * 100 if len(fired) > 0 else float('nan')

    return {
        'dataset':      name,
        'n_test':       n_test,
        'pinelis_pct':  pinelis_pct,
        'gauss_pct':    gauss_pct,
        'nc_acc':       nc_acc_all,
        'nc_acc_fired': nc_acc_fired,
    }


def fmt(v):
    return r'\textemdash' if np.isnan(v) else f'{v:.1f}'


def main():
    rows = []
    for name, stem, ds_key in DATASETS:
        r = process(name, stem, ds_key)
        if r is not None:
            rows.append(r)
            print(f"  {name:10s}: Pinelis={r['pinelis_pct']:.1f}%  "
                  f"Gauss={r['gauss_pct']:.1f}%  "
                  f"NC acc={r['nc_acc']:.1f}%  "
                  f"NC acc|fired={fmt(r['nc_acc_fired'])}%  "
                  f"(n_test≈{r['n_test']})", flush=True)

    print(f"\n{'Dataset':10s} {'n_test':>7} {'Pinelis%':>10} "
          f"{'Gauss%':>10} {'NC acc%':>9} {'NC acc|fired%':>14}")
    print('-' * 64)
    for r in rows:
        print(f"{r['dataset']:10s} {r['n_test']:>7} {r['pinelis_pct']:>10.1f} "
              f"{r['gauss_pct']:>10.1f} {r['nc_acc']:>9.1f} "
              f"{fmt(r['nc_acc_fired']):>14}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r'\begin{table}[t]',
        r'\centering',
        (r'\caption{Per-prediction certificate firing rates on the six Paper~II '
         r'benchmark datasets, averaged across seeds, folds, and filtrations. '
         r'\emph{Pinelis} and \emph{Gaussian} columns report the fraction of '
         r'test graphs for which $r_m < \tfrac{1}{2}\widehat\Delta_{\hat c}$ '
         r'under the respective tail bound. '
         r'\emph{NC acc\,|\,fired} is the nearest-centroid accuracy restricted '
         r'to folds where the Gaussian certificate fires on at least one test '
         r'point; ``\textemdash'' indicates no folds fired.}'),
        r'\label{tab:certificate_firing}',
        r'\begin{tabular}{lrrrr}',
        r'\toprule',
        (r'Dataset & $n_{\text{test}}$ & Pinelis (\%) '
         r'& Gaussian (\%) & NC acc\,|\,fired (\%) \\'),
        r'\midrule',
    ]
    for r in rows:
        lines.append(
            f"{r['dataset']} & {r['n_test']} & "
            f"{fmt(r['pinelis_pct'])} & "
            f"{fmt(r['gauss_pct'])} & "
            f"{fmt(r['nc_acc_fired'])} \\\\"
        )
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']

    OUT_PATH.write_text('\n'.join(lines))
    print(f'\nWrote {OUT_PATH}', flush=True)


if __name__ == '__main__':
    main()
