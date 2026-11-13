"""
recompute_cert_firing_quarter.py
--------------------------------
Re-run of the Pinelis-Bernstein certificate firing table under the
Honest empirical firing condition
and J5KD (W6):

  Theorem 5.1(a) fires when  r_m < Delta/2  (POPULATION gap Delta).
  Prop 3.2 gives |Delta_hat - Delta| <= 2 r_m on the coverage event,
  so to certify r_m < Delta/2 from the EMPIRICAL Delta_hat we need
      Delta_hat - 2 r_m > 2 r_m   <=>   r_m < Delta_hat / 4   (not /2).

This script reports, per dataset, the firing rate under the OLD
(r < Delta_hat/2) and CORRECTED (r < Delta_hat/4) conditions, for all
three radii, and the headline count "fires on X of 12 benchmarks".

It also adds the Pt-3 linear Bernstein correction to radius (iii):
  r_vP = sqrt(2 tr(Sigma) L / m) + 4 R L / (3 m),   L = log(2k/alpha),
using Sigma_op_max as the trace proxy (stable rank <= 1.17 on these
benchmarks; see W5 caveat) and R_max_tr as R.

Mirrors data plumbing of aggregate_certificate_firing_vp.py.
Does NOT overwrite that script's outputs.
"""
from __future__ import annotations
from pathlib import Path
import math

import pandas as pd
from scipy.stats import chi2

ALPHA = 0.05
DATASETS = [
    ('Orbit5k', 'orbit5k_records_newweights', None,         5, 500, 'alpha_H1'),
    ('MUTAG',   'mutag_records_newweights',   'mutag',      2,  19, None),
    ('COX2',    'cox2_records_newweights',    'cox2',       2,  47, None),
    ('DHFR',    'dhfr_records_newweights',    'dhfr',       2,  76, None),
    ('PTC',     'ptc_records_newweights',     'ptc',        2,  34, None),
    ('NCI1',    'nci1_records_newweights',    'nci1',       2, 411, None),
    ('PROTEINS','proteins_records_newweights','proteins',   2, 111, None),
    ('DD',      'dd_records_newweights',      'dd',         2, 118, None),
    ('IMDB-B',  'imdbb_records_newweights',   'imdb-b',     2, 100, None),
    ('IMDB-M',  'imdbm_records_newweights',   'imdb-m',     3, 150, None),
    ('NCI109',  'nci109_records_newweights',  'nci109',     2, 413, None),
]
RECORDS_DIR = Path('results/filtration_grid')
MAH_DIR     = Path('results/filtration_grid/analysis/full46_newweights')
CANONICAL_CFG = ('proxy', 5)


def L_of(k, alpha=ALPHA):
    return math.log(2.0 * k / alpha)


def r_pinelis(R, m, k):
    if not R or R <= 0 or m <= 0:
        return float('inf')
    return 2.0 * R * math.sqrt(2.0 * L_of(k) / m)      # paper radius (i): centred, factor 2 (fixed 2026-11-08)


def r_var_pinelis(sigma_op, R, m, k):
    """Pt-3 corrected: variance term + linear Bernstein correction."""
    if not sigma_op or sigma_op <= 0 or m <= 0:
        return float('inf')
    L = L_of(k)
    return math.sqrt(2.0 * sigma_op * L / m) + 4.0 * (R or 0.0) * L / (3.0 * m)


def r_chi2(sigma_op, m, ell, k):
    if not sigma_op or sigma_op <= 0 or m <= 0 or ell <= 0:
        return float('inf')
    q = float(chi2.ppf(1.0 - ALPHA / k, df=int(ell)))
    return math.sqrt(sigma_op * q / m)


def per_fold_selection(mah_csv, allowed):
    df = pd.read_csv(mah_csv)
    mcol = 'mahalanobis' if 'mahalanobis' in df.columns else 'rho_mahalanobis'
    df = df.dropna(subset=[mcol])
    if allowed is not None:
        df = df[df['filt'].isin(allowed)]
    df = df.sort_values(['seed', 'fold', mcol], ascending=[True, True, False])
    return df.groupby(['seed', 'fold']).head(1)[['seed', 'fold', 'filt']]


def process(display, rec_stem, mah_stem, k, fb):
    rec = pd.read_csv(RECORDS_DIR / f'{rec_stem}.csv')
    cfg = rec[(rec['tau_method'] == CANONICAL_CFG[0]) &
              (rec['n_scales'] == CANONICAL_CFG[1])]
    rec = cfg if not cfg.empty else rec
    if mah_stem is None:
        rows = [{'seed': s, 'fold': f, 'filt': fb}
                for s in sorted(rec['seed'].unique())
                for f in sorted(rec['fold'].unique())]
        sel = pd.DataFrame(rows)
    else:
        mah_csv = MAH_DIR / f'mahalanobis_{mah_stem}.csv'
        if not mah_csv.exists():
            return {'dataset': display, 'status': f'MISSING {mah_csv}'}
        sel = per_fold_selection(mah_csv, set(rec['filt'].dropna().unique()))
    m = sel.merge(rec, on=['seed', 'fold', 'filt'], how='inner')
    if m.empty:
        return {'dataset': display, 'status': 'no rows'}

    m['r_pin'] = m.apply(lambda r: r_pinelis(r['R_max_tr'], r['m_min'], k), axis=1)
    m['r_vp'] = m.apply(
        lambda r: r_var_pinelis(r['Sigma_op_max'], r['R_max_tr'], r['m_min'], k), axis=1)
    m['r_chi'] = m.apply(lambda r: r_chi2(r['Sigma_op_max'], r['m_min'], r['ell'], k), axis=1)

    out = {'dataset': display, 'k': k, 'n_folds': len(m), 'status': 'OK'}
    for tag, thresh in [('half', 0.5), ('quarter', 0.25)]:
        d = thresh * m['delta_hat']
        for rad in ['pin', 'vp', 'chi']:
            fire = (m[f'r_{rad}'] < d)
            out[f'fire_{rad}_{tag}'] = 100.0 * fire.mean()
        ncv = m[m['r_vp'] < d]['nc_acc']
        out[f'nc_vp_{tag}'] = 100.0 * ncv.mean() if len(ncv) else float('nan')
    return out


def main():
    rows = [process(d, rs, ms, k, fb) for d, rs, ms, k, _, fb in DATASETS]
    ok = [r for r in rows if r.get('status') == 'OK']

    print(f"\n{'Dataset':9s} {'vP/2%':>7} {'vP/4%':>7}  "
          f"{'Pin/2%':>7} {'Pin/4%':>7}  {'NC|vP/2':>8} {'NC|vP/4':>8}")
    print('-' * 64)
    for r in ok:
        nc2 = r['nc_vp_half']; nc4 = r['nc_vp_quarter']
        print(f"{r['dataset']:9s} {r['fire_vp_half']:>6.1f}% {r['fire_vp_quarter']:>6.1f}%  "
              f"{r['fire_pin_half']:>6.1f}% {r['fire_pin_quarter']:>6.1f}%  "
              f"{(nc2 if nc2==nc2 else 0):>7.1f}% {(nc4 if nc4==nc4 else 0):>7.1f}%")

    def count(tag, rad='vp', cut=0.0):
        return sum(1 for r in ok if r[f'fire_{rad}_{tag}'] > cut)
    n = len(ok)
    print('-' * 64)
    print(f"vP fires (>0%)   OLD r<Dhat/2: {count('half')}/{n}   "
          f"CORRECTED r<Dhat/4: {count('quarter')}/{n}")
    print(f"vP fires (>=50%) OLD r<Dhat/2: {count('half', cut=50)}/{n}   "
          f"CORRECTED r<Dhat/4: {count('quarter', cut=50)}/{n}")
    print(f"vP fires (=100%) OLD r<Dhat/2: {count('half', cut=99.9)}/{n}   "
          f"CORRECTED r<Dhat/4: {count('quarter', cut=99.9)}/{n}")

    out = Path('results/paper_II/cert_firing_quarter_compare.csv')
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ok).to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == '__main__':
    main()
