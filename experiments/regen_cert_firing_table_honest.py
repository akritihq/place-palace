"""
regen_cert_firing_table_honest.py
---------------------------------
Regenerate tab:cert_firing (TMLR Paper I) with HONEST, theorem-consistent
numbers, replacing the submission table.

Changes vs the submission table (audit_variance_aware_pinelis.py):
  * Pinelis radius uses the theorem form r_Pin = 2R sqrt(2L/m)  (eq:rm-pinelis);
    the submission table dropped the factor 2.
  * Variance-aware radius carries the linear Bernstein term (eq:rm-vp):
        r_vP = sqrt(2 tr(Σ) L / m) + 4 R L / (3 m),   tr(Σ) ≈ ‖Σ̂‖_op.
  * Firing tested against the honest empirical threshold Δ̂/4 (Prop 3.2 / Pt.5),
    NOT Δ̂/2, for all three radii.
Radii reported as per-fold medians; Fire % is the per-fold firing fraction.
Reads the current per-fold records, which reproduce the submission firing
inputs exactly (ell=4003, R=6.699, σ=1.229, Δ̂=1.570 on MUTAG deg+hks10).
"""
from __future__ import annotations
import math
from pathlib import Path
import pandas as pd
from scipy.stats import chi2

RECORDS_DIR = Path('results/filtration_grid')
ALPHA = 0.05

# (display, records_stem, k, filt, latex_filt)  -- table order & committed descriptor
DATASETS = [
    ('MUTAG', 'mutag_records_newweights', 2, 'deg+hks10', r'deg+HKS$_{10}$'),
    ('PROTEINS', 'proteins_records_newweights', 2, 'deg+ricci', r'deg+ricci'),
    ('NCI1', 'nci1_records_newweights', 2, 'hks_t10', r'HKS$_{10}$'),
    ('NCI109', 'nci109_records_newweights', 2, 'hks_t10', r'HKS$_{10}$'),
    ('DHFR', 'dhfr_records_newweights', 2, 'hks_t10', r'HKS$_{10}$'),
    ('DD', 'dd_records_newweights', 2, 'degree', r'degree'),
    ('REDDIT-5K', 'reddit5k_records_newweights', 5, 'closeness', r'closeness'),
    ('COX2', 'cox2_records_newweights', 2, 'jaccard+hks10', r'jaccard+HKS$_{10}$'),
    ('PTC', 'ptc_records_newweights', 2, 'deg+betw', r'deg+betw'),
    ('IMDB-B', 'imdbb_records_newweights', 2, 'degree', r'degree'),
    ('IMDB-M', 'imdbm_records_newweights', 3, 'betw+ricci', r'betw+ricci'),
    ('Orbit5k', 'orbit5k_records_newweights', 5, 'alpha_H1', r'alpha~$H_1$'),
]


def fmt(x, d=2):
    return f"{x:.{d}f}" if x >= 0.1 or x == 0 else f"{x:.3f}"


def process(name, stem, k, filt, latex_filt):
    df = pd.read_csv(RECORDS_DIR / f'{stem}.csv')
    s = df[df.filt == filt].dropna(
        subset=['R_max_tr', 'Sigma_op_max', 'delta_hat', 'm_min', 'ell']).copy()
    if not len(s):
        return None
    L = math.log(2 * k / ALPHA)
    R, sig, m, dh, ell = (s.R_max_tr, s.Sigma_op_max, s.m_min, s.delta_hat, s.ell)
    q = chi2.ppf(1 - ALPHA / k, ell)                       # per-fold χ² quantile
    r_pin = 2 * R * (2 * L / m).pow(0.5)                    # theorem eq:rm-pinelis
    r_vp = (2 * sig * L / m).pow(0.5) + 4 * R * L / (3 * m) # honest eq:rm-vp
    r_g = (sig * q / m).pow(0.5)                            # radius (ii)
    quarter = dh / 4                                       # honest threshold
    return dict(
        dataset=name, filt=latex_filt, m_min=int(m.median()),
        r_pin=r_pin.median(), r_vp=r_vp.median(), r_g=r_g.median(),
        quarter=quarter.median(),
        fire_pin=100 * (r_pin < quarter).mean(),
        fire_vp=100 * (r_vp < quarter).mean(),
        fire_g=100 * (r_g < quarter).mean(),
    )


def main():
    rows = [r for r in (process(*d) for d in DATASETS) if r]
    print(f"{'dataset':10s} {'m_min':>6} {'r_Pin':>7} {'r_vP':>7} {'r_G':>8} "
          f"{'Δ̂/4':>7} {'Pin%':>5} {'vP%':>6} {'G%':>5}")
    tex = []
    for r in rows:
        vp = r['fire_vp']
        vpcell = (r"$\mathbf{" + f"{vp:.0f}" + r"\%}$") if vp > 0 else r"$0\%$"
        tex.append(
            f"{r['dataset']:11s} & {r['filt']:18s} & ${r['m_min']}$ & "
            f"${fmt(r['r_pin'])}$ & ${fmt(r['r_vp'],3)}$ & ${fmt(r['r_g'])}$ & "
            f"${fmt(r['quarter'],3)}$ & ${r['fire_pin']:.0f}\\%$ & "
            f"{vpcell} & ${r['fire_g']:.0f}\\%$ \\\\")
        print(f"{r['dataset']:10s} {r['m_min']:>6d} {r['r_pin']:>7.2f} "
              f"{r['r_vp']:>7.3f} {r['r_g']:>8.2f} {r['quarter']:>7.3f} "
              f"{r['fire_pin']:>4.0f}% {vp:>5.0f}% {r['fire_g']:>4.0f}%")
    fires = [r['dataset'] for r in rows if r['fire_vp'] > 0]
    print(f"\nHonest vP firing (>0%): {len(fires)}/12  {fires}")
    out = Path('results/paper_I/cert_firing_table_honest.csv')
    pd.DataFrame(rows).to_csv(out, index=False)
    (Path('results/paper_I/cert_firing_table_honest.tex')).write_text('\n'.join(tex))
    print(f"Wrote {out} and .tex rows")


if __name__ == '__main__':
    main()
