"""
recompute_cert_firing_honest_perfold.py
---------------------------------------
J5KD-W5/W6 + 6nmp-Pt3/Pt5 (TMLR round-1): EXACT per-fold honest firing.

The submission firing table (results/paper_I/cert_firing_variance_pinelis.csv,
reproduced by audit_variance_aware_pinelis.py) used the CLEAN variance-aware
radius r_vP = sqrt(2 tr(Σ) L / m) tested against Δ̂/2. The honest radius carries
the linear Bernstein term and the empirical firing condition needs Δ̂/4
(Prop 3.2: |Δ̂-Δ| ≤ 2 r_m):

    r_vP_honest = sqrt(2 tr(Σ) L / m) + 4 R L / (3 m),   L = log(2k/α)
    honest firing:  r_vP_honest < Δ̂/4.

This script recomputes firing PER FOLD (not from per-dataset medians), over the
same per-fold records that reproduce the frozen submission table exactly
(verified: MUTAG deg+hks10 median ell=4003, R=6.699, σ=1.229, Δ̂=1.570). tr(Σ) is
taken as ‖Σ̂‖_op (empirical stable rank ≤ 1.17 on the audited datasets).

No re-embedding: reads results/filtration_grid/<stem>.csv.
"""
from __future__ import annotations
import math
from pathlib import Path
import pandas as pd

RECORDS_DIR = Path('results/filtration_grid')
ALPHA = 0.05

# (display, records_stem, k, committed headline descriptor) -- matches
# experiments/audit_variance_aware_pinelis.py (the submission firing table).
DATASETS = [
    ('MUTAG', 'mutag_records_newweights', 2, 'deg+hks10'),
    ('PROTEINS', 'proteins_records_newweights', 2, 'deg+ricci'),
    ('NCI1', 'nci1_records_newweights', 2, 'hks_t10'),
    ('NCI109', 'nci109_records_newweights', 2, 'hks_t10'),
    ('DHFR', 'dhfr_records_newweights', 2, 'hks_t10'),
    ('DD', 'dd_records_newweights', 2, 'degree'),
    ('REDDIT-5K', 'reddit5k_records_newweights', 5, 'closeness'),
    ('COX2', 'cox2_records_newweights', 2, 'jaccard+hks10'),
    ('PTC', 'ptc_records_newweights', 2, 'deg+betw'),
    ('IMDB-B', 'imdbb_records_newweights', 2, 'degree'),
    ('IMDB-M', 'imdbm_records_newweights', 3, 'betw+ricci'),
    ('Orbit5k', 'orbit5k_records_newweights', 5, 'alpha_H1'),
]


def process(name, stem, k, filt):
    df = pd.read_csv(RECORDS_DIR / f'{stem}.csv')
    s = df[df.filt == filt].dropna(
        subset=['R_max_tr', 'Sigma_op_max', 'delta_hat', 'm_min']).copy()
    if not len(s):
        return None
    L = math.log(2 * k / ALPHA)
    R, sig, m, dh = s.R_max_tr, s.Sigma_op_max, s.m_min, s.delta_hat
    r_vp_old = (2 * sig * L / m).pow(0.5)              # clean submission radius
    r_vp_hon = r_vp_old + 4 * R * L / (3 * m)          # + linear Bernstein term
    return dict(
        dataset=name, filt=filt, n_fold=len(s),
        fire_old_half=100 * (r_vp_old < dh / 2).mean(),      # reproduces submission
        fire_hon_half=100 * (r_vp_hon < dh / 2).mean(),      # Pt3 only
        fire_hon_quarter=100 * (r_vp_hon < dh / 4).mean(),   # Pt3 + Pt5 (honest)
    )


def main():
    rows = [r for r in (process(*d) for d in DATASETS) if r]
    o = pd.DataFrame(rows)
    print(o.to_string(index=False))
    thr = 0.0
    old = o[o.fire_old_half > thr].dataset.tolist()
    hon = o[o.fire_hon_quarter > thr].dataset.tolist()
    print(f"\nSubmission (old radius, Δ̂/2):     {len(old)}/12  {old}")
    print(f"Honest (Pt3 term) + Δ̂/2:          "
          f"{int((o.fire_hon_half > thr).sum())}/12")
    print(f"Honest (Pt3 term) + Δ̂/4 (final):  {len(hon)}/12  {hon}")
    out = Path('results/paper_I/cert_firing_honest_perfold.csv')
    o.to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == '__main__':
    main()
