"""
selector_ablation_committed.py
------------------------------
J5KD-W9 (TMLR round-1 revision): protocol-matched closed-form headline.

Computes the CLOSED-FORM Mahalanobis selector's end-to-end PLACE-linear
accuracy on the SAME 15-descriptor committed candidate pool used for the
Table exp1 headline (tab:graph_filt), so the number is comparable to the
borrowed baselines. Per (seed, fold): pick argmax rho_Mahalanobis over the
15 committed descriptors, report that descriptor's held-out linear_acc.
Also reports the 15-pool in-pool oracle (argmax test linear_acc) and the
random-pick baseline, all at the canonical config the Mahalanobis stat is
defined at (proxy, n_scales=10 -- the config of tab:selection_ranks and of
the mahalanobis_<ds>.csv files).

No re-embedding: reads existing records + mahalanobis stat files.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

RECORDS_DIR = Path('results/filtration_grid')
MAH_DIR = Path('results/filtration_grid/analysis/full46_newweights')
CANON = ('proxy', 10)   # native config of mahalanobis_<ds>.csv & tab:selection_ranks
ACC = 'linear_acc'

# 15 committed descriptors (tab:graph_filt / Sec 6.2), mapped to record keys.
COMMITTED = [
    'degree', 'betweenness', 'closeness', 'clustering', 'core_number',
    'jaccard', 'ricci', 'forman_ricci', 'hks_t10',            # singletons (9)
    'deg+betw', 'deg+ricci', 'deg+hks10', 'betw+ricci',
    'ricci+hks10', 'jaccard+hks10',                            # pairs (6)
]

# (display, records_stem, mahalanobis_stem)
DATASETS = [
    ('MUTAG', 'mutag', 'mutag'), ('PROTEINS', 'proteins', 'proteins'),
    ('NCI1', 'nci1', 'nci1'), ('COX2', 'cox2', 'cox2'),
    ('DHFR', 'dhfr', 'dhfr'), ('PTC', 'ptc', 'ptc'),
    ('DD', 'dd', 'dd'), ('IMDB-B', 'imdbb', 'imdb-b'),
    ('IMDB-M', 'imdbm', 'imdb-m'), ('NCI109', 'nci109', 'nci109'),
    ('REDDIT-5K', 'reddit5k', 'reddit-5k'),
]

# published Table exp1 headline oracle (best-of-120, train-selected) for reference
PUB_ORACLE = {
    'MUTAG': 88.4, 'PROTEINS': 71.5, 'NCI1': 71.3, 'COX2': 80.0, 'DHFR': 77.6,
    'PTC': 59.3, 'DD': 76.3, 'IMDB-B': 66.4, 'IMDB-M': 44.5, 'NCI109': 70.6,
    'REDDIT-5K': 46.2,
}


def load(ds_stem, mah_stem):
    rec = pd.read_csv(RECORDS_DIR / f'{ds_stem}_records_newweights.csv')
    rec = rec[(rec.tau_method == CANON[0]) & (rec.n_scales == CANON[1])]
    mah = pd.read_csv(MAH_DIR / f'mahalanobis_{mah_stem}.csv')
    m = rec.merge(mah[['seed', 'fold', 'filt', 'mahalanobis']],
                  on=['seed', 'fold', 'filt'], how='inner')
    m = m[m.filt.isin(COMMITTED)]
    return m.dropna(subset=[ACC, 'mahalanobis'])


def per_fold(m):
    mah_acc, rand_acc, oracle_acc = [], [], []
    for _, g in m.groupby(['seed', 'fold']):
        if len(g) < 2:
            continue
        mah_acc.append(g.loc[g['mahalanobis'].idxmax(), ACC])
        rand_acc.append(g[ACC].mean())
        oracle_acc.append(g[ACC].max())
    return np.array(mah_acc), np.array(rand_acc), np.array(oracle_acc)


def main():
    rows = []
    print(f"\n{'Dataset':10s} {'|pool|':>6} {'Mahal':>12} {'Random':>12} "
          f"{'Ora(15,p10)':>12} {'PubOra(120)':>11}")
    print('-' * 70)
    for disp, rec_stem, mah_stem in DATASETS:
        try:
            m = load(rec_stem, mah_stem)
        except FileNotFoundError as e:
            print(f"{disp:10s}  -- missing: {e}")
            continue
        mah, rnd, ora = per_fold(m)
        if len(mah) == 0:
            print(f"{disp:10s}  -- no folds (committed keys present: "
                  f"{sorted(set(m.filt.unique()))})")
            continue
        pool = int(m.groupby(['seed', 'fold']).size().median())
        f = lambda a: (100 * a.mean(), 100 * a.std())
        (mm, ms), (rm, rs), (om, os) = f(mah), f(rnd), f(ora)
        rows.append(dict(dataset=disp, pool=pool,
                         mahalanobis=mm, mahalanobis_std=ms,
                         random=rm, random_std=rs,
                         oracle15=om, oracle15_std=os,
                         pub_oracle120=PUB_ORACLE.get(disp, float('nan'))))
        print(f"{disp:10s} {pool:>6d} {mm:>6.1f}+/-{ms:>4.1f} "
              f"{rm:>6.1f}+/-{rs:>4.1f} {om:>6.1f}+/-{os:>4.1f} "
              f"{PUB_ORACLE.get(disp, float('nan')):>11.1f}")
    df = pd.DataFrame(rows)
    out = Path('results/tables'); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / 'selector_ablation_committed.csv', index=False)
    if len(df):
        print('-' * 70)
        print(f"{'MEAN':10s} {'':>6} {df.mahalanobis.mean():>6.1f}"
              f"{'':>7}{df.random.mean():>6.1f}{'':>7}"
              f"{df.oracle15.mean():>6.1f}{'':>7}{df.pub_oracle120.mean():>11.1f}")
    print(f"\nWrote {out/'selector_ablation_committed.csv'}")


if __name__ == '__main__':
    main()
