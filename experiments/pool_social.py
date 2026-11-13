"""
experiments/pool_social.py
--------------------------
Shared filtration pool for Wang's social / protein graph datasets

    PROTEINS, DD, IMDB-B, IMDB-M, REDDIT-5K.

All five follow the Wang 2019 / Paper I social convention:
  • ordinary persistence
  • H0 only, using the union of sublevel and superlevel sets
  • no node labels (Wang uses Jaccard uniformly and doesn't touch atom-
    type or amino-acid labels even on PROTEINS / DD)

Pool size
  23 atomic singles + 31 curated pairs = 54 configs.

Each runner (exp_grid_imdbm.py, exp_grid_imdbb.py, exp_grid_proteins.py,
exp_grid_dd.py, exp_grid_reddit5k.py) imports POOL from here and passes
it through to grid_common.build_graph_diagrams(extended=False,
union_h0=True).

Paper I Table 4 best filtrations for each dataset are all covered:

    PROTEINS   ricci+hks10
    DD         degree
    IMDB-B     degree
    IMDB-M     degree
    REDDIT-5K  deg+betw

Wang's filtration (jaccard) is present as an atomic entry so that Wang's
per-dataset performance on PLACE+jaccard is directly readable from the
CSV.
"""
from __future__ import annotations

_DIMS = [0]


# ── 23 atomic singles ────────────────────────────────────────────────────
ATOMIC_SINGLES = [
    # Node centralities (7)
    {'name': 'degree',              'filts': ['degree'],              'dims': _DIMS},
    {'name': 'betweenness',         'filts': ['betweenness'],         'dims': _DIMS},
    {'name': 'closeness',           'filts': ['closeness'],           'dims': _DIMS},
    {'name': 'eigenvector',         'filts': ['eigenvector'],         'dims': _DIMS},
    {'name': 'pagerank',            'filts': ['pagerank'],            'dims': _DIMS},
    {'name': 'clustering',          'filts': ['clustering'],          'dims': _DIMS},
    {'name': 'core_number',         'filts': ['core_number'],         'dims': _DIMS},
    # Edge descriptors (4) — jaccard here is Wang's filtration
    {'name': 'jaccard',             'filts': ['jaccard'],             'dims': _DIMS},
    {'name': 'adamic_adar',         'filts': ['adamic_adar'],         'dims': _DIMS},
    {'name': 'resource_allocation', 'filts': ['resource_allocation'], 'dims': _DIMS},
    {'name': 'edge_betweenness',    'filts': ['edge_betweenness'],    'dims': _DIMS},
    # HKS t ∈ {0.1, 0.25, 0.5, 1, 2, 5, 10, 25, 50, 100} (10)
    {'name': 'hks_t0.1',            'filts': ['hks_t0.1'],            'dims': _DIMS},
    {'name': 'hks_t0.25',           'filts': ['hks_t0.25'],           'dims': _DIMS},
    {'name': 'hks_t0.5',            'filts': ['hks_t0.5'],            'dims': _DIMS},
    {'name': 'hks_t1',              'filts': ['hks_t1'],              'dims': _DIMS},
    {'name': 'hks_t2',              'filts': ['hks_t2'],              'dims': _DIMS},
    {'name': 'hks_t5',              'filts': ['hks_t5'],              'dims': _DIMS},
    {'name': 'hks_t10',             'filts': ['hks_t10'],             'dims': _DIMS},
    {'name': 'hks_t25',             'filts': ['hks_t25'],             'dims': _DIMS},
    {'name': 'hks_t50',             'filts': ['hks_t50'],             'dims': _DIMS},
    {'name': 'hks_t100',            'filts': ['hks_t100'],            'dims': _DIMS},
    # Curvature (2) — needed for PROTEINS (Paper I Table 4 picks ricci+hks10)
    {'name': 'ricci',               'filts': ['ricci'],               'dims': _DIMS},
    {'name': 'forman_ricci',        'filts': ['forman_ricci'],        'dims': _DIMS},
]


# ── 31 curated pairs ─────────────────────────────────────────────────────
ATOMIC_PAIRS = [
    # degree × each (12 — no deg+nodelabel, no social label)
    {'name': 'deg+betw',         'filts': ['degree', 'betweenness'],        'dims': _DIMS},
    {'name': 'deg+closeness',    'filts': ['degree', 'closeness'],          'dims': _DIMS},
    {'name': 'deg+eigen',        'filts': ['degree', 'eigenvector'],        'dims': _DIMS},
    {'name': 'deg+pagerank',     'filts': ['degree', 'pagerank'],           'dims': _DIMS},
    {'name': 'deg+clustering',   'filts': ['degree', 'clustering'],         'dims': _DIMS},
    {'name': 'deg+core',         'filts': ['degree', 'core_number'],        'dims': _DIMS},
    {'name': 'deg+jaccard',      'filts': ['degree', 'jaccard'],            'dims': _DIMS},
    {'name': 'deg+ricci',        'filts': ['degree', 'ricci'],              'dims': _DIMS},
    {'name': 'deg+forman',       'filts': ['degree', 'forman_ricci'],       'dims': _DIMS},
    {'name': 'deg+hks1',         'filts': ['degree', 'hks_t1'],             'dims': _DIMS},
    {'name': 'deg+hks10',        'filts': ['degree', 'hks_t10'],            'dims': _DIMS},
    {'name': 'deg+hks100',       'filts': ['degree', 'hks_t100'],           'dims': _DIMS},

    # betweenness × other (6)
    {'name': 'betw+closeness',   'filts': ['betweenness', 'closeness'],     'dims': _DIMS},
    {'name': 'betw+jaccard',     'filts': ['betweenness', 'jaccard'],       'dims': _DIMS},
    {'name': 'betw+ricci',       'filts': ['betweenness', 'ricci'],         'dims': _DIMS},
    {'name': 'betw+forman',      'filts': ['betweenness', 'forman_ricci'],  'dims': _DIMS},
    {'name': 'betw+hks1',        'filts': ['betweenness', 'hks_t1'],        'dims': _DIMS},
    {'name': 'betw+hks10',       'filts': ['betweenness', 'hks_t10'],       'dims': _DIMS},

    # Curvature × HKS (3) — `ricci+hks10` is Paper I's PROTEINS winner
    {'name': 'ricci+hks1',       'filts': ['ricci', 'hks_t1'],              'dims': _DIMS},
    {'name': 'ricci+hks10',      'filts': ['ricci', 'hks_t10'],             'dims': _DIMS},
    {'name': 'forman+hks10',     'filts': ['forman_ricci', 'hks_t10'],      'dims': _DIMS},

    # HKS × HKS — widened (5)
    {'name': 'hks0.1+hks100',    'filts': ['hks_t0.1', 'hks_t100'],         'dims': _DIMS},
    {'name': 'hks1+hks10',       'filts': ['hks_t1', 'hks_t10'],            'dims': _DIMS},
    {'name': 'hks0.5+hks5',      'filts': ['hks_t0.5', 'hks_t5'],           'dims': _DIMS},
    {'name': 'hks2+hks25',       'filts': ['hks_t2', 'hks_t25'],            'dims': _DIMS},
    {'name': 'hks0.1+hks10',     'filts': ['hks_t0.1', 'hks_t10'],          'dims': _DIMS},

    # Jaccard × HKS (1) — Wang's descriptor × multiscale
    {'name': 'jaccard+hks10',    'filts': ['jaccard', 'hks_t10'],           'dims': _DIMS},

    # Structural × curvature (2)
    {'name': 'clustering+ricci', 'filts': ['clustering', 'ricci'],          'dims': _DIMS},
    {'name': 'core+ricci',       'filts': ['core_number', 'ricci'],         'dims': _DIMS},

    # Edge descriptor × HKS (2)
    {'name': 'adamic+hks10',     'filts': ['adamic_adar', 'hks_t10'],       'dims': _DIMS},
    {'name': 'edge_betw+hks10',  'filts': ['edge_betweenness', 'hks_t10'],  'dims': _DIMS},
]


POOL = ATOMIC_SINGLES + ATOMIC_PAIRS
