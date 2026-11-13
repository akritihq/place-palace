"""
experiments/pool_chemical.py
----------------------------
Shared filtration pool for Wang's chemical graph datasets

    MUTAG, NCI1, NCI109, PTC, COX2, DHFR.

All six datasets follow the Wang 2019 / Paper I chemical convention:
  • extended persistence
  • both H0 and H1 dimensions merged before top-K truncation
  • node labels available (atom type, one integer per vertex)

Pool size
  24 atomic singles + 40 curated pairs = 64 configs.

Each runner (exp_grid_mutag.py, exp_grid_nci1.py, …) imports POOL from
here and passes it through to grid_common.build_graph_diagrams.
"""
from __future__ import annotations

_DIMS = [0, 1]


# ── 24 atomic singles ────────────────────────────────────────────────────
ATOMIC_SINGLES = [
    # Node centralities (7)
    {'name': 'degree',              'filts': ['degree'],              'dims': _DIMS},
    {'name': 'betweenness',         'filts': ['betweenness'],         'dims': _DIMS},
    {'name': 'closeness',           'filts': ['closeness'],           'dims': _DIMS},
    {'name': 'eigenvector',         'filts': ['eigenvector'],         'dims': _DIMS},
    {'name': 'pagerank',            'filts': ['pagerank'],            'dims': _DIMS},
    {'name': 'clustering',          'filts': ['clustering'],          'dims': _DIMS},
    {'name': 'core_number',         'filts': ['core_number'],         'dims': _DIMS},
    # Edge descriptors (4)
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
    # Curvature (2)
    {'name': 'ricci',               'filts': ['ricci'],               'dims': _DIMS},
    {'name': 'forman_ricci',        'filts': ['forman_ricci'],        'dims': _DIMS},
    # Node label (atom type) (1)
    {'name': 'nodelabel',           'filts': ['nodelabel'],           'dims': _DIMS},
]


# ── 40 curated pairs ─────────────────────────────────────────────────────
ATOMIC_PAIRS = [
    # degree × each (13)
    {'name': 'deg+betw',         'filts': ['degree', 'betweenness'],        'dims': _DIMS},
    {'name': 'deg+closeness',    'filts': ['degree', 'closeness'],          'dims': _DIMS},
    {'name': 'deg+eigen',        'filts': ['degree', 'eigenvector'],        'dims': _DIMS},
    {'name': 'deg+pagerank',     'filts': ['degree', 'pagerank'],           'dims': _DIMS},
    {'name': 'deg+clustering',   'filts': ['degree', 'clustering'],         'dims': _DIMS},
    {'name': 'deg+core',         'filts': ['degree', 'core_number'],        'dims': _DIMS},
    {'name': 'deg+jaccard',     'filts': ['degree', 'jaccard'],             'dims': _DIMS},
    {'name': 'deg+ricci',        'filts': ['degree', 'ricci'],              'dims': _DIMS},
    {'name': 'deg+forman',       'filts': ['degree', 'forman_ricci'],       'dims': _DIMS},
    {'name': 'deg+hks1',         'filts': ['degree', 'hks_t1'],             'dims': _DIMS},
    {'name': 'deg+hks10',        'filts': ['degree', 'hks_t10'],            'dims': _DIMS},
    {'name': 'deg+hks100',       'filts': ['degree', 'hks_t100'],           'dims': _DIMS},
    {'name': 'deg+nodelabel',    'filts': ['degree', 'nodelabel'],          'dims': _DIMS},

    # betweenness × other (6)
    {'name': 'betw+closeness',   'filts': ['betweenness', 'closeness'],     'dims': _DIMS},
    {'name': 'betw+jaccard',     'filts': ['betweenness', 'jaccard'],       'dims': _DIMS},
    {'name': 'betw+ricci',       'filts': ['betweenness', 'ricci'],         'dims': _DIMS},
    {'name': 'betw+forman',      'filts': ['betweenness', 'forman_ricci'],  'dims': _DIMS},
    {'name': 'betw+hks1',        'filts': ['betweenness', 'hks_t1'],        'dims': _DIMS},
    {'name': 'betw+hks10',       'filts': ['betweenness', 'hks_t10'],       'dims': _DIMS},

    # Curvature × HKS (3)
    {'name': 'ricci+hks1',       'filts': ['ricci', 'hks_t1'],              'dims': _DIMS},
    {'name': 'ricci+hks10',      'filts': ['ricci', 'hks_t10'],             'dims': _DIMS},
    {'name': 'forman+hks10',     'filts': ['forman_ricci', 'hks_t10'],      'dims': _DIMS},

    # HKS × HKS — widened (5)
    {'name': 'hks0.1+hks100',    'filts': ['hks_t0.1', 'hks_t100'],         'dims': _DIMS},
    {'name': 'hks1+hks10',       'filts': ['hks_t1', 'hks_t10'],            'dims': _DIMS},
    {'name': 'hks0.5+hks5',      'filts': ['hks_t0.5', 'hks_t5'],           'dims': _DIMS},
    {'name': 'hks2+hks25',       'filts': ['hks_t2', 'hks_t25'],            'dims': _DIMS},
    {'name': 'hks0.1+hks10',     'filts': ['hks_t0.1', 'hks_t10'],          'dims': _DIMS},

    # Node label × chemically-relevant — widened for NCI1/NCI109 (6)
    {'name': 'nodelabel+ricci',     'filts': ['nodelabel', 'ricci'],         'dims': _DIMS},
    {'name': 'nodelabel+betw',      'filts': ['nodelabel', 'betweenness'],   'dims': _DIMS},
    {'name': 'nodelabel+hks10',     'filts': ['nodelabel', 'hks_t10'],       'dims': _DIMS},
    {'name': 'nodelabel+jaccard',   'filts': ['nodelabel', 'jaccard'],       'dims': _DIMS},
    {'name': 'nodelabel+forman',    'filts': ['nodelabel', 'forman_ricci'],  'dims': _DIMS},
    {'name': 'nodelabel+hks1',      'filts': ['nodelabel', 'hks_t1'],        'dims': _DIMS},
    {'name': 'nodelabel+clustering','filts': ['nodelabel', 'clustering'],    'dims': _DIMS},
    {'name': 'nodelabel+adamic',    'filts': ['nodelabel', 'adamic_adar'],   'dims': _DIMS},

    # Jaccard × HKS (1 existing)
    {'name': 'jaccard+hks10',    'filts': ['jaccard', 'hks_t10'],           'dims': _DIMS},

    # Structural × curvature (2 new)
    {'name': 'clustering+ricci', 'filts': ['clustering', 'ricci'],          'dims': _DIMS},
    {'name': 'core+ricci',       'filts': ['core_number', 'ricci'],         'dims': _DIMS},

    # Edge descriptor × HKS (2 new)
    {'name': 'adamic+hks10',     'filts': ['adamic_adar', 'hks_t10'],       'dims': _DIMS},
    {'name': 'edge_betw+hks10',  'filts': ['edge_betweenness', 'hks_t10'],  'dims': _DIMS},
]


POOL = ATOMIC_SINGLES + ATOMIC_PAIRS
