"""
exp_ballester_rieck_demo.py
---------------------------
6ydS Q3 (TMLR round-1 revision): demonstrate that PLACE's descriptor
selector is filtration-agnostic by dropping a *non-sublevel* filtration
into the pool -- the Ballester-Rieck Vietoris-Rips-on-graphs generator
[Ballester & Rieck, "On the Expressivity of Persistent Homology in
Graph Learning"]: build the VR complex on the shortest-path metric d_G,
where sigma enters at f_V(sigma) = max_{u,v in sigma} d_G(u,v), and take
H0 + H1. This carries higher-order (clique) information beyond the
sublevel H0/H1 filtrations the paper currently uses.

We compute VR-SP persistence, embed with the SAME PLACE pipeline
(embed_fold), classify with the SAME linear SVM (fit_linear_svm), and
compare head-to-head with a sublevel 'degree' baseline on identical
splits. Small datasets only (demo): MUTAG, PTC.

Run (background):
  python experiments/exp_ballester_rieck_demo.py
Output: results/tables/ballester_rieck_demo.csv
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import networkx as nx
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))

from utils.datasets import load_tu_dataset
from utils.persistence import graph_to_persistence
from grid_common import embed_fold, fit_linear_svm

N_SEEDS = 5
N_SPLITS = 10
TEST_FRAC = 0.10
TOP_K = 64


def _truncate_top_k(dgm, k=TOP_K):
    if len(dgm) <= k:
        return dgm
    pers = dgm[:, 1] - dgm[:, 0]
    return dgm[np.argsort(-pers)[:k]]


def vr_sp_diagram(G, max_dim=1):
    """Vietoris-Rips persistence on the shortest-path metric (H0+H1)."""
    import gudhi
    nodes = sorted(G.nodes())
    n = len(nodes)
    if n == 0:
        return np.zeros((0, 2))
    idx = {v: i for i, v in enumerate(nodes)}
    D = np.full((n, n), np.inf)
    for u, dist in nx.all_pairs_shortest_path_length(G):
        for v, d in dist.items():
            D[idx[u], idx[v]] = d
    np.fill_diagonal(D, 0.0)
    finite = D[np.isfinite(D)]
    cap = (finite.max() + 1.0) if finite.size else 1.0
    D[~np.isfinite(D)] = cap  # disconnected pairs enter at diameter+1
    rc = gudhi.RipsComplex(distance_matrix=D, max_edge_length=cap)
    st = rc.create_simplex_tree(max_dimension=max_dim + 1)
    st.compute_persistence()
    parts = []
    for dim in range(max_dim + 1):
        for b, d in st.persistence_intervals_in_dimension(dim):
            if np.isfinite(d) and d > b:
                parts.append((b, d))
    return _truncate_top_k(np.array(parts)) if parts else np.zeros((0, 2))


def sublevel_diagram(G, filtration='degree', max_dim=1):
    """H0+H1 sublevel diagram via the existing pipeline, merged."""
    dgms = graph_to_persistence(G, filtration=filtration, max_dim=max_dim)
    parts = [d for d in dgms if len(d)]
    return _truncate_top_k(np.vstack(parts)) if parts else np.zeros((0, 2))


def evaluate(diagrams, labels, n_classes, tag):
    accs = []
    for seed in range(N_SEEDS):
        sss = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=TEST_FRAC,
                                     random_state=seed)
        for tr, te in sss.split(np.zeros(len(labels)), labels):
            X_tr, X_te, _ = embed_fold(diagrams, labels, tr, te, n_classes,
                                       tau_method='proxy')
            acc, _ = fit_linear_svm(X_tr, labels[tr], X_te, labels[te], seed)
            if acc == acc:
                accs.append(acc)
    a = np.array(accs)
    return 100 * a.mean(), 100 * a.std(), len(a)


def main():
    out_rows = []
    for name in ['MUTAG', 'PTC_MR']:
        try:
            graphs, labels = load_tu_dataset(name)
        except Exception as e:
            print(f"{name}: load failed ({e}); skipping")
            continue
        labels = np.asarray(labels)
        n_classes = int(labels.max()) + 1
        print(f"\n=== {name}: {len(graphs)} graphs, {n_classes} classes ===")

        t0 = time.time()
        vr = [vr_sp_diagram(G) for G in graphs]
        print(f"  VR-SP diagrams: {time.time()-t0:.1f}s; "
              f"mean |dgm|={np.mean([len(d) for d in vr]):.1f}")
        sub = [sublevel_diagram(G, 'degree') for G in graphs]

        for tag, dgms in [('VR-shortest-path (Ballester-Rieck)', vr),
                          ('sublevel degree (baseline)', sub)]:
            m, s, n = evaluate(dgms, labels, n_classes, tag)
            print(f"  {tag:38s}: {m:5.1f} +/- {s:4.1f}  (n={n})")
            out_rows.append(dict(dataset=name, filtration=tag,
                                 acc_mean=m, acc_std=s, n=n))

    import pandas as pd
    outdir = ROOT / 'results' / 'tables'
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(outdir / 'ballester_rieck_demo.csv', index=False)
    print(f"\nWrote {outdir/'ballester_rieck_demo.csv'}")


if __name__ == '__main__':
    main()
