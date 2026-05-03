"""
utils/persistence.py
--------------------
Compute persistence diagrams from graphs and point clouds.

Graph filtrations:
  - degree        : node degree (fast, always available)
  - betweenness   : betweenness centrality
  - jaccard       : Jaccard edge similarity (social/protein graphs)
  - ricci         : Ollivier-Ricci curvature (α=0.5)
  - hks_t1/t10    : Heat Kernel Signature at diffusion time t=1 or t=10

Point clouds: Vietoris-Rips via ripser (for Orbit5k).

Each function returns a list of np.ndarray of shape (n_pts, 2),
one array per homological dimension, where each row is (birth, death).
Infinite death values are clipped to a finite maximum.

Dependencies
------------
  pip install gudhi networkx scipy numpy
  pip install GraphRicciCurvature  # for Ricci curvature (optional)
  pip install ripser               # for point cloud persistence (optional)
"""

from __future__ import annotations
import numpy as np
import networkx as nx
from pathlib import Path
from typing import List, Tuple
import pickle, os

# ── optional imports (fail loudly only when the function is called) ─────
def _require_gudhi():
    try:
        import gudhi
        return gudhi
    except ImportError:
        raise ImportError(
            "gudhi is required for graph persistence diagrams.\n"
            "Install with: pip install gudhi"
        )

def _require_ripser():
    try:
        import ripser
        return ripser
    except ImportError:
        raise ImportError(
            "ripser is required for Vietoris–Rips persistence.\n"
            "Install with: pip install ripser"
        )

def _require_ricci():
    try:
        from GraphRicciCurvature.OllivierRicci import OllivierRicci
        return OllivierRicci
    except ImportError:
        raise ImportError(
            "GraphRicciCurvature is required for Ricci curvature filtration.\n"
            "Install with: pip install GraphRicciCurvature"
        )

ROOT     = Path(__file__).resolve().parents[1]
DIAG_DIR = ROOT / "data" / "diagrams"
DIAG_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
# Graph → filtration → persistence diagram
# ══════════════════════════════════════════════════════════════════════════

def _hks_filtration(G: nx.Graph, t: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Heat Kernel Signature (HKS) filtration (Section 5.1 of the manuscript).

    HKS(v, t) = sum_k  exp(-lambda_k * t) * phi_k(v)^2

    where (lambda_k, phi_k) are eigenpairs of the normalised graph
    Laplacian.  Edges inherit the maximum of their endpoint values.
    """
    from scipy.sparse.linalg import eigsh
    from scipy.sparse import issparse

    nodes = sorted(G.nodes())
    node_idx = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)

    if n == 0:
        return np.array([]), np.array([])

    # Normalised graph Laplacian: L_norm = I - D^{-1/2} A D^{-1/2}
    L_norm = nx.normalized_laplacian_matrix(G).toarray().astype(float)

    # Full eigendecomposition (graphs are small, typically < 1000 nodes)
    eigenvalues, eigenvectors = np.linalg.eigh(L_norm)

    # HKS(v, t) = sum_k exp(-lambda_k * t) * phi_k(v)^2
    # eigenvalues:  (n,)
    # eigenvectors: (n, n) — column k is phi_k
    weights = np.exp(-eigenvalues * t)                 # (n,)
    phi_sq = eigenvectors ** 2                          # (n, n)
    node_val = phi_sq @ weights                         # (n,)

    # Edges inherit the maximum of their endpoint values
    edges = list(G.edges())
    if edges:
        u_arr = np.array([node_idx[u] for u, _ in edges])
        v_arr = np.array([node_idx[v] for _, v in edges])
        edge_val = np.maximum(node_val[u_arr], node_val[v_arr])
    else:
        edge_val = np.array([])

    return node_val, edge_val


def _degree_filtration(G: nx.Graph) -> Tuple[np.ndarray, np.ndarray]:
    """
    Node-degree descriptor function.
    Returns (node_values, edge_values) arrays (0-indexed nodes).
    """
    nodes = sorted(G.nodes())
    node_idx = {v: i for i, v in enumerate(nodes)}
    node_val = np.array([G.degree(v) for v in nodes], dtype=float)
    edges = list(G.edges())
    if edges:
        u_arr = np.array([node_idx[u] for u, _ in edges])
        v_arr = np.array([node_idx[v] for _, v in edges])
        edge_val = np.maximum(node_val[u_arr], node_val[v_arr])
    else:
        edge_val = np.array([])
    return node_val, edge_val


def _nodelabel_filtration(G: nx.Graph) -> Tuple[np.ndarray, np.ndarray]:
    """
    Node-label filtration: f(v) = label(v) as integer.

    Uses the 'label' attribute stored by load_tu_dataset.
    Edges inherit the maximum of their endpoint labels (sublevel convention).
    """
    nodes = sorted(G.nodes())
    node_idx = {v: i for i, v in enumerate(nodes)}
    node_val = np.array([G.nodes[v].get('label', 0) for v in nodes], dtype=float)
    edges = list(G.edges())
    if edges:
        u_arr = np.array([node_idx[u] for u, _ in edges])
        v_arr = np.array([node_idx[v] for _, v in edges])
        edge_val = np.maximum(node_val[u_arr], node_val[v_arr])
    else:
        edge_val = np.array([])
    return node_val, edge_val


def _betweenness_filtration(G: nx.Graph) -> Tuple[np.ndarray, np.ndarray]:
    """
    Betweenness centrality filtration.

    Node value: betweenness centrality (fraction of shortest paths through v).
    Edge value: max of endpoint betweenness values (sublevel convention).
    """
    nodes = sorted(G.nodes())
    node_idx = {v: i for i, v in enumerate(nodes)}
    bc = nx.betweenness_centrality(G)
    node_val = np.array([bc[v] for v in nodes], dtype=float)
    edges = list(G.edges())
    if edges:
        u_arr = np.array([node_idx[u] for u, _ in edges])
        v_arr = np.array([node_idx[v] for _, v in edges])
        edge_val = np.maximum(node_val[u_arr], node_val[v_arr])
    else:
        edge_val = np.array([])
    return node_val, edge_val


def _jaccard_filtration(G: nx.Graph) -> Tuple[np.ndarray, np.ndarray]:
    """
    Jaccard-index edge similarity descriptor (Wang & Zhao, NeurIPS 2019).

    J(u,v) = |N(u) ∩ N(v)| / |N(u) ∪ N(v)|

    Used for social/protein graphs (PROTEINS, DD, IMDB-B/M, REDDIT)
    with H0 only, union of sublevel + superlevel ordinary persistence.
    """
    nodes = sorted(G.nodes())
    node_idx = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)

    # Jaccard index per edge
    edges = list(G.edges())
    edge_val = np.zeros(len(edges))
    for k, (u, v) in enumerate(edges):
        Nu = set(G.neighbors(u))
        Nv = set(G.neighbors(v))
        inter = len(Nu & Nv)
        union = len(Nu | Nv)
        edge_val[k] = inter / union if union > 0 else 0.0

    # Extend to nodes: min of adjacent edges (sublevel convention)
    node_val = np.full(n, np.inf)
    for k, (u, v) in enumerate(edges):
        node_val[node_idx[u]] = min(node_val[node_idx[u]], edge_val[k])
        node_val[node_idx[v]] = min(node_val[node_idx[v]], edge_val[k])
    node_val[node_val == np.inf] = 0.0

    return node_val, edge_val


def _ricci_filtration(G: nx.Graph) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ollivier-Ricci curvature filtration (Wang & Zhao, NeurIPS 2019, Appendix).

    κ^α(x,y) = 1 − W(m^α_x, m^α_y) / d(x,y)
    with α = 0.5  (lazy random walk: mass α on vertex, (1−α)/deg on neighbours),
    as specified in the paper's appendix.

    Filtration value: f_c(e) = κ(e).  Values can be negative (bridge/tree edges)
    or positive (well-connected/ring edges).

    Disconnected graphs: curvature computed per connected component; isolated
    nodes are assigned filtration value 0.

    Returns (node_values, edge_values) arrays (same node ordering as sorted(G.nodes())).
    """
    OllivierRicci = _require_ricci()

    nodes = sorted(G.nodes())
    node_idx = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    edges = list(G.edges())

    # Compute curvature per connected component (OllivierRicci requires connected input)
    edge_curvature: dict = {}
    for component in nx.connected_components(G):
        Gc = G.subgraph(component).copy()
        if len(Gc.edges()) == 0:
            continue
        orc = OllivierRicci(Gc, alpha=0.5, verbose="ERROR", proc=1)
        orc.compute_ricci_curvature()
        for u, v in orc.G.edges():
            kappa = orc.G[u][v].get('ricciCurvature', 0.0)
            edge_curvature[(u, v)] = kappa
            edge_curvature[(v, u)] = kappa

    # f_c(e) = κ(e)  — raw curvature, no negation, no shift
    edge_val = np.array([
        edge_curvature.get((u, v), edge_curvature.get((v, u), 0.0))
        for u, v in edges
    ])

    # Extend to nodes: f(v) = min_{e ∋ v} f(e)  (sublevel convention for edge-based f)
    node_val = np.full(n, np.inf)
    for k, (u, v) in enumerate(edges):
        node_val[node_idx[u]] = min(node_val[node_idx[u]], edge_val[k])
        node_val[node_idx[v]] = min(node_val[node_idx[v]], edge_val[k])
    node_val[node_val == np.inf] = 0.0  # isolated nodes

    return node_val, edge_val


# ── additional node-centrality filtrations ────────────────────────────────

def _extend_from_nodes(G: nx.Graph, node_val: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Sublevel convention: edge value = max of endpoint node values."""
    nodes = sorted(G.nodes())
    node_idx = {v: i for i, v in enumerate(nodes)}
    edges = list(G.edges())
    if edges:
        u_arr = np.array([node_idx[u] for u, _ in edges])
        v_arr = np.array([node_idx[v] for _, v in edges])
        edge_val = np.maximum(node_val[u_arr], node_val[v_arr])
    else:
        edge_val = np.array([])
    return node_val, edge_val


def _extend_from_edges(G: nx.Graph, edge_map: dict
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Given an (u,v) -> scalar dict, derive edge and node arrays.

    Node value = min over incident edges (sublevel convention for edge-based f).
    Isolated nodes default to 0.
    """
    nodes = sorted(G.nodes())
    node_idx = {v: i for i, v in enumerate(nodes)}
    edges = list(G.edges())
    if not edges:
        return np.zeros(len(nodes)), np.array([])
    edge_val = np.array([edge_map.get((u, v), edge_map.get((v, u), 0.0))
                         for u, v in edges], dtype=float)
    node_val = np.full(len(nodes), np.inf)
    for k, (u, v) in enumerate(edges):
        i, j = node_idx[u], node_idx[v]
        if edge_val[k] < node_val[i]: node_val[i] = edge_val[k]
        if edge_val[k] < node_val[j]: node_val[j] = edge_val[k]
    node_val[node_val == np.inf] = 0.0
    return node_val, edge_val


def _closeness_filtration(G: nx.Graph) -> Tuple[np.ndarray, np.ndarray]:
    """Closeness centrality (Freeman 1978).  Averaged over components."""
    nodes = sorted(G.nodes())
    cc = nx.closeness_centrality(G)
    node_val = np.array([cc[v] for v in nodes], dtype=float)
    return _extend_from_nodes(G, node_val)


def _eigenvector_filtration(G: nx.Graph) -> Tuple[np.ndarray, np.ndarray]:
    """Eigenvector centrality.  Falls back to uniform if no convergence."""
    nodes = sorted(G.nodes())
    try:
        ec = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-4)
    except (nx.PowerIterationFailedConvergence, nx.NetworkXError):
        ec = {v: 1.0 / max(1, len(nodes)) for v in nodes}
    node_val = np.array([ec.get(v, 0.0) for v in nodes], dtype=float)
    return _extend_from_nodes(G, node_val)


def _pagerank_filtration(G: nx.Graph) -> Tuple[np.ndarray, np.ndarray]:
    """PageRank stationary distribution."""
    nodes = sorted(G.nodes())
    try:
        pr = nx.pagerank(G, max_iter=500, tol=1e-4)
    except nx.PowerIterationFailedConvergence:
        pr = {v: 1.0 / max(1, len(nodes)) for v in nodes}
    node_val = np.array([pr.get(v, 0.0) for v in nodes], dtype=float)
    return _extend_from_nodes(G, node_val)


def _clustering_filtration(G: nx.Graph) -> Tuple[np.ndarray, np.ndarray]:
    """Local clustering coefficient."""
    nodes = sorted(G.nodes())
    cl = nx.clustering(G)
    node_val = np.array([cl.get(v, 0.0) for v in nodes], dtype=float)
    return _extend_from_nodes(G, node_val)


def _core_number_filtration(G: nx.Graph) -> Tuple[np.ndarray, np.ndarray]:
    """k-core decomposition: each node labelled by its core number."""
    nodes = sorted(G.nodes())
    # core_number requires a graph without self-loops
    H = G.copy()
    H.remove_edges_from(nx.selfloop_edges(H))
    try:
        cn = nx.core_number(H)
    except nx.NetworkXError:
        cn = {v: 0 for v in nodes}
    node_val = np.array([cn.get(v, 0) for v in nodes], dtype=float)
    return _extend_from_nodes(G, node_val)


def _adamic_adar_filtration(G: nx.Graph) -> Tuple[np.ndarray, np.ndarray]:
    """Adamic-Adar edge similarity:
        AA(u, v) = sum_{w in N(u) ∩ N(v)} 1 / log(deg(w))
    """
    edge_map = {}
    for u, v in G.edges():
        common = set(G.neighbors(u)) & set(G.neighbors(v))
        s = 0.0
        for w in common:
            deg_w = G.degree(w)
            if deg_w > 1:
                s += 1.0 / np.log(deg_w)
        edge_map[(u, v)] = s
    return _extend_from_edges(G, edge_map)


def _resource_allocation_filtration(G: nx.Graph) -> Tuple[np.ndarray, np.ndarray]:
    """Resource-allocation edge similarity:
        RA(u, v) = sum_{w in N(u) ∩ N(v)} 1 / deg(w)
    """
    edge_map = {}
    for u, v in G.edges():
        common = set(G.neighbors(u)) & set(G.neighbors(v))
        s = sum(1.0 / G.degree(w) for w in common if G.degree(w) > 0)
        edge_map[(u, v)] = s
    return _extend_from_edges(G, edge_map)


def _edge_betweenness_filtration(G: nx.Graph) -> Tuple[np.ndarray, np.ndarray]:
    """Edge-betweenness centrality (Girvan & Newman 2002)."""
    eb = nx.edge_betweenness_centrality(G)
    # nx returns (u,v) with u < v; normalize lookup
    edge_map = {tuple(sorted(e)): w for e, w in eb.items()}
    # derive per-edge in original edge order
    nodes = sorted(G.nodes())
    node_idx = {v: i for i, v in enumerate(nodes)}
    edges = list(G.edges())
    if not edges:
        return np.zeros(len(nodes)), np.array([])
    edge_val = np.array([edge_map.get(tuple(sorted(e)), 0.0) for e in edges],
                        dtype=float)
    node_val = np.full(len(nodes), np.inf)
    for k, (u, v) in enumerate(edges):
        i, j = node_idx[u], node_idx[v]
        if edge_val[k] < node_val[i]: node_val[i] = edge_val[k]
        if edge_val[k] < node_val[j]: node_val[j] = edge_val[k]
    node_val[node_val == np.inf] = 0.0
    return node_val, edge_val


def _forman_ricci_filtration(G: nx.Graph) -> Tuple[np.ndarray, np.ndarray]:
    """Combinatorial Forman-Ricci curvature on undirected graphs:
        F(u, v) = 4 − deg(u) − deg(v) + 3·|triangles through (u, v)|

    The triangle term is the standard simplicial correction (Sreejith et al.
    2016; Samal et al. 2018).  Node value = min over incident edges.
    """
    edge_map = {}
    for u, v in G.edges():
        deg_u, deg_v = G.degree(u), G.degree(v)
        triangles = len(set(G.neighbors(u)) & set(G.neighbors(v)))
        edge_map[(u, v)] = 4.0 - deg_u - deg_v + 3.0 * triangles
    return _extend_from_edges(G, edge_map)


def graph_to_persistence(G: nx.Graph,
                          filtration: str = "degree",
                          max_dim: int = 1,
                          max_death: float | None = None,
                          extended: bool = False,
                          union_h0: bool = False,
                          ) -> List[np.ndarray]:
    """
    Compute persistence diagrams for a graph.

    Parameters
    ----------
    G         : networkx.Graph
    filtration: "degree", "betweenness", "jaccard", "ricci", "hks_t1", "hks_t10"
    max_dim   : highest homological dimension (0 or 1)
    max_death : clip infinite bars to this value (default: max finite value)
    extended  : if True, compute extended persistence for all dimensions
                (Cohen-Steiner et al.). Wang uses this for Ricci datasets (H0+H1).
    union_h0  : if True, compute the union of sublevel and superlevel H0 ordinary
                persistence (Wang & Zhao NeurIPS 2019, Appendix B.2).
                Used for Jaccard/social datasets (H0 only).

    Returns
    -------
    diagrams : list of np.ndarray shape (n_pts, 2), one per dimension 0..max_dim
               Points on or below the diagonal are excluded.
    """
    gudhi = _require_gudhi()

    if filtration == "degree":
        node_val, edge_val = _degree_filtration(G)
    elif filtration == "nodelabel":
        node_val, edge_val = _nodelabel_filtration(G)
    elif filtration == "betweenness":
        node_val, edge_val = _betweenness_filtration(G)
    elif filtration == "closeness":
        node_val, edge_val = _closeness_filtration(G)
    elif filtration == "eigenvector":
        node_val, edge_val = _eigenvector_filtration(G)
    elif filtration == "pagerank":
        node_val, edge_val = _pagerank_filtration(G)
    elif filtration == "clustering":
        node_val, edge_val = _clustering_filtration(G)
    elif filtration == "core_number":
        node_val, edge_val = _core_number_filtration(G)
    elif filtration == "jaccard":
        node_val, edge_val = _jaccard_filtration(G)
    elif filtration == "adamic_adar":
        node_val, edge_val = _adamic_adar_filtration(G)
    elif filtration == "resource_allocation":
        node_val, edge_val = _resource_allocation_filtration(G)
    elif filtration == "edge_betweenness":
        node_val, edge_val = _edge_betweenness_filtration(G)
    elif filtration == "ricci":
        node_val, edge_val = _ricci_filtration(G)
    elif filtration == "forman_ricci":
        node_val, edge_val = _forman_ricci_filtration(G)
    elif filtration.startswith("hks_t"):
        t_str = filtration[len("hks_t"):]
        try:
            t_val = float(t_str)
        except ValueError:
            raise ValueError(f"Cannot parse HKS time from '{filtration}'. "
                             f"Use e.g. hks_t0.5, hks_t1, hks_t10, hks_t25.")
        node_val, edge_val = _hks_filtration(G, t=t_val)
    else:
        raise ValueError(
            f"Unknown filtration '{filtration}'. Choose from: "
            f"degree, betweenness, closeness, eigenvector, pagerank, "
            f"clustering, core_number, jaccard, adamic_adar, "
            f"resource_allocation, edge_betweenness, ricci, forman_ricci, "
            f"hks_t<float>, nodelabel"
        )

    nodes = sorted(G.nodes())
    node_idx = {v: i for i, v in enumerate(nodes)}
    edges = list(G.edges())

    def _build_st(nv, ev):
        s = gudhi.SimplexTree()
        for i in range(len(nodes)):
            s.insert([i], filtration=float(nv[i]))
        for k, (u, v) in enumerate(edges):
            s.insert([node_idx[u], node_idx[v]],
                     filtration=float(ev[k]) if len(ev) > 0 else 0.0)
        return s

    def _ordinary_h0(nv, ev, drop_essential=False):
        """Sublevel H0 pairs for a given (node, edge) filtration.
        If drop_essential=True, discard the essential component (infinite death)
        instead of clipping it — used for the union of sublevel+superlevel."""
        s = _build_st(nv, ev)
        s.compute_persistence(persistence_dim_max=True)
        raw = s.persistence_intervals_in_dimension(0)
        if len(raw) == 0:
            return np.zeros((0, 2))
        pts = np.array(raw)
        finite = np.isfinite(pts[:, 1])
        if drop_essential:
            pts = pts[finite]
        else:
            max_f = max(float(nv.max()), float(ev.max()) if len(ev) > 0 else 0.0)
            md = (pts[finite, 1].max() if finite.any() else max_f + 1.0) \
                 if max_death is None else max_death
            pts[~finite, 1] = md
        if len(pts) == 0:
            return np.zeros((0, 2))
        return pts[pts[:, 1] > pts[:, 0]]

    if union_h0:
        # Wang & Zhao (NeurIPS 2019) Appendix B.2, footnote 4:
        # Union of sublevel and superlevel H0 ordinary persistence.
        #
        # Superlevel H0 of f = sublevel H0 of (−f).
        # For a pair (b', d') from the −f sublevel diagram with b' < d':
        #   birth_f = −d',  death_f = −b'  (both > 0 for Jaccard ∈ [0,1])
        #   and  birth_f < death_f  so the converted pairs are above diagonal ✓
        # The essential component of −f (inf death) yields birth_f = −∞ and
        # is discarded.
        sub_pts = _ordinary_h0(node_val, edge_val, drop_essential=True)
        neg_pts = _ordinary_h0(-node_val, -edge_val, drop_essential=True)
        if len(neg_pts) > 0:
            sup_pts = np.column_stack([-neg_pts[:, 1], -neg_pts[:, 0]])
            sup_pts = sup_pts[sup_pts[:, 1] > sup_pts[:, 0]]
        else:
            sup_pts = np.zeros((0, 2))

        h0 = np.vstack([sub_pts, sup_pts]) \
             if (len(sub_pts) + len(sup_pts)) > 0 else np.zeros((0, 2))
        diagrams = [h0]

        # Higher dims: standard sublevel (Wang uses H0 only, but keep general)
        if max_dim > 0:
            st = _build_st(node_val, edge_val)
            st.compute_persistence(persistence_dim_max=True)
            max_filt = max(float(node_val.max()),
                           float(edge_val.max()) if len(edge_val) > 0 else 0.0)
            for dim in range(1, max_dim + 1):
                pairs = st.persistence_intervals_in_dimension(dim)
                if len(pairs) == 0:
                    diagrams.append(np.zeros((0, 2)))
                    continue
                pts = np.array(pairs)
                finite_mask = np.isfinite(pts[:, 1])
                md = (pts[finite_mask, 1].max() if finite_mask.any() else max_filt + 1.0) \
                     if max_death is None else max_death
                pts[~finite_mask, 1] = md
                diagrams.append(pts[pts[:, 1] > pts[:, 0]])
        return diagrams

    if extended:
        # Shift filtration to [0, ∞) so that the extended+ H0 pair (born at
        # f_min, killed at f_max in the superlevel) is never mapped onto the
        # diagonal.  Shifting by a constant does not change the persistence
        # diagram (only translates all coordinates equally).
        f_min = min(node_val.min(),
                    edge_val.min() if len(edge_val) > 0 else node_val.min())
        if f_min < 0:
            node_val = node_val - f_min
            edge_val = edge_val - f_min

    # Build gudhi SimplexTree
    st = _build_st(node_val, edge_val)

    if extended:
        # Extended persistence (Cohen-Steiner et al.): captures features that
        # survive from the sublevel to the superlevel filtration.
        #
        # gudhi.SimplexTree.extended_persistence() returns 4 groups:
        #   [0] ordinary   — born/killed in sublevel
        #   [1] relative   — born/killed in superlevel (coords negated by gudhi)
        #   [2] extended_+ — born in sublevel, killed in superlevel
        #   [3] extended_- — born in superlevel, killed in sublevel
        # After the non-negative shift above, abs() + min/max correctly maps
        # all groups into the upper half-plane without collapsing to the diagonal.
        st.extend_filtration()
        st.compute_persistence()

        groups = st.extended_persistence()
        diagrams = []
        for dim in range(max_dim + 1):
            pairs = []
            for group in groups:
                for (d, (b, d_val)) in group:
                    if d == dim and np.isfinite(b) and np.isfinite(d_val):
                        ab, ad = abs(b), abs(d_val)
                        pairs.append([min(ab, ad), max(ab, ad)])
            if len(pairs) == 0:
                diagrams.append(np.zeros((0, 2)))
            else:
                pts = np.array(pairs)
                pts = pts[pts[:, 1] > pts[:, 0] + 1e-10]
                diagrams.append(pts)
    else:
        # Standard (ordinary) persistence
        # persistence_dim_max=True ensures H_d is computed for a
        # d-dimensional complex (default only goes up to d-1).
        st.compute_persistence(persistence_dim_max=True)

        # Max filtration value in the complex — used to clip infinite bars
        max_filt = max(float(node_val.max()),
                       float(edge_val.max()) if len(edge_val) > 0 else 0.0)

        diagrams = []
        for dim in range(max_dim + 1):
            pairs = st.persistence_intervals_in_dimension(dim)
            if len(pairs) == 0:
                diagrams.append(np.zeros((0, 2)))
                continue
            pts = np.array(pairs)
            finite_mask = np.isfinite(pts[:, 1])
            if max_death is None:
                md = pts[finite_mask, 1].max() if finite_mask.any() else max_filt + 1.0
            else:
                md = max_death
            pts[~finite_mask, 1] = md
            # keep only points strictly above diagonal
            pts = pts[pts[:, 1] > pts[:, 0]]
            diagrams.append(pts)

    return diagrams


def _dims_tag(dims: List[int]) -> str:
    """'H0', 'H1', 'H0H1', etc. — used in cache filenames."""
    return "".join(f"H{d}" for d in sorted(dims))


def graphs_to_persistence_cached(graphs: list,
                                  dataset_name: str,
                                  filtration: str = "degree",
                                  dims: List[int] | None = None,
                                  max_dim: int | None = None,
                                  extended: bool = False,
                                  union_h0: bool = False,
                                  recompute: bool = False
                                  ) -> List[dict]:
    """
    Compute (or load cached) persistence diagrams for a list of graphs.

    Parameters
    ----------
    graphs       : list of networkx.Graph objects
    dataset_name : name for caching
    filtration   : "degree", "betweenness", "jaccard", "ricci", "hks_t1", "hks_t10"
    dims         : list of homological dimensions to store, e.g. [0], [1], [0, 1]
    max_dim      : legacy shorthand — equivalent to dims=list(range(max_dim+1))
    extended     : if True, compute extended persistence
    union_h0     : if True, union of sublevel + superlevel H0 (Wang Jaccard setup)
    recompute    : if True, recompute even if cached

    Returns
    -------
    list of length len(graphs); each element is a dict {dim: np.ndarray of shape (n,2)}.
    """
    if dims is None and max_dim is None:
        dims = [0]
    elif dims is None:
        dims = list(range(max_dim + 1))
    dims = sorted(set(dims))
    compute_max_dim = max(dims)

    mode_suffix = "_ext" if extended else ("_union" if union_h0 else "")
    cache_path = DIAG_DIR / f"{dataset_name}_{filtration}_{_dims_tag(dims)}{mode_suffix}.pkl"
    if cache_path.exists() and not recompute:
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    mode_str = "extended" if extended else ("union_h0" if union_h0 else "ordinary")
    print(f"Computing persistence for {dataset_name} "
          f"({filtration}, {_dims_tag(dims)}, {mode_str}) ...")
    all_diags = []
    for i, G in enumerate(graphs):
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(graphs)}")
        raw = graph_to_persistence(G, filtration=filtration,
                                   max_dim=compute_max_dim, extended=extended,
                                   union_h0=union_h0)
        all_diags.append({d: raw[d] for d in dims})

    with open(cache_path, "wb") as f:
        pickle.dump(all_diags, f)

    return all_diags


# ══════════════════════════════════════════════════════════════════════════
# Point cloud → persistence  (Experiments 5–7)
# ══════════════════════════════════════════════════════════════════════════

def pointcloud_to_persistence(pts: np.ndarray,
                               max_dim: int = 1,
                               max_death: float | None = None,
                               method: str = "alpha",
                               extended: bool = False,
                               ) -> List[np.ndarray]:
    """
    Persistence diagram for a point cloud.

    Parameters
    ----------
    pts      : (n, d) array
    max_dim  : highest homological dimension
    max_death: clip infinite bars
    method   : "alpha" (Alpha complex via gudhi, default),
               "density" (KDE sublevel on Alpha complex), or
               "rips" (Vietoris-Rips via ripser).
    extended : if True, compute extended persistence (ordinary + relative +
               extended_+ + extended_-).  Only supported for alpha/density.

    Returns
    -------
    diagrams : list of np.ndarray per dimension
    """
    if method == "alpha":
        return _alpha_persistence(pts, max_dim, max_death, extended=extended)
    elif method == "density":
        return _density_persistence(pts, max_dim, max_death, extended=extended)
    elif method == "knn":
        return _knn_persistence(pts, max_dim, max_death)
    elif method == "kde":
        return _kde_persistence(pts, max_dim, max_death, extended=extended)
    elif method == "coord_x":
        return _coord_persistence(pts, max_dim, max_death, axis=0)
    elif method == "coord_y":
        return _coord_persistence(pts, max_dim, max_death, axis=1)
    elif method == "eccentricity":
        return _eccentricity_persistence(pts, max_dim, max_death)
    elif method == "radial":
        return _radial_persistence(pts, max_dim, max_death)
    elif method.startswith("knn_k"):
        k = int(method.split("knn_k")[1])
        return _knn_persistence(pts, max_dim, max_death, k=k)
    elif method.startswith("density_k"):
        k = int(method.split("density_k")[1])
        return _density_persistence(pts, max_dim, max_death, k=k)
    elif method == "rips":
        if extended:
            raise ValueError("Extended persistence not supported for Rips")
        return _rips_persistence(pts, max_dim, max_death)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose: alpha, density, knn, "
                         f"kde, coord_x, coord_y, eccentricity, radial, "
                         f"knn_k<N>, density_k<N>, rips")


def _st_ordinary_persistence(st, max_dim: int,
                              max_death: float | None,
                              sqrt_filtration: bool = False
                              ) -> List[np.ndarray]:
    """Extract ordinary persistence diagrams from a computed SimplexTree."""
    st.compute_persistence()
    diagrams = []
    for dim in range(max_dim + 1):
        raw = st.persistence_intervals_in_dimension(dim)
        if len(raw) == 0:
            diagrams.append(np.zeros((0, 2)))
            continue
        pts_dim = np.array(raw)
        if sqrt_filtration:
            pts_dim = np.sqrt(np.maximum(pts_dim, 0.0))
        finite_mask = np.isfinite(pts_dim[:, 1])
        if max_death is None:
            md = pts_dim[finite_mask, 1].max() if finite_mask.any() else 1.0
        else:
            md = max_death
        pts_dim[~finite_mask, 1] = md
        pts_dim = pts_dim[pts_dim[:, 1] > pts_dim[:, 0]]
        diagrams.append(pts_dim)
    return diagrams


def _st_extended_persistence(st, max_dim: int,
                              sqrt_filtration: bool = False
                              ) -> List[np.ndarray]:
    """Extract extended persistence diagrams from a SimplexTree.

    Extended persistence (Cohen-Steiner et al.) collects pairs from 4 groups:
      ordinary, relative, extended_+, extended_-.
    All pairs are mapped to the upper half-plane via abs() + min/max.
    """
    # Shift filtration to [0, ∞) before extending
    filt_vals = [st.filtration(s) for s, _ in st.get_simplices()]
    f_min = min(filt_vals)
    if f_min < 0:
        for simplex, fv in st.get_simplices():
            st.assign_filtration(simplex, fv - f_min)
        st.make_filtration_non_decreasing()

    st.extend_filtration()
    st.compute_persistence()
    groups = st.extended_persistence()

    diagrams = []
    for dim in range(max_dim + 1):
        pairs = []
        for group in groups:
            for (d, (b, d_val)) in group:
                if d == dim and np.isfinite(b) and np.isfinite(d_val):
                    ab, ad = abs(b), abs(d_val)
                    if sqrt_filtration:
                        ab, ad = np.sqrt(max(ab, 0.0)), np.sqrt(max(ad, 0.0))
                    pairs.append([min(ab, ad), max(ab, ad)])
        if len(pairs) == 0:
            diagrams.append(np.zeros((0, 2)))
        else:
            pts = np.array(pairs)
            pts = pts[pts[:, 1] > pts[:, 0] + 1e-10]
            diagrams.append(pts)
    return diagrams


def _alpha_persistence(pts: np.ndarray,
                       max_dim: int = 1,
                       max_death: float | None = None,
                       extended: bool = False,
                       ) -> List[np.ndarray]:
    """Alpha complex persistence via gudhi (O(n log n) for 2D)."""
    gudhi = _require_gudhi()
    ac = gudhi.AlphaComplex(points=pts)
    st = ac.create_simplex_tree()
    # Alpha returns squared radii; sqrt is applied inside the helpers
    if extended:
        return _st_extended_persistence(st, max_dim, sqrt_filtration=True)
    return _st_ordinary_persistence(st, max_dim, max_death, sqrt_filtration=True)


def _density_persistence(pts: np.ndarray,
                         max_dim: int = 1,
                         max_death: float | None = None,
                         extended: bool = False,
                         k: int | None = None,
                         ) -> List[np.ndarray]:
    """
    DTM (distance-to-measure) sublevel persistence on the Alpha complex.

    For each point, computes the average distance to its k nearest
    neighbors (DTM_k).  Low DTM = high density.  The DTM values are
    used as a vertex filtration on the Alpha complex (higher simplices
    inherit max of their vertices).

    DTM produces much more local variation than KDE, creating many
    topological transitions (births/deaths) in the sublevel filtration.

    Parameters
    ----------
    pts       : (n, d) point cloud
    max_dim   : highest homological dimension
    max_death : clip infinite bars
    extended  : if True, compute extended persistence
    k         : number of neighbors for DTM (default: sqrt(n))
    """
    from scipy.spatial import KDTree
    gudhi = _require_gudhi()

    n = len(pts)
    if k is None:
        k = max(2, int(np.sqrt(n)))

    # DTM: average distance to k nearest neighbors (excluding self)
    tree = KDTree(pts)
    dists, _ = tree.query(pts, k=k + 1)   # (n, k+1) — includes self at dist 0
    dtm_vals = dists[:, 1:].mean(axis=1)   # (n,) — exclude self

    # Build Alpha complex, then override filtration with DTM
    ac = gudhi.AlphaComplex(points=pts)
    st = ac.create_simplex_tree()

    for simplex, _ in st.get_simplices():
        new_filt = max(dtm_vals[v] for v in simplex)
        st.assign_filtration(simplex, new_filt)
    st.make_filtration_non_decreasing()

    if extended:
        return _st_extended_persistence(st, max_dim, sqrt_filtration=False)
    return _st_ordinary_persistence(st, max_dim, max_death, sqrt_filtration=False)


def _kde_persistence(pts: np.ndarray,
                     max_dim: int = 1,
                     max_death: float | None = None,
                     extended: bool = False,
                     bandwidth: float | None = None,
                     ) -> List[np.ndarray]:
    """
    KDE sublevel persistence on the Alpha complex (ECS paper bifiltration).

    Matches Hacquard & Lebovici (JMLR 2024): function-alpha bifiltration
    with f(v) = -KDE(v) (post-composed with x -> -x so sublevel sets
    grow from high-density to low-density regions).

    Uses Gaussian KDE with Scott's bandwidth by default.

    Parameters
    ----------
    pts       : (n, d) point cloud
    max_dim   : highest homological dimension
    max_death : clip infinite bars
    extended  : if True, compute extended persistence
    bandwidth : KDE bandwidth (default: Scott's rule)
    """
    from scipy.stats import gaussian_kde
    gudhi = _require_gudhi()

    # Gaussian KDE with Scott's bandwidth
    kde = gaussian_kde(pts.T, bw_method='scott' if bandwidth is None else bandwidth)
    kde_vals = kde(pts.T)  # (n,) density at each point

    # Post-compose with x -> -x (high density = low filtration value)
    neg_kde = -kde_vals

    # Build Alpha complex, then override filtration with -KDE
    ac = gudhi.AlphaComplex(points=pts)
    st = ac.create_simplex_tree()

    for simplex, _ in st.get_simplices():
        new_filt = max(neg_kde[v] for v in simplex)
        st.assign_filtration(simplex, new_filt)
    st.make_filtration_non_decreasing()

    if extended:
        return _st_extended_persistence(st, max_dim, sqrt_filtration=False)
    return _st_ordinary_persistence(st, max_dim, max_death, sqrt_filtration=False)


def _coord_persistence(pts: np.ndarray,
                       max_dim: int = 1,
                       max_death: float | None = None,
                       axis: int = 0,
                       ) -> List[np.ndarray]:
    """
    Sublevel persistence of a coordinate function on the Alpha complex.

    f(v) = pts[v, axis].  Higher simplices inherit max of their vertices.
    Captures spatial distribution along the chosen axis.
    """
    gudhi = _require_gudhi()
    ac = gudhi.AlphaComplex(points=pts)
    st = ac.create_simplex_tree()

    coord_vals = pts[:, axis]
    for simplex, _ in st.get_simplices():
        new_filt = max(coord_vals[v] for v in simplex)
        st.assign_filtration(simplex, new_filt)
    st.make_filtration_non_decreasing()

    return _st_ordinary_persistence(st, max_dim, max_death, sqrt_filtration=False)


def _eccentricity_persistence(pts: np.ndarray,
                              max_dim: int = 1,
                              max_death: float | None = None,
                              ) -> List[np.ndarray]:
    """
    Sublevel persistence of the eccentricity function on the Alpha complex.

    f(v) = max_{u} d(v, u).  Points near the cloud center have low
    eccentricity; outliers have high eccentricity.  Captures global shape.
    """
    from scipy.spatial.distance import pdist, squareform
    gudhi = _require_gudhi()

    dist_mat = squareform(pdist(pts))
    ecc_vals = dist_mat.max(axis=1)

    ac = gudhi.AlphaComplex(points=pts)
    st = ac.create_simplex_tree()

    for simplex, _ in st.get_simplices():
        new_filt = max(ecc_vals[v] for v in simplex)
        st.assign_filtration(simplex, new_filt)
    st.make_filtration_non_decreasing()

    return _st_ordinary_persistence(st, max_dim, max_death, sqrt_filtration=False)


def _radial_persistence(pts: np.ndarray,
                        max_dim: int = 1,
                        max_death: float | None = None,
                        ) -> List[np.ndarray]:
    """
    Sublevel persistence of radial distance from the cloud centroid,
    on the Alpha complex.

    f(v) = ||pts[v] - centroid||.  Captures radial mass distribution.
    """
    gudhi = _require_gudhi()

    centroid = pts.mean(axis=0)
    r_vals = np.linalg.norm(pts - centroid, axis=1)

    ac = gudhi.AlphaComplex(points=pts)
    st = ac.create_simplex_tree()
    for simplex, _ in st.get_simplices():
        new_filt = max(r_vals[v] for v in simplex)
        st.assign_filtration(simplex, new_filt)
    st.make_filtration_non_decreasing()

    return _st_ordinary_persistence(st, max_dim, max_death, sqrt_filtration=False)


def _knn_persistence(pts: np.ndarray,
                     max_dim: int = 1,
                     max_death: float | None = None,
                     k: int | None = None,
                     ) -> List[np.ndarray]:
    """
    k-NN graph persistence: builds a flag complex from the k-nearest-neighbor
    graph, with edge filtration = Euclidean distance between neighbors.

    The k-NN graph's topology is directly shaped by local density:
    dense regions have short edges (low filtration), sparse regions have
    long edges.  This produces persistence diagrams that capture
    density-driven topological features, complementary to Alpha complex.

    Parameters
    ----------
    pts       : (n, d) point cloud
    max_dim   : highest homological dimension
    max_death : clip infinite bars
    k         : number of neighbors (default: sqrt(n))
    """
    from sklearn.neighbors import kneighbors_graph
    gudhi = _require_gudhi()

    n = len(pts)
    if k is None:
        k = max(2, int(np.sqrt(n)))

    # Build symmetric k-NN graph with distance weights
    A = kneighbors_graph(pts, n_neighbors=k, mode='distance', include_self=False)
    A = A.maximum(A.T)  # symmetrize

    st = gudhi.SimplexTree()
    for i in range(n):
        st.insert([i], filtration=0.0)

    rows, cols = A.nonzero()
    for i, j in zip(rows, cols):
        if i < j:
            st.insert([i, j], filtration=A[i, j])

    # Flag complex expansion (adds triangles where all 3 edges exist)
    st.expansion(max_dim + 1)
    st.compute_persistence()

    diagrams = []
    for dim in range(max_dim + 1):
        raw = st.persistence_intervals_in_dimension(dim)
        if len(raw) == 0:
            diagrams.append(np.zeros((0, 2)))
            continue
        pts_dim = np.array(raw)
        finite_mask = np.isfinite(pts_dim[:, 1])
        if max_death is None:
            md = pts_dim[finite_mask, 1].max() if finite_mask.any() else 1.0
        else:
            md = max_death
        pts_dim[~finite_mask, 1] = md
        pts_dim = pts_dim[pts_dim[:, 1] > pts_dim[:, 0]]
        diagrams.append(pts_dim)

    return diagrams


def pointclouds_to_persistence_cached(point_clouds: list,
                                       dataset_name: str,
                                       dims: List[int] | None = None,
                                       method: str = "alpha",
                                       extended: bool = False,
                                       recompute: bool = False
                                       ) -> List[dict]:
    """
    Compute (or load cached) persistence diagrams for a list of point clouds.

    Parameters
    ----------
    point_clouds : list of (n, d) arrays
    dataset_name : name for caching
    dims         : homological dimensions to store, e.g. [0, 1]
    method       : "alpha", "density", or "rips"
    extended     : if True, compute extended persistence
    recompute    : if True, recompute even if cached

    Returns
    -------
    list of dicts {dim: np.ndarray of shape (n, 2)}, one per point cloud.
    """
    if dims is None:
        dims = [0, 1]
    dims = sorted(set(dims))
    max_dim = max(dims)

    ext_tag = "_ext" if extended else ""
    cache_path = DIAG_DIR / f"{dataset_name}_{method}_{_dims_tag(dims)}{ext_tag}.pkl"
    if cache_path.exists() and not recompute:
        print(f"Loading cached diagrams from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    mode_str = "extended" if extended else "ordinary"
    print(f"Computing {method} persistence for {dataset_name} "
          f"({_dims_tag(dims)}, {mode_str}, {len(point_clouds)} clouds) ...")
    all_diags = []
    for i, pc in enumerate(point_clouds):
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(point_clouds)}")
        raw = pointcloud_to_persistence(pc, max_dim=max_dim, method=method,
                                        extended=extended)
        all_diags.append({d: raw[d] for d in dims})

    with open(cache_path, "wb") as f:
        pickle.dump(all_diags, f)
    print(f"Cached → {cache_path}")

    return all_diags


def _rips_persistence(pts: np.ndarray,
                      max_dim: int = 1,
                      max_death: float | None = None
                      ) -> List[np.ndarray]:
    """Vietoris-Rips persistence via ripser."""
    ripser_mod = _require_ripser()
    result = ripser_mod.ripser(pts, maxdim=max_dim)
    raw = result["dgms"]

    diagrams = []
    for dim in range(max_dim + 1):
        pts_dim = raw[dim].copy()
        finite_mask = np.isfinite(pts_dim[:, 1])
        if max_death is None:
            md = pts_dim[finite_mask, 1].max() if finite_mask.any() else 1.0
        else:
            md = max_death
        pts_dim[~finite_mask, 1] = md
        pts_dim = pts_dim[pts_dim[:, 1] > pts_dim[:, 0]]
        diagrams.append(pts_dim)

    return diagrams


