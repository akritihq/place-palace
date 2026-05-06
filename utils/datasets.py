"""
utils/datasets.py
-----------------
Dataset loaders.

TU Dortmund graph datasets (MUTAG, PROTEINS, NCI1, etc.):
  Auto-downloaded from https://www.chrsmrrs.com/graphkerneldatasets on first use.
  Raw files cached under data/raw/<name>/.
  Returns (list[nx.Graph], np.ndarray[int]).

Orbit5k (synthetic point clouds):
  Dynamical system orbits on [0,1]^2 with 5 parameter values.
  Returns (list[np.ndarray], np.ndarray[int]).
"""

import io
import os
import pickle
import urllib.request
import zipfile
import numpy as np
import networkx as nx
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[1]
RAW_DIR   = ROOT / "data" / "raw"
PROC_DIR  = ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
# TU Dortmund loader
# ══════════════════════════════════════════════════════════════════════════

_TU_BASE_URL = "https://www.chrsmrrs.com/graphkerneldatasets"


def download_tu_dataset(name: str) -> None:
    """Download and unzip a TU Dortmund dataset into data/raw/<name>/."""
    url = f"{_TU_BASE_URL}/{name}.zip"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {name} from {url} ...")
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(RAW_DIR)
    print(f"Extracted to {RAW_DIR / name}/")


def load_tu_dataset(name: str):
    """
    Load a TU Dortmund dataset from data/raw/<name>/.

    Expected files (standard TU format):
      <name>_A.txt          — edge list  (1-indexed, comma-separated)
      <name>_graph_indicator.txt  — graph id per node (1-indexed)
      <name>_graph_labels.txt     — one label per graph
      <name>_node_labels.txt      — (optional) node labels

    Returns
    -------
    graphs : list[nx.Graph]
    labels : np.ndarray[int]   (0-indexed class labels)
    """
    cache = PROC_DIR / f"{name}.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)

    folder = RAW_DIR / name
    if not folder.exists():
        download_tu_dataset(name)

    def _read(fname):
        with open(folder / fname) as fh:
            return [line.strip() for line in fh if line.strip()]

    # graph indicator: node i (1-indexed) belongs to graph graph_id[i-1]
    graph_indicator = np.array([int(x) for x in _read(f"{name}_graph_indicator.txt")])
    n_graphs = graph_indicator.max()

    # raw labels (may be -1/1 or 1..k)
    raw_labels = np.array([int(x) for x in _read(f"{name}_graph_labels.txt")])
    unique_labels = np.unique(raw_labels)
    label_map = {v: i for i, v in enumerate(unique_labels)}
    labels = np.array([label_map[v] for v in raw_labels])

    # optional node labels
    node_label_file = folder / f"{name}_node_labels.txt"
    if node_label_file.exists():
        node_labels = [int(x) for x in _read(f"{name}_node_labels.txt")]
    else:
        node_labels = None

    # build one nx.Graph per graph-id
    graphs = [nx.Graph() for _ in range(n_graphs)]
    for node_idx, gid in enumerate(graph_indicator):
        g = graphs[gid - 1]
        attr = {"label": node_labels[node_idx]} if node_labels else {}
        g.add_node(node_idx, **attr)

    edges_raw = _read(f"{name}_A.txt")
    for line in edges_raw:
        u_str, v_str = line.split(",")
        u, v = int(u_str.strip()) - 1, int(v_str.strip()) - 1   # 0-indexed
        gid = graph_indicator[u] - 1
        graphs[gid].add_edge(u, v)

    # save cache
    with open(cache, "wb") as f:
        pickle.dump((graphs, labels), f)

    return graphs, labels


# ══════════════════════════════════════════════════════════════════════════
# Orbit5k  (synthetic point clouds)
# ══════════════════════════════════════════════════════════════════════════

# Dynamical system parameters for 5 classes (Kim et al., PLLay 2020).
_ORBIT_R = [2.5, 3.5, 4.0, 4.1, 4.3]


def _orbit_point_cloud(r: float, n_pts: int,
                       rng: np.random.Generator) -> np.ndarray:
    """
    Generate one orbit of the iterated map on [0,1]^2:

        x_{n+1} = (x_n + r * y_n * (1 - y_n)) mod 1
        y_{n+1} = (y_n + r * x_{n+1} * (1 - x_{n+1})) mod 1

    starting from a uniform random initial point.

    Returns (n_pts, 2) array.
    """
    x, y = rng.random(), rng.random()
    pts = np.empty((n_pts, 2))
    for i in range(n_pts):
        x = (x + r * y * (1 - y)) % 1.0
        y = (y + r * x * (1 - x)) % 1.0
        pts[i] = [x, y]
    return pts


def load_orbit5k(n_per_class: int = 1000, n_pts: int = 1000,
                 noise: float = 0.0, seed: int = 0) -> tuple:
    """
    Generate Orbit5k: 5 classes × n_per_class point clouds.

    Each point cloud is an orbit of a 2D dynamical system with
    parameter r ∈ {2.5, 3.5, 4.0, 4.1, 4.3}.  Different r values
    produce orbits with qualitatively different topological structure.

    Parameters
    ----------
    n_per_class : number of orbits per class (default 1000 → 5000 total)
    n_pts       : points per orbit (default 1000)
    noise       : probability of replacing each point with uniform noise
    seed        : random seed (default 0 for reproducibility)

    Returns
    -------
    point_clouds : list of (n_pts, 2) arrays
    labels       : np.ndarray[int] (0-indexed)
    """
    tag = f"Orbit5k_{n_per_class}_n{n_pts}_noise{noise:.2f}_s{seed}"
    cache = PROC_DIR / f"{tag}.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)

    rng = np.random.default_rng(seed)
    point_clouds, labels = [], []
    for cls, r in enumerate(_ORBIT_R):
        for _ in range(n_per_class):
            pc = _orbit_point_cloud(r, n_pts, rng)
            if noise > 0:
                mask = rng.random(n_pts) < noise
                pc[mask] = rng.random((mask.sum(), 2))
            point_clouds.append(pc)
            labels.append(cls)

    labels = np.array(labels)
    with open(cache, "wb") as f:
        pickle.dump((point_clouds, labels), f)

    return point_clouds, labels


# ══════════════════════════════════════════════════════════════════════════
# SynthGraphs5  (synthetic graph families for proof-of-concept)
# ══════════════════════════════════════════════════════════════════════════

def _make_synth_graph(rng: np.random.Generator, graph_class: int,
                      n_nodes: int = 20) -> nx.Graph:
    """
    Five random graph families with distinct topological structure:
      0 : Erdos-Renyi  G(n, 0.10)  — sparse, many components
      1 : Erdos-Renyi  G(n, 0.30)  — moderate density
      2 : Erdos-Renyi  G(n, 0.50)  — dense, few loops
      3 : Barabasi-Albert  m=2     — scale-free, hub structure
      4 : Barabasi-Albert  m=4     — denser scale-free
    """
    s = int(rng.integers(1e9))
    if graph_class == 0:
        return nx.erdos_renyi_graph(n_nodes, 0.10, seed=s)
    elif graph_class == 1:
        return nx.erdos_renyi_graph(n_nodes, 0.30, seed=s)
    elif graph_class == 2:
        return nx.erdos_renyi_graph(n_nodes, 0.50, seed=s)
    elif graph_class == 3:
        return nx.barabasi_albert_graph(n_nodes, 2, seed=s)
    elif graph_class == 4:
        return nx.barabasi_albert_graph(n_nodes, 4, seed=s)
    else:
        raise ValueError(f"Unknown graph class {graph_class}")


def load_synthgraphs5(n_per_class: int = 1000, n_nodes: int = 20,
                      seed: int = 42) -> tuple:
    """
    Generate SynthGraphs5: 5 classes × n_per_class graphs.

    Returns
    -------
    graphs : list[nx.Graph]
    labels : np.ndarray[int]
    """
    cache = PROC_DIR / f"SynthGraphs5_{n_per_class}_n{n_nodes}.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)

    rng = np.random.default_rng(seed)
    graphs, labels = [], []
    for cls in range(5):
        for _ in range(n_per_class):
            graphs.append(_make_synth_graph(rng, cls, n_nodes))
            labels.append(cls)

    labels = np.array(labels)
    with open(cache, "wb") as f:
        pickle.dump((graphs, labels), f)

    return graphs, labels


# ══════════════════════════════════════════════════════════════════════════
# Synthetic two-sample families  (Experiments 5–7)
# ══════════════════════════════════════════════════════════════════════════

def noisy_circle_pointcloud(n: int, radius: float, sigma: float,
                             rng: np.random.Generator) -> np.ndarray:
    """
    n points uniformly on a circle of given radius, plus Gaussian noise sigma.
    Returns array of shape (n, 2).
    """
    angles = rng.uniform(0, 2 * np.pi, n)
    pts = radius * np.column_stack([np.cos(angles), np.sin(angles)])
    pts += rng.normal(0, sigma, pts.shape)
    return pts


def erdos_renyi_graph(n: int, p: float,
                      rng: np.random.Generator) -> nx.Graph:
    return nx.erdos_renyi_graph(n, p, seed=int(rng.integers(1e9)))


# ══════════════════════════════════════════════════════════════════════════
# Convenience dispatcher
# ══════════════════════════════════════════════════════════════════════════

DATASET_LOADERS = {
    # Chemical (Ricci filtration, H0+H1, extended persistence)
    "MUTAG":      lambda: load_tu_dataset("MUTAG"),
    "NCI1":       lambda: load_tu_dataset("NCI1"),
    "NCI109":     lambda: load_tu_dataset("NCI109"),
    "PTC":        lambda: load_tu_dataset("PTC_MR"),
    # Protein / social (Jaccard filtration, H0, ordinary persistence)
    "PROTEINS":   lambda: load_tu_dataset("PROTEINS"),
    "DD":         lambda: load_tu_dataset("DD"),
    "IMDB-B":     lambda: load_tu_dataset("IMDB-BINARY"),
    "IMDB-M":     lambda: load_tu_dataset("IMDB-MULTI"),
    "REDDIT-5K":  lambda: load_tu_dataset("REDDIT-MULTI-5K"),
    "REDDIT-12K": lambda: load_tu_dataset("REDDIT-MULTI-12K"),
    # Other TU datasets
    "ENZYMES":    lambda: load_tu_dataset("ENZYMES"),
    # Synthetic
    "SynthGraphs5": lambda: load_synthgraphs5(),
    "COX2":       lambda: load_tu_dataset("COX2"),
    "DHFR":       lambda: load_tu_dataset("DHFR"),
    "Orbit5k":      lambda: load_orbit5k(n_per_class=1000, n_pts=1000, seed=0),
}

def load_dataset(name: str):
    if name not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset '{name}'. "
                         f"Choose from {list(DATASET_LOADERS)}")
    return DATASET_LOADERS[name]()
