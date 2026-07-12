"""
exp_tda_gnn_baseline_pii.py
---------------------------

TDA-GNN baseline integration scaffolding for Paper II
(JMLR-blocker #10).  When activated and a TDA-GNN package is
installed, this script runs TOGL or persistence-augmented GIN on
the chemical pool under PALACE's protocol (5 seeds x 10 folds,
matched hyperparameter budget) and writes results to
results/paper_II/tables/tda_gnn_baseline.csv for direct
incorporation into Table~\\ref{tab:graph_comparison}.

Status: SCAFFOLDING (not yet wired to a TDA-GNN backend).
The script is intentionally a stub: it (a) defines the
protocol-matched runner interface, (b) loads the same datasets
PALACE uses, (c) checks for a TDA-GNN backend at import time and
exits with a clear message if none is found, and (d) when a
backend is present, runs the baseline and writes the CSV.

Backend options (any one suffices):
  - TOGL via the official codebase
    (https://github.com/BorgwardtLab/TOGL).  Install from source.
  - torch_topological  (`pip install torch_topological`)
    provides a TOGL-compatible PersistenceLayer.
  - Roll-your-own PyG GIN with a persistence-image readout
    (PyG is already installed).  Cheaper to integrate but is a
    re-derivation of \\citet{Hofer2020}'s persistence-image GIN
    rather than TOGL itself.

Cluster usage (Pegasus@GW), once a backend is installed:
  python experiments/exp_tda_gnn_baseline_pii.py
  python experiments/exp_tda_gnn_baseline_pii.py MUTAG DHFR
  python experiments/exp_tda_gnn_baseline_pii.py --backend togl

The full sweep is 5 datasets x 5 seeds x 10 folds = 250 cells.
Each cell trains one GNN end-to-end on K~200 hidden units; runs
on the order of ~5-15 minutes per cell on a single GPU,
~20-60 hours total on one GPU; cluster batch parallelizes over
(dataset, seed) pairs.
"""
from __future__ import annotations
import argparse
import csv
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

DATASETS = ["MUTAG", "PTC", "COX2", "DHFR", "NCI1"]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
N_FOLDS = 10
BACKENDS = ["togl", "torch_topological", "pyg_pi_gin"]

# Map PII dataset name -> (PyG TU name, PALACE-headline filtration list)
DATASET_HEADLINES = {
    "MUTAG":  ("MUTAG",  ["degree", "hks_t10"]),
    "PTC":    ("PTC_MR", ["degree", "betweenness"]),
    "COX2":   ("COX2",   ["jaccard", "hks_t10"]),
    "DHFR":   ("DHFR",   ["hks_t10"]),
    "NCI1":   ("NCI1",   ["degree", "hks_t10"]),
}

PI_GRID = 16        # 16 x 16 = 256 features per filtration
PI_SIGMA = 0.05
PI_MAX_P = 1.0
HIDDEN_DIM = 128
N_GIN_LAYERS = 3
N_EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4

OUT_CSV = Path('results/paper_II/tables/tda_gnn_baseline.csv')


def detect_backend(preferred: str | None = None) -> str | None:
    """Return the first available backend among `preferred`-then-others, or None."""
    candidates = [preferred] if preferred else []
    candidates += [b for b in BACKENDS if b not in candidates]
    for b in candidates:
        if b is None:
            continue
        if b == "togl":
            try:
                importlib.import_module("togl")
                return "togl"
            except ImportError:
                continue
        if b == "torch_topological":
            try:
                importlib.import_module("torch_topological")
                return "torch_topological"
            except ImportError:
                continue
        if b == "pyg_pi_gin":
            try:
                importlib.import_module("torch_geometric")
                return "pyg_pi_gin"  # roll-your-own; always available if PyG is
            except ImportError:
                continue
    return None


def _normalize_diagram_box(diagrams):
    """Rescale all diagrams jointly so that max death = PI_MAX_P.

    This keeps PIs on a common reference grid across the dataset.
    """
    max_d = 0.0
    for d in diagrams:
        if len(d) > 0:
            max_d = max(max_d, float(d[:, 1].max()))
    if max_d == 0:
        max_d = 1.0
    scale = PI_MAX_P / max_d
    return [d * scale if len(d) > 0 else d for d in diagrams], scale


def _load_palace_diagrams(name: str, filtrations: list) -> list:
    from exp_noninterference_audit import load_combined_diagrams, filter_topN, N_MAX
    diagrams = load_combined_diagrams(name, filtrations)
    return filter_topN(diagrams, N_MAX)


def _ensure_node_features(data, max_degree: int = 32):
    """Provide one-hot degree features when x is None (e.g. NCI1)."""
    import torch
    from torch_geometric.utils import degree
    if data.x is not None:
        return data
    deg = degree(data.edge_index[0], data.num_nodes, dtype=torch.long).clamp_(max=max_degree)
    data.x = torch.nn.functional.one_hot(deg, num_classes=max_degree + 1).float()
    return data


def run_one_cell(dataset: str, seed: int, backend: str) -> list:
    """Run one 10-fold CV cell with the chosen backend (pyg_pi_gin only)."""
    if backend != "pyg_pi_gin":
        raise NotImplementedError(
            f"Backend '{backend}' wiring is left to the revision run."
        )

    import numpy as np
    import torch
    from torch_geometric.datasets import TUDataset
    from torch_geometric.loader import DataLoader
    from sklearn.model_selection import StratifiedKFold

    from embedding.pi_gin import diagrams_to_pi, PIGIN

    pyg_name, filt_list = DATASET_HEADLINES[dataset]

    # Load PyG dataset (atomic numbers / edge labels are already one-hot for
    # the chemicals; NCI1 has no node labels, so we hot-encode degree).
    pyg_ds = TUDataset(root='data/pyg_tu', name=pyg_name, use_node_attr=False)
    pyg_ds = [_ensure_node_features(g.clone()) for g in pyg_ds]
    in_dim = pyg_ds[0].x.shape[1]

    # Load PALACE-headline diagrams + compute PI features
    palace_dgms = _load_palace_diagrams(dataset, filt_list)
    palace_dgms, _ = _normalize_diagram_box(palace_dgms)

    # Crop to the joint cardinality (PyG and PALACE caches sometimes differ
    # by 1-2 graphs from preprocessing edge cases)
    n_g = min(len(pyg_ds), len(palace_dgms))
    pyg_ds = pyg_ds[:n_g]
    palace_dgms = palace_dgms[:n_g]
    pi_features = diagrams_to_pi(palace_dgms, grid=PI_GRID, sigma=PI_SIGMA,
                                 max_p=PI_MAX_P, weight='linear')
    pi_features = torch.from_numpy(pi_features)
    pi_dim = pi_features.shape[1]

    # Labels for stratification
    labels = np.asarray([int(g.y.item()) for g in pyg_ds])
    n_classes = int(labels.max()) + 1

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    rows = []
    for fold, (tr, te) in enumerate(skf.split(np.zeros(n_g), labels)):
        torch.manual_seed(seed * 100 + fold)

        tr_loader = DataLoader([pyg_ds[i] for i in tr], batch_size=BATCH_SIZE,
                               shuffle=True)
        te_loader = DataLoader([pyg_ds[i] for i in te], batch_size=BATCH_SIZE,
                               shuffle=False)

        # PI features in the same order as the dataset; we slice per batch
        pi_tr_index = {idx: pi_features[idx] for idx in tr}
        pi_te_index = {idx: pi_features[idx] for idx in te}

        model = PIGIN(in_dim=in_dim, hidden_dim=HIDDEN_DIM,
                      n_classes=n_classes, pi_dim=pi_dim,
                      n_layers=N_GIN_LAYERS, dropout=0.5)
        opt = torch.optim.Adam(model.parameters(),
                               lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        loss_fn = torch.nn.CrossEntropyLoss()

        # Pre-compute the per-batch PI tensors using torch_geometric's batch
        # ordering; rebuilding the loader with PI as a tensor in each Data
        # object is cleaner.  We reattach pi as data.pi.
        for idx, g in enumerate(pyg_ds):
            g.pi = pi_features[idx].unsqueeze(0)  # (1, pi_dim)

        tr_loader = DataLoader([pyg_ds[i] for i in tr], batch_size=BATCH_SIZE,
                               shuffle=True)
        te_loader = DataLoader([pyg_ds[i] for i in te], batch_size=BATCH_SIZE,
                               shuffle=False)

        for epoch in range(N_EPOCHS):
            model.train()
            for batch in tr_loader:
                opt.zero_grad()
                out = model(batch.x, batch.edge_index, batch.batch, batch.pi)
                loss = loss_fn(out, batch.y)
                loss.backward()
                opt.step()

        # eval
        model.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for batch in te_loader:
                out = model(batch.x, batch.edge_index, batch.batch, batch.pi)
                correct += int((out.argmax(-1) == batch.y).sum())
                total += int(batch.y.shape[0])
        acc = correct / total if total else 0.0
        rows.append({
            'dataset': dataset, 'backend': backend, 'seed': seed,
            'fold': fold, 'acc': acc,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('datasets', nargs='*',
                        help='Subset of datasets (default: all five)')
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    parser.add_argument('--backend', choices=BACKENDS, default=None,
                        help='Force a specific backend (auto-detected by default)')
    parser.add_argument('--out', type=Path, default=OUT_CSV)
    args = parser.parse_args()

    backend = detect_backend(args.backend)
    if backend is None:
        sys.exit(
            "No TDA-GNN backend detected.  Install one of:\n"
            "  - TOGL (https://github.com/BorgwardtLab/TOGL)\n"
            "  - torch_topological  (pip install torch_topological)\n"
            "Or run with --backend pyg_pi_gin to use the\n"
            "persistence-image-GIN roll-your-own (requires hooking the\n"
            "model definition; see module docstring).\n"
        )
    print(f"Using TDA-GNN backend: {backend}", flush=True)

    selected = [d for d in DATASETS if not args.datasets or d in args.datasets]
    if not selected:
        sys.exit(f"No datasets matched {args.datasets}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['dataset', 'backend', 'seed', 'fold', 'acc']
    new_file = not args.out.exists()
    with open(args.out, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
        for dataset in selected:
            for seed in args.seeds:
                rows = run_one_cell(dataset, seed, backend)
                writer.writerows(rows)
                f.flush()
    print(f"\nWrote {args.out}")


if __name__ == '__main__':
    main()
