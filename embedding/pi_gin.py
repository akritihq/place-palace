"""
PI-GIN: a roll-your-own persistence-image-augmented GIN baseline.

Used by experiments/exp_tda_gnn_baseline_pii.py to provide a TDA-GNN
baseline for Paper II's tab:graph_comparison.  This is a small reference
implementation following Hofer et al. (2017) -- not TOGL (Horn 2022),
which has a more elaborate persistence-aware readout.  The architecture is

    g  -GIN-> h_g                                       \
                                                         concat -> MLP -> y
    diagrams  -persistence-image-> p_g                  /

where h_g is the sum-pool of node embeddings from a 3-layer GIN and
p_g is the concatenation of per-filtration persistence images.

This module deliberately stays minimal (~150 lines).  No tunable
classifier head, fixed PI grid, no margin/weight-function tuning.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool


def diagrams_to_pi(diagrams: list, grid: int = 16,
                   sigma: float = 0.05, max_p: float = 1.0,
                   weight: str = 'linear') -> np.ndarray:
    """
    Persistence-image batch encoder.

    Parameters
    ----------
    diagrams : list of (m_i, 2) numpy arrays of (birth, death) pairs.
    grid     : PI is a grid x grid pixel image.
    sigma    : Gaussian smoothing standard deviation in birth-persistence
               coordinates (after rescaling to [0, 1]).
    max_p    : upper coordinate of birth-persistence box (rescale factor).
    weight   : 'linear' = persistence-weighted (Adams 2017 standard);
               'unit' = unweighted.

    Returns
    -------
    Array of shape (len(diagrams), grid * grid) of flattened PI features.
    """
    grid_b = np.linspace(0, max_p, grid + 1)[:-1] + max_p / (2 * grid)
    grid_p = np.linspace(0, max_p, grid + 1)[:-1] + max_p / (2 * grid)
    Bg, Pg = np.meshgrid(grid_b, grid_p, indexing='ij')
    out = np.empty((len(diagrams), grid * grid), dtype=np.float32)
    for i, dgm in enumerate(diagrams):
        if len(dgm) == 0:
            out[i] = 0.0
            continue
        b = dgm[:, 0].astype(np.float32) / max_p
        p = (dgm[:, 1] - dgm[:, 0]).astype(np.float32) / max_p
        # filter non-finite (essential classes) by clipping persistence
        finite = np.isfinite(p) & np.isfinite(b)
        b, p = b[finite], p[finite]
        if len(b) == 0:
            out[i] = 0.0
            continue
        w = p if weight == 'linear' else np.ones_like(p)
        # img(x, y) = sum_i w_i * exp(-((x-b_i)^2 + (y-p_i)^2) / (2 sigma^2))
        img = np.zeros((grid, grid), dtype=np.float32)
        for bi, pi, wi in zip(b, p, w):
            img += wi * np.exp(
                -((Bg - bi) ** 2 + (Pg - pi) ** 2) / (2 * sigma ** 2)
            )
        out[i] = img.flatten()
    return out


class PIGIN(nn.Module):
    """
    Persistence-image-augmented GIN.

    Forward pass takes a PyG batch and a per-graph PI feature tensor
    (precomputed offline; same row-order as `batch.batch.unique()`).
    """

    def __init__(self, in_dim: int, hidden_dim: int, n_classes: int,
                 pi_dim: int, n_layers: int = 3, dropout: float = 0.5):
        super().__init__()
        self.gins = nn.ModuleList()
        d = in_dim
        for _ in range(n_layers):
            self.gins.append(GINConv(nn.Sequential(
                nn.Linear(d, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )))
            d = hidden_dim
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + pi_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x, edge_index, batch, pi):
        h = x
        for layer in self.gins:
            h = F.relu(layer(h, edge_index))
        h_g = global_add_pool(h, batch)
        z = torch.cat([h_g, pi], dim=-1)
        return self.head(z)
