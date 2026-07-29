"""In-memory dynamics dataset and dense batching.

All REMUL tasks reduce to: given per-node initial positions, velocities and
scalar features, predict per-node target positions. Graphs within a dataset have
a fixed number of nodes (N-body: fixed #bodies; MD17: fixed per molecule; MoCap:
fixed #joints), so we use a simple dense ``(B, N, *)`` batching, which keeps the
Transformer / MLP / GNN / EGNN implementations uniform.
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader


class DynamicsDataset(Dataset):
    """Holds stacked tensors for a split.

    Args:
        pos:    (S, N, 3) initial positions (already center-of-mass subtracted).
        vel:    (S, N, 3) initial velocities.
        h:      (S, N, F) invariant scalar node features.
        target: (S, N, 3) target positions (same frame as ``pos``).
        edge_index: optional (2, E) LongTensor shared by all graphs (fixed graph),
            or ``None`` for fully connected.
    """

    def __init__(self, pos, vel, h, target, edge_index=None):
        self.pos = pos.float()
        self.vel = vel.float()
        self.h = h.float()
        self.target = target.float()
        self.edge_index = edge_index

    def __len__(self):
        return self.pos.shape[0]

    def __getitem__(self, idx):
        return {
            "pos": self.pos[idx],
            "vel": self.vel[idx],
            "h": self.h[idx],
            "target": self.target[idx],
        }

    @property
    def num_nodes(self):
        return self.pos.shape[1]

    @property
    def num_node_features(self):
        return self.h.shape[-1]


def collate_dynamics(items):
    return {
        "pos": torch.stack([it["pos"] for it in items]),
        "vel": torch.stack([it["vel"] for it in items]),
        "h": torch.stack([it["h"] for it in items]),
        "target": torch.stack([it["target"] for it in items]),
    }


def make_loader(dataset: DynamicsDataset, batch_size: int, shuffle: bool = False,
                num_workers: int = 0) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_dynamics,
    )


def fully_connected_edge_index(n: int, device="cpu") -> torch.Tensor:
    """(2, N*(N-1)) edge index without self-loops for a single graph."""
    idx = torch.arange(n, device=device)
    row = idx.repeat_interleave(n)
    col = idx.repeat(n)
    mask = row != col
    return torch.stack([row[mask], col[mask]], dim=0)
