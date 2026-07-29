"""Unconstrained models trained with REMUL: Transformer, MLP, and GNN.

These impose no equivariance in their architecture (that is the whole point of
REMUL). Each takes a dense batch dict and predicts per-node target positions
``(B, N, 3)``. Positions/velocities are consumed as ordinary input features.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _in_features(num_node_features: int) -> int:
    # [pos(3), vel(3), scalar features]
    return 6 + num_node_features


def _node_inputs(batch) -> torch.Tensor:
    return torch.cat([batch["pos"], batch["vel"], batch["h"]], dim=-1)


class Transformer(nn.Module):
    """Standard Transformer encoder over nodes-as-tokens (unconstrained)."""

    def __init__(self, num_node_features, num_nodes, cfg):
        super().__init__()
        d = cfg.channels
        self.embed = nn.Linear(_in_features(num_node_features), d)
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.num_heads,
            dim_feedforward=d * 2,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.head = nn.Linear(d, 3)

    def forward(self, batch):
        z = self.embed(_node_inputs(batch))
        z = self.encoder(z)
        return batch["pos"] + self.head(z)


class MLP(nn.Module):
    """Fully-connected MLP over the flattened node set (fixed N)."""

    def __init__(self, num_node_features, num_nodes, cfg):
        super().__init__()
        self.num_nodes = num_nodes
        in_dim = _in_features(num_node_features) * num_nodes
        h = cfg.mlp_hidden
        layers, prev = [], in_dim
        for _ in range(cfg.num_layers):
            layers += [nn.Linear(prev, h), nn.SiLU()]
            prev = h
        layers += [nn.Linear(prev, num_nodes * 3)]
        self.net = nn.Sequential(*layers)

    def forward(self, batch):
        b = batch["pos"].shape[0]
        x = _node_inputs(batch).reshape(b, -1)
        delta = self.net(x).reshape(b, self.num_nodes, 3)
        return batch["pos"] + delta


class GNN(nn.Module):
    """Non-equivariant message-passing GNN (EGNN's counterpart, Satorras 2021).

    Coordinates are fed as ordinary node features and the coordinate output is an
    unconstrained linear head, so the model is not rotation-equivariant.
    """

    def __init__(self, num_node_features, num_nodes, cfg):
        super().__init__()
        d = cfg.hidden_dim
        self.num_layers = cfg.num_layers
        self.embed = nn.Linear(_in_features(num_node_features), d)
        self.msg = nn.ModuleList()
        self.upd = nn.ModuleList()
        for _ in range(cfg.num_layers):
            self.msg.append(nn.Sequential(nn.Linear(2 * d + 1, d), nn.SiLU(), nn.Linear(d, d), nn.SiLU()))
            self.upd.append(nn.Sequential(nn.Linear(2 * d, d), nn.SiLU(), nn.Linear(d, d)))
        self.head = nn.Linear(d, 3)

    def forward(self, batch):
        pos = batch["pos"]
        h = self.embed(_node_inputs(batch))  # (B, N, d)
        b, n, _ = h.shape
        # squared distance as an extra (rotation-invariant, but coords also fed in)
        dist2 = ((pos.unsqueeze(2) - pos.unsqueeze(1)) ** 2).sum(-1, keepdim=True)  # (B,N,N,1)
        for msg, upd in zip(self.msg, self.upd):
            hi = h.unsqueeze(2).expand(b, n, n, h.shape[-1])
            hj = h.unsqueeze(1).expand(b, n, n, h.shape[-1])
            m = msg(torch.cat([hi, hj, dist2], dim=-1)).sum(dim=2)  # (B,N,d)
            h = h + upd(torch.cat([h, m], dim=-1))
        return pos + self.head(h)
