"""E(n)-Equivariant Graph Neural Network (Satorras et al., 2021).

Faithful dense implementation with velocity, matching the N-body / MD17 variant
used as the paper's equivariant baseline. Coordinate updates only ever add
rotation-covariant vectors ((x_i - x_j) and velocity), so the model is exactly
E(3)-equivariant: rotating/translating the input rotates/translates the output.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class EGNNLayer(nn.Module):
    def __init__(self, hidden_dim, use_velocity=True):
        super().__init__()
        self.use_velocity = use_velocity
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        if use_velocity:
            self.vel_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))

    def forward(self, h, x, v):
        b, n, _ = h.shape
        rel = x.unsqueeze(2) - x.unsqueeze(1)          # (B,N,N,3) x_i - x_j
        dist2 = (rel ** 2).sum(-1, keepdim=True)        # (B,N,N,1)
        hi = h.unsqueeze(2).expand(b, n, n, h.shape[-1])
        hj = h.unsqueeze(1).expand(b, n, n, h.shape[-1])
        m = self.edge_mlp(torch.cat([hi, hj, dist2], dim=-1))  # (B,N,N,d)
        # coordinate update (normalized aggregation)
        coord_w = self.coord_mlp(m)                     # (B,N,N,1)
        agg = (rel * coord_w).sum(dim=2) / (n - 1)      # (B,N,3)
        if self.use_velocity:
            v = self.vel_mlp(h) * v + agg
            x = x + v
        else:
            x = x + agg
        # node update
        m_sum = m.sum(dim=2)                            # (B,N,d)
        h = h + self.node_mlp(torch.cat([h, m_sum], dim=-1))
        return h, x, v


class EGNN(nn.Module):
    def __init__(self, num_node_features, num_nodes, cfg, use_velocity=True):
        super().__init__()
        d = cfg.hidden_dim
        self.embed = nn.Linear(num_node_features, d)
        self.use_velocity = use_velocity
        self.layers = nn.ModuleList([EGNNLayer(d, use_velocity) for _ in range(cfg.num_layers)])

    def forward(self, batch):
        h = self.embed(batch["h"])
        x = batch["pos"]
        v = batch["vel"] if self.use_velocity else torch.zeros_like(x)
        for layer in self.layers:
            h, x, v = layer(h, x, v)
        return x
