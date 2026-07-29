"""Message Passing Neural Network baseline (Gilmer et al., 2017).

Invariant edge features (squared distance) with scalar node updates and a
coordinate head. Not rotation-equivariant by construction — included as a
non-equivariant baseline in the charged N-body comparison (Table 8).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MPNN(nn.Module):
    def __init__(self, num_node_features, num_nodes, cfg):
        super().__init__()
        d = cfg.hidden_dim
        in_dim = 6 + num_node_features  # pos(3) + vel(3) + h
        self.embed = nn.Linear(in_dim, d)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * d + 1, d), nn.SiLU(),
            nn.Linear(d, d), nn.SiLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * d, d), nn.SiLU(),
            nn.Linear(d, d),
        )
        self.layers = cfg.num_layers
        self.head = nn.Linear(d, 3)

    def forward(self, batch):
        pos = batch["pos"]
        h_in = torch.cat([batch["pos"], batch["vel"], batch["h"]], dim=-1)
        h = self.embed(h_in)
        b, n, _ = h.shape
        dist2 = ((pos.unsqueeze(2) - pos.unsqueeze(1)) ** 2).sum(-1, keepdim=True)
        for _ in range(self.layers):
            hi = h.unsqueeze(2).expand(b, n, n, h.shape[-1])
            hj = h.unsqueeze(1).expand(b, n, n, h.shape[-1])
            m = self.edge_mlp(torch.cat([hi, hj, dist2], dim=-1)).sum(dim=2)
            h = h + self.node_mlp(torch.cat([h, m], dim=-1))
        return pos + self.head(h)
