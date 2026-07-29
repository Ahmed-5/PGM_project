"""MLP-family baselines from the Motion-Capture comparison (Table 2):

* EMLP  (Finzi et al., 2021): SO(3)-equivariant MLP. Compact node-wise
  reimplementation using scalar + vector channels with equivariant linear maps
  and gated nonlinearities.
* RPP   (Finzi et al., 2021): Residual Pathway Priors = an equivariant path plus
  an unconstrained (free) path summed together; the free path is meant to carry a
  small prior (implemented via a scale factor / higher effective regularization).
* PER   (Kim et al., 2023): Projection-based Equivariance Regularizer applied to a
  standard MLP. The architecture is a plain MLP; the equivariance regularization
  is provided by the REMUL/DA training objective.

These are not the original authors' code but capture each method's defining idea.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .unconstrained import MLP, _node_inputs


class _EquivLinear(nn.Module):
    """SO(3)-equivariant node-wise layer over scalar + vector channels."""

    def __init__(self, scal_in, vec_in, scal_out, vec_out):
        super().__init__()
        self.s_lin = nn.Linear(scal_in + vec_in, scal_out)  # scalars <- scalars + vec norms
        self.v_lin = nn.Linear(vec_in, vec_out, bias=False)
        self.gate = nn.Linear(scal_in + vec_in, vec_out)

    def forward(self, s, V):
        norms = (V * V).sum(-1)                         # (B,N,vec_in) invariant
        s_cat = torch.cat([s, norms], dim=-1)
        s_out = torch.nn.functional.silu(self.s_lin(s_cat))
        V_out = self.v_lin(V.transpose(-1, -2)).transpose(-1, -2)
        g = self.gate(s_cat).unsqueeze(-1)
        return s_out, V_out * torch.sigmoid(g)


class EMLP(nn.Module):
    def __init__(self, num_node_features, num_nodes, cfg):
        super().__init__()
        d = cfg.mlp_hidden
        vec = 8
        self.v_embed = nn.Linear(2, vec, bias=False)
        self.s_embed = nn.Linear(num_node_features, d)
        self.layers = nn.ModuleList()
        s_dim, v_dim = d, vec
        for _ in range(cfg.num_layers):
            self.layers.append(_EquivLinear(s_dim, v_dim, d, vec))
            s_dim, v_dim = d, vec
        self.v_read = nn.Linear(vec, 1, bias=False)

    def forward(self, batch):
        pos, vel = batch["pos"], batch["vel"]
        s = self.s_embed(batch["h"])
        V = self.v_embed(torch.stack([pos, vel], dim=-2).transpose(-1, -2)).transpose(-1, -2)
        for layer in self.layers:
            s, V = layer(s, V)
        delta = self.v_read(V.transpose(-1, -2)).transpose(-1, -2).squeeze(-2)
        return pos + delta


class RPP(nn.Module):
    """Equivariant path (EMLP) + unconstrained free path (MLP)."""

    def __init__(self, num_node_features, num_nodes, cfg):
        super().__init__()
        self.equiv = EMLP(num_node_features, num_nodes, cfg)
        self.free = MLP(num_node_features, num_nodes, cfg)
        # small prior weight on the free path (RPP favours the equivariant path)
        self.free_scale = 0.1

    def forward(self, batch):
        equiv_out = self.equiv(batch)                    # includes +pos
        free_delta = self.free(batch) - batch["pos"]     # free path residual only
        return equiv_out + self.free_scale * free_delta


class PER(nn.Module):
    """Standard MLP; equivariance enforced by the training-time regularizer."""

    def __init__(self, num_node_features, num_nodes, cfg):
        super().__init__()
        self.net = MLP(num_node_features, num_nodes, cfg)

    def forward(self, batch):
        return self.net(batch)
