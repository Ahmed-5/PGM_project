"""Geometric Algebra Transformer (GATr) baseline — compact reimplementation.

Not the authors' code. A faithful GATr (Brehmer et al., 2023) operates on 16-dim
projective-geometric-algebra multivectors with equilinear layers, the geometric
product, and equivariant attention. Here we implement a compact SO(3)-equivariant
transformer over geometric-algebra-style channels: each node carries scalar
channels and vector channels, mixed by equivariant linear maps and a
geometric-product-style bilinear (dot products -> scalars, cross products ->
vectors), with attention weights computed from invariants. The vector output
provides the coordinate update, so the model is SO(3)-equivariant by construction.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GeometricBilinear(nn.Module):
    """Geometric-product-style interaction between vector channels.

    Produces new scalars from pairwise dot products and new vectors from
    cross products, both SO(3)-equivariant.
    """

    def __init__(self, vec_channels, scal_channels):
        super().__init__()
        self.dot_proj = nn.Linear(vec_channels, scal_channels)
        self.cross_a = nn.Linear(vec_channels, vec_channels, bias=False)
        self.cross_b = nn.Linear(vec_channels, vec_channels, bias=False)

    def forward(self, s, V):  # s:(B,N,Cs)  V:(B,N,Cv,3)
        dots = (V * V).sum(-1)                       # (B,N,Cv) invariant norms^2
        s_new = self.dot_proj(dots)
        a = self.cross_a(V.transpose(-1, -2)).transpose(-1, -2)
        b = self.cross_b(V.transpose(-1, -2)).transpose(-1, -2)
        V_new = torch.cross(a, b, dim=-1)            # (B,N,Cv,3) equivariant
        return s_new, V_new


class EquivariantChannelMix(nn.Module):
    """Linear mixing: scalars mix freely; vector channels mix linearly; scalars
    can rescale vectors (all SO(3)-equivariant)."""

    def __init__(self, scal, vec):
        super().__init__()
        self.s_lin = nn.Linear(scal, scal)
        self.v_lin = nn.Linear(vec, vec, bias=False)
        self.gate = nn.Linear(scal, vec)

    def forward(self, s, V):
        s2 = self.s_lin(s)
        Vm = self.v_lin(V.transpose(-1, -2)).transpose(-1, -2)
        g = self.gate(s).unsqueeze(-1)               # (B,N,vec,1)
        return s2, Vm * g


class GATrBlock(nn.Module):
    def __init__(self, scal, vec, num_heads):
        super().__init__()
        self.mix = EquivariantChannelMix(scal, vec)
        self.bilinear = GeometricBilinear(vec, scal)
        self.merge_s = nn.Linear(2 * scal, scal)
        self.num_heads = num_heads
        # attention from invariants (scalars + vector norms)
        self.to_q = nn.Linear(scal + vec, num_heads)
        self.to_k = nn.Linear(scal + vec, num_heads)
        self.s_out = nn.Linear(scal, scal)
        self.v_out = nn.Linear(vec, vec, bias=False)

    def _invariants(self, s, V):
        return torch.cat([s, (V * V).sum(-1)], dim=-1)

    def forward(self, s, V):
        s1, V1 = self.mix(s, V)
        sb, Vb = self.bilinear(s1, V1)
        s1 = self.merge_s(torch.cat([s1, sb], dim=-1))
        V1 = V1 + Vb
        inv = self._invariants(s1, V1)
        q = self.to_q(inv)                            # (B,N,H)
        k = self.to_k(inv)
        att = torch.softmax(torch.einsum("bih,bjh->bijh", q, k) / (q.shape[-1] ** 0.5), dim=2)
        alpha = att.mean(-1).unsqueeze(-1)            # (B,N,N,1) scalar weights
        s_agg = (alpha * s1.unsqueeze(1)).sum(dim=2)
        V_agg = (alpha.unsqueeze(-1) * V1.unsqueeze(1)).sum(dim=2)
        return s + self.s_out(s_agg), V + self.v_out(V_agg.transpose(-1, -2)).transpose(-1, -2)


class GATr(nn.Module):
    def __init__(self, num_node_features, num_nodes, cfg):
        super().__init__()
        scal = cfg.hidden_dim
        vec = cfg.num_multivectors
        self.s_embed = nn.Linear(num_node_features, scal)
        self.v_embed = nn.Linear(2, vec, bias=False)  # pos & vel -> vec channels
        self.blocks = nn.ModuleList([GATrBlock(scal, vec, cfg.num_heads) for _ in range(cfg.num_layers)])
        self.v_read = nn.Linear(vec, 1, bias=False)

    def forward(self, batch):
        pos, vel = batch["pos"], batch["vel"]
        s = self.s_embed(batch["h"])                  # (B,N,scal)
        V_in = torch.stack([pos, vel], dim=-2)        # (B,N,2,3)
        V = self.v_embed(V_in.transpose(-1, -2)).transpose(-1, -2)  # (B,N,vec,3)
        for blk in self.blocks:
            s, V = blk(s, V)
        delta = self.v_read(V.transpose(-1, -2)).transpose(-1, -2).squeeze(-2)  # (B,N,3)
        return pos + delta
