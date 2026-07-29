"""Additional equivariant baselines built on the EGNN backbone.

These are compact, self-contained reimplementations (not the original authors'
code) that preserve E(3)-equivariance by construction: they only ever transform
invariant scalar channels or add rotation-covariant vectors to coordinates. They
capture the defining idea of each method rather than every architectural detail:

* EGNO  (Xu et al., 2024): EGNN + a Fourier operator on the invariant node
  features (the paper notes EGNO "employs additional Fourier features").
* HEGNN (Cen et al., 2024): EGNN enriched with high-degree steerable
  (spherical-harmonic) invariants in the messages ("high-degree steerable
  features").
* GMN   (Huang et al., 2022): an equivariant second-order (mechanics) update
  maintaining velocity and acceleration.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .egnn import EGNNLayer


class _FourierNodeOperator(nn.Module):
    """1D Fourier neural operator over the hidden channel of each node."""

    def __init__(self, dim, modes):
        super().__init__()
        self.modes = min(modes, dim // 2 + 1)
        scale = 1.0 / dim
        self.weight = nn.Parameter(scale * torch.randn(self.modes, dtype=torch.cfloat))
        self.bias = nn.Linear(dim, dim)

    def forward(self, h):  # h: (B, N, d) invariant
        ft = torch.fft.rfft(h, dim=-1)
        ft_out = torch.zeros_like(ft)
        ft_out[..., : self.modes] = ft[..., : self.modes] * self.weight
        spec = torch.fft.irfft(ft_out, n=h.shape[-1], dim=-1)
        return h + self.bias(spec)


class EGNO(nn.Module):
    def __init__(self, num_node_features, num_nodes, cfg, use_velocity=True):
        super().__init__()
        d = cfg.hidden_dim
        self.embed = nn.Linear(num_node_features, d)
        self.use_velocity = use_velocity
        self.layers = nn.ModuleList([EGNNLayer(d, use_velocity) for _ in range(cfg.num_layers)])
        self.fno = nn.ModuleList([_FourierNodeOperator(d, cfg.num_fourier_modes) for _ in range(cfg.num_layers)])

    def forward(self, batch):
        h = self.embed(batch["h"])
        x = batch["pos"]
        v = batch["vel"] if self.use_velocity else torch.zeros_like(x)
        for layer, fno in zip(self.layers, self.fno):
            h, x, v = layer(h, x, v)
            h = fno(h)  # transforms invariant scalars only -> stays equivariant
        return x


class _HEGNNLayer(nn.Module):
    """EGNN layer whose edge messages also see high-degree spherical-harmonic
    invariants of the relative direction (norms per degree are rotation-invariant
    scalars derived from the degree-l harmonics)."""

    def __init__(self, hidden_dim, max_degree=4, use_velocity=True):
        super().__init__()
        self.max_degree = max_degree
        self.use_velocity = use_velocity
        # number of extra invariant scalars: one per degree (0..L) via |Y_l|
        extra = max_degree + 1
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1 + extra, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.coord_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1, bias=False))
        self.node_mlp = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        if use_velocity:
            self.vel_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))

    def _degree_invariants(self, rel):
        from e3nn import o3
        b, n, _, _ = rel.shape
        flat = rel.reshape(-1, 3)
        norm = flat.norm(dim=-1, keepdim=True)
        unit = flat / (norm + 1e-8)
        invs = []
        for l in range(self.max_degree + 1):
            y = o3.spherical_harmonics(l, unit, normalize=True, normalization="component")
            invs.append(y.norm(dim=-1, keepdim=True))  # rotation-invariant scalar
        inv = torch.cat(invs, dim=-1).reshape(b, n, n, -1)
        return inv

    def forward(self, h, x, v):
        b, n, _ = h.shape
        rel = x.unsqueeze(2) - x.unsqueeze(1)
        dist2 = (rel ** 2).sum(-1, keepdim=True)
        inv = self._degree_invariants(rel)
        hi = h.unsqueeze(2).expand(b, n, n, h.shape[-1])
        hj = h.unsqueeze(1).expand(b, n, n, h.shape[-1])
        m = self.edge_mlp(torch.cat([hi, hj, dist2, inv], dim=-1))
        agg = (rel * self.coord_mlp(m)).sum(dim=2) / (n - 1)
        if self.use_velocity:
            v = self.vel_mlp(h) * v + agg
            x = x + v
        else:
            x = x + agg
        h = h + self.node_mlp(torch.cat([h, m.sum(dim=2)], dim=-1))
        return h, x, v


class HEGNN(nn.Module):
    def __init__(self, num_node_features, num_nodes, cfg, use_velocity=True):
        super().__init__()
        d = cfg.hidden_dim
        self.embed = nn.Linear(num_node_features, d)
        self.use_velocity = use_velocity
        self.layers = nn.ModuleList(
            [_HEGNNLayer(d, cfg.num_degrees, use_velocity) for _ in range(cfg.num_layers)]
        )

    def forward(self, batch):
        h = self.embed(batch["h"])
        x = batch["pos"]
        v = batch["vel"] if self.use_velocity else torch.zeros_like(x)
        for layer in self.layers:
            h, x, v = layer(h, x, v)
        return x


class GMN(nn.Module):
    """Equivariant second-order (mechanics) network: maintains velocity and an
    aggregated acceleration, integrating them to update coordinates."""

    def __init__(self, num_node_features, num_nodes, cfg):
        super().__init__()
        d = cfg.hidden_dim
        self.embed = nn.Linear(num_node_features, d)
        self.num_layers = cfg.num_layers
        self.edge_mlp = nn.ModuleList()
        self.acc_mlp = nn.ModuleList()
        self.node_mlp = nn.ModuleList()
        self.damp = nn.ModuleList()
        for _ in range(cfg.num_layers):
            self.edge_mlp.append(nn.Sequential(nn.Linear(2 * d + 2, d), nn.SiLU(), nn.Linear(d, d), nn.SiLU()))
            self.acc_mlp.append(nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1, bias=False)))
            self.node_mlp.append(nn.Sequential(nn.Linear(2 * d, d), nn.SiLU(), nn.Linear(d, d)))
            self.damp.append(nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1)))

    def forward(self, batch):
        h = self.embed(batch["h"])
        x = batch["pos"]
        v = batch["vel"]
        b, n, _ = h.shape
        for i in range(self.num_layers):
            rel = x.unsqueeze(2) - x.unsqueeze(1)
            dist2 = (rel ** 2).sum(-1, keepdim=True)
            relv = v.unsqueeze(2) - v.unsqueeze(1)
            velinv = (relv * rel).sum(-1, keepdim=True)  # invariant coupling
            hi = h.unsqueeze(2).expand(b, n, n, h.shape[-1])
            hj = h.unsqueeze(1).expand(b, n, n, h.shape[-1])
            m = self.edge_mlp[i](torch.cat([hi, hj, dist2, velinv], dim=-1))
            acc = (rel * self.acc_mlp[i](m)).sum(dim=2) / (n - 1)  # equivariant acceleration
            v = self.damp[i](h) * v + acc                          # integrate acceleration
            x = x + v                                              # integrate velocity
            h = h + self.node_mlp[i](torch.cat([h, m.sum(dim=2)], dim=-1))
        return x
