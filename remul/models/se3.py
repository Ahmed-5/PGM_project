"""SE(3)-Transformer / Tensor Field Network baseline (e3nn-based).

Compact reimplementation (not the authors' code) of an SE(3)-equivariant
attention network in the spirit of Fuchs et al. (2020) / Thomas et al. (2018).
Node features are e3nn irreps (type-0 scalars + type-1 vectors); messages are
equivariant tensor products of neighbour features with spherical harmonics of
the relative direction, weighted by a learned radial function and a scalar
attention coefficient. The type-1 output channel provides the coordinate update,
so the whole model is exactly SE(3)-equivariant.

Set ``model.name = 'tfn'`` to disable attention (plain Tensor Field Network).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from e3nn import o3
from e3nn.nn import Gate
from e3nn.math import soft_one_hot_linspace


class _SE3Layer(nn.Module):
    def __init__(self, irreps_in, irreps_out, irreps_sh, num_basis=16, attention=True):
        super().__init__()
        self.attention = attention
        self.irreps_sh = irreps_sh
        self.num_basis = num_basis

        # Gated nonlinearity: scalars pass through, vectors are gated by scalars.
        irreps_scalars = o3.Irreps([(mul, ir) for mul, ir in irreps_out if ir.l == 0])
        irreps_gated = o3.Irreps([(mul, ir) for mul, ir in irreps_out if ir.l > 0])
        irreps_gates = o3.Irreps([(mul, "0e") for mul, _ in irreps_gated])
        self.gate = Gate(irreps_scalars, [torch.nn.functional.silu],
                         irreps_gates, [torch.sigmoid], irreps_gated)

        self.tp = o3.FullyConnectedTensorProduct(
            irreps_in, irreps_sh, self.gate.irreps_in, shared_weights=False
        )
        self.radial = nn.Sequential(
            nn.Linear(num_basis, 64), nn.SiLU(), nn.Linear(64, self.tp.weight_numel)
        )
        self.self_connection = o3.Linear(irreps_in, self.gate.irreps_in)
        if attention:
            self.att = nn.Sequential(nn.Linear(num_basis, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, node_feat, pos):
        b, n, _ = node_feat.shape
        rel = pos.unsqueeze(2) - pos.unsqueeze(1)              # (B,N,N,3)
        dist = rel.norm(dim=-1)                                 # (B,N,N)
        sh = o3.spherical_harmonics(self.irreps_sh, rel, normalize=True, normalization="component")
        basis = soft_one_hot_linspace(
            dist.reshape(-1), 0.0, dist.max().item() + 1e-3, self.num_basis,
            basis="smooth_finite", cutoff=True,
        ).reshape(b, n, n, self.num_basis)
        weight = self.radial(basis)                            # (B,N,N,weight_numel)
        fj = node_feat.unsqueeze(1).expand(b, n, n, node_feat.shape[-1])
        messages = self.tp(fj, sh, weight)                     # (B,N,N,irreps)
        if self.attention:
            logits = self.att(basis).squeeze(-1)               # (B,N,N)
            alpha = torch.softmax(logits, dim=2).unsqueeze(-1)
            agg = (messages * alpha).sum(dim=2)
        else:
            agg = messages.sum(dim=2) / (n - 1)
        out = agg + self.self_connection(node_feat)
        return self.gate(out)


class SE3Transformer(nn.Module):
    def __init__(self, num_node_features, num_nodes, cfg, attention=True):
        super().__init__()
        d = cfg.se3_channels
        lmax = max(1, cfg.num_degrees - 1)
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)
        irreps_hidden = o3.Irreps(f"{d}x0e + {d}x1o")
        self.embed = o3.Linear(o3.Irreps(f"{num_node_features}x0e + 1x1o"), irreps_hidden)
        self.num_node_features = num_node_features
        self.layers = nn.ModuleList()
        for _ in range(cfg.num_layers):
            self.layers.append(_SE3Layer(irreps_hidden, irreps_hidden, self.irreps_sh, attention=attention))
        self.readout = o3.Linear(irreps_hidden, o3.Irreps("1x1o"))

    def _embed_input(self, batch):
        # scalars = h ; single type-1 vector = velocity
        b, n, _ = batch["h"].shape
        feat = torch.cat([batch["h"], batch["vel"]], dim=-1)  # (B,N,F+3)
        return self.embed(feat)

    def forward(self, batch):
        pos = batch["pos"]
        f = self._embed_input(batch)
        for layer in self.layers:
            f = f + layer(f, pos)
        delta = self.readout(f)  # (B,N,3) type-1 -> equivariant vector
        return pos + delta
