"""
Sanity checks for EquivarianceLoss (v3 API).

Print-based test in the style of test_gnns.py: toy networks with KNOWN
symmetry properties must get ~0 loss, and networks violating the symmetry
must get a clearly nonzero loss. Exits non-zero if any check fails.

v3 API contract (see equivariance_loss.py):
    loss_fn = EquivarianceLoss(group_type=..., feature_type=..., num_samples=...)
    main_loss, loss_dict = loss_fn(network_fn, positions, features, edge_index, batch)
    network_fn(positions, features, edge_index, batch, return_layer_outputs=True)
        -> (final_output, layer_outputs_list)
    layer_outputs_list items: {'layer_idx', 'representation', ...}
"""

import sys

import torch
import torch.nn as nn

from equivariance_loss import EquivarianceLoss

TOL_ZERO = 1e-8      # "exactly equivariant" up to float error
TOL_NONZERO = 1e-3   # "clearly violated"


def make_batch():
    """Two small graphs: 5 nodes and 4 nodes, ring edges, random features."""
    torch.manual_seed(0)
    n1, n2, feat_dim = 5, 4, 6
    positions = torch.randn(n1 + n2, 3)
    features = torch.randn(n1 + n2, feat_dim)

    def ring(n, offset):
        src = torch.arange(n) + offset
        dst = (torch.arange(n) + 1) % n + offset
        return torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])

    edge_index = torch.cat([ring(n1, 0), ring(n2, n1)], dim=1)
    batch = torch.cat([torch.zeros(n1), torch.ones(n2)]).long()
    return positions, features, edge_index, batch


# ---------------- Toy networks (known symmetry properties) ----------------

def nodewise_net(pos, feat, edge, batch, return_layer_outputs=True):
    """Permutation-equivariant, position-ignoring: f(x)_i = W x_i (shared W)."""
    W = torch.arange(1, feat.shape[1] * 3 + 1, dtype=feat.dtype).reshape(feat.shape[1], 3)
    out = feat @ W  # [N, 3]
    return out, []


def pos_passthrough_net(pos, feat, edge, batch, return_layer_outputs=True):
    """Rotation-equivariant: f(x) = pos (output IS the position vector)."""
    return pos, []


def pos_offset_net(pos, feat, edge, batch, return_layer_outputs=True):
    """NOT rotation-equivariant: f(x) = pos + const (offset doesn't rotate)."""
    return pos + torch.tensor([1.0, 2.0, 3.0]), []


def vector_layer_net(pos, feat, edge, batch, return_layer_outputs=True):
    """PaiNN-shaped layer outputs: scalar [N, H] + vector [N, H, 3] layers."""
    n = pos.shape[0]
    h = 4
    layers = [{
        'layer_idx': 0,
        'representation': feat[:, :h] if feat.shape[1] >= h else feat.repeat(1, h)[:, :h],
        'vector_representation': pos.unsqueeze(1).expand(n, h, 3).contiguous(),
        'edge_index': edge,
        'batch': batch,
    }]
    return pos, layers


class TrainablePosNet(nn.Module):
    """Trainable, position-sensitive scalar net (for the gradient check)."""

    def __init__(self, feat_dim, hidden=8):
        super().__init__()
        self.lin = nn.Linear(feat_dim + 3, hidden)

    def forward(self, pos, feat, edge, batch, return_layer_outputs=True):
        h = self.lin(torch.cat([feat, pos], dim=-1))
        return h, []


# ---------------- Checks ----------------

def run_checks():
    positions, features, edge_index, batch = make_batch()
    results = []

    def check(name, value, op, threshold):
        ok = (value < threshold) if op == '<' else (value > threshold)
        results.append((name, value, op, threshold, ok))

    # 1. Permutation-equivariant net -> ~0 permutation loss (both feature types)
    for ftype in ('invariant', 'equivariant'):
        loss_fn = EquivarianceLoss('permutation', num_samples=2, normalize=False,
                                   feature_type=ftype)
        loss, _ = loss_fn(nodewise_net, positions, features, edge_index, batch)
        check(f"permutation / nodewise net / {ftype}", loss.item(), '<', TOL_ZERO)

    # 2. Rotation-equivariant net (returns pos) -> ~0 so3 loss, equivariant features
    loss_fn = EquivarianceLoss('so3', num_samples=2, normalize=False,
                               feature_type='equivariant')
    loss, _ = loss_fn(pos_passthrough_net, positions, features, edge_index, batch)
    check("so3 / pos passthrough / equivariant", loss.item(), '<', TOL_ZERO)

    # 3. Same net under 'invariant' feature type -> LARGE so3 loss (it is not invariant)
    loss_fn = EquivarianceLoss('so3', num_samples=2, normalize=False,
                               feature_type='invariant')
    loss, _ = loss_fn(pos_passthrough_net, positions, features, edge_index, batch)
    check("so3 / pos passthrough / invariant", loss.item(), '>', TOL_NONZERO)

    # 4. Non-equivariant net (pos + const) -> LARGE so3 loss, equivariant features
    loss_fn = EquivarianceLoss('so3', num_samples=2, normalize=False,
                               feature_type='equivariant')
    loss, _ = loss_fn(pos_offset_net, positions, features, edge_index, batch)
    check("so3 / pos+offset net / equivariant", loss.item(), '>', TOL_NONZERO)

    # 5. [N, H, 3] vector layer outputs (PaiNN shape) must not crash, finite losses
    loss_fn = EquivarianceLoss('so3', num_samples=2, normalize=False,
                               feature_type='equivariant')
    loss, loss_dict = loss_fn(vector_layer_net, positions, features, edge_index, batch)
    finite = torch.isfinite(loss).item() and all(
        torch.isfinite(v).item() for v in loss_dict.values())
    check("so3 / [N,H,3] layer outputs finite", 0.0 if finite else 1.0, '<', TOL_ZERO)
    check("so3 / vector layer loss ~0 (pos-derived)", loss.item(), '<', 1e-6)

    # 6. Gradient flows through the loss into a trainable net
    net = TrainablePosNet(features.shape[1])
    loss_fn = EquivarianceLoss('so3', num_samples=2, normalize=True,
                               feature_type='invariant')
    loss, _ = loss_fn(net, positions, features, edge_index, batch)
    loss.backward()
    grad = net.lin.weight.grad
    grad_ok = grad is not None and torch.isfinite(grad).all().item() and grad.abs().sum().item() > 0
    check("so3 / trainable net gets nonzero grads", 0.0 if grad_ok else 1.0, '<', TOL_ZERO)

    return results


if __name__ == '__main__':
    print("=" * 78)
    print("EQUIVARIANCE LOSS SANITY CHECKS (v3 API)")
    print("=" * 78)
    print(f"{'Check':<48} {'Value':>10} {'Want':>10} Status")
    print("-" * 78)

    results = run_checks()
    failures = 0
    for name, value, op, threshold, ok in results:
        status = "✓" if ok else "✗"
        failures += 0 if ok else 1
        print(f"{name:<48} {value:>10.3e} {op}{threshold:>9.0e} {status}")

    print("=" * 78)
    if failures:
        print(f"{failures} check(s) FAILED")
        sys.exit(1)
    print("All checks passed")
