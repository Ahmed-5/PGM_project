"""Aggregation / harness sanity checks (companion to test_equiloss.py).

Where test_equiloss.py exercises the EquivarianceLoss *primitive*, this file
guards the group/layer *aggregation* in train.py, the DepthScheduler init, the
permutation/geometric reduction parity, the OOD/E' evaluation, and the
gradient-accumulation accounting — i.e. the bug fixes A, B, C, D, E, F, G.

Run:  python test_aggregation.py   (CPU-only; exits non-zero on failure)
"""
from __future__ import annotations

import glob
import math
import os
import shutil
import sys

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from config import ExperimentConfig
from equivariant_gnn import BaseGNN
from equivariance_loss import EquivarianceLoss
from train import (compute_equivariance_losses, initialize_equivariance_losses,
                   _resolve_group_weights, build_optimizer, train_epoch)
from schedulers import DepthScheduler
from logger import get_logger

DEVICE = "cpu"
_TMP_PREFIX = "__test_agg__"
_results = []


def _record(name, ok, detail=""):
    _results.append((name, ok, detail))


def _make_config(model_type="gcn", num_layers=3, groups=("so3",),
                 group_weights=None, use_pos=True, use_positions=True,
                 in_channels=4, hidden=8, accumulation_steps=1,
                 normalize_group_weights=True, total_strength=1.0):
    cfg = ExperimentConfig(experiment_name=_TMP_PREFIX, device=DEVICE)
    cfg.model.model_type = model_type
    cfg.model.num_layers = num_layers
    cfg.model.in_channels = in_channels
    cfg.model.hidden_channels = hidden
    cfg.model.out_channels = 1
    cfg.model.use_pos = use_pos
    cfg.model.dropout = 0.0
    cfg.data.use_positions = use_positions
    cfg.equivariance.symmetry_groups = list(groups)
    cfg.equivariance.group_weights = dict(group_weights) if group_weights else {
        g: 0.1 for g in groups}
    cfg.equivariance.num_samples = 2
    cfg.equivariance.normalize = True
    cfg.equivariance.normalize_group_weights = normalize_group_weights
    cfg.equivariance.total_equivariance_strength = total_strength
    cfg.training.accumulation_steps = accumulation_steps
    cfg.training.use_amp = False
    cfg.training.grad_clip = 0.0
    cfg.training.optimizer = "adam"
    return cfg


def _make_model(cfg):
    return BaseGNN(
        in_channels=cfg.model.in_channels,
        hidden_channels=cfg.model.hidden_channels,
        out_channels=cfg.model.out_channels,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        model_type=cfg.model.model_type,
        spatial_dim=cfg.model.spatial_dim,
        use_pos=cfg.model.use_pos,
    ).to(DEVICE)


def _make_graph(n, in_channels, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, in_channels, generator=g)
    pos = torch.randn(n, 3, generator=g)
    # simple ring edges (undirected)
    src = torch.arange(n)
    dst = (src + 1) % n
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    y = torch.randn(1, generator=g)
    return Data(x=x, pos=pos, edge_index=edge_index, y=y)


def _make_batch(in_channels=4, sizes=(4, 5), seed=0):
    from torch_geometric.data import Batch
    graphs = [_make_graph(n, in_channels, seed + i) for i, n in enumerate(sizes)]
    return Batch.from_data_list(graphs).to(DEVICE)


# --------------------------------------------------------------------------- #
# A + F: final layer not double-counted; positional weight indexing is safe.
# --------------------------------------------------------------------------- #
def test_final_layer_not_double_counted():
    torch.manual_seed(0)
    cfg = _make_config(num_layers=3, groups=("so3",))
    model = _make_model(cfg)
    batch = _make_batch(cfg.model.in_channels)
    eq_losses = initialize_equivariance_losses(cfg)
    lw = torch.tensor([0.2, 0.5, 1.3], device=DEVICE)

    torch.manual_seed(123)
    total, d = compute_equivariance_losses(
        model, batch, eq_losses, cfg, DEVICE, layer_weights=lw,
        apply_group_weights=False)

    # exactly num_layers per-layer entries, and no layer_3 (the old extra slot)
    layer_keys = sorted(k for k in d if k.startswith("eq_loss/so3_layer_"))
    has_no_extra = "eq_loss/so3_layer_3" not in d
    n_layers_ok = layer_keys == [f"eq_loss/so3_layer_{i}" for i in range(3)]

    L = [d[f"eq_loss/so3_layer_{i}"] for i in range(3)]
    expected = sum(lw[i] * L[i] for i in range(3))
    main = d["eq_loss/so3_main"]
    buggy = expected + lw[2] * main  # what the old double-counting produced

    matches_clean = torch.allclose(total, expected, atol=1e-5)
    differs_from_buggy = not torch.allclose(total, buggy, atol=1e-5) and float(main) > 1e-8

    ok = has_no_extra and n_layers_ok and matches_clean and differs_from_buggy
    _record("A/F: no final-layer double count", ok,
            f"total={float(total):.4e} clean={float(expected):.4e} buggy={float(buggy):.4e}")


# --------------------------------------------------------------------------- #
# C: group-weight normalization -> equal total strength across group counts.
# --------------------------------------------------------------------------- #
def test_group_weight_normalization():
    c1 = _make_config(groups=("so3",), group_weights={"so3": 0.1})
    c3 = _make_config(groups=("so3", "translation", "permutation"),
                      group_weights={"so3": 0.1, "translation": 0.1, "permutation": 0.1})
    w1 = _resolve_group_weights(c1)
    w3 = _resolve_group_weights(c3)
    s1, s3 = sum(w1.values()), sum(w3.values())
    ok = abs(s1 - 1.0) < 1e-6 and abs(s3 - 1.0) < 1e-6 and abs(s1 - s3) < 1e-6
    _record("C: group weights normalized to equal total", ok,
            f"sum1={s1:.4f} sum3={s3:.4f}")


# --------------------------------------------------------------------------- #
# B: apply_group_weights=False -> result independent of group_weights.
# --------------------------------------------------------------------------- #
def test_unweighted_is_group_weight_free():
    cfg_a = _make_config(groups=("so3",), group_weights={"so3": 0.1})
    cfg_b = _make_config(groups=("so3",), group_weights={"so3": 5.0})
    model = _make_model(cfg_a)
    batch = _make_batch(cfg_a.model.in_channels)
    eq_a = initialize_equivariance_losses(cfg_a)
    eq_b = initialize_equivariance_losses(cfg_b)

    torch.manual_seed(7)
    ta, _ = compute_equivariance_losses(model, batch, eq_a, cfg_a, DEVICE,
                                        layer_weights=None, apply_group_weights=False)
    torch.manual_seed(7)
    tb, _ = compute_equivariance_losses(model, batch, eq_b, cfg_b, DEVICE,
                                        layer_weights=None, apply_group_weights=False)
    ok = torch.allclose(ta, tb, atol=1e-6) and float(ta) > 0
    _record("B: unweighted metric ignores group_weights", ok,
            f"a={float(ta):.6e} b={float(tb):.6e}")


# --------------------------------------------------------------------------- #
# D: learnable schedule init recovers the intended target (both modules).
# --------------------------------------------------------------------------- #
def test_learnable_init_matches_target():
    import relaxed.schedulers as rel_sched
    N, a0, beta = 4, 1.0, 0.3
    target = torch.tensor([a0 * math.exp(-beta * l) for l in range(N)])
    ok = True
    detail = []
    for label, cls in (("legacy", DepthScheduler),
                       ("relaxed", rel_sched.DepthScheduler)):
        ds = cls(num_layers=N, schedule_type="learnable", alpha_0=a0, beta=beta)
        got = ds.get_all_alphas().detach()
        close = torch.allclose(got, target, atol=1e-4)
        ok = ok and close
        detail.append(f"{label}:{'ok' if close else 'BAD'}")
    _record("D: learnable init == alpha_0*exp(-beta*l)", ok, " ".join(detail))


# --------------------------------------------------------------------------- #
# G: permutation-path reduction is an element-wise MEAN (feature-dim invariant),
#    not a sum (which would scale ~ feature_dim).
# --------------------------------------------------------------------------- #
def test_reduction_parity():
    # A deliberately non-permutation-equivariant network_fn: zero out the FIRST
    # (local index 0) node's row. The per-element squared error pattern is then
    # independent of the feature dimension, so an element-wise mean reduction is
    # invariant to F while a sum reduction scales linearly with F.
    def make_net(feat_dim):
        def net(pos, feat, edges, batch, return_layer_outputs=True):
            out = feat.clone()
            out[0] = 0.0
            return out, []
        return net

    losses = {}
    for F in (2, 16):
        eq = EquivarianceLoss(group_type="permutation", num_samples=1, normalize=False)
        n = 6
        feat = torch.ones(n, F)
        pos = torch.zeros(n, 3)
        edge_index = torch.stack([torch.arange(n), (torch.arange(n) + 1) % n])
        batch = torch.zeros(n, dtype=torch.long)
        torch.manual_seed(0)
        main, _ = eq(make_net(F), pos, feat, edge_index, batch)
        losses[F] = float(main)
    ratio = losses[16] / max(losses[2], 1e-12)
    # element-wise mean -> ratio ~1; old sum-based code -> ratio ~8 (16/2).
    ok = 0.5 < ratio < 2.0 and losses[2] > 0
    _record("G: permutation reduction is feature-dim invariant", ok,
            f"L(F=2)={losses[2]:.4e} L(F=16)={losses[16]:.4e} ratio={ratio:.2f}")


# --------------------------------------------------------------------------- #
# OOD/E' evaluation: invariant model -> E'~0; position-sensitive model -> E'>0.
# --------------------------------------------------------------------------- #
def test_ood_eval_correct():
    from relaxed.engines.graph import evaluate_ood
    from torch_geometric.data import Batch
    graphs = [_make_graph(n, 4, seed=i) for i, n in enumerate((4, 5, 6))]
    loader = DataLoader(graphs, batch_size=2, shuffle=False)

    # invariant model: use_pos=False -> ignores positions -> E' ~ 0
    cfg_inv = _make_config(use_pos=False, use_positions=True)
    m_inv = _make_model(cfg_inv)
    ood_inv = evaluate_ood(m_inv, loader, DEVICE, cfg_inv, groups=["so3"], num_rotations=4)

    # sensitive model: use_pos=True -> rotation-sensitive -> E' > 0
    cfg_sen = _make_config(use_pos=True, use_positions=True)
    m_sen = _make_model(cfg_sen)
    ood_sen = evaluate_ood(m_sen, loader, DEVICE, cfg_sen, groups=["so3"], num_rotations=4)

    einv = ood_inv.get("so3/E_prime", None)
    esen = ood_sen.get("so3/E_prime", None)
    ok = (einv is not None and esen is not None
          and einv < 1e-5 and esen > 1e-4 and "so3/MAE" in ood_sen)
    _record("OOD: invariant E'~0, sensitive E'>0", ok,
            f"E'_inv={einv} E'_sen={esen}")


# --------------------------------------------------------------------------- #
# E: trailing gradient-accumulation group is flushed (optimizer steps).
# --------------------------------------------------------------------------- #
def test_train_epoch_accounting():
    torch.manual_seed(0)
    # accumulation_steps (8) > number of batches (3): WITHOUT the trailing-flush
    # fix the optimizer never steps and params never change.
    cfg = _make_config(use_pos=False, use_positions=False, accumulation_steps=8)
    cfg.equivariance.symmetry_groups = []          # isolate the accounting path
    cfg.logging.logger_type = "none"
    model = _make_model(cfg)
    graphs = [_make_graph(n, cfg.model.in_channels, seed=i)
              for i, n in enumerate((4, 5, 6, 4, 5, 6))]
    loader = DataLoader(graphs, batch_size=2, shuffle=False)  # 3 batches of 2 graphs
    optimizer = build_optimizer(model.parameters(), cfg)
    depth = DepthScheduler(num_layers=cfg.model.num_layers, schedule_type="constant",
                           alpha_0=1.0)
    logger = get_logger(cfg)
    before = [p.detach().clone() for p in model.parameters()]
    train_epoch(model, loader, optimizer, DEVICE, torch.nn.MSELoss(), {},
                logger, 1, cfg, depth)
    changed = any(not torch.equal(b, p) for b, p in zip(before, model.parameters()))
    _record("E: trailing accumulation group is flushed", changed,
            f"params_changed={changed}")


def _cleanup():
    for root in ("checkpoints", "outputs"):
        for d in glob.glob(os.path.join(root, f"{_TMP_PREFIX}*")):
            shutil.rmtree(d, ignore_errors=True)


def main():
    tests = [
        test_final_layer_not_double_counted,
        test_group_weight_normalization,
        test_unweighted_is_group_weight_free,
        test_learnable_init_matches_target,
        test_reduction_parity,
        test_ood_eval_correct,
        test_train_epoch_accounting,
    ]
    for t in tests:
        try:
            t()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            _record(t.__name__, False, f"EXCEPTION: {e}")

    _cleanup()

    print("=" * 78)
    print("AGGREGATION / HARNESS SANITY CHECKS")
    print("=" * 78)
    print(f"{'Check':52s}{'Status':8s}Detail")
    print("-" * 78)
    all_ok = True
    for name, ok, detail in _results:
        all_ok = all_ok and ok
        print(f"{name:52s}{'PASS' if ok else 'FAIL':8s}{detail}")
    print("=" * 78)
    print("All checks passed" if all_ok else "SOME CHECKS FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
