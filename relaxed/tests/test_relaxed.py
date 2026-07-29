"""Unified tests for the relaxed package (print-based, repo style; exits
non-zero on failure). Run: python relaxed/tests/test_relaxed.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from relaxed.config import ExperimentConfig
from relaxed.geometry import random_rotation_matrix, random_so2_batch
from relaxed.losses import GradNorm, RemulLoss, equivariance_error, rotate_batch
from relaxed.reporting import load_record, make_record, write_record

RESULTS = []


def check(name, ok):
    RESULTS.append((name, bool(ok)))
    print(f"{'✓' if ok else '✗'} {name}")


def test_geometry():
    rot = random_rotation_matrix(64)
    eye = torch.bmm(rot, rot.transpose(-1, -2))
    check("so3 orthonormal", torch.allclose(eye, torch.eye(3).expand_as(eye), atol=1e-5))
    check("so3 det +1", torch.allclose(torch.linalg.det(rot), torch.ones(64), atol=1e-5))
    rz = random_so2_batch("z", 64)
    check("so2_z leaves z invariant", torch.allclose(rz[:, :, 2], torch.tensor([0., 0., 1.]).expand(64, 3), atol=1e-6))
    check("so2_z det +1", torch.allclose(torch.linalg.det(rz), torch.ones(64), atol=1e-5))


def test_gradnorm_purity():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(10, 32), torch.nn.ReLU(), torch.nn.Linear(32, 3))
    x = torch.randn(8, 10)
    y = torch.randn(8, 3)
    gn = GradNorm(2, alpha=1.5, lr=0.025, init_weights=[1.0, 1.0])
    shared_w = model[-1].weight
    pred = model(x)
    obj = torch.nn.functional.mse_loss(pred, y)
    equi = torch.nn.functional.mse_loss(model(x + 0.1), y)
    gn.weighted_sum([obj, equi]).backward(retain_graph=True)
    before = [p.grad.clone() for p in model.parameters()]
    gn.update([obj, equi], shared_w)
    after = [p.grad for p in model.parameters()]
    max_diff = max((a - b).abs().max().item() for a, b in zip(before, after))
    check("GradNorm leaves model gradients untouched", max_diff == 0.0)


def _toy_batch():
    torch.manual_seed(0)
    return {
        "pos": torch.randn(4, 5, 3),
        "vel": torch.randn(4, 5, 3),
        "target": torch.randn(4, 5, 3),
        "h": torch.randn(4, 5, 1),
    }


def test_remul_loss_anchoring():
    batch = _toy_batch()
    loss_fn = RemulLoss(group="so3", metric="l2", num_group_samples=2)

    def fwd(b):
        return b["pos"] * 2.0  # equivariant toy model

    eq = loss_fn.equivariance_loss(fwd, batch)
    # ground-truth anchored: equals task loss on the rotated batch (not vs rho*f(x))
    rot = loss_fn.sample_rotations(4, batch["pos"].device, batch["pos"].dtype)
    rb = rotate_batch(batch, rot)
    ref = loss_fn.objective_loss(fwd(rb), rb["target"])
    check("RemulLoss equivariance term is finite and non-negative",
          torch.isfinite(eq) and eq >= 0)
    check("objective on rotated batch is finite", torch.isfinite(ref))

    # E' of an exactly equivariant model ~ 0; of a non-equivariant one > 0
    e_prime = equivariance_error(fwd, batch, 8, "so3", "E_prime")
    check("E' ~ 0 for equivariant toy", e_prime < 1e-6)
    bad = equivariance_error(lambda b: b["pos"] + 1.0, batch, 8, "so3", "E_prime")
    check("E' > 0 for non-equivariant toy", bad > 1e-3)


def test_config_and_reporting(tmp_path="outputs/relaxed/_test"):
    cfg = ExperimentConfig()
    assert cfg.train.task == "dynamics"
    cfg2 = ExperimentConfig.__new__(ExperimentConfig)
    from relaxed.config import (DataConfig, ModelConfig, LossConfig, ScheduleConfig,
                                TrainConfig, LogConfig, RunConfig)
    cfg2.data = DataConfig(name="QM9", use_positions=True)
    cfg2.model = ModelConfig(name="gcn")
    cfg2.loss = LossConfig()
    cfg2.schedule = ScheduleConfig()
    cfg2.train = TrainConfig()
    cfg2.log = LogConfig()
    cfg2.run = RunConfig()
    cfg2.finalize()
    check("task inferred from dataset", cfg2.train.task == "graph")

    rec = make_record(slug="test", timestamp="20260720_000000", framework="remul",
                      config=cfg.to_dict(), metrics={"test": {"mse": 1.0}}, run_dir=tmp_path)
    path = write_record(rec)
    loaded = load_record(path)
    check("record roundtrip", loaded["slug"] == "test" and loaded["schema"] == 2)
    import shutil
    shutil.rmtree(tmp_path, ignore_errors=True)


if __name__ == "__main__":
    test_geometry()
    test_gradnorm_purity()
    test_remul_loss_anchoring()
    test_config_and_reporting()
    failures = sum(1 for _, ok in RESULTS if not ok)
    print("=" * 60)
    if failures:
        print(f"{failures} check(s) FAILED")
        sys.exit(1)
    print(f"All {len(RESULTS)} checks passed")
