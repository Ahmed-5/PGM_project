"""Compute-time benchmark (paper Section 6.4 / Figure 3 style).

Measures parameters, training throughput (ms/step, forward+backward), and
inference latency (ms/batch, no-grad forward) for unconstrained models vs
equivariant baselines, on both tasks:

* dynamics: N-body dimensions (B=64, N=4) — transformer/mlp/gnn vs
  egnn/se3_transformer/tfn/gatr (the paper's Fig. 3 comparison);
* graph: QM9-like PyG batch (128 graphs x ~15 nodes) — gcn vs egnn.

Usage: python -m relaxed.benchmark [--device cuda] [--csv results/benchmark.csv]
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import torch

from .config import ExperimentConfig


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _time_fn(fn, iters: int, warmup: int, device) -> float:
    """Median milliseconds per call."""
    for _ in range(warmup):
        fn()
    _sync(device)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return times[len(times) // 2]


def _dyn_batch(batch_size: int, num_nodes: int, feat: int, device) -> dict:
    return {
        "pos": torch.randn(batch_size, num_nodes, 3, device=device),
        "vel": torch.randn(batch_size, num_nodes, 3, device=device),
        "h": torch.randn(batch_size, num_nodes, feat, device=device),
        "mass": torch.ones(batch_size, num_nodes, 1, device=device),
        "target": torch.randn(batch_size, num_nodes, 3, device=device),
    }


def bench_dynamics(device, iters: int, warmup: int) -> list[dict]:
    from remul.models import build_model as build_dyn
    from remul.config import ModelConfig as DynModelConfig

    specs = {
        "transformer": dict(channels=384, num_layers=10, num_heads=8),
        "mlp": dict(mlp_hidden=680, num_layers=3),
        "gnn": dict(hidden_dim=64, num_layers=4),
        "egnn": dict(hidden_dim=64, num_layers=4),
        "se3_transformer": dict(),
        "tfn": dict(),
        "gatr": dict(channels=128, num_layers=12, num_heads=8, num_multivectors=16),
    }
    batch = _dyn_batch(64, 4, 1, device)
    rows = []
    for name, overrides in specs.items():
        cfg = DynModelConfig(name=name, **overrides)
        model = build_dyn(cfg, 1, 4).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        params = sum(p.numel() for p in model.parameters())

        def train_step():
            opt.zero_grad()
            pred = model(batch)
            loss = torch.nn.functional.mse_loss(pred, batch["target"])
            loss.backward()
            opt.step()

        def infer():
            with torch.no_grad():
                model(batch)

        ms_train = _time_fn(train_step, iters, warmup, device)
        ms_infer = _time_fn(infer, iters, warmup, device)
        rows.append({"task": "dynamics", "model": name, "params": params,
                     "train_ms_per_step": round(ms_train, 3),
                     "inference_ms_per_batch": round(ms_infer, 3)})
        print(f"  dyn {name:<16} params={params:>10,} train={ms_train:8.3f} ms/step "
              f"infer={ms_infer:7.3f} ms/batch")
        del model, opt
    return rows


def bench_graph(device, iters: int, warmup: int) -> list[dict]:
    from equivariant_gnn import BaseGNN

    torch.manual_seed(0)
    num_graphs, nodes = 128, 15
    n = num_graphs * nodes
    x = torch.randn(n, 11, device=device)
    pos = torch.randn(n, 3, device=device)
    src = torch.randint(0, n, (2 * n,), device=device)
    dst = torch.randint(0, n, (2 * n,), device=device)
    edge_index = torch.stack([src, dst])
    batch_idx = torch.arange(num_graphs, device=device).repeat_interleave(nodes)
    y = torch.randn(num_graphs, device=device)

    rows = []
    for name in ("gcn", "egnn"):
        model = BaseGNN(in_channels=11, hidden_channels=128, out_channels=1,
                        num_layers=4, model_type=name).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        params = sum(p.numel() for p in model.parameters())

        def train_step():
            opt.zero_grad()
            pred = model(x, pos, edge_index, batch_idx).squeeze()
            loss = torch.nn.functional.mse_loss(pred, y)
            loss.backward()
            opt.step()

        def infer():
            with torch.no_grad():
                model(x, pos, edge_index, batch_idx)

        ms_train = _time_fn(train_step, iters, warmup, device)
        ms_infer = _time_fn(infer, iters, warmup, device)
        rows.append({"task": "graph", "model": name, "params": params,
                     "train_ms_per_step": round(ms_train, 3),
                     "inference_ms_per_batch": round(ms_infer, 3)})
        print(f"  graph {name:<15} params={params:>10,} train={ms_train:8.3f} ms/step "
              f"infer={ms_infer:7.3f} ms/batch")
        del model, opt
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--csv", default="results/benchmark.csv")
    args = parser.parse_args()

    device = args.device
    print(f"Benchmark on {device} (iters={args.iters}, warmup={args.warmup})")
    rows = bench_dynamics(device, args.iters, args.warmup) + \
           bench_graph(device, args.iters, args.warmup)

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV -> {args.csv}")


if __name__ == "__main__":
    main()
