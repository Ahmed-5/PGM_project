"""Recompute the UNWEIGHTED functional equivariance error on the test set for
completed ablation runs, and add it to each run's ``test_metrics.json``.

The weighted ``test/eq_loss_total`` reported by training is the *objective*
value (alpha_0 flows through the DepthScheduler layer weights), which is
degenerate for the baseline (alpha_0 = 0 -> reads 0). For a fair comparison of
how equivariant each trained model actually is, this script evaluates the same
layer-wise functional loss with all-ones layer weights (no alpha_0, no group
weights) on the test split, for every run that has a best checkpoint.

Usage:
    python recompute_unweighted_eq.py --pattern 'QM9_*'
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil

import torch
from torch_geometric.loader import DataLoader

from config import ExperimentConfig
from load_dataset import load_dataset
from equivariant_gnn import BaseGNN
from train import initialize_equivariance_losses, compute_equivariance_losses
from utils import load_checkpoint

RUN_DIR_RE = re.compile(r'^(?P<exp>.+)_(?P<ts>\d{8}_\d{6})$')


@torch.no_grad()
def unweighted_eq_error(model, loader, eq_losses, config, device):
    """Mean total equivariance loss (all-ones layer weights) + per-group means."""
    model.eval()
    total, n_batches = 0.0, 0
    per_group: dict[str, float] = {}
    for batch in loader:
        batch = batch.to(device)
        eq_total, eq_dict = compute_equivariance_losses(
            model, batch, eq_losses, config, device, layer_weights=None,
            apply_group_weights=False)
        total += float(eq_total)
        n_batches += 1
        for key, value in eq_dict.items():
            if key.endswith('_total'):
                per_group[key] = per_group.get(key, 0.0) + float(value)
    n_batches = max(n_batches, 1)
    return total / n_batches, {k: v / n_batches for k, v in per_group.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pattern', default='QM9_*')
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset_cache = {}

    run_dirs = sorted(d for d in glob.glob(os.path.join('checkpoints', args.pattern))
                      if RUN_DIR_RE.match(os.path.basename(d)))
    print(f"Found {len(run_dirs)} run dir(s)")

    # ExperimentConfig.load() creates fresh run dirs as a side effect; remove
    # only dirs that did not exist before this script started AND are empty.
    # (Never delete pre-existing empty dirs — they may belong to in-flight runs.)
    pre_existing = set()
    for root in ('checkpoints', 'outputs'):
        pre_existing.update(os.path.join(root, d) for d in os.listdir(root))

    for ckpt_dir in run_dirs:
        run_name = os.path.basename(ckpt_dir)
        config_files = glob.glob(os.path.join(ckpt_dir, '*_config.json'))
        best_files = glob.glob(os.path.join(ckpt_dir, '*_best.pt'))
        metrics_file = os.path.join('outputs', run_name, 'test_metrics.json')
        if not config_files or not best_files or not os.path.exists(metrics_file):
            print(f"  skip {run_name} (missing config/checkpoint/metrics)")
            continue

        config = ExperimentConfig.load(config_files[0])
        config.training.batch_size = args.batch_size

        # Same seeded splits as training; cache across runs.
        if 'datasets' not in dataset_cache:
            dataset_cache['datasets'] = load_dataset(config)
        _, _, test_dataset = dataset_cache['datasets']
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        model = BaseGNN(
            in_channels=config.model.in_channels,
            hidden_channels=config.model.hidden_channels,
            out_channels=config.model.out_channels,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout,
            model_type=config.model.model_type,
            spatial_dim=config.model.spatial_dim,
            num_heads=config.model.num_heads,
            num_gaussians=config.model.num_gaussians,
            num_spherical=config.model.num_spherical,
            cutoff=config.model.cutoff,
            update_coords=config.model.update_coords,
            max_ell=config.model.max_ell,
            num_degrees=config.model.num_degrees,
            use_pos=config.model.use_pos,
        ).to(device)
        load_checkpoint(model, None, best_files[0], device=device)

        eq_losses = {k: v.to(device) for k, v in initialize_equivariance_losses(config).items()}
        if not eq_losses:
            print(f"  skip {run_name} (no symmetry groups configured)")
            continue

        eq_total, per_group = unweighted_eq_error(model, test_loader, eq_losses, config, device)

        with open(metrics_file) as f:
            metrics = json.load(f)
        metrics['test/eq_loss_unweighted'] = eq_total
        for key, value in per_group.items():
            metrics[f"test/unweighted/{key}"] = value
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        groups_str = ' '.join(f"{k.replace('eq_loss/', '').replace('_total', '')}={v:.2e}"
                              for k, v in sorted(per_group.items()))
        print(f"  {run_name}: unweighted eq={eq_total:.4e} ({groups_str})")

    # Clean up only brand-new empty dirs created by ExperimentConfig.load().
    for root in ('checkpoints', 'outputs'):
        for d in glob.glob(os.path.join(root, '*_*')):
            if d in pre_existing:
                continue
            try:
                if not os.listdir(d):
                    shutil.rmtree(d)
            except OSError:
                pass


if __name__ == '__main__':
    main()
