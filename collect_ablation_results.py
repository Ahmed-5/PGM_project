"""Aggregate group x schedule ablation runs into a table and CSV.

Scans ``checkpoints/<pattern>_<timestamp>/`` for run configs and the matching
``outputs/<pattern>_<timestamp>/test_metrics.json`` written by train.py, keeps
the latest run per experiment name, and prints a compact comparison table
(test MAE/RMSE/R2 + equivariance loss) plus an optional CSV.

Usage:
    python collect_ablation_results.py --pattern 'QM9_*'
    python collect_ablation_results.py --pattern 'QM9_*' --csv results/group_schedule_ablation.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from typing import Any

RUN_DIR_RE = re.compile(r'^(?P<exp>.+)_(?P<ts>\d{8}_\d{6})$')


def find_runs(pattern: str, checkpoints_root: str = 'checkpoints',
              outputs_root: str = 'outputs') -> list[dict[str, Any]]:
    """Return one record per experiment (latest timestamp wins)."""
    runs: dict[str, dict[str, Any]] = {}
    for ckpt_dir in sorted(glob.glob(os.path.join(checkpoints_root, pattern))):
        run_name = os.path.basename(ckpt_dir)
        m = RUN_DIR_RE.match(run_name)
        if not m:
            continue
        exp_name, timestamp = m.group('exp'), m.group('ts')
        config_files = glob.glob(os.path.join(ckpt_dir, '*_config.json'))
        metrics_file = os.path.join(outputs_root, run_name, 'test_metrics.json')
        if not config_files or not os.path.exists(metrics_file):
            continue  # run still in progress or failed before test eval
        with open(config_files[0]) as f:
            cfg = json.load(f)
        with open(metrics_file) as f:
            metrics = json.load(f)
        if exp_name not in runs or timestamp > runs[exp_name]['timestamp']:
            runs[exp_name] = {
                'experiment': exp_name,
                'timestamp': timestamp,
                'model': cfg['model']['model_type'],
                'use_pos': cfg['model'].get('use_pos', False),
                'groups': ','.join(cfg['equivariance']['symmetry_groups']),
                'strategy': cfg['equivariance']['layer_weight_strategy'],
                'alpha_0': cfg['scheduler']['alpha_0'],
                'stochastic_p': cfg['equivariance']['stochastic_probability'],
                'epochs_trained': None,  # filled from checkpoint name if needed
                'test_loss': metrics.get('test/loss'),
                'test_task_loss': metrics.get('test/task_loss'),
                'test_eq_loss': metrics.get('test/eq_loss_total'),
                'test_MAE': metrics.get('test/MAE'),
                'test_RMSE': metrics.get('test/RMSE'),
                'test_R2': metrics.get('test/R2'),
                'test_eq_unweighted': metrics.get('test/eq_loss_unweighted'),
            }
    return list(runs.values())


def arm_of(run: dict[str, Any]) -> str:
    """Human-readable ablation arm label."""
    if run['alpha_0'] == 0.0 or run['stochastic_p'] == 0.0:
        return 'baseline (no eq loss)'
    return f"{run['strategy']} | {run['groups']}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pattern', default='QM9_*',
                        help="Experiment-name glob prefix (default 'QM9_*')")
    parser.add_argument('--csv', default=None, help='Optional CSV output path')
    args = parser.parse_args()

    pattern = args.pattern
    runs = find_runs(pattern)
    if not runs:
        print(f"No completed runs found for pattern '{args.pattern}'")
        return

    runs.sort(key=lambda r: (r['model'], r['experiment']))

    print('=' * 116)
    print(f"{'experiment':<34} {'model':<6} {'ablation arm':<40} "
          f"{'MAE':>8} {'RMSE':>8} {'R2':>7} {'eq(w)':>8} {'eq(unw)':>8}")
    print('-' * 116)
    for r in runs:
        mae = f"{r['test_MAE']:.4f}" if r['test_MAE'] is not None else '-'
        rmse = f"{r['test_RMSE']:.4f}" if r['test_RMSE'] is not None else '-'
        r2 = f"{r['test_R2']:.4f}" if r['test_R2'] is not None else '-'
        eq = f"{r['test_eq_loss']:.2e}" if r['test_eq_loss'] is not None else '-'
        equ = f"{r['test_eq_unweighted']:.2e}" if r.get('test_eq_unweighted') is not None else '-'
        print(f"{r['experiment']:<34} {r['model']:<6} {arm_of(r):<40} "
              f"{mae:>8} {rmse:>8} {r2:>7} {eq:>8} {equ:>8}")
    print('=' * 116)

    # Best per model by test MAE
    by_model: dict[str, list[dict[str, Any]]] = {}
    for r in runs:
        by_model.setdefault(r['model'], []).append(r)
    for model, group in by_model.items():
        valid = [r for r in group if r['test_MAE'] is not None]
        if valid:
            best = min(valid, key=lambda r: r['test_MAE'])
            print(f"Best {model}: {best['experiment']} "
                  f"(MAE {best['test_MAE']:.4f}, arm: {arm_of(best)})")

    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or '.', exist_ok=True)
        with open(args.csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(runs[0].keys()) + ['arm'])
            writer.writeheader()
            for r in runs:
                writer.writerow({**r, 'arm': arm_of(r)})
        print(f"CSV written to {args.csv}")


if __name__ == '__main__':
    main()
