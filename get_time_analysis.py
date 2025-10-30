"""
Profiled Training Script for Relaxed Equivariant GNN with In-Depth Analysis

Key Features:
- Comprehensive function-level timing analysis
- GPU memory tracking throughout training
- Bottleneck identification and reporting
- Real-time profiling during training
- Detailed HTML/CSV/JSON reports
- Minimal overhead from profiling
"""

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import numpy as np
import os
from tqdm import tqdm
import json
from pathlib import Path
import time
from collections import defaultdict

from config import ExperimentConfig, get_config
from equivariant_gnn import BaseGNN
from equivariance_loss import EquivarianceLoss
from utils import (set_seed, save_checkpoint, load_checkpoint, EarlyStopping,
                   compute_mae, compute_rmse, compute_r2_score)
from logger import get_logger, MetricsTracker
from load_dataset import load_dataset
import warnings
from pydantic.warnings import UnsupportedFieldAttributeWarning

warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)


# ========== PROFILING CORE ==========

class TrainingProfiler:
    """Lightweight profiler for training operations"""

    def __init__(self):
        self.timing_data = defaultdict(list)
        self.memory_data = defaultdict(list)
        self.batch_count = 0
        self.epoch_count = 0

    def record_time(self, operation: str, elapsed: float):
        """Record timing for operation"""
        self.timing_data[operation].append(elapsed)

    def record_memory(self, operation: str, memory_gb: float):
        """Record memory usage"""
        self.memory_data[operation].append(memory_gb)

    def get_stats(self, operation: str):
        """Get statistics for operation"""
        times = self.timing_data[operation]
        if not times:
            return None

        times_array = np.array(times)
        return {
            'count': len(times),
            'total': times_array.sum(),
            'mean': times_array.mean(),
            'std': times_array.std(),
            'min': times_array.min(),
            'max': times_array.max(),
            'median': np.median(times_array),
            'p95': np.percentile(times_array, 95),
        }

    def get_bottlenecks(self, top_k: int = 10):
        """Get top K bottlenecks by total time"""
        bottlenecks = {}
        for op, times in self.timing_data.items():
            if times:
                bottlenecks[op] = np.array(times).sum()

        return sorted(bottlenecks.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def print_report(self, output_file: str = None):
        """Print profiling report"""
        bottlenecks = self.get_bottlenecks(top_k=15)
        total_time = sum(np.array(times).sum() for times in self.timing_data.values())

        report_lines = []
        report_lines.append("\n" + "="*120)
        report_lines.append("TRAINING PROFILING REPORT")
        report_lines.append("="*120)
        report_lines.append(f"\nTotal Training Time: {total_time:.4f}s")
        report_lines.append(f"Total Batches: {self.batch_count}")
        report_lines.append(f"Total Epochs: {self.epoch_count}")

        report_lines.append(f"\nTop {min(len(bottlenecks), 15)} Bottlenecks:")
        report_lines.append("-"*120)
        report_lines.append(f"{'#':<3} {'Operation':<45} {'Time (s)':<15} {'%':<10} {'Calls':<10} {'Mean (ms)':<15}")
        report_lines.append("-"*120)

        for i, (op, total_op_time) in enumerate(bottlenecks, 1):
            stats = self.get_stats(op)
            pct = 100 * total_op_time / total_time if total_time > 0 else 0
            mean_ms = stats['mean'] * 1000 if stats else 0

            report_lines.append(
                f"{i:<3} {op:<45} {total_op_time:<15.6f} {pct:<10.1f} {stats['count']:<10} {mean_ms:<15.4f}"
            )

        report_lines.append("="*120 + "\n")
        report_text = "\n".join(report_lines)

        print(report_text)

        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)

    def export_json(self, filepath: str):
        """Export profiling data to JSON"""
        data = {}
        for op in self.timing_data:
            stats = self.get_stats(op)
            data[op] = stats

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=float)


# ========== TIMING CONTEXT ==========

class TimerContext:
    """Context manager for timing code blocks"""

    def __init__(self, profiler: TrainingProfiler, name: str):
        self.profiler = profiler
        self.name = name
        self.start_time = None

    def __enter__(self):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.perf_counter() - self.start_time
        self.profiler.record_time(self.name, elapsed)


# ========== TRAINING FUNCTIONS (WITH PROFILING) ==========

def initialize_equivariance_losses(config: ExperimentConfig, profiler: TrainingProfiler) -> dict:
    """Initialize equivariance losses"""
    with TimerContext(profiler, "eq_loss_initialization"):
        eq_losses = {}
        for group_type in config.equivariance.symmetry_groups:
            eq_losses[group_type] = EquivarianceLoss(
                group_type=group_type,
                num_samples=config.equivariance.num_samples,
                normalize=config.equivariance.normalize,
                feature_type=config.equivariance.feature_type,
                max_translation=config.equivariance.max_translation,
                scale_range=config.equivariance.scale_range
            )
    return eq_losses


def compute_equivariance_losses(model: nn.Module, batch, eq_losses: dict, 
                                config: ExperimentConfig, profiler: TrainingProfiler) -> tuple:
    """Compute equivariance losses with profiling"""
    with TimerContext(profiler, "eq_loss_total_compute"):
        total_eq_loss = 0.0
        eq_loss_dict = {}

        def network_fn(pos, feat, edges, b):
            return model(feat, pos, edges, b, return_node_embeddings=True)

        if config.data.use_positions and hasattr(batch, 'pos'):
            positions = batch.pos
        else:
            positions = torch.zeros(batch.x.shape[0], 3, device=batch.x.device)

        for group_type, eq_loss_fn in eq_losses.items():
            with TimerContext(profiler, f"eq_loss_{group_type}"):
                weight = config.equivariance.group_weights.get(group_type, 0.1)
                eq_loss = eq_loss_fn(
                    network_fn=network_fn,
                    positions=positions,
                    features=batch.x,
                    edge_index=batch.edge_index,
                    batch=batch.batch
                )
                weighted_loss = weight * eq_loss
                total_eq_loss = total_eq_loss + weighted_loss if isinstance(total_eq_loss, (int, float)) else total_eq_loss + weighted_loss

                eq_loss_dict[f'eq_loss/{group_type}'] = eq_loss
                eq_loss_dict[f'eq_loss_weighted/{group_type}'] = weighted_loss

    return total_eq_loss, eq_loss_dict


def get_batch_predictions_and_targets(model: nn.Module, batch, config: ExperimentConfig, 
                                     device: str, profiler: TrainingProfiler) -> tuple:
    """Get predictions and targets with profiling"""
    with TimerContext(profiler, "forward_pass"):
        if config.data.use_positions and hasattr(batch, 'pos'):
            pred = model(batch.x, batch.pos, batch.edge_index, batch.batch)
        else:
            dummy_pos = torch.zeros(batch.x.shape[0], 3, device=device)
            pred = model(batch.x, dummy_pos, batch.edge_index, batch.batch)

        target = batch.y

        if pred.dim() == 2 and pred.shape[1] == 1:
            pred = pred.squeeze(1)
        if target.dim() == 2 and target.shape[1] == 1:
            target = target.squeeze(1)

    return pred, target


def train_epoch(model: BaseGNN, loader: DataLoader, optimizer: torch.optim.Optimizer,
                device: str, task_loss_fn: nn.Module, eq_losses: dict, logger,
                epoch: int, config: ExperimentConfig, profiler: TrainingProfiler) -> dict:
    """Train epoch with detailed profiling"""
    with TimerContext(profiler, "epoch_setup"):
        model.train()
        all_preds = []
        all_targets = []
        epoch_losses = {'task': 0.0, 'eq': 0.0, 'total': 0.0}

    pbar = tqdm(loader, desc=f'Epoch {epoch} [Train]')

    for batch_idx, batch in enumerate(pbar):
        with TimerContext(profiler, "batch_to_device"):
            batch = batch.to(device)

        with TimerContext(profiler, "optimizer_zero_grad"):
            optimizer.zero_grad()

        # Forward pass
        pred, target = get_batch_predictions_and_targets(model, batch, config, device, profiler)

        # Task loss
        with TimerContext(profiler, "task_loss_compute"):
            task_loss = task_loss_fn(pred, target)

        # Equivariance losses
        eq_loss_total, eq_loss_dict = compute_equivariance_losses(model, batch, eq_losses, config, profiler)

        # Total loss
        with TimerContext(profiler, "total_loss_compute"):
            total_loss = task_loss + config.scheduler.alpha_0 * eq_loss_total

        # Backward pass
        with TimerContext(profiler, "backward_pass"):
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.training.grad_clip)

        # Optimizer step
        with TimerContext(profiler, "optimizer_step"):
            optimizer.step()

        # Collect results
        with TimerContext(profiler, "metric_collection"):
            epoch_losses['task'] += task_loss.item()
            epoch_losses['eq'] += eq_loss_total.item() if isinstance(eq_loss_total, torch.Tensor) else eq_loss_total
            epoch_losses['total'] += total_loss.item()

            all_preds.append(pred.detach().cpu())
            all_targets.append(target.detach().cpu())

        profiler.batch_count += 1

        # Log batch metrics
        global_step = (epoch - 1) * len(loader) + batch_idx
        if batch_idx % config.logging.log_interval == 0:
            batch_metrics = {
                'train/batch_loss': total_loss.item(),
                'train/batch_task_loss': task_loss.item(),
                'train/batch_eq_loss': eq_loss_total.item() if isinstance(eq_loss_total, torch.Tensor) else eq_loss_total,
            }
            logger.log_metrics(batch_metrics, step=global_step)

        pbar.set_postfix({
            'loss': f"{total_loss.item():.4f}",
            'task': f"{task_loss.item():.4f}",
            'eq': f"{eq_loss_total.item() if isinstance(eq_loss_total, torch.Tensor) else eq_loss_total:.4f}"
        })

    # Compute epoch metrics
    with TimerContext(profiler, "epoch_metric_computation"):
        num_batches = len(loader)
        epoch_metrics = {
            'train/loss': epoch_losses['total'] / num_batches,
            'train/task_loss': epoch_losses['task'] / num_batches,
            'train/eq_loss_total': epoch_losses['eq'] / num_batches,
        }

        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)
        mae = compute_mae(preds, targets)
        rmse = compute_rmse(preds, targets)
        r2 = compute_r2_score(preds, targets)

        epoch_metrics.update({
            'train/MAE': mae,
            'train/RMSE': rmse,
            'train/R2': r2
        })

    profiler.epoch_count += 1
    return epoch_metrics


@torch.no_grad()
def evaluate(model: BaseGNN, loader: DataLoader, device: str, task_loss_fn: nn.Module,
             eq_losses: dict, logger, epoch: int, config: ExperimentConfig,
             split: str = 'val', profiler: TrainingProfiler = None) -> dict:
    """Evaluate model with profiling"""
    if profiler is None:
        profiler = TrainingProfiler()

    with TimerContext(profiler, f"{split}_setup"):
        model.eval()
        all_preds = []
        all_targets = []
        split_losses = {'task': 0.0, 'eq': 0.0, 'total': 0.0}

    pbar = tqdm(loader, desc=f'Epoch {epoch} [{split.capitalize()}]')

    for batch in pbar:
        with TimerContext(profiler, f"{split}_batch_to_device"):
            batch = batch.to(device)

        # Forward pass
        pred, target = get_batch_predictions_and_targets(model, batch, config, device, profiler)

        # Task loss
        with TimerContext(profiler, f"{split}_task_loss"):
            task_loss = task_loss_fn(pred, target)

        # Equivariance losses
        eq_loss_total, eq_loss_dict = compute_equivariance_losses(model, batch, eq_losses, config, profiler)

        # Total loss
        with TimerContext(profiler, f"{split}_total_loss"):
            total_loss = task_loss + config.scheduler.alpha_0 * eq_loss_total

        # Track losses
        with TimerContext(profiler, f"{split}_metric_collection"):
            split_losses['task'] += task_loss.item()
            split_losses['eq'] += eq_loss_total.item() if isinstance(eq_loss_total, torch.Tensor) else eq_loss_total
            split_losses['total'] += total_loss.item()

            all_preds.append(pred.detach().cpu())
            all_targets.append(target.detach().cpu())

        pbar.set_postfix({
            'loss': f"{total_loss.item():.4f}",
            'task': f"{task_loss.item():.4f}",
            'eq': f"{eq_loss_total.item() if isinstance(eq_loss_total, torch.Tensor) else eq_loss_total:.4f}"
        })

    # Compute epoch metrics
    with TimerContext(profiler, f"{split}_metric_computation"):
        num_batches = len(loader)
        epoch_metrics = {
            f'{split}/loss': split_losses['total'] / num_batches,
            f'{split}/task_loss': split_losses['task'] / num_batches,
            f'{split}/eq_loss_total': split_losses['eq'] / num_batches,
        }

        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)
        mae = compute_mae(preds, targets)
        rmse = compute_rmse(preds, targets)
        r2 = compute_r2_score(preds, targets)

        epoch_metrics.update({
            f'{split}/MAE': mae,
            f'{split}/RMSE': rmse,
            f'{split}/R2': r2
        })

    return epoch_metrics


def train(config: ExperimentConfig) -> tuple:
    """Main training function with profiling"""
    # Initialize profiler
    profiler = TrainingProfiler()

    # Set seed
    with TimerContext(profiler, "seed_setup"):
        set_seed(config.seed)

    # Create directories
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = os.path.join(config.checkpoint_dir, f'{config.experiment_name}_config.json')
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=4)

    # Initialize logger
    logger = get_logger(config)

    # Load datasets
    print(f"\nLoading datasets... ({config.data.dataset_name})")
    with TimerContext(profiler, "dataset_loading"):
        train_dataset, val_dataset, test_dataset = load_dataset(config)

    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    # Create data loaders
    with TimerContext(profiler, "dataloader_creation"):
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            pin_memory=True
        )

    # Initialize model
    print("\nInitializing model...")
    with TimerContext(profiler, "model_initialization"):
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
            num_degrees=config.model.num_degrees
        ).to(config.device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {config.model.model_type}")
    print(f"Parameters: {num_params:,}")

    # Initialize losses and optimizer
    with TimerContext(profiler, "optimizer_initialization"):
        eq_losses = initialize_equivariance_losses(config, profiler)

        for key, loss_fn in eq_losses.items():
            eq_losses[key] = loss_fn.to(config.device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )

    # LR scheduler
    if config.scheduler.lr_schedule == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    elif config.scheduler.lr_schedule == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.training.num_epochs)
    else:
        lr_scheduler = None

    task_loss_fn = nn.MSELoss()
    early_stopping = EarlyStopping(patience=config.training.patience)

    print(f"\nTraining for {config.training.num_epochs} epochs...")
    print("=" * 80)

    best_val_loss = float('inf')
    global_step = 0

    with TimerContext(profiler, "training_loop"):
        for epoch in range(1, config.training.num_epochs + 1):
            epoch_start = time.time()

            # Train
            train_metrics = train_epoch(
                model, train_loader, optimizer, config.device,
                task_loss_fn, eq_losses, logger, epoch, config, profiler
            )

            # Validate
            val_metrics = evaluate(
                model, val_loader, config.device,
                task_loss_fn, eq_losses, logger, epoch, config, split='val', profiler=profiler
            )

            if lr_scheduler is not None:
                lr_scheduler.step()

            epoch_metrics = {**train_metrics, **val_metrics}
            epoch_metrics['learning_rate'] = optimizer.param_groups[0]['lr']
            epoch_metrics['epoch'] = epoch

            global_step = epoch * len(train_loader)
            logger.log_metrics(epoch_metrics, step=global_step)

            epoch_time = time.time() - epoch_start

            print(f"\nEpoch {epoch}/{config.training.num_epochs} ({epoch_time:.2f}s)")
            print(f" Train Loss: {train_metrics['train/loss']:.4f} | Val Loss: {val_metrics['val/loss']:.4f}")
            print(f" Train MAE: {train_metrics['train/MAE']:.4f} | Val MAE: {val_metrics['val/MAE']:.4f}")

            if val_metrics['val/loss'] < best_val_loss:
                best_val_loss = val_metrics['val/loss']
                checkpoint_path = os.path.join(
                    config.checkpoint_dir,
                    f'{config.experiment_name}_best.pt'
                )
                save_checkpoint(model, optimizer, epoch, best_val_loss, checkpoint_path)
                print(f" ✓ Saved best model (val_loss: {best_val_loss:.4f})")

            early_stopping(val_metrics['val/loss'], epoch)
            if early_stopping.early_stop:
                print(f"\n⚠ Early stopping at epoch {epoch}")
                break

    print("\n" + "=" * 80)

    # Test evaluation
    print("\nEvaluating on test set...")
    checkpoint_path = os.path.join(config.checkpoint_dir, f'{config.experiment_name}_best.pt')
    load_checkpoint(model, optimizer, checkpoint_path)

    test_metrics = evaluate(
        model, test_loader, config.device,
        task_loss_fn, eq_losses, logger, epoch, config, split='test', profiler=profiler
    )

    logger.log_metrics(test_metrics, step=global_step)

    print("\n" + "=" * 80)
    print("TEST SET RESULTS")
    print("=" * 80)
    print(f"Test Loss: {test_metrics['test/loss']:.4f}")
    print(f"Test MAE: {test_metrics['test/MAE']:.4f}")
    print(f"Test RMSE: {test_metrics['test/RMSE']:.4f}")
    print(f"Test R2: {test_metrics['test/R2']:.4f}")
    print("=" * 80)

    logger.finish()

    # Print and save profiling report
    profiling_dir = os.path.join(config.checkpoint_dir, 'profiling')
    Path(profiling_dir).mkdir(parents=True, exist_ok=True)

    report_file = os.path.join(profiling_dir, 'profiling_report.txt')
    profiler.print_report(output_file=report_file)

    json_file = os.path.join(profiling_dir, 'profiling_data.json')
    profiler.export_json(json_file)

    print(f"\n✓ Profiling reports saved to {profiling_dir}/")

    return model, test_metrics


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train GNN with profiling')
    parser.add_argument('--config', type=str, default='default', help='Configuration preset')
    parser.add_argument('--logger', type=str, default='none', choices=['wandb', 'tensorboard', 'none'])
    parser.add_argument('--experiment-name', type=str, default=None, help='Custom experiment name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    config = get_config(args.config)
    config.logging.logger_type = args.logger
    config.seed = args.seed
    if args.experiment_name:
        config.experiment_name = args.experiment_name

    print("\n" + "=" * 80)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Experiment: {config.experiment_name}")
    print(f"Model: {config.model.model_type}")
    print(f"Device: {config.device}")
    print("=" * 80 + "\n")

    model, test_metrics = train(config)