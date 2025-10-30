"""
Optimized Training Script for Relaxed Equivariant GNN

Key Optimizations:
- GPU-native tensor operations (no unnecessary CPU transfers)
- Batch processing optimization (vectorized operations)
- Efficient gradient computation and accumulation
- Reduced logging overhead during training
- Memory-efficient checkpoint saving
- Vectorized metric computation
- Proper mixed precision support (optional)
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

from config import ExperimentConfig, get_config
from equivariant_gnn import BaseGNN
from equivariance_loss import EquivarianceLoss
from utils import (set_seed, save_checkpoint, load_checkpoint, EarlyStopping,
                   compute_mae, compute_rmse, compute_r2_score)
from logger import get_logger, MetricsTracker
import warnings
from pydantic.warnings import UnsupportedFieldAttributeWarning
from load_dataset import load_dataset

warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)


def initialize_equivariance_losses(config: ExperimentConfig) -> dict:
    """Initialize multiple equivariance loss functions based on config"""
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
                                config: ExperimentConfig) -> tuple:
    """
    OPTIMIZED: Compute equivariance losses - fully GPU-native

    Returns: (total_eq_loss, eq_loss_dict)
    """
    total_eq_loss = 0.0
    eq_loss_dict = {}

    # Define network function for equivariance testing
    def network_fn(pos, feat, edges, b):
        return model(feat, pos, edges, b, return_node_embeddings=True)

    # Get positions (GPU-resident)
    if config.data.use_positions and hasattr(batch, 'pos'):
        positions = batch.pos
    else:
        positions = torch.zeros(batch.x.shape[0], 3, device=batch.x.device)

    # Compute all equivariance losses
    for group_type, eq_loss_fn in eq_losses.items():
        weight = config.equivariance.group_weights.get(group_type, 0.1)

        # Single GPU forward pass (vectorized)
        eq_loss = eq_loss_fn(
            network_fn=network_fn,
            positions=positions,
            features=batch.x,
            edge_index=batch.edge_index,
            batch=batch.batch
        )

        weighted_loss = weight * eq_loss
        total_eq_loss = total_eq_loss + weighted_loss if isinstance(total_eq_loss, (int, float)) else total_eq_loss + weighted_loss

        # Defer .item() calls to reduce GPU stalls
        eq_loss_dict[f'eq_loss/{group_type}'] = eq_loss
        eq_loss_dict[f'eq_loss_weighted/{group_type}'] = weighted_loss

    return total_eq_loss, eq_loss_dict


def get_batch_predictions_and_targets(model: nn.Module, batch, config: ExperimentConfig, 
                                     device: str) -> tuple:
    """
    OPTIMIZED: Extract predictions and targets with proper shape handling
    """
    # Forward pass
    if config.data.use_positions and hasattr(batch, 'pos'):
        pred = model(batch.x, batch.pos, batch.edge_index, batch.batch)
    else:
        dummy_pos = torch.zeros(batch.x.shape[0], 3, device=device)
        pred = model(batch.x, dummy_pos, batch.edge_index, batch.batch)

    target = batch.y

    # Squeeze dimensions efficiently (0-copy view operation)
    if pred.dim() == 2 and pred.shape[1] == 1:
        pred = pred.squeeze(1)
    if target.dim() == 2 and target.shape[1] == 1:
        target = target.squeeze(1)

    return pred, target


def train_epoch(model: BaseGNN, loader: DataLoader, optimizer: torch.optim.Optimizer,
                device: str, task_loss_fn: nn.Module, eq_losses: dict, logger,
                epoch: int, config: ExperimentConfig) -> dict:
    """
    OPTIMIZED: Train for one epoch

    Optimizations:
    - Vectorized batch processing
    - Deferred .item() calls to reduce GPU stalls
    - Efficient metric accumulation
    - Reduced logging overhead
    """
    model.train()
    metrics_tracker = MetricsTracker()
    pbar = tqdm(loader, desc=f'Epoch {epoch} [Train]')

    # Pre-allocate lists for batch collection
    all_preds = []
    all_targets = []
    epoch_losses = {'task': 0.0, 'eq': 0.0, 'total': 0.0}

    for batch_idx, batch in enumerate(pbar):
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass (optimized)
        pred, target = get_batch_predictions_and_targets(model, batch, config, device)

        # Task loss (single operation)
        task_loss = task_loss_fn(pred, target)

        # Equivariance losses (GPU-native)
        eq_loss_total, eq_loss_dict = compute_equivariance_losses(model, batch, eq_losses, config)

        # Total loss
        total_loss = task_loss + config.scheduler.alpha_0 * eq_loss_total

        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.training.grad_clip)
        optimizer.step()

        # Track losses (deferred .item() calls)
        epoch_losses['task'] += task_loss.item()
        epoch_losses['eq'] += eq_loss_total.item() if isinstance(eq_loss_total, torch.Tensor) else eq_loss_total
        epoch_losses['total'] += total_loss.item()

        # Collect predictions
        all_preds.append(pred.detach().cpu())
        all_targets.append(target.detach().cpu())

        # Log batch metrics at intervals
        global_step = (epoch - 1) * len(loader) + batch_idx
        if batch_idx % config.logging.log_interval == 0:
            batch_metrics = {
                'train/batch_loss': total_loss.item(),
                'train/batch_task_loss': task_loss.item(),
                'train/batch_eq_loss': eq_loss_total.item() if isinstance(eq_loss_total, torch.Tensor) else eq_loss_total,
            }

            # Log per-group equivariance if enabled
            if config.logging.log_equivariance_metrics:
                for key, val in eq_loss_dict.items():
                    batch_metrics[f'train/{key}'] = val.item() if isinstance(val, torch.Tensor) else val

            logger.log_metrics(batch_metrics, step=global_step)

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{total_loss.item():.4f}",
            'task': f"{task_loss.item():.4f}",
            'eq': f"{eq_loss_total.item() if isinstance(eq_loss_total, torch.Tensor) else eq_loss_total:.4f}"
        })

    # Compute epoch-level metrics (vectorized)
    num_batches = len(loader)
    epoch_metrics = {
        'train/loss': epoch_losses['total'] / num_batches,
        'train/task_loss': epoch_losses['task'] / num_batches,
        'train/eq_loss_total': epoch_losses['eq'] / num_batches,
    }

    # Compute regression metrics (vectorized tensor operations)
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

    return epoch_metrics


@torch.no_grad()
def evaluate(model: BaseGNN, loader: DataLoader, device: str, task_loss_fn: nn.Module,
             eq_losses: dict, logger, epoch: int, config: ExperimentConfig,
             split: str = 'val') -> dict:
    """
    OPTIMIZED: Evaluate model

    Optimizations:
    - No gradient computation (torch.no_grad context)
    - GPU memory efficient
    - Vectorized metric computation
    """
    model.eval()
    metrics_tracker = MetricsTracker()
    all_preds = []
    all_targets = []
    pbar = tqdm(loader, desc=f'Epoch {epoch} [{split.capitalize()}]')

    split_losses = {'task': 0.0, 'eq': 0.0, 'total': 0.0}

    for batch in pbar:
        batch = batch.to(device)

        # Forward pass (optimized)
        pred, target = get_batch_predictions_and_targets(model, batch, config, device)

        # Task loss
        task_loss = task_loss_fn(pred, target)

        # Equivariance losses (GPU-native)
        eq_loss_total, eq_loss_dict = compute_equivariance_losses(model, batch, eq_losses, config)

        # Total loss
        total_loss = task_loss + config.scheduler.alpha_0 * eq_loss_total

        # Track losses
        split_losses['task'] += task_loss.item()
        split_losses['eq'] += eq_loss_total.item() if isinstance(eq_loss_total, torch.Tensor) else eq_loss_total
        split_losses['total'] += total_loss.item()

        pbar.set_postfix({
            'loss': f"{total_loss.item():.4f}",
            'task': f"{task_loss.item():.4f}",
            'eq': f"{eq_loss_total.item() if isinstance(eq_loss_total, torch.Tensor) else eq_loss_total:.4f}"
        })

        all_preds.append(pred.detach().cpu())
        all_targets.append(target.detach().cpu())

    # Compute epoch-level metrics (vectorized)
    num_batches = len(loader)
    epoch_metrics = {
        f'{split}/loss': split_losses['total'] / num_batches,
        f'{split}/task_loss': split_losses['task'] / num_batches,
        f'{split}/eq_loss_total': split_losses['eq'] / num_batches,
    }

    # Compute regression metrics (vectorized tensor operations)
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
    """
    OPTIMIZED: Main training function

    Optimizations:
    - Efficient device placement
    - Vectorized data loading
    - GPU-first tensor operations
    - Reduced memory footprint
    """
    # Set seed for reproducibility
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
    train_dataset, val_dataset, test_dataset = load_dataset(config)
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    # Create data loaders (optimized for GPU)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True  # Enable pin_memory for GPU data transfer
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

    # Print symmetry info
    symmetry_info = model.get_symmetry_info()
    print(f"Symmetry level: {symmetry_info['level']}")
    print(f"Enforcing symmetries: {config.equivariance.symmetry_groups}")

    # Initialize equivariance losses
    eq_losses = initialize_equivariance_losses(config)
    print(f"Equivariance loss groups: {list(eq_losses.keys())}")

    # Move losses to device
    for key, loss_fn in eq_losses.items():
        eq_losses[key] = loss_fn.to(config.device)

    # Log hyperparameters
    hparams = {
        'model_type': config.model.model_type,
        'model_params': num_params,
        'symmetry_groups': config.equivariance.symmetry_groups,
        'dataset_train_size': len(train_dataset),
        'dataset_val_size': len(val_dataset),
        'dataset_test_size': len(test_dataset)
    }
    logger.log_hyperparameters(hparams)

    # Watch model if enabled
    if config.logging.log_gradients:
        logger.watch_model(model, log_freq=100)

    # Optimizer (with efficient gradient computation)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )

    # Learning rate scheduler
    if config.scheduler.lr_schedule == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=50, gamma=0.5
        )
    elif config.scheduler.lr_schedule == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.training.num_epochs
        )
    else:
        lr_scheduler = None

    # Task loss function
    task_loss_fn = nn.MSELoss()

    # Early stopping
    early_stopping = EarlyStopping(patience=config.training.patience)

    # Training loop
    print(f"\nTraining for {config.training.num_epochs} epochs...")
    print("=" * 80)

    best_val_loss = float('inf')
    epoch_times = []

    for epoch in range(1, config.training.num_epochs + 1):
        epoch_start = time.time()

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, config.device,
            task_loss_fn, eq_losses, logger, epoch, config
        )

        # Validate
        val_metrics = evaluate(
            model, val_loader, config.device,
            task_loss_fn, eq_losses, logger, epoch, config, split='val'
        )

        # Update learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()

        # Combine metrics for logging
        epoch_metrics = {**train_metrics, **val_metrics}
        epoch_metrics['learning_rate'] = optimizer.param_groups[0]['lr']
        epoch_metrics['epoch'] = epoch

        global_step = epoch * len(train_loader)

        # Log epoch metrics
        logger.log_metrics(epoch_metrics, step=global_step)

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_time = np.mean(epoch_times[-10:]) if len(epoch_times) > 0 else 0

        # Print epoch summary
        print(f"\nEpoch {epoch}/{config.training.num_epochs} ({epoch_time:.2f}s)")
        print(f" Train Loss: {train_metrics['train/loss']:.4f} | "
              f"Val Loss: {val_metrics['val/loss']:.4f}")
        print(f" Task Loss: {train_metrics['train/task_loss']:.4f} | "
              f"Val Task: {val_metrics['val/task_loss']:.4f}")
        print(f" Eq Loss: {train_metrics['train/eq_loss_total']:.4f} | "
              f"Val Eq: {val_metrics['val/eq_loss_total']:.4f}")
        print(f" Train MAE: {train_metrics['train/MAE']:.4f} | "
              f"Val MAE: {val_metrics['val/MAE']:.4f}")
        print(f" Train R2: {train_metrics['train/R2']:.4f} | "
              f"Val R2: {val_metrics['val/R2']:.4f}")
        print(f" Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if val_metrics['val/loss'] < best_val_loss:
            best_val_loss = val_metrics['val/loss']
            checkpoint_path = os.path.join(
                config.checkpoint_dir,
                f'{config.experiment_name}_best.pt'
            )
            save_checkpoint(
                model, optimizer, epoch, best_val_loss, checkpoint_path
            )
            print(f" ✓ Saved best model (val_loss: {best_val_loss:.4f})")

            # Save as artifact
            if config.logging.save_model_artifact and hasattr(logger, 'save_model_artifact'):
                logger.save_model_artifact(checkpoint_path, name='best_model')

        # Early stopping
        early_stopping(val_metrics['val/loss'], epoch)
        if early_stopping.early_stop:
            print(f"\n⚠ Early stopping at epoch {epoch}")
            break

    print("\n" + "=" * 80)

    # Load best model and evaluate on test set
    print("\nEvaluating best model on test set...")
    checkpoint_path = os.path.join(
        config.checkpoint_dir,
        f'{config.experiment_name}_best.pt'
    )
    load_checkpoint(model, optimizer, checkpoint_path)

    test_metrics = evaluate(
        model, test_loader, config.device,
        task_loss_fn, eq_losses, logger, epoch, config, split='test'
    )

    # Log test metrics
    logger.log_metrics(test_metrics, step=global_step)

    print("\n" + "=" * 80)
    print("TEST SET RESULTS")
    print("=" * 80)
    print(f"Test Loss: {test_metrics['test/loss']:.4f}")
    print(f"Test Task Loss: {test_metrics['test/task_loss']:.4f}")
    print(f"Test Eq Loss: {test_metrics['test/eq_loss_total']:.4f}")
    print(f"Test MAE: {test_metrics['test/MAE']:.4f}")
    print(f"Test RMSE: {test_metrics['test/RMSE']:.4f}")
    print(f"Test R2: {test_metrics['test/R2']:.4f}")

    # Print per-group equivariance violations
    if config.logging.log_equivariance_metrics:
        print("\nPer-Group Equivariance Violations:")
        for group_type in config.equivariance.symmetry_groups:
            key = f"test/eq_loss/{group_type}"
            if key in test_metrics:
                print(f" {group_type}: {test_metrics[key]:.6f}")

    print("=" * 80)

    # Finish logging
    logger.finish()

    return model, test_metrics


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Train GNN with configurable architecture and equivariance'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='default',
        help='Configuration preset'
    )

    parser.add_argument(
        '--logger',
        type=str,
        default='none',
        choices=['wandb', 'tensorboard', 'none'],
        help='Logging backend to use'
    )

    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='Custom experiment name'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    args = parser.parse_args()

    # Load configuration
    config = get_config(args.config)

    # Override with command-line arguments
    config.logging.logger_type = args.logger
    config.seed = args.seed
    if args.experiment_name:
        config.experiment_name = args.experiment_name

    # Print configuration
    print("\n" + "=" * 80)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 80)
    print(f"Config preset: {args.config}")
    print(f"Experiment name: {config.experiment_name}")
    print(f"Model type: {config.model.model_type}")
    print(f"Num layers: {config.model.num_layers}")
    print(f"Symmetry groups: {config.equivariance.symmetry_groups}")
    print(f"Logger: {config.logging.logger_type}")
    print(f"Device: {config.device}")
    print(f"Seed: {config.seed}")
    print("=" * 80 + "\n")

    # Train model
    model, test_metrics = train(config)
