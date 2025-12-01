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
from torch.amp import GradScaler, autocast
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
import math
from schedulers import DepthScheduler

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
                              config, device: str, 
                              layer_weights: torch.Tensor = None) -> tuple:  # <--- Added Arg
    """
    Compute equivariance losses with pre-calculated layer weights.
    """
    total_eq_loss = torch.tensor(0.0, device=device)
    eq_loss_dict = {}

    if not eq_losses:
        return total_eq_loss, eq_loss_dict

    # 1. Setup Data
    if config.data.use_positions and hasattr(batch, 'pos'):
        positions = batch.pos
    else:
        positions = torch.zeros(batch.x.shape[0], 3, device=device)

    # 2. Run model to get layers
    final_x, layers_x = model(batch.x, positions, batch.edge_index, batch.batch, 
                            return_layer_outputs=True, return_node_embeddings=True)
    
    num_actual_layers = len(layers_x) + 1
    
    # Safety check: Ensure weight vector matches actual model depth
    if layer_weights is None:
        layer_weights = torch.ones(num_actual_layers, device=device)
    elif len(layer_weights) != num_actual_layers:
        # Handle mismatch (e.g., if config.num_layers != actual output length)
        # Truncate or pad
        if len(layer_weights) > num_actual_layers:
             layer_weights = layer_weights[:num_actual_layers]
        else:
             # This case is rarer, but just use what we have
             pass

    # 3. Iterate Groups
    for group_type, eq_loss_fn in eq_losses.items():
        group_weight = config.equivariance.group_weights.get(group_type, 0.1)
        
        num_graphs = batch.batch.max().item() + 1
        
        # Define wrapper to match EquivarianceLoss signature (pos, x, ...) -> (out, layers)
        # and handle argument swapping for BaseGNN (x, pos, ...)
        def model_wrapper(p, f, e, b, return_layer_outputs=True):
            # BaseGNN expects (x, pos, edge_index, batch)
            return model(f, p, e, b, return_layer_outputs=True, return_node_embeddings=True)

        # Compute all losses at once (Optimized v3 API)
        # Returns: main_loss (final layer), loss_dict (intermediate layers)
        main_loss, group_layer_losses = eq_loss_fn(model_wrapper, positions, batch.x, batch.edge_index, batch.batch)
        
        group_total_loss = torch.tensor(0.0, device=device)
        
        # 4. Aggregate weighted losses
        # Intermediate layers
        for layer_idx in range(num_actual_layers - 1):
            key = f"layer_{layer_idx}"
            if key in group_layer_losses:
                l_loss = group_layer_losses[key]
                w_l = layer_weights[layer_idx]
                group_total_loss += l_loss * w_l
                eq_loss_dict[f'eq_loss/{group_type}_{key}'] = l_loss.detach()
        
        # Final layer
        final_layer_idx = num_actual_layers - 1
        w_final = layer_weights[final_layer_idx]
        group_total_loss += main_loss * w_final
        eq_loss_dict[f'eq_loss/{group_type}_layer_{final_layer_idx}'] = main_loss.detach()
            
        total_eq_loss += group_weight * group_total_loss
        eq_loss_dict[f'eq_loss/{group_type}_total'] = group_total_loss.detach()

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


def train_epoch(model, loader, optimizer, device, task_loss_fn, eq_losses, 
                logger, epoch, config, depth_scheduler) -> dict:
    model.train()
    
    # [NEW] Initialize Scaler for Mixed Precision
    use_amp = config.training.use_amp
    scaler = GradScaler(enabled=use_amp)
    
    epoch_losses = {'task': 0.0, 'eq': 0.0, 'total': 0.0}
    all_preds = []
    all_targets = []

    # DepthScheduler is now passed in
    
    pbar = tqdm(loader, desc=f'Epoch {epoch} [Train]')
    
    optimizer.zero_grad()
    

    
    for batch_idx, batch in enumerate(pbar):
        batch = batch.to(device)

        # Get the weight vector for this batch (important for learnable weights)
        layer_weights = depth_scheduler.get_all_alphas()
        
        # [NEW] Stochastic Probability Check
        # Only compute expensive equivariance loss on a subset of batches
        do_equivariance = (torch.rand(1).item() < config.equivariance.stochastic_probability)
        
        # Mixed Precision Context
        with autocast(enabled=use_amp, device_type=device.split(':')[0]):
            # 1. Forward Pass (Task)
            if config.data.use_positions and hasattr(batch, 'pos'):
                pred = model(batch.x, batch.pos, batch.edge_index, batch.batch)
            else:
                dummy_pos = torch.zeros(batch.x.shape[0], 3, device=device)
                pred = model(batch.x, dummy_pos, batch.edge_index, batch.batch)
                
            target = batch.y
            if pred.dim() > 1 and target.dim() == 1: pred = pred.squeeze()
            
            task_loss = task_loss_fn(pred, target)
            
            # 2. Equivariance Loss (Conditional)
            eq_loss_total = torch.tensor(0.0, device=device)
            eq_loss_dict = {}
            if do_equivariance and len(eq_losses) > 0:
                eq_loss_total, eq_loss_dict = compute_equivariance_losses(model, batch, eq_losses, config, device, layer_weights=layer_weights)
                
                # Scale loss up because we are only applying it p% of the time
                # This keeps the expected gradient magnitude consistent
                scale_factor = 1.0 / config.equivariance.stochastic_probability
                eq_loss_total = eq_loss_total * scale_factor
            
            # Combine losses
            alpha = config.scheduler.alpha_0
            total_loss = task_loss + (alpha * eq_loss_total)
            
            # Normalize loss for gradient accumulation
            total_loss = total_loss / config.training.accumulation_steps

            all_preds.append(pred.detach().float().cpu())
            all_targets.append(target.detach().float().cpu())

        # 3. Backward Pass (Scaled)
        scaler.scale(total_loss).backward()
        
        # 4. Optimizer Step (Accumulated)
        if (batch_idx + 1) % config.training.accumulation_steps == 0:
            if config.training.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Logging & Metrics
        # Undo the accumulation division for logging
        log_task = task_loss.item()
        log_eq = eq_loss_total.item() if do_equivariance else 0.0
        log_total = log_task + (alpha * log_eq)

        epoch_losses['task'] += log_task
        epoch_losses['eq'] += log_eq
        epoch_losses['total'] += log_total
        
        # [NEW] Step-wise logging
        current_step = (epoch - 1) * len(loader) + batch_idx
        step_metrics = {
            'train/step_loss': log_total,
            'train/step_task_loss': log_task,
            'train/step_eq_loss': log_eq,
        }

        # Add per-layer equivariance losses
        if do_equivariance:
             for k, v in eq_loss_dict.items():
                 step_metrics[k] = v.item() if isinstance(v, torch.Tensor) else v
        
        # Log layer weights
        if layer_weights is not None:
            weights_np = layer_weights.detach().cpu().numpy()
            for i, w in enumerate(weights_np):
                step_metrics[f'layer_weights/layer_{i}'] = w
                
        logger.log_metrics(step_metrics, step=current_step)
        
        all_preds.append(pred.detach().float().cpu())
        all_targets.append(target.detach().float().cpu())

        # Update Progress Bar
        pbar.set_postfix({
            'loss': f"{log_total:.4f}", 
            'eq': f"{log_eq:.4f}" if do_equivariance else "-"
        })
        # break  # TEMPORARY: REMOVE THIS LINE TO RUN FULL EPOCH

    # Aggregate Metrics
    num_batches = len(loader)
    metrics = {
        'train/loss': epoch_losses['total'] / num_batches,
        'train/task_loss': epoch_losses['task'] / num_batches,
        'train/eq_loss_total': epoch_losses['eq'] / num_batches, # Average over all batches (zeros included)
    }
    
    # Regression metrics
    if len(all_preds) > 0:
        preds_cat = torch.cat(all_preds)
        targets_cat = torch.cat(all_targets)
        metrics.update({
            'train/MAE': compute_mae(preds_cat, targets_cat),
            'train/RMSE': compute_rmse(preds_cat, targets_cat),
            'train/R2': compute_r2_score(preds_cat, targets_cat)
        })
        
    return metrics

@torch.inference_mode()
def evaluate(model: BaseGNN, loader: DataLoader, device: str, task_loss_fn: nn.Module,
             eq_losses: dict, logger, epoch: int, config: ExperimentConfig,
             split: str = 'val') -> dict:
    """
    OPTIMIZED: Evaluate model

    Optimizations:
    - Inference Mode (faster than no_grad)
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
        eq_loss_total, eq_loss_dict = compute_equivariance_losses(model, batch, eq_losses, config, device)

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

    # [OPTIMIZATION] Enable Tensor Cores
    torch.set_float32_matmul_precision('high')

    # Create directories
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = os.path.join(config.checkpoint_dir, f'{config.experiment_name}_config.json')
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=4)

    # Initialize logger
    logger = get_logger(config)

    print("Device:", config.device)

    # Load datasets
    print(f"\nLoading datasets... ({config.data.dataset_name})")
    train_dataset, val_dataset, test_dataset = load_dataset(config)
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    # Create data loaders (optimized for GPU)
    # Create data loaders (optimized for GPU)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,  # Enable pin_memory for GPU data transfer
        persistent_workers=config.data.persistent_workers if config.data.num_workers > 0 else False,
        prefetch_factor=config.data.prefetch_factor if config.data.num_workers > 0 else None
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
        persistent_workers=config.data.persistent_workers if config.data.num_workers > 0 else False,
        prefetch_factor=config.data.prefetch_factor if config.data.num_workers > 0 else None
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
        persistent_workers=config.data.persistent_workers if config.data.num_workers > 0 else False,
        prefetch_factor=config.data.prefetch_factor if config.data.num_workers > 0 else None
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
        num_degrees=config.model.num_degrees,
        use_pos=config.model.use_pos
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

    # Initialize DepthScheduler
    # We determine the number of layers from config.
    # Note: We usually check (num_layers hidden) + 1 (final embedding)
    num_check_layers = config.model.num_layers + 1
    
    # Map config parameters to scheduler args
    scheduler_strategy = getattr(config.equivariance, 'layer_weight_strategy', 'constant')
    decay_rate = getattr(config.equivariance, 'layer_decay_rate', 0.5)
    
    # Convert decay_rate to beta for exponential schedules
    beta_val = -math.log(max(decay_rate, 1e-6))
    
    depth_scheduler = DepthScheduler(
        num_layers=num_check_layers,
        schedule_type=scheduler_strategy,
        alpha_0=1.0,
        beta=beta_val,
        gamma=0.1
    ).to(config.device)

    # Optimizer (with efficient gradient computation)
    # Add depth_scheduler parameters if learnable
    params = list(model.parameters())
    if scheduler_strategy == 'learnable':
        params += list(depth_scheduler.parameters())
        print("Added DepthScheduler parameters to optimizer")

    optimizer = torch.optim.Adam(
        params,
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
    elif config.scheduler.lr_schedule == 'plateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=config.scheduler.plateau_patience, factor=config.scheduler.plateau_factor
        )
    elif config.scheduler.lr_schedule == 'exponential':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=config.scheduler.exponential_decay_rate
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
            task_loss_fn, eq_losses, logger, epoch, config, depth_scheduler
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
        print(f" Eq Loss: {train_metrics.get('train/eq_loss_total', 0.0):.4f} | "
              f"Val Eq: {val_metrics.get('val/eq_loss_total', 0.0):.4f}")
        print(f" Train MAE: {train_metrics.get('train/MAE', 0.0):.4f} | "
              f"Val MAE: {val_metrics.get('val/MAE', 0.0):.4f}")
        print(f" Train R2: {train_metrics.get('train/R2', 0.0):.4f} | "
              f"Val R2: {val_metrics.get('val/R2', 0.0):.4f}")
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
