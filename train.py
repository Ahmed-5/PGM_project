"""
Training script for Relaxed Equivariant GNN with logging support
"""

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import ZINC, QM9
from torch_geometric.transforms import OneHotDegree
import numpy as np
import os
from tqdm import tqdm
import json
from pathlib import Path
import matplotlib.pyplot as plt

from config import ExperimentConfig, get_config
from relaxed_gnn import RelaxedEquivariantGNN
from utils import set_seed, save_checkpoint, load_checkpoint, EarlyStopping, compute_mae, compute_rmse, compute_r2_score, AtomDegreeOneHot, OneHotEncoder
from logger import get_logger, MetricsTracker
import warnings
from pydantic.warnings import UnsupportedFieldAttributeWarning

warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)

def load_dataset(config: ExperimentConfig):
    """Load dataset based on configuration"""
    if config.data.dataset_name == 'ZINC':
        # the transform should one-hot encode the node features
        train_dataset = ZINC(
            root=config.data.root,
            subset=config.data.subset,
            # transform=AtomDegreeOneHot(num_atom_types=28, max_degree=10),
            transform=OneHotEncoder(num_classes=28, feature_index=0),
            split='train'
        )
        val_dataset = ZINC(
            root=config.data.root,
            subset=config.data.subset,
            # transform=AtomDegreeOneHot(num_atom_types=28, max_degree=10),
            transform=OneHotEncoder(num_classes=28, feature_index=0),
            split='val'
        )
        test_dataset = ZINC(
            root=config.data.root,
            subset=config.data.subset,
            # transform=AtomDegreeOneHot(num_atom_types=28, max_degree=10),
            transform=OneHotEncoder(num_classes=28, feature_index=0),
            split='test'
        )
    elif config.data.dataset_name == 'QM9':
        dataset = QM9(root=config.data.root)
        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
    else:
        raise ValueError(f"Unknown dataset: {config.data.dataset_name}")
    
    return train_dataset, val_dataset, test_dataset


def plot_alpha_schedule(alphas: np.ndarray, schedule_type: str, save_path: str = None):
    """Plot layer-wise alpha values"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    layers = np.arange(len(alphas))
    ax.plot(layers, alphas, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax.fill_between(layers, 0, alphas, alpha=0.3, color='#2E86AB')
    
    ax.set_xlabel('Layer Index', fontsize=14, fontweight='bold')
    ax.set_ylabel('α (Equivariance Weight)', fontsize=14, fontweight='bold')
    ax.set_title(f'Depth-Adaptive Schedule: {schedule_type}', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(-0.5, len(alphas) - 0.5)
    ax.set_ylim(0, max(alphas) * 1.1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def train_epoch(
    model: RelaxedEquivariantGNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    task_loss_fn: nn.Module,
    logger,
    epoch: int,
    config: ExperimentConfig
):
    """Train for one epoch"""
    model.train()
    
    metrics_tracker = MetricsTracker()
    
    pbar = tqdm(loader, desc=f'Epoch {epoch} [Train]')

    all_preds = []
    all_targets = []

    for batch_idx, batch in enumerate(pbar):
        batch = batch.to(device)
        
        optimizer.zero_grad()

        # Compute loss
        pred, loss, loss_dict = model.compute_total_loss(
            x=batch.x,
            edge_index=batch.edge_index,
            batch=batch.batch,
            y_true=batch.y,
            task_loss_fn=task_loss_fn
        )
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track metrics
        metrics_tracker.update({
            'train/loss': loss_dict['total_loss'],
            'train/task_loss': loss_dict['task_loss'],
            'train/eq_loss': loss_dict['eq_loss_total'],
            'train/eq_measure': loss_dict.get('eq_loss_measure_total', 0.0)
        })

        all_preds.append(pred.detach().cpu())
        all_targets.append(batch.y.detach().cpu())
        
        # Log batch metrics at intervals
        global_step = (epoch - 1) * len(loader) + batch_idx
        if batch_idx % config.logging.log_interval == 0:
            batch_metrics = {
                'train/batch_loss': loss_dict['total_loss'],
                'train/batch_task_loss': loss_dict['task_loss'],
                'train/batch_eq_loss': loss_dict['eq_loss_total'],
                'train/batch_eq_measure': loss_dict.get('eq_loss_measure_total', 0.0),
            }
            
            # Log layer-wise metrics if enabled
            if config.logging.log_layer_outputs:
                for i, (eq_loss, alpha, eq_measure) in enumerate(zip(
                    loss_dict['layer_eq_losses'], 
                    loss_dict['layer_alphas'],
                    loss_dict['layer_eq_loss_measures']
                )):
                    batch_metrics[f'train/layer_{i}_eq_loss'] = eq_loss
                    batch_metrics[f'train/layer_{i}_alpha'] = alpha
                    batch_metrics[f'train/layer_{i}_eq_measure'] = eq_measure

            logger.log_metrics(batch_metrics, step=global_step)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss_dict['total_loss']:.4f}",
            'task': f"{loss_dict['task_loss']:.4f}",
            'eq': f"{loss_dict['eq_loss_total']:.4f}",
            'eq_est': f"{loss_dict.get('eq_loss_measure_total', 0.0):.4f}"
        })

    # Get epoch averages
    epoch_metrics = metrics_tracker.get_averages()

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    # Compute additional regression metrics
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
def evaluate(
    model: RelaxedEquivariantGNN,
    loader: DataLoader,
    device: str,
    task_loss_fn: nn.Module,
    logger,
    epoch: int,
    config: ExperimentConfig,
    split: str = 'val'
):
    """Evaluate model"""
    model.eval()
    
    metrics_tracker = MetricsTracker()

    all_preds = []
    all_targets = []
    
    pbar = tqdm(loader, desc=f'Epoch {epoch} [{split.capitalize()}]')
    for batch in pbar:
        batch = batch.to(device)
        
        # Compute loss
        pred, _, loss_dict = model.compute_total_loss(
            x=batch.x,
            edge_index=batch.edge_index,
            batch=batch.batch,
            y_true=batch.y,
            task_loss_fn=task_loss_fn
        )
        
        # Track metrics
        metrics_tracker.update({
            f'{split}/loss': loss_dict['total_loss'],
            f'{split}/task_loss': loss_dict['task_loss'],
            f'{split}/eq_loss': loss_dict['eq_loss_total'],
            f'{split}/eq_measure': loss_dict.get('eq_loss_measure_total', 0.0)
        })
        
        pbar.set_postfix({
            'loss': f"{loss_dict['total_loss']:.4f}",
            'task': f"{loss_dict['task_loss']:.4f}",
            'eq': f"{loss_dict['eq_loss_total']:.4f}",
            'eq_est': f"{loss_dict.get('eq_loss_measure_total', 0.0):.4f}"
        })

        all_preds.append(pred.detach().cpu())
        all_targets.append(batch.y.detach().cpu())

    # Get epoch averages
    epoch_metrics = metrics_tracker.get_averages()

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    # Compute additional regression metrics
    mae = compute_mae(preds, targets)
    rmse = compute_rmse(preds, targets)
    r2 = compute_r2_score(preds, targets)
    
    epoch_metrics = metrics_tracker.get_averages()
    epoch_metrics.update({
        f'{split}/MAE': mae,
        f'{split}/RMSE': rmse,
        f'{split}/R2': r2
    })
    
    return epoch_metrics


def train(config: ExperimentConfig):
    """Main training function"""
    
    # Set seed for reproducibility
    set_seed(config.seed)
    
    # Create directories
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # save config to file
    config_path = os.path.join(config.checkpoint_dir, f'{config.experiment_name}_config.json')
    with open(config_path, 'w') as f:
        json.dump(config.dict(), f, indent=4)
    
    # Initialize logger
    logger = get_logger(config)
    
    # Load datasets
    print(f"\nLoading datasets... ({config.data.dataset_name})")
    train_dataset, val_dataset, test_dataset = load_dataset(config)
    
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers
    )
    
    # Initialize model
    print("\nInitializing model...")
    model = RelaxedEquivariantGNN(
        in_channels=config.model.in_channels,
        hidden_channels=config.model.hidden_channels,
        out_channels=config.model.out_channels,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
        gnn_type=config.model.gnn_type,
        schedule_type=config.scheduler.schedule_type,
        alpha_0=config.scheduler.alpha_0,
        beta=config.scheduler.beta,
        gamma=config.scheduler.gamma
    ).to(config.device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Log hyperparameters
    hparams = {
        'model_params': num_params,
        'dataset_train_size': len(train_dataset),
        'dataset_val_size': len(val_dataset),
        'dataset_test_size': len(test_dataset)
    }
    logger.log_hyperparameters(hparams)
    
    # Watch model gradients if enabled
    if config.logging.log_gradients:
        logger.watch_model(model, log_freq=100)
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # Learning rate scheduler
    if config.training.scheduler_lr == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=50, gamma=0.5
        )
    elif config.training.scheduler_lr == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.training.num_epochs
        )
    else:
        lr_scheduler = None
    
    # Task loss function
    task_loss_fn = nn.MSELoss()
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config.training.patience)
    
    # Log initial alpha schedule
    initial_alphas = model.get_scheduler_alphas().detach().cpu().numpy()
    alpha_fig = plot_alpha_schedule(
        initial_alphas, 
        config.scheduler.schedule_type,
        save_path=os.path.join(config.checkpoint_dir, 'alpha_schedule_initial.png')
    )
    logger.log_image('schedule/initial_alphas', alpha_fig, step=0)
    plt.close(alpha_fig)
    
    # Training loop
    print(f"\nTraining for {config.training.num_epochs} epochs...")
    print("=" * 80)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, config.training.num_epochs + 1):
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, config.device, 
            task_loss_fn, logger, epoch, config
        )
        
        # Validate
        val_metrics = evaluate(
            model, val_loader, config.device, 
            task_loss_fn, logger, epoch, config, split='val'
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
        
        # Log current alpha values for learnable schedule
        if config.scheduler.schedule_type == 'learnable':
            current_alphas = model.get_scheduler_alphas().detach().cpu().numpy()
            for i, alpha in enumerate(current_alphas):
                logger.log_metrics({f'schedule/alpha_layer_{i}': alpha}, step=global_step)
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{config.training.num_epochs}")
        print(f"  Train Loss: {train_metrics['train/loss']:.4f} | "
              f"Val Loss: {val_metrics['val/loss']:.4f}")
        print(f"  Task Loss:  {train_metrics['train/task_loss']:.4f} | "
              f"Val Task: {val_metrics['val/task_loss']:.4f}")
        print(f"  Eq Loss:    {train_metrics['train/eq_loss']:.4f} | "
              f"Val Eq: {val_metrics['val/eq_loss']:.4f}")
        print(f"  Eq Est: {train_metrics['train/eq_measure']:.4f} | "
              f"Val Eq Est: {val_metrics['val/eq_measure']:.4f}")
        print(f"  Train MAE: {train_metrics['train/MAE']:.4f} | "
              f"Val MAE: {val_metrics['val/MAE']:.4f}")
        print(f"  Train RMSE: {train_metrics['train/RMSE']:.4f} | "
              f"Val RMSE: {val_metrics['val/RMSE']:.4f}")
        print(f"  Train R2: {train_metrics['train/R2']:.4f} | "
              f"Val R2: {val_metrics['val/R2']:.4f}")
        print(f"  LR: {epoch_metrics['learning_rate']:.6f}")
        
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
            print(f"  ✓ Saved best model (val_loss: {best_val_loss:.4f})")
            
            # Save as wandb artifact
            if hasattr(logger, 'save_model_artifact'):
                logger.save_model_artifact(checkpoint_path, name='best_model')
        else:
            print(f"  ✗ No improvement (best_val_loss: {best_val_loss:.4f})")
        
        # Early stopping
        early_stopping(val_metrics['val/loss'])
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
        task_loss_fn, logger, epoch, config, split='test'
    )
    
    # Log test metrics
    logger.log_metrics(test_metrics, step=global_step)
    
    print("\n" + "=" * 80)
    print("TEST SET RESULTS")
    print("=" * 80)
    print(f"Test Loss:      {test_metrics['test/loss']:.4f}")
    print(f"Test Task Loss: {test_metrics['test/task_loss']:.4f}")
    print(f"Test Eq Loss:   {test_metrics['test/eq_loss']:.4f}")
    print(f"Test Eq Est:    {test_metrics['test/eq_measure']:.4f}")
    print("=" * 80)
    
    # Save final alpha schedule
    final_alphas = model.get_scheduler_alphas().detach().cpu().numpy()
    alpha_fig = plot_alpha_schedule(
        final_alphas,
        config.scheduler.schedule_type,
        save_path=os.path.join(config.checkpoint_dir, 'alpha_schedule_final.png')
    )
    logger.log_image('schedule/final_alphas', alpha_fig, step=global_step)
    plt.close(alpha_fig)
    
    # Save alpha values
    alphas_path = os.path.join(config.checkpoint_dir, f'{config.experiment_name}_alphas.npy')
    np.save(alphas_path, final_alphas)
    print(f"\nFinal layer alphas: {final_alphas}")
    
    # Finish logging
    logger.finish()
    
    return model, test_metrics


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train Relaxed Equivariant GNN with configurable logging'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='default',
        choices=['default', 'baseline', 'constant_alpha', 
                'exponential_decay', 'linear_decay', 'learnable', 'deep_network'],
        help='Configuration preset'
    )
    parser.add_argument(
        '--logger',
        type=str,
        default='tensorboard',
        choices=['wandb', 'tensorboard', 'none'],
        help='Logging backend to use'
    )
    parser.add_argument(
        '--wandb-project',
        type=str,
        default='relaxed-equivariance-gnn',
        help='Weights & Biases project name'
    )
    parser.add_argument(
        '--wandb-entity',
        type=str,
        default=None,
        help='Weights & Biases entity (username or team)'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='Custom experiment name (overrides config default)'
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
    config.logging.wandb_project = args.wandb_project
    config.logging.wandb_entity = args.wandb_entity
    config.seed = args.seed
    
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    
    # Print configuration
    print("\n" + "=" * 80)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 80)
    print(f"Config preset:     {args.config}")
    print(f"Experiment name:   {config.experiment_name}")
    print(f"Logger:            {config.logging.logger_type}")
    print(f"GNN type:          {config.model.gnn_type}")
    print(f"Num layers:        {config.model.num_layers}")
    print(f"Schedule type:     {config.scheduler.schedule_type}")
    print(f"Alpha_0:           {config.scheduler.alpha_0}")
    print(f"Device:            {config.device}")
    print(f"Seed:              {config.seed}")
    print("=" * 80 + "\n")
    
    # Train model
    model, test_metrics = train(config)
