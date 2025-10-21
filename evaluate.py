"""
Evaluation script for Relaxed Equivariant GNN
Loads a saved checkpoint and evaluates on test dataset
"""

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import ZINC, QM9
import numpy as np
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt

from config import ExperimentConfig, get_config
from relaxed_gnn import RelaxedEquivariantGNN
from utils import set_seed, load_checkpoint
from logger import get_logger, MetricsTracker

def load_dataset(config: ExperimentConfig):
    """Load dataset based on configuration"""
    if config.data.dataset_name == 'ZINC':
        train_dataset = ZINC(
            root=config.data.root,
            subset=config.data.subset,
            split='train'
        )
        val_dataset = ZINC(
            root=config.data.root,
            subset=config.data.subset,
            split='val'
        )
        test_dataset = ZINC(
            root=config.data.root,
            subset=config.data.subset,
            split='test'
        )
    elif config.data.dataset_name == 'QM9':
        dataset = QM9(root=config.data.root)
        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
    else:
        raise ValueError(f"Unknown dataset: {config.data.dataset_name}")
    
    return train_dataset, val_dataset, test_dataset

@torch.no_grad()
def evaluate(
    model: RelaxedEquivariantGNN,
    loader: DataLoader,
    device: str,
    task_loss_fn: nn.Module,
    logger,
    config: ExperimentConfig,
    split: str = 'test'
):
    """Evaluate model"""
    model.eval()
    metrics_tracker = MetricsTracker()
    for batch in loader:
        batch = batch.to(device)
        _, _, loss_dict = model.compute_total_loss(
            x=batch.x,
            edge_index=batch.edge_index,
            batch=batch.batch,
            y_true=batch.y,
            task_loss_fn=task_loss_fn
        )
        metrics_tracker.update({
            f'{split}/loss': loss_dict['total_loss'],
            f'{split}/task_loss': loss_dict['task_loss'],
            f'{split}/eq_loss': loss_dict['eq_loss_total']
        })

    epoch_metrics = metrics_tracker.get_averages()
    return epoch_metrics

def main(config_name: str = 'default', checkpoint_path: str = None, logger_type: str = 'none'):
    config = get_config(config_name)
    config.logging.logger_type = logger_type
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    set_seed(config.seed)
    
    # Setup logger
    logger = get_logger(config)
    
    # Load datasets
    _, _, test_dataset = load_dataset(config)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers
    )
    
    # Initialize model
    model = RelaxedEquivariantGNN(
        # in_channels=config.model.in_channels,
        in_channels=1,
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
    
    # Load checkpoint weights
    if checkpoint_path is None:
        checkpoint_path = os.path.join(config.checkpoint_dir, f'{config.experiment_name}_best.pt')
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    load_checkpoint(model, optimizer=None, path=checkpoint_path)  # optimizer=None since not needed for eval
    
    # Task loss function
    task_loss_fn = nn.MSELoss()
    
    # Evaluate
    metrics = evaluate(
        model=model,
        loader=test_loader,
        device=config.device,
        task_loss_fn=task_loss_fn,
        logger=logger,
        config=config,
        split='test'
    )
    
    print(f"Test Loss:      {metrics['test/loss']:.4f}")
    print(f"Test Task Loss: {metrics['test/task_loss']:.4f}")
    print(f"Test Eq Loss:   {metrics['test/eq_loss']:.4f}")
    print(f"Test Eq Est:    {metrics['test/eq_measure']:.4f}")
    
    # Log metrics
    logger.log_metrics(metrics, step=0)
    logger.finish()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='default',
                        help='Experiment config name to use')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--logger', type=str, default='none',
                        choices=['wandb', 'tensorboard', 'none'],
                        help="Logging backend")
    args = parser.parse_args()
    
    main(config_name=args.config, checkpoint_path=args.checkpoint, logger_type=args.logger)
