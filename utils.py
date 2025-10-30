"""
Utility functions with improved error handling and flexibility
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.transforms import OneHotDegree
import numpy as np
import random
from typing import Dict, Any, Optional
from pathlib import Path


class OneHotEncoder:
    """Flexible one-hot encoding for node features"""
    
    def __init__(self, num_classes: int, feature_index: int = 0, append: bool = False):
        self.num_classes = num_classes
        self.feature_index = feature_index
        self.append = append
    
    def __call__(self, data: Data) -> Data:
        if data.x is None:
            raise ValueError("Data has no node features (x)")
        
        features = data.x[:, self.feature_index].long()
        one_hot = F.one_hot(features, num_classes=self.num_classes).float()
        
        if self.append:
            data.x = torch.cat([data.x, one_hot], dim=-1)
        else:
            data.x = one_hot
        
        return data


class AtomDegreeOneHot:
    """Combined atom type and degree one-hot encoding"""
    
    def __init__(self, num_atom_types: int, max_degree: int):
        self.num_atom_types = num_atom_types
        self.degree_transform = OneHotDegree(max_degree)
    
    def __call__(self, data: Data) -> Data:
        if data.x is None or data.x.shape[1] == 0:
            raise ValueError("Data must have atom type features")
        
        atom_types = data.x[:, 0].long()
        atom_one_hot = F.one_hot(atom_types, num_classes=self.num_atom_types).float()
        data.x = atom_one_hot
        data = self.degree_transform(data)
        
        return data


def set_seed(seed: int = 42, deterministic: bool = True):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
    additional_state: Optional[Dict[str, Any]] = None
):
    """Save comprehensive checkpoint"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    if additional_state:
        state.update(additional_state)
    
    torch.save(state, path)


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    path: str,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """Load checkpoint with device handling"""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


class EarlyStopping:
    """Early stopping with best model tracking"""
    
    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, val_loss: float, epoch: int) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
            return False
        
        if self.mode == 'min':
            improved = val_loss < self.best_loss - self.min_delta
        else:
            improved = val_loss > self.best_loss + self.min_delta
        
        if improved:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count trainable and total parameters"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {'trainable': trainable, 'total': total, 'frozen': total - trainable}


def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean Absolute Error"""
    return torch.mean(torch.abs(pred - target)).item()


def compute_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Root Mean Squared Error"""
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


def compute_r2_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    """R² Score with numerical stability"""
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - pred) ** 2)
    
    if ss_tot < 1e-10:
        return 0.0
    
    r2 = 1 - ss_res / ss_tot
    return r2.item()


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Compute all regression metrics"""
    return {
        'mae': compute_mae(pred, target),
        'rmse': compute_rmse(pred, target),
        'r2': compute_r2_score(pred, target),
    }
