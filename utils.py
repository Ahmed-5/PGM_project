"""
Utility functions for training and evaluation
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.transforms import OneHotDegree
import numpy as np
import random
import os
from typing import Dict, Any
import warnings
from pydantic.warnings import UnsupportedFieldAttributeWarning

warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)

class OneHotEncoder:
    def __init__(self, num_classes: int, feature_index: int):
        self.num_classes = num_classes
        self.feature_index = feature_index
    
    def __call__(self, data: Data):
        features = data.x[:, self.feature_index].long()
        one_hot = torch.nn.functional.one_hot(features, num_classes=self.num_classes).float()
        # data.x = torch.cat([data.x, one_hot], dim=-1)
        data.x = one_hot
        return data

class AtomDegreeOneHot:
    def __init__(self, num_atom_types, max_degree):
        self.num_atom_types = num_atom_types
        self.degree_transform = OneHotDegree(max_degree)
    
    def __call__(self, data: Data):
        atom_types = data.x[:, 0].long()  # assuming atom types are stored here
        atom_one_hot = torch.nn.functional.one_hot(atom_types, num_classes=self.num_atom_types).float()
        
        # Replace data.x with one-hot atom types
        data.x = atom_one_hot
        
        # Apply degree one-hot to augment data.x
        data = self.degree_transform(data)
        return data
    
def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str
):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, path)

def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str
) -> Dict[str, Any]:
    """Load model checkpoint"""
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 20, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Mean Absolute Error"""
    return torch.mean(torch.abs(pred - target)).item()

def compute_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Root Mean Squared Error"""
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()

def compute_r2_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Coefficient of Determination (R^2)"""
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - pred) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2.item()