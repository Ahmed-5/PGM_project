"""
Depth-adaptive and learning rate scheduling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List


class DepthScheduler(nn.Module):
    """Layer-wise equivariance weight scheduling"""
    
    def __init__(
        self,
        num_layers: int,
        schedule_type: str = 'exponential',
        alpha_0: float = 1.0,
        beta: float = 0.1,
        gamma: float = 0.1
    ):
        super().__init__()
        self.num_layers = num_layers
        self.schedule_type = schedule_type
        self.alpha_0 = alpha_0
        self.beta = beta
        self.gamma = gamma
        
        if schedule_type == 'learnable':
            init_values = alpha_0 * np.exp(-beta * np.arange(num_layers))
            self.alpha = nn.Parameter(torch.tensor(init_values, dtype=torch.float32))
        else:
            self.register_buffer('alpha', torch.zeros(num_layers))
            self._compute_schedule()
    
    def _compute_schedule(self):
        """Pre-compute fixed schedules"""
        if self.schedule_type == 'constant':
            self.alpha.fill_(self.alpha_0)
        
        elif self.schedule_type == 'exponential':
            for l in range(self.num_layers):
                self.alpha[l] = self.alpha_0 * np.exp(-self.beta * l)
        
        elif self.schedule_type == 'linear':
            for l in range(self.num_layers):
                self.alpha[l] = max(0.0, self.alpha_0 - self.gamma * l)
        
        elif self.schedule_type == 'inverse':
            for l in range(self.num_layers):
                self.alpha[l] = self.alpha_0 / (1.0 + self.beta * l)
        
        elif self.schedule_type == 'u_shaped':
            mid = self.num_layers // 2
            for l in range(self.num_layers):
                distance_from_mid = abs(l - mid) / max(mid, 1)
                self.alpha[l] = self.alpha_0 * (0.5 + 0.5 * distance_from_mid)
    
    def get_alpha(self, layer_idx: int) -> torch.Tensor:
        """Get weight for specific layer"""
        if self.schedule_type == 'learnable':
            return F.softplus(self.alpha[layer_idx])
        return self.alpha[layer_idx]
    
    def get_all_alphas(self) -> torch.Tensor:
        """Get all layer weights"""
        if self.schedule_type == 'learnable':
            return F.softplus(self.alpha)
        return self.alpha.clone()
    
    def forward(self, layer_idx: int) -> torch.Tensor:
        return self.get_alpha(layer_idx)


class WarmupScheduler:
    """Learning rate warmup scheduler"""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        base_lr: float,
        warmup_start_lr: float = 1e-6
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.warmup_start_lr = warmup_start_lr
        self.current_epoch = 0
    
    def step(self):
        """Update learning rate"""
        if self.current_epoch < self.warmup_epochs:
            lr = self.warmup_start_lr + (
                self.base_lr - self.warmup_start_lr
            ) * self.current_epoch / self.warmup_epochs
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        
        self.current_epoch += 1


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config,
    steps_per_epoch: Optional[int] = None
):
    """Factory for learning rate schedulers"""
    
    if config.scheduler.lr_schedule == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.scheduler.lr_step_size,
            gamma=config.scheduler.lr_gamma
        )
    
    elif config.scheduler.lr_schedule == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.num_epochs,
            eta_min=1e-6
        )
    
    elif config.scheduler.lr_schedule == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.scheduler.lr_gamma,
            patience=10,
            verbose=True
        )
    
    elif config.scheduler.lr_schedule == 'exponential':
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.scheduler.lr_gamma
        )
    
    else:  # 'none'
        return None
