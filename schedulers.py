"""
Depth-adaptive scheduling strategies for layer-wise equivariance weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
from pydantic.warnings import UnsupportedFieldAttributeWarning

warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)

class DepthScheduler(nn.Module):
    """
    Schedules layer-wise equivariance weights α_l based on depth
    """
    
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
        
        # For learnable schedule, initialize parameters
        if schedule_type == 'learnable':
            # Initialize with exponential decay as prior
            init_values = alpha_0 * np.exp(-beta * np.arange(num_layers))
            self.alpha = nn.Parameter(torch.tensor(init_values, dtype=torch.float32))
        else:
            self.register_buffer('alpha', torch.zeros(num_layers))
            self._compute_schedule()
    
    def _compute_schedule(self):
        """Pre-compute schedule for non-learnable types"""
        if self.schedule_type == 'constant':
            self.alpha.fill_(self.alpha_0)
        
        elif self.schedule_type == 'exponential':
            for l in range(self.num_layers):
                self.alpha[l] = self.alpha_0 * np.exp(-self.beta * l)
        
        elif self.schedule_type == 'linear':
            for l in range(self.num_layers):
                self.alpha[l] = max(0.0, self.alpha_0 - self.gamma * l)
        
        elif self.schedule_type == 'inverse':
            # 1 / (1 + β*l) decay
            for l in range(self.num_layers):
                self.alpha[l] = self.alpha_0 / (1.0 + self.beta * l)
        
        elif self.schedule_type == 'u_shaped':
            # Higher weight on early and late layers, lower in middle
            mid = self.num_layers // 2
            for l in range(self.num_layers):
                distance_from_mid = abs(l - mid) / mid
                self.alpha[l] = self.alpha_0 * (0.5 + 0.5 * distance_from_mid)
    
    def get_alpha(self, layer_idx: int) -> torch.Tensor:
        """
        Get equivariance weight for specific layer
        
        Args:
            layer_idx: Layer index (0-indexed)
            
        Returns:
            α_l for this layer
        """
        if self.schedule_type == 'learnable':
            # Apply softplus to ensure positivity
            return F.softplus(self.alpha[layer_idx])
        else:
            return self.alpha[layer_idx]
    
    def get_all_alphas(self) -> torch.Tensor:
        """Get all layer weights"""
        if self.schedule_type == 'learnable':
            return F.softplus(self.alpha)
        else:
            return self.alpha.clone()
    
    def forward(self, layer_idx: int) -> torch.Tensor:
        """Forward call returns alpha for given layer"""
        return self.get_alpha(layer_idx)
    
    def plot_schedule(self):
        """Utility to visualize the schedule"""
        import matplotlib.pyplot as plt
        
        alphas = self.get_all_alphas().detach().cpu().numpy()
        layers = np.arange(self.num_layers)
        
        plt.figure(figsize=(8, 5))
        plt.plot(layers, alphas, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Layer Index', fontsize=12)
        plt.ylabel('α_l (Equivariance Weight)', fontsize=12)
        plt.title(f'Depth-Adaptive Schedule: {self.schedule_type}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
