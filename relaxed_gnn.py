"""
Relaxed Equivariant GNN with multitask loss and depth-adaptive scheduling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List

from baseline_gnn import BaseGNN
from equivariance_loss import EquivarianceLoss
from schedulers import DepthScheduler
import warnings
from pydantic.warnings import UnsupportedFieldAttributeWarning

warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)

class RelaxedEquivariantGNN(nn.Module):
    """
    GNN with relaxed equivariance via multitask learning
    L_total = L_task + Σ_l α_l * L_eq^(l)
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 4,
        dropout: float = 0.5,
        gnn_type: str = 'GCN',
        schedule_type: str = 'exponential',
        alpha_0: float = 1.0,
        beta: float = 0.1,
        gamma: float = 0.1,
        eq_loss_type: str = 'permutation',
        eq_num_samples: int = 1
    ):
        super().__init__()
        
        # Base GNN model
        self.gnn = BaseGNN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            gnn_type=gnn_type
        )
        
        # Depth-adaptive scheduler
        self.scheduler = DepthScheduler(
            num_layers=num_layers,
            schedule_type=schedule_type,
            alpha_0=alpha_0,
            beta=beta,
            gamma=gamma
        )
        
        # Equivariance loss module
        self.eq_loss_fn = EquivarianceLoss(
            group_type=eq_loss_type,
            num_samples=eq_num_samples,
            normalize=True
        )
        
        self.num_layers = num_layers
        
    def forward(self, x, edge_index, batch):
        """Standard forward pass for inference"""
        pred, _ = self.gnn(x, edge_index, batch, return_layer_outputs=False)
        return pred
    
    def compute_total_loss(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        y_true: torch.Tensor,
        task_loss_fn: nn.Module = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute multitask loss: task loss + weighted equivariance losses
        
        Args:
            x: Node features
            edge_index: Edge indices
            batch: Batch assignment
            y_true: Ground truth labels
            task_loss_fn: Loss function for task (defaults to MSE for regression)
            
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with loss components for logging
        """
        # Forward pass with layer outputs
        pred, layer_outputs = self.gnn(
            x, edge_index, batch, return_layer_outputs=True
        )
        
        # Task loss (regression: MSE, classification: CrossEntropy)
        if task_loss_fn is None:
            task_loss_fn = nn.MSELoss()
        
        L_task = task_loss_fn(pred.squeeze(), y_true)
        
        # Layer-wise equivariance losses
        L_eq_total = torch.tensor(0.0, device=x.device)
        L_eq_loss_measure_total = torch.tensor(0.0, device=x.device)
        layer_eq_loss_measures = []
        layer_eq_losses = []
        layer_alphas = []
        
        for layer_data in layer_outputs:
            layer_idx = layer_data['layer_idx']
            representation = layer_data['representation']
            layer_edge_index = layer_data['edge_index']
            layer_batch = layer_data['batch']
            
            # Get depth-adaptive weight
            alpha_l = self.scheduler.get_alpha(layer_idx)
            
            # Compute equivariance loss for this layer
            L_eq_l = self.eq_loss_fn(
                representation,
                layer_edge_index,
                layer_batch,
                conv_layer=None  # Simplified version
            )
            
            # Weighted sum
            L_eq_total += alpha_l * L_eq_l
            L_eq_loss_measure_total += L_eq_l
            
            # Track for logging
            layer_eq_losses.append(L_eq_l.item())
            layer_eq_loss_measures.append(L_eq_l.item())
            layer_alphas.append(alpha_l.item())
        
        # Total loss
        total_loss = L_task + L_eq_total
        
        # Prepare loss dictionary for logging
        loss_dict = {
            'total_loss': total_loss.item(),
            'task_loss': L_task.item(),
            'eq_loss_total': L_eq_total.item(),
            'eq_loss_measure_total': L_eq_loss_measure_total.item(),
            'layer_eq_losses': layer_eq_losses,
            'layer_alphas': layer_alphas,
            'layer_eq_loss_measures': layer_eq_loss_measures
        }
        
        return total_loss, loss_dict
    
    def get_scheduler_alphas(self) -> torch.Tensor:
        """Get current alpha values from scheduler"""
        return self.scheduler.get_all_alphas()
    
    def reset_parameters(self):
        """Reset all learnable parameters"""
        self.gnn.reset_parameters()
        if self.scheduler.schedule_type == 'learnable':
            self.scheduler._compute_schedule()
