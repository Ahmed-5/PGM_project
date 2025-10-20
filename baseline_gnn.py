"""
Base GNN architectures with layer-wise output tracking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, global_mean_pool, global_add_pool
from typing import List, Dict, Tuple

class BaseGNN(nn.Module):
    """
    Base Graph Neural Network with multiple architecture options
    Tracks intermediate layer outputs for equivariance loss computation
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 4,
        dropout: float = 0.5,
        gnn_type: str = 'GCN'
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.gnn_type = gnn_type
        
        # Build convolutional layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        self.convs.append(self._build_conv_layer(in_channels, hidden_channels, gnn_type))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(self._build_conv_layer(hidden_channels, hidden_channels, gnn_type))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Output layer
        self.convs.append(self._build_conv_layer(hidden_channels, hidden_channels, gnn_type))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Final prediction head
        self.lin = nn.Linear(hidden_channels, out_channels)
        
    def _build_conv_layer(self, in_channels: int, out_channels: int, gnn_type: str):
        """Factory method for different GNN layer types"""
        if gnn_type == 'GCN':
            return GCNConv(in_channels, out_channels)
        elif gnn_type == 'GIN':
            mlp = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels)
            )
            return GINConv(mlp)
        elif gnn_type == 'GraphSAGE':
            return SAGEConv(in_channels, out_channels)
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        batch: torch.Tensor,
        return_layer_outputs: bool = False
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Forward pass with optional layer output tracking
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment [num_nodes]
            return_layer_outputs: Whether to return intermediate representations
            
        Returns:
            predictions: Graph-level predictions [batch_size, out_channels]
            layer_outputs: List of dicts with layer info (if requested)
        """
        layer_outputs = []
        
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            # Apply convolution
            # print("type of x:", type(x), "shape of x:", x.shape, "type of edge_index:", type(edge_index), "shape of edge_index:", edge_index.shape)
            x = conv(x.float(), edge_index)
            
            # Store pre-activation representation for equivariance loss
            if return_layer_outputs:
                layer_outputs.append({
                    'layer_idx': i,
                    'representation': x.clone(),
                    'edge_index': edge_index,
                    'batch': batch
                })
            
            # Batch normalization
            x = bn(x)
            
            # ReLU activation (except last layer)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Final prediction
        x = self.lin(x)
        
        return x, layer_outputs
    
    def reset_parameters(self):
        """Reset all learnable parameters"""
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.batch_norms:
            bn.reset_parameters()
        self.lin.reset_parameters()


class GINWithEdgeFeatures(nn.Module):
    """
    Graph Isomorphism Network variant that handles edge features
    Useful for molecular datasets
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 4,
        dropout: float = 0.5,
        edge_dim: int = None
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.convs.append(GINConv(mlp))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            mlp = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
            self.convs.append(GINConv(mlp))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Output layer
        mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.convs.append(GINConv(mlp))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Prediction head
        self.lin = nn.Linear(hidden_channels, out_channels)
        
    def forward(self, x, edge_index, batch, return_layer_outputs=False):
        layer_outputs = []
        
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            
            if return_layer_outputs:
                layer_outputs.append({
                    'layer_idx': i,
                    'representation': x.clone(),
                    'edge_index': edge_index,
                    'batch': batch
                })
            
            x = bn(x)
            
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = global_add_pool(x, batch)  # GIN typically uses sum pooling
        x = self.lin(x)
        
        return x, layer_outputs
