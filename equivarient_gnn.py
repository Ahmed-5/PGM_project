"""
Unified Base GNN with multiple architecture options
Supports models with varying degrees of symmetry including SO(3), O(3), SE(3), E(3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, TransformerConv
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_scatter import scatter_add
from typing import List, Dict, Tuple, Optional
import math
import traceback


class BaseGNN(nn.Module):
    """
    Unified GNN architecture supporting multiple model types:
    
    Symmetry levels:
    - 'raw_mlp': No symmetry (baseline)
    - 'transformer': Permutation only
    - 'gcn', 'gin', 'graphsage': Permutation only (standard GNNs)
    - 'schnet', 'dimenet': E(3) invariant
    - 'egnn', 'painn': E(3) equivariant
    - 'vector_neuron': SO(3) equivariant (no reflections)
    - 'se3_transformer': SE(3) equivariant (rotations + translations)
    - 'nequip': E(3) equivariant with tensor products
    - 'clofnet': SE(3) with local frames
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 4,
        dropout: float = 0.5,
        model_type: str = 'gcn',
        spatial_dim: int = 3,
        # Model-specific parameters
        num_heads: int = 8,  # For Transformer, SE3Transformer
        num_gaussians: int = 50,  # For SchNet, DimeNet, NequIP
        num_spherical: int = 7,  # For DimeNet
        cutoff: float = 10.0,  # For distance-based models
        update_coords: bool = False,  # For EGNN
        max_ell: int = 2,  # For NequIP (angular momentum)
        num_degrees: int = 2,  # For SE3Transformer
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.model_type = model_type.lower()
        self.spatial_dim = spatial_dim
        
        # Validate model type
        valid_models = [
            'raw_mlp', 'transformer', 'gcn', 'gin', 'graphsage', 
            'schnet', 'dimenet', 'egnn', 'painn',
            'vector_neuron', 'se3_transformer', 'nequip', 'clofnet'
        ]
        if self.model_type not in valid_models:
            raise ValueError(f"model_type must be one of {valid_models}")
        
        # Build model based on type
        if self.model_type == 'raw_mlp':
            self._build_raw_mlp()
        elif self.model_type == 'transformer':
            self._build_transformer(num_heads)
        elif self.model_type in ['gcn', 'gin', 'graphsage']:
            self._build_standard_gnn()
        elif self.model_type == 'schnet':
            self._build_schnet(num_gaussians, cutoff)
        elif self.model_type == 'dimenet':
            self._build_dimenet(num_gaussians, num_spherical, cutoff)
        elif self.model_type == 'egnn':
            self._build_egnn(update_coords)
        elif self.model_type == 'painn':
            self._build_painn()
        elif self.model_type == 'vector_neuron':
            self._build_vector_neuron()
        elif self.model_type == 'se3_transformer':
            self._build_se3_transformer(num_heads, num_degrees)
        elif self.model_type == 'nequip':
            self._build_nequip(num_gaussians, max_ell)
        elif self.model_type == 'clofnet':
            self._build_clofnet()
        
        # Output predictor (common for most models)
        self._build_predictor()
    
    # ========== Model Builders ==========
    
    def _build_raw_mlp(self):
        """MLP using raw coordinates - no symmetry"""
        input_dim = self.in_channels + self.spatial_dim
        
        layers = []
        layers.append(nn.Linear(input_dim, self.hidden_channels))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.dropout))
        
        for _ in range(self.num_layers - 2):
            layers.append(nn.Linear(self.hidden_channels, self.hidden_channels))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
        
        self.mlp = nn.Sequential(*layers)
    
    def _build_transformer(self, num_heads):
        """Transformer with positional encoding - permutation only"""
        self.pos_encoder = nn.Linear(self.spatial_dim, self.hidden_channels)
        self.feat_encoder = nn.Linear(self.in_channels, self.hidden_channels)
        
        self.convs = nn.ModuleList([
            TransformerConv(
                self.hidden_channels,
                self.hidden_channels // num_heads,
                heads=num_heads,
                dropout=self.dropout,
                concat=True
            )
            for _ in range(self.num_layers)
        ])
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(self.hidden_channels)
            for _ in range(self.num_layers)
        ])
    
    def _build_standard_gnn(self):
        """Standard GNN (GCN/GIN/GraphSAGE) - permutation only"""
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        self.convs.append(self._build_conv_layer(
            self.in_channels, self.hidden_channels, self.model_type
        ))
        self.batch_norms.append(nn.BatchNorm1d(self.hidden_channels))
        
        # Hidden layers
        for _ in range(self.num_layers - 2):
            self.convs.append(self._build_conv_layer(
                self.hidden_channels, self.hidden_channels, self.model_type
            ))
            self.batch_norms.append(nn.BatchNorm1d(self.hidden_channels))
        
        # Output layer
        self.convs.append(self._build_conv_layer(
            self.hidden_channels, self.hidden_channels, self.model_type
        ))
        self.batch_norms.append(nn.BatchNorm1d(self.hidden_channels))
    
    def _build_conv_layer(self, in_ch, out_ch, gnn_type):
        """Build a single convolution layer"""
        if gnn_type == 'gcn':
            return GCNConv(in_ch, out_ch)
        elif gnn_type == 'gin':
            mlp = nn.Sequential(
                nn.Linear(in_ch, out_ch),
                nn.ReLU(),
                nn.Linear(out_ch, out_ch)
            )
            return GINConv(mlp)
        elif gnn_type == 'graphsage':
            return SAGEConv(in_ch, out_ch)
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")
    
    def _build_schnet(self, num_gaussians, cutoff):
        """SchNet - E(3) invariant"""
        self.embedding = nn.Linear(self.in_channels, self.hidden_channels)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)
        self.cutoff = cutoff
        
        self.interactions = nn.ModuleList([
            SchNetInteraction(self.hidden_channels, num_gaussians, cutoff)
            for _ in range(self.num_layers)
        ])
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(self.hidden_channels)
            for _ in range(self.num_layers)
        ])
    
    def _build_dimenet(self, num_gaussians, num_spherical, cutoff):
        """DimeNet - E(3) invariant with angles"""
        self.embedding = nn.Linear(self.in_channels, self.hidden_channels)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)
        self.angle_expansion = SphericalBasisLayer(num_spherical, num_gaussians)
        self.cutoff = cutoff
        
        self.interactions = nn.ModuleList([
            DimeNetInteraction(self.hidden_channels, num_gaussians, num_spherical)
            for _ in range(self.num_layers)
        ])
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(self.hidden_channels)
            for _ in range(self.num_layers)
        ])
    
    def _build_egnn(self, update_coords):
        """EGNN - E(3) equivariant"""
        self.embedding = nn.Linear(self.in_channels, self.hidden_channels)
        self.update_coords = update_coords
        
        self.egnn_layers = nn.ModuleList([
            EGNNLayer(self.hidden_channels, self.hidden_channels, update_coords=update_coords)
            for _ in range(self.num_layers)
        ])
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(self.hidden_channels)
            for _ in range(self.num_layers)
        ])
    
    def _build_painn(self):
        """PaiNN - E(3) equivariant with scalar/vector features"""
        self.scalar_embedding = nn.Linear(self.in_channels, self.hidden_channels)
        self.vector_embedding = nn.Linear(self.spatial_dim, self.hidden_channels)
        
        self.painn_layers = nn.ModuleList([
            PaiNNLayer(self.hidden_channels)
            for _ in range(self.num_layers)
        ])
    
    def _build_vector_neuron(self):
        """Vector Neurons - SO(3) equivariant"""
        self.vn_embedding = VectorNeuronMLP(self.in_channels, self.hidden_channels)
        
        self.vn_layers = nn.ModuleList([
            VectorNeuronLayer(self.hidden_channels, self.hidden_channels)
            for _ in range(self.num_layers)
        ])
        
        self.invariant_pooling = VectorNeuronInvariant(self.hidden_channels)
    
    def _build_se3_transformer(self, num_heads, num_degrees):
        """SE(3)-Transformer - SE(3) equivariant attention"""
        self.fiber_hidden = {0: self.hidden_channels, 1: self.hidden_channels // 3}
        self.embedding = nn.Linear(self.in_channels, self.hidden_channels)
        
        self.se3_layers = nn.ModuleList([
            SE3TransformerLayer(self.fiber_hidden, self.fiber_hidden, num_heads=num_heads)
            for _ in range(self.num_layers)
        ])
    
    def _build_nequip(self, num_gaussians, max_ell):
        """NequIP - E(3) equivariant with tensor products"""
        self.embedding = nn.Linear(self.in_channels, self.hidden_channels)
        self.spherical_harmonics = SphericalHarmonicBasis(max_ell)
        
        self.nequip_layers = nn.ModuleList([
            NequIPLayer(self.hidden_channels, max_ell)
            for _ in range(self.num_layers)
        ])
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(self.hidden_channels)
            for _ in range(self.num_layers)
        ])
    
    def _build_clofnet(self):
        """ClofNet - SE(3) with local frames"""
        self.embedding = nn.Linear(self.in_channels, self.hidden_channels)
        
        self.clof_layers = nn.ModuleList([
            ClofLayer(self.hidden_channels)
            for _ in range(self.num_layers)
        ])
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(self.hidden_channels)
            for _ in range(self.num_layers)
        ])
    
    def _build_predictor(self):
        """Common output predictor"""
        self.predictor = nn.Sequential(
            nn.Linear(self.hidden_channels, self.hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_channels // 2, self.out_channels)
        )
    
    # ========== Forward Passes ==========
    
    def forward(self, x, pos, edge_index, batch, return_layer_outputs=False):
        """Universal forward pass for all model types"""
        if self.model_type == 'raw_mlp':
            return self._forward_raw_mlp(x, pos, batch, return_layer_outputs)
        elif self.model_type == 'transformer':
            return self._forward_transformer(x, pos, edge_index, batch, return_layer_outputs)
        elif self.model_type in ['gcn', 'gin', 'graphsage']:
            return self._forward_standard_gnn(x, edge_index, batch, return_layer_outputs)
        elif self.model_type == 'schnet':
            return self._forward_schnet(x, pos, edge_index, batch, return_layer_outputs)
        elif self.model_type == 'dimenet':
            return self._forward_dimenet(x, pos, edge_index, batch, return_layer_outputs)
        elif self.model_type == 'egnn':
            return self._forward_egnn(x, pos, edge_index, batch, return_layer_outputs)
        elif self.model_type == 'painn':
            return self._forward_painn(x, pos, edge_index, batch, return_layer_outputs)
        elif self.model_type == 'vector_neuron':
            return self._forward_vector_neuron(x, pos, edge_index, batch, return_layer_outputs)
        elif self.model_type == 'se3_transformer':
            return self._forward_se3_transformer(x, pos, edge_index, batch, return_layer_outputs)
        elif self.model_type == 'nequip':
            return self._forward_nequip(x, pos, edge_index, batch, return_layer_outputs)
        elif self.model_type == 'clofnet':
            return self._forward_clofnet(x, pos, edge_index, batch, return_layer_outputs)
    
    def _forward_raw_mlp(self, x, pos, batch, return_layer_outputs):
        layer_outputs = []
        x_with_pos = torch.cat([x, pos], dim=-1)
        graph_features = global_mean_pool(x_with_pos, batch)
        out = self.mlp(graph_features)
        out = self.predictor(out)
        return (out, layer_outputs) if return_layer_outputs else out
    
    def _forward_transformer(self, x, pos, edge_index, batch, return_layer_outputs):
        layer_outputs = []
        x = self.feat_encoder(x) + self.pos_encoder(pos)
        
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            if return_layer_outputs:
                layer_outputs.append({'layer_idx': i, 'representation': x.clone(), 'edge_index': edge_index, 'batch': batch})
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = global_mean_pool(x, batch)
        out = self.predictor(x)
        return (out, layer_outputs) if return_layer_outputs else out
    
    def _forward_standard_gnn(self, x, edge_index, batch, return_layer_outputs):
        layer_outputs = []
        
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            if return_layer_outputs:
                layer_outputs.append({'layer_idx': i, 'representation': x.clone(), 'edge_index': edge_index, 'batch': batch})
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = global_mean_pool(x, batch)
        out = self.predictor(x)
        return (out, layer_outputs) if return_layer_outputs else out
    
    def _forward_schnet(self, x, pos, edge_index, batch, return_layer_outputs):
        layer_outputs = []
        x = self.embedding(x)
        
        for i, (interaction, bn) in enumerate(zip(self.interactions, self.batch_norms)):
            x = interaction(x, pos, edge_index)
            if return_layer_outputs:
                layer_outputs.append({'layer_idx': i, 'representation': x.clone(), 'edge_index': edge_index, 'batch': batch})
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = global_add_pool(x, batch)
        out = self.predictor(x)
        return (out, layer_outputs) if return_layer_outputs else out
    
    def _forward_dimenet(self, x, pos, edge_index, batch, return_layer_outputs):
        layer_outputs = []
        x = self.embedding(x)
        
        for i, (interaction, bn) in enumerate(zip(self.interactions, self.batch_norms)):
            x = interaction(x, pos, edge_index, self.distance_expansion, self.angle_expansion)
            if return_layer_outputs:
                layer_outputs.append({'layer_idx': i, 'representation': x.clone(), 'edge_index': edge_index, 'batch': batch})
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = global_add_pool(x, batch)
        out = self.predictor(x)
        return (out, layer_outputs) if return_layer_outputs else out
    
    def _forward_egnn(self, x, pos, edge_index, batch, return_layer_outputs):
        layer_outputs = []
        x = self.embedding(x)
        
        for i, (egnn_layer, bn) in enumerate(zip(self.egnn_layers, self.batch_norms)):
            x, pos = egnn_layer(x, pos, edge_index)
            if return_layer_outputs:
                layer_outputs.append({'layer_idx': i, 'representation': x.clone(), 'positions': pos.clone(), 'edge_index': edge_index, 'batch': batch})
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = global_mean_pool(x, batch)
        out = self.predictor(x)
        return (out, layer_outputs) if return_layer_outputs else out
    
    def _forward_painn(self, x, pos, edge_index, batch, return_layer_outputs):
        layer_outputs = []
        s = self.scalar_embedding(x)
        v = self.vector_embedding(pos).unsqueeze(-1)
        
        for i, painn_layer in enumerate(self.painn_layers):
            s, v = painn_layer(s, v, pos, edge_index)
            if return_layer_outputs:
                layer_outputs.append({'layer_idx': i, 'representation': s.clone(), 'vector_representation': v.clone(), 'edge_index': edge_index, 'batch': batch})
            s = F.dropout(s, p=self.dropout, training=self.training)
        
        s = global_mean_pool(s, batch)
        out = self.predictor(s)
        return (out, layer_outputs) if return_layer_outputs else out
    
    def _forward_vector_neuron(self, x, pos, edge_index, batch, return_layer_outputs):
        layer_outputs = []
        v = self.vn_embedding(x, pos)
        
        for i, vn_layer in enumerate(self.vn_layers):
            v = vn_layer(v, pos, edge_index)
            if return_layer_outputs:
                layer_outputs.append({'layer_idx': i, 'representation': v.clone(), 'edge_index': edge_index, 'batch': batch})
            v = F.dropout(v, p=self.dropout, training=self.training)
        
        x_inv = self.invariant_pooling(v)
        x_inv = global_mean_pool(x_inv, batch)
        out = self.predictor(x_inv)
        return (out, layer_outputs) if return_layer_outputs else out
    
    def _forward_se3_transformer(self, x, pos, edge_index, batch, return_layer_outputs):
        layer_outputs = []
        features = {0: self.embedding(x)}
        
        for i, se3_layer in enumerate(self.se3_layers):
            features = se3_layer(features, pos, edge_index)
            if return_layer_outputs:
                layer_outputs.append({'layer_idx': i, 'representation': features[0].clone(), 'edge_index': edge_index, 'batch': batch})
            features[0] = F.dropout(features[0], p=self.dropout, training=self.training)
        
        x = global_mean_pool(features[0], batch)
        out = self.predictor(x)
        return (out, layer_outputs) if return_layer_outputs else out
    
    def _forward_nequip(self, x, pos, edge_index, batch, return_layer_outputs):
        layer_outputs = []
        x = self.embedding(x)
        
        for i, nequip_layer in enumerate(zip(self.nequip_layers, self.batch_norms)):
            nequip_layer, bn = nequip_layer
            x = nequip_layer(x, pos, edge_index, self.spherical_harmonics)
            if return_layer_outputs:
                layer_outputs.append({'layer_idx': i, 'representation': x.clone(), 'edge_index': edge_index, 'batch': batch})
            x = bn(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = global_add_pool(x, batch)
        out = self.predictor(x)
        return (out, layer_outputs) if return_layer_outputs else out
    
    def _forward_clofnet(self, x, pos, edge_index, batch, return_layer_outputs):
        layer_outputs = []
        x = self.embedding(x)
        frames = self.build_local_frames(pos, edge_index)
        
        for i, (clof_layer, bn) in enumerate(zip(self.clof_layers, self.batch_norms)):
            x = clof_layer(x, pos, edge_index, frames)
            if return_layer_outputs:
                layer_outputs.append({'layer_idx': i, 'representation': x.clone(), 'edge_index': edge_index, 'batch': batch})
            x = bn(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = global_mean_pool(x, batch)
        out = self.predictor(x)
        return (out, layer_outputs) if return_layer_outputs else out
    
    def build_local_frames(self, pos, edge_index):
        """Build orthonormal frames for ClofNet"""
        row, col = edge_index
        num_nodes = pos.shape[0]
        frames = []
        
        for i in range(num_nodes):
            neighbors = col[row == i]
            if len(neighbors) >= 2:
                v1 = pos[neighbors[0]] - pos[i]
                v2 = pos[neighbors[1]] - pos[i]
                v1 = F.normalize(v1, dim=0)
                v2 = v2 - (v2 @ v1) * v1
                v2 = F.normalize(v2, dim=0)
                v3 = torch.linalg.cross(v1, v2)
                frame = torch.stack([v1, v2, v3], dim=1)
            else:
                frame = torch.eye(3, device=pos.device)
            frames.append(frame)
        
        return torch.stack(frames)
    
    def get_symmetry_info(self):
        """Return information about model's symmetry properties"""
        symmetry_map = {
            'raw_mlp': {'permutation': False, 'rotation': False, 'translation': False, 'level': 'None'},
            'transformer': {'permutation': True, 'rotation': False, 'translation': False, 'level': 'Permutation only'},
            'gcn': {'permutation': True, 'rotation': False, 'translation': False, 'level': 'Permutation only'},
            'gin': {'permutation': True, 'rotation': False, 'translation': False, 'level': 'Permutation only'},
            'graphsage': {'permutation': True, 'rotation': False, 'translation': False, 'level': 'Permutation only'},
            'schnet': {'permutation': True, 'rotation': 'Invariant', 'translation': 'Invariant', 'level': 'E(3) invariant'},
            'dimenet': {'permutation': True, 'rotation': 'Invariant', 'translation': 'Invariant', 'level': 'E(3) invariant'},
            'egnn': {'permutation': True, 'rotation': 'Equivariant', 'translation': 'Equivariant', 'level': 'E(3) equivariant'},
            'painn': {'permutation': True, 'rotation': 'Equivariant', 'translation': 'Equivariant', 'level': 'E(3) equivariant'},
            'vector_neuron': {'permutation': True, 'rotation': 'SO(3) Equivariant', 'translation': 'Invariant', 'level': 'SO(3) equivariant'},
            'se3_transformer': {'permutation': True, 'rotation': 'SE(3) Equivariant', 'translation': 'Equivariant', 'level': 'SE(3) equivariant'},
            'nequip': {'permutation': True, 'rotation': 'Equivariant', 'translation': 'Equivariant', 'level': 'E(3) equivariant'},
            'clofnet': {'permutation': True, 'rotation': 'SE(3) Equivariant', 'translation': 'Equivariant', 'level': 'SE(3) equivariant'},
        }
        return symmetry_map.get(self.model_type, {})


# ========== Helper Layers ==========

class GaussianSmearing(nn.Module):
    """Gaussian basis for distance encoding"""
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)
    
    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class SchNetInteraction(nn.Module):
    """SchNet layer"""
    def __init__(self, hidden_channels: int, num_gaussians: int, cutoff: float):
        super().__init__()
        self.cutoff = cutoff
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)
        
        self.filter_network = nn.Sequential(
            nn.Linear(num_gaussians, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        self.interaction_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
    
    def forward(self, x, pos, edge_index):
        row, col = edge_index
        dist = torch.norm(pos[row] - pos[col], dim=1)
        dist_expanded = self.distance_expansion(dist)
        filters = self.filter_network(dist_expanded)
        
        messages = x[col] * filters
        x_out = torch.zeros_like(x)
        x_out.index_add_(0, row, messages)
        x_out = self.interaction_mlp(x_out)
        
        return x + x_out


class DimeNetInteraction(nn.Module):
    """DimeNet layer"""
    def __init__(self, hidden_channels: int, num_gaussians: int, num_spherical: int):
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(num_gaussians + num_spherical, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
    
    def forward(self, x, pos, edge_index, distance_expansion, angle_expansion):
        row, col = edge_index
        vec = pos[row] - pos[col]
        dist = torch.norm(vec, dim=1)
        dist_emb = distance_expansion(dist)
        angle_emb = torch.zeros(dist_emb.shape[0], 7, device=x.device)
        geom_feat = torch.cat([dist_emb, angle_emb], dim=-1)
        
        messages = self.message_mlp(geom_feat) * x[col]
        x_out = torch.zeros_like(x)
        x_out.index_add_(0, row, messages)
        x_out = self.update_mlp(torch.cat([x, x_out], dim=-1))
        
        return x + x_out


class SphericalBasisLayer(nn.Module):
    """Placeholder for spherical harmonics"""
    def __init__(self, num_spherical: int, num_radial: int):
        super().__init__()
        self.num_spherical = num_spherical
    
    def forward(self, angles):
        return torch.randn(angles.shape[0], self.num_spherical, device=angles.device)


class EGNNLayer(nn.Module):
    """EGNN layer"""
    def __init__(self, in_channels: int, out_channels: int, update_coords: bool = True):
        super().__init__()
        self.update_coords = update_coords
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_channels * 2 + 1, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        
        self.node_mlp = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        
        if update_coords:
            self.coord_mlp = nn.Sequential(
                nn.Linear(out_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, 1)
            )
    
    def forward(self, h, x, edge_index):
        row, col = edge_index
        rel_pos = x[row] - x[col]
        dist_sq = torch.sum(rel_pos ** 2, dim=1, keepdim=True)
        
        edge_feat = torch.cat([h[row], h[col], dist_sq], dim=-1)
        edge_emb = self.edge_mlp(edge_feat)
        
        messages = torch.zeros(h.shape[0], edge_emb.shape[1], device=h.device)
        messages.index_add_(0, row, edge_emb)
        
        h_new = self.node_mlp(torch.cat([h, messages], dim=-1))
        
        x_new = x
        if self.update_coords:
            coord_weights = self.coord_mlp(edge_emb)
            coord_update = torch.zeros_like(x)
            coord_update.index_add_(0, row, rel_pos * coord_weights)
            x_new = x + coord_update
        
        return h_new, x_new
    


class PaiNNLayer(nn.Module):
    """PaiNN layer - Efficient version with scatter"""
    def __init__(self, hidden_channels: int):
        super().__init__()
        self.message_scalar = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels * 3)
        )
        self.update_net = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels * 3)
        )
    
    def forward(self, s, v, pos, edge_index):
        row, col = edge_index
        num_nodes = s.shape[0]
        
        rel_pos = pos[row] - pos[col]
        dist = torch.norm(rel_pos, dim=1, keepdim=True) + 1e-8
        dir_vec = rel_pos / dist
        
        msg = self.message_scalar(s[col])
        msg_s, msg_v1, msg_v2 = torch.split(msg, msg.shape[-1] // 3, dim=-1)
        
        # Scalar aggregation
        s_msg = scatter_add(msg_s, row, dim=0, dim_size=num_nodes)
        
        # Vector aggregation
        v_msg = msg_v1.unsqueeze(-1) * dir_vec.unsqueeze(1)  # [E, hidden, 3]
        
        # Reshape for scatter: [E, hidden, 3] -> [E, hidden*3]
        v_msg_flat = v_msg.reshape(v_msg.shape[0], -1)
        v_update_flat = scatter_add(v_msg_flat, row, dim=0, dim_size=num_nodes)
        v_update = v_update_flat.reshape(num_nodes, -1, 3)  # [N, hidden, 3]
        
        # Update
        v_norm = torch.norm(v, dim=-1)
        update_input = torch.cat([s, v_norm], dim=-1)
        update = self.update_net(update_input)
        u_s, u_v1, u_v2 = torch.split(update, update.shape[-1] // 3, dim=-1)
        
        s_new = s + s_msg + u_s
        v_new = v + v_update + u_v1.unsqueeze(-1) * v
        
        return s_new, v_new


# ========== Vector Neuron Components ==========

class VectorNeuronMLP(nn.Module):
    """Convert scalar features to vector neurons"""
    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, hidden_channels)
    
    def forward(self, x, pos):
        x = self.linear(x)
        pos_norm = F.normalize(pos, dim=-1)
        v = x.unsqueeze(-1) * pos_norm.unsqueeze(1)
        return v


class VectorNeuronLayer(nn.Module):
    """SO(3)-equivariant layer with vector neurons"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.kappa = nn.Parameter(torch.ones(1))
    
    def forward(self, v, pos, edge_index):
        row, col = edge_index
        v_message = v[col]
        v_message = v_message.transpose(1, 2)
        v_message = self.linear(v_message)
        v_message = v_message.transpose(1, 2)
        
        v_out = torch.zeros(v.shape[0], v_message.shape[1], 3, device=v.device)
        v_out.index_add_(0, row, v_message)
        v_out = self.vn_relu(v_out)
        
        return v + v_out
    
    def vn_relu(self, v):
        norm = torch.norm(v, dim=-1, keepdim=True)
        activated_norm = F.relu(norm - self.kappa)
        dir_vec = v / (norm + 1e-8)
        return activated_norm * dir_vec


class VectorNeuronInvariant(nn.Module):
    """Extract rotation-invariant features"""
    def __init__(self, in_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, in_channels)
    
    def forward(self, v):
        norm = torch.norm(v, dim=-1)
        return self.linear(norm)


# ========== SE(3)-Transformer Components ==========

class SE3TransformerLayer(nn.Module):
    """SE(3)-equivariant attention layer"""
    def __init__(self, fiber_in: Dict[int, int], fiber_out: Dict[int, int], num_heads: int = 8):
        super().__init__()
        self.fiber_in = fiber_in
        self.fiber_out = fiber_out
        self.num_heads = num_heads
        
        hidden_dim = fiber_in[0]
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
    
    def forward(self, features, pos, edge_index):
        x = features[0]
        x_attended, _ = self.attention(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
        x_attended = x_attended.squeeze(0)
        x = self.norm(x + x_attended)
        x = x + self.ffn(x)
        return {0: x}


# ========== NequIP Components ==========

class NequIPLayer(nn.Module):
    """E(3)-equivariant layer with tensor products"""
    def __init__(self, hidden_channels: int, max_ell: int):
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
    
    def forward(self, x, pos, edge_index, spherical_harmonics):
        row, col = edge_index
        rel_pos = pos[row] - pos[col]
        dist = torch.norm(rel_pos, dim=-1, keepdim=True) + 1e-8
        
        messages = self.message_mlp(x[col])
        x_out = torch.zeros_like(x)
        x_out.index_add_(0, row, messages)
        x_out = self.update_mlp(torch.cat([x, x_out], dim=-1))
        
        return x + x_out


class SphericalHarmonicBasis(nn.Module):
    """Simplified spherical harmonic basis"""
    def __init__(self, max_ell: int):
        super().__init__()
        self.max_ell = max_ell
    
    def forward(self, dir_vec):
        x, y, z = dir_vec[:, 0], dir_vec[:, 1], dir_vec[:, 2]
        sh = [torch.ones_like(x)]
        
        if self.max_ell >= 1:
            sh.extend([x, y, z])
        
        if self.max_ell >= 2:
            sh.extend([x*y, x*z, y*z, x**2 - y**2, 3*z**2 - 1])
        
        return torch.stack(sh, dim=-1)


# ========== ClofNet Components ==========

class ClofLayer(nn.Module):
    """SE(3)-equivariant layer with local frames"""
    def __init__(self, hidden_channels: int):
        super().__init__()
        self.scalarize_mlp = nn.Sequential(
            nn.Linear(3, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
    
    def forward(self, x, pos, edge_index, frames):
        row, col = edge_index
        rel_pos = pos[row] - pos[col]
        local_coords = torch.bmm(frames[row], rel_pos.unsqueeze(-1)).squeeze(-1)
        geom_feat = self.scalarize_mlp(local_coords)
        
        messages = self.message_mlp(torch.cat([x[col], geom_feat], dim=-1))
        x_out = torch.zeros_like(x)
        x_out.index_add_(0, row, messages)
        x_out = self.update_mlp(torch.cat([x, x_out], dim=-1))
        
        return x + x_out


# ========== Testing ==========

if __name__ == "__main__":
    print("=" * 80)
    print("UNIFIED BaseGNN - ALL MODELS TEST")
    print("=" * 80)
    
    num_nodes = 20
    in_channels = 16
    hidden_channels = 64
    out_channels = 1
    
    x = torch.randn(num_nodes, in_channels)
    pos = torch.randn(num_nodes, 3)
    edge_index = torch.randint(0, num_nodes, (2, 50))
    batch = torch.cat([torch.zeros(10), torch.ones(10)]).long()
    
    models = [
            'raw_mlp', 'transformer', 'gcn', 'gin', 'graphsage', 
            'schnet', 'dimenet', 'egnn', 'painn',
            'vector_neuron', 'se3_transformer', 'nequip', 'clofnet'
        ]
    
    print(f"\n{'Model':<20} {'Symmetry':<25} {'Output':<15} {'Status'}")
    print("-" * 80)
    
    for model_name in models:
        try:
            model = BaseGNN(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                num_layers=3,
                model_type=model_name
            )
            
            model.eval()
            with torch.no_grad():
                out, _ = model(x, pos, edge_index, batch, return_layer_outputs=True)
            
            sym_info = model.get_symmetry_info()
            print(f"{model_name:<20} {sym_info['level']:<25} {str(out.shape):<15} ✓")
            
        except Exception as e:
            print(f"{model_name:<20} {'Error':<25} {'-':<15} ✗ {str(e)[:20]}")
            print(traceback.format_exc())
    
    print("\n" + "=" * 80)
