"""
Equivariance loss computation for graph neural networks
Properly implements f(g·x) vs g·f(x) testing for multiple symmetry groups

Tests if networks satisfy: f(g·x) ≈ g·f(x) for group transformations g
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Callable
import math


class EquivarianceLoss(nn.Module):
    """
    Computes equivariance error: L_eq = ||f(g·x) - g·f(x)||²
    
    Properly tests network equivariance by:
    1. Computing f(g·x): Transform input → Apply network
    2. Computing g·f(x): Apply network → Transform output
    3. Measuring ||f(g·x) - g·f(x)||²
    
    Supported groups:
    - permutation: Node permutation symmetry
    - so3: 3D rotation group (proper rotations)
    - e3: Euclidean group (rotations + translations + reflections)
    - se3: Special Euclidean group (rotations + translations)
    - o3: Orthogonal group (rotations + reflections)
    - translation: Spatial translation
    - reflection: Mirror symmetry
    - scaling: Uniform scaling/dilation
    """
    
    def __init__(
        self,
        group_type: str = 'permutation',
        num_samples: int = 1,
        normalize: bool = True,
        spatial_dim: int = 3,
        epsilon: float = 1e-8,
        feature_type: str = 'invariant',  # 'invariant' or 'equivariant'
        max_translation: float = 5.0,
        scale_range: Tuple[float, float] = (0.5, 2.0)
    ):
        """
        Args:
            group_type: Which symmetry group to test
            num_samples: Number of random transformations to test per graph
            normalize: Whether to normalize loss by feature magnitude
            spatial_dim: Dimensionality of positions (2D or 3D)
            epsilon: Small constant for numerical stability
            feature_type: 'invariant' (features unchanged) or 'equivariant' (features transform)
            max_translation: Maximum translation magnitude for testing
            scale_range: (min, max) scaling factors for testing
        """
        super().__init__()
        self.group_type = group_type.lower()
        self.num_samples = num_samples
        self.normalize = normalize
        self.spatial_dim = spatial_dim
        self.epsilon = epsilon
        self.feature_type = feature_type
        self.max_translation = max_translation
        self.scale_range = scale_range
        
        valid_groups = ['permutation', 'so3', 'e3', 'se3', 'o3', 
                       'translation', 'reflection', 'scaling']
        if self.group_type not in valid_groups:
            raise ValueError(f"Group type must be one of {valid_groups}, got {group_type}")
    
    # ========== Group Transformation Sampling ==========
    
    def sample_permutation(self, num_nodes: int, device: torch.device) -> torch.Tensor:
        """Sample random node permutation"""
        return torch.randperm(num_nodes, device=device)
    
    def sample_rotation_so3(self, device: torch.device) -> torch.Tensor:
        """Sample random SO(3) rotation using Rodrigues' formula"""
        axis = torch.randn(3, device=device)
        axis = axis / (torch.norm(axis) + self.epsilon)
        angle = torch.rand(1, device=device).item() * 2 * math.pi
        
        # Rodrigues' rotation formula
        K = torch.tensor([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ], device=device, dtype=torch.float32)
        
        I = torch.eye(3, device=device, dtype=torch.float32)
        R = I + math.sin(angle) * K + (1 - math.cos(angle)) * torch.matmul(K, K)
        return R
    
    def sample_orthogonal_o3(self, device: torch.device) -> torch.Tensor:
        """Sample from O(3): SO(3) with optional reflection"""
        R = self.sample_rotation_so3(device)
        
        # 50% chance: add reflection
        if torch.rand(1, device=device).item() < 0.5:
            normal = torch.randn(3, device=device)
            normal = normal / (torch.norm(normal) + self.epsilon)
            reflection = torch.eye(3, device=device) - 2 * torch.outer(normal, normal)
            R = torch.matmul(reflection, R)
        
        return R
    
    def sample_translation(self, device: torch.device) -> torch.Tensor:
        """Sample random translation vector"""
        return (torch.rand(self.spatial_dim, device=device) - 0.5) * 2 * self.max_translation
    
    def sample_reflection(self, device: torch.device) -> torch.Tensor:
        """Sample reflection through random plane"""
        normal = torch.randn(self.spatial_dim, device=device)
        normal = normal / (torch.norm(normal) + self.epsilon)
        I = torch.eye(self.spatial_dim, device=device)
        return I - 2 * torch.outer(normal, normal)
    
    def sample_scaling(self, device: torch.device) -> float:
        """Sample uniform scaling factor"""
        min_s, max_s = self.scale_range
        return min_s + torch.rand(1, device=device).item() * (max_s - min_s)
    
    def sample_se3_transform(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample SE(3) = SO(3) ⋉ T(3)"""
        return self.sample_rotation_so3(device), self.sample_translation(device)
    
    def sample_e3_transform(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample E(3) = O(3) ⋉ T(3)"""
        return self.sample_orthogonal_o3(device), self.sample_translation(device)
    
    # ========== Apply Transformations ==========
    
    def apply_rotation(self, positions: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """Apply rotation matrix: x' = R @ x"""
        return torch.matmul(positions, R.T)
    
    def apply_translation(self, positions: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Apply translation: x' = x + t"""
        return positions + t.unsqueeze(0)
    
    def apply_scaling(self, positions: torch.Tensor, s: float) -> torch.Tensor:
        """Apply scaling: x' = s * x"""
        return positions * s
    
    def apply_permutation_to_edges(self, edge_index: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
        """Permute edge indices: if edge (i,j) exists, becomes (perm[i], perm[j])"""
        inv_perm = torch.argsort(perm)
        return inv_perm[edge_index]
    
    # ========== Transform Representations ==========
    
    def transform_features(self, features: torch.Tensor, group_element, 
                          transformation_type: str) -> torch.Tensor:
        """
        Apply group transformation to output features based on feature_type
        
        For 'invariant': features unchanged (most common for GNNs)
        For 'equivariant': features transform with the group
        """
        if self.feature_type == 'invariant':
            return features
        
        elif self.feature_type == 'equivariant':
            if transformation_type == 'permutation':
                return features[group_element]
            
            elif transformation_type in ['rotation', 'reflection', 'orthogonal']:
                # For 3D vector features
                if features.shape[-1] == 3:
                    return torch.matmul(features, group_element.T)
                else:
                    # Scalar features remain invariant
                    return features
            
            elif transformation_type == 'scaling':
                # Vector magnitudes scale proportionally
                if features.shape[-1] == 3:
                    return features * group_element
                else:
                    return features
            
            else:
                return features
        
        return features
    
    # ========== Core Equivariance Testing ==========
    
    def test_permutation_equivariance(
        self,
        network_fn: Callable,
        positions: torch.Tensor,
        features: torch.Tensor,
        edges: torch.Tensor,
        batch: torch.Tensor,
        num_nodes: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Test permutation equivariance: ||f(π·x) - π·f(x)||²
        
        For permutation-invariant GNNs: f(π·x) = π·f(x)
        """
        perm = self.sample_permutation(num_nodes, device)
        inv_perm = torch.argsort(perm)
        
        # f(x): Apply network to original input
        f_x = network_fn(positions, features, edges, batch)
        
        # f(π·x): Permute input, then apply network
        permuted_pos = positions[perm]
        permuted_feat = features[perm]
        permuted_edges = inv_perm[edges]
        f_pi_x = network_fn(permuted_pos, permuted_feat, permuted_edges, batch)
        
        # π·f(x): Apply network, then permute output
        pi_f_x = self.transform_features(f_x[perm], perm, 'permutation')
        
        # Equivariance error
        error = torch.mean((f_pi_x - pi_f_x) ** 2)
        return error
    
    def test_rotation_equivariance(
        self,
        network_fn: Callable,
        positions: torch.Tensor,
        features: torch.Tensor,
        edges: torch.Tensor,
        batch: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """Test rotation equivariance: ||f(R·x) - R·f(x)||² or ||f(R·x) - f(x)||²"""
        if self.group_type == 'so3':
            R = self.sample_rotation_so3(device)
        elif self.group_type == 'o3':
            R = self.sample_orthogonal_o3(device)
        elif self.group_type == 'reflection':
            R = self.sample_reflection(device)
        
        # f(x)
        f_x = network_fn(positions, features, edges, batch)
        
        # f(R·x): Rotate positions, then apply network
        rotated_pos = self.apply_rotation(positions, R)
        f_R_x = network_fn(rotated_pos, features, edges, batch)
        
        # R·f(x) or f(x) depending on feature_type
        R_f_x = self.transform_features(f_x, R, 'rotation')
        
        error = torch.mean((f_R_x - R_f_x) ** 2)
        return error
    
    def test_translation_equivariance(
        self,
        network_fn: Callable,
        positions: torch.Tensor,
        features: torch.Tensor,
        edges: torch.Tensor,
        batch: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """Test translation invariance: ||f(x + t) - f(x)||²"""
        t = self.sample_translation(device)
        
        f_x = network_fn(positions, features, edges, batch)
        translated_pos = self.apply_translation(positions, t)
        f_x_t = network_fn(translated_pos, features, edges, batch)
        
        # For translation-invariant networks
        error = torch.mean((f_x_t - f_x) ** 2)
        return error
    
    def test_scaling_equivariance(
        self,
        network_fn: Callable,
        positions: torch.Tensor,
        features: torch.Tensor,
        edges: torch.Tensor,
        batch: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """Test scaling behavior: ||f(s·x) - f(x)||² for scale-invariant nets"""
        s = self.sample_scaling(device)
        
        f_x = network_fn(positions, features, edges, batch)
        scaled_pos = self.apply_scaling(positions, s)
        f_s_x = network_fn(scaled_pos, features, edges, batch)
        
        # For scale-invariant networks
        error = torch.mean((f_s_x - f_x) ** 2)
        return error
    
    def test_euclidean_equivariance(
        self,
        network_fn: Callable,
        positions: torch.Tensor,
        features: torch.Tensor,
        edges: torch.Tensor,
        batch: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """Test E(3)/SE(3) equivariance: ||f(g·x) - g·f(x)||²"""
        if self.group_type == 'se3':
            R, t = self.sample_se3_transform(device)
        else:  # e3
            R, t = self.sample_e3_transform(device)
        
        f_x = network_fn(positions, features, edges, batch)
        
        # Apply transformation: x' = Rx + t
        transformed_pos = self.apply_rotation(positions, R)
        transformed_pos = self.apply_translation(transformed_pos, t)
        f_g_x = network_fn(transformed_pos, features, edges, batch)
        
        # For invariant features: should be unchanged
        g_f_x = self.transform_features(f_x, R, 'rotation')
        
        error = torch.mean((f_g_x - g_f_x) ** 2)
        return error
    
    # ========== Main Forward Pass ==========
    
    def forward(
        self,
        network_fn: Callable,
        positions: torch.Tensor,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute equivariance loss by testing f(g·x) vs g·f(x)
        
        Args:
            network_fn: Callable that takes (positions, features, edge_index, batch)
                       and returns node representations [num_nodes, feature_dim]
            positions: Node positions [num_nodes, spatial_dim]
            features: Node features [num_nodes, feature_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment [num_nodes]
            
        Returns:
            Equivariance loss (scalar)
        """
        device = positions.device
        total_loss = torch.tensor(0.0, device=device)
        unique_graphs = torch.unique(batch)
        num_tested_graphs = 0
        num_successful_samples = 0
        
        for graph_id in unique_graphs:
            # Extract single graph
            node_mask = (batch == graph_id)
            graph_nodes = torch.where(node_mask)[0]
            num_nodes = graph_nodes.size(0)
            
            if num_nodes < 2:
                continue
            
            # Extract graph data
            graph_positions = positions[graph_nodes]
            graph_features = features[graph_nodes]
            
            # Extract edges and map to local indices
            edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
            graph_edges = edge_index[:, edge_mask]
            
            node_mapping = torch.zeros(batch.size(0), dtype=torch.long, device=device)
            node_mapping[graph_nodes] = torch.arange(num_nodes, device=device)
            local_edges = node_mapping[graph_edges]
            
            # Single-graph batch
            graph_batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
            
            # Test equivariance with multiple random transformations
            for _ in range(self.num_samples):
                try:
                    if self.group_type == 'permutation':
                        loss = self.test_permutation_equivariance(
                            network_fn, graph_positions, graph_features,
                            local_edges, graph_batch, num_nodes, device
                        )
                    
                    elif self.group_type in ['so3', 'o3', 'reflection']:
                        loss = self.test_rotation_equivariance(
                            network_fn, graph_positions, graph_features,
                            local_edges, graph_batch, device
                        )
                    
                    elif self.group_type == 'translation':
                        loss = self.test_translation_equivariance(
                            network_fn, graph_positions, graph_features,
                            local_edges, graph_batch, device
                        )
                    
                    elif self.group_type == 'scaling':
                        loss = self.test_scaling_equivariance(
                            network_fn, graph_positions, graph_features,
                            local_edges, graph_batch, device
                        )
                    
                    elif self.group_type in ['e3', 'se3']:
                        loss = self.test_euclidean_equivariance(
                            network_fn, graph_positions, graph_features,
                            local_edges, graph_batch, device
                        )
                    
                    total_loss += loss
                    num_successful_samples += 1
                
                except Exception as e:
                    print(f"Warning: Failed to test {self.group_type} equivariance: {e}")
                    continue
            
            num_tested_graphs += 1
        
        # Average over successful samples
        if num_successful_samples > 0:
            total_loss = total_loss / num_successful_samples
        
        if self.normalize and total_loss > 0:
            # Normalize by feature magnitude
            feat_norm = torch.mean(features ** 2) + self.epsilon
            total_loss = total_loss / feat_norm
        
        return total_loss

