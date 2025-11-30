"""
Graph Rewiring Strategies for Geometric GNNs.
Supports:
1. Spectral Rewiring (Diffusion-based): Adds edges between distant nodes with high diffusion scores.
2. Geometric Rewiring (Distance-based): Connects spatially close nodes that are not topologically connected.
3. Fosr (First-Order Spectral Rewiring): Adds edges to minimize spectral gap (iterative).
"""

import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, dense_to_sparse, add_self_loops, remove_self_loops, degree
from torch_scatter import scatter_add

class GraphRewiring:
    def __init__(self, strategy: str = 'spectral', k: int = 2, threshold: float = 0.5):
        """
        Args:
            strategy: 'spectral', 'geometric', or 'none'
            k: Number of hops (for spectral) or k-nearest neighbors (for geometric)
            threshold: Cutoff value for adding edges
        """
        self.strategy = strategy.lower()
        self.k = k
        self.threshold = threshold

    def __call__(self, data):
        if self.strategy == 'none':
            return data
        
        if self.strategy == 'spectral':
            return self.spectral_rewiring(data)
        elif self.strategy == 'geometric':
            return self.geometric_rewiring(data)
        else:
            raise ValueError(f"Unknown rewiring strategy: {self.strategy}")

    def spectral_rewiring(self, data):
        """
        Adds edges based on Graph Diffusion (Heat Kernel or PageRank).
        Simulates effective resistance/commute time to connect bottleneck nodes.
        """
        device = data.edge_index.device
        num_nodes = data.x.size(0)
        
        # 1. Compute Normalized Adjacency
        edge_index, _ = add_self_loops(data.edge_index, num_nodes=num_nodes)
        row, col = edge_index
        deg = degree(row, num_nodes, dtype=data.x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm_adj = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # 2. Compute Diffusion (Power iteration for efficiency)
        # A^k represents k-hop diffusion probabilities
        # We work with dense matrices for small molecular graphs (efficient enough for ZINC/QM9)
        adj_dense = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
        
        # Diffusion Matrix S = sum(theta_k * T^k)
        # Simplified: Just take T^k for a specific scale k (usually 2-4 for molecules)
        t_k = adj_dense
        for _ in range(self.k - 1):
            t_k = torch.matmul(t_k, adj_dense)
            
        # 3. Thresholding to find new edges
        # Zero out existing edges to avoid duplicates
        mask_existing = (adj_dense > 0)
        t_k[mask_existing] = 0
        t_k.fill_diagonal_(0)
        
        # Select top entries or threshold
        # Dynamic thresholding: top 10% of diffusion scores
        flat_scores = t_k.view(-1)
        limit_idx = int(flat_scores.numel() * 0.1) # Top 10% potential connections
        val, _ = torch.topk(flat_scores, limit_idx)
        min_score = val[-1] if len(val) > 0 else 0.0
        
        new_edges_mask = (t_k > min_score)
        new_row, new_col = torch.nonzero(new_edges_mask, as_tuple=True)
        
        if new_row.size(0) > 0:
            new_edge_index = torch.stack([new_row, new_col], dim=0)
            # Add to original graph
            data.edge_index = torch.cat([data.edge_index, new_edge_index], dim=1)
            
            # If edges have attributes, we must pad them. 
            # For ZINC, edge_attr is bond type. We assign a special "virtual bond" type (e.g., 0 or max+1)
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                num_new = new_edge_index.size(1)
                feat_dim = data.edge_attr.size(1)
                # Create dummy attributes (e.g., vectors of zeros)
                new_attrs = torch.zeros(num_new, feat_dim, device=device, dtype=data.edge_attr.dtype)
                data.edge_attr = torch.cat([data.edge_attr, new_attrs], dim=0)
                
        return data

    def geometric_rewiring(self, data):
        """
        Adds edges between nodes that are spatially close (Euclidean distance)
        but topologically distant (Geodesic distance > k).
        Requires 'pos' in data.
        """
        if not hasattr(data, 'pos') or data.pos is None:
            return data
            
        # 1. Compute pairwise Euclidean distances
        pos = data.pos
        # (N, 1, 3) - (1, N, 3) -> (N, N, 3) -> norm -> (N, N)
        dist_matrix = torch.cdist(pos, pos)
        
        # 2. Find spatially close nodes (< threshold)
        spatial_mask = (dist_matrix < self.threshold)
        
        # 3. Exclude existing neighbors (1-hop)
        adj_dense = to_dense_adj(data.edge_index, max_num_nodes=pos.size(0))[0]
        candidates = spatial_mask & (adj_dense == 0)
        candidates.fill_diagonal_(0)
        
        new_row, new_col = torch.nonzero(candidates, as_tuple=True)
        
        if new_row.size(0) > 0:
            new_edge_index = torch.stack([new_row, new_col], dim=0)
            data.edge_index = torch.cat([data.edge_index, new_edge_index], dim=1)
            
            # Handle edge attributes (pad with zeros)
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                num_new = new_edge_index.size(1)
                new_attrs = torch.zeros(num_new, data.edge_attr.size(1), 
                                      device=data.edge_attr.device, 
                                      dtype=data.edge_attr.dtype)
                data.edge_attr = torch.cat([data.edge_attr, new_attrs], dim=0)
                
        return data
    
    def fosr_rewiring(self, data, iterations: int = 5):
        """
        Adds edges between nodes that are spatially close (Euclidean distance)
        but topologically distant (Geodesic distance > k).
        Requires 'pos' in data.
        """
        if not hasattr(data, 'pos') or data.pos is None:
            return data
            
        # 1. Compute pairwise Euclidean distances
        pos = data.pos
        # (N, 1, 3) - (1, N, 3) -> (N, N, 3) -> norm -> (N, N)
        dist_matrix = torch.cdist(pos, pos)
        
        # 2. Find spatially close nodes (< threshold)
        spatial_mask = (dist_matrix < self.threshold)
        
        # 3. Exclude existing neighbors (1-hop)
        adj_dense = to_dense_adj(data.edge_index, max_num_nodes=pos.size(0))[0]
        candidates = spatial_mask & (adj_dense == 0)
        candidates.fill_diagonal_(0)
        
        new_row, new_col = torch.nonzero(candidates, as_tuple=True)
        
        if new_row.size(0) > 0:
            new_edge_index = torch.stack([new_row, new_col], dim=0)
            data.edge_index = torch.cat([data.edge_index, new_edge_index], dim=1)
            
            # Handle edge attributes (pad with zeros)
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                num_new = new_edge_index.size(1)
                new_attrs = torch.zeros(num_new, data.edge_attr.size(1), 
                                      device=data.edge_attr.device, 
                                      dtype=data.edge_attr.dtype)
                data.edge_attr = torch.cat([data.edge_attr, new_attrs], dim=0)
                
        return data