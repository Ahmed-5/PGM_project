import torch
import torch.nn as nn
from equivariance_loss import EquivarianceLoss

# ========== Test Networks ==========

class DistanceGNN(nn.Module):
    """
    Simple distance-based GNN (approximately E(3)-invariant)
    Uses pairwise distances, which are invariant to rotations and translations
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, positions, features, edge_index, batch):
        num_nodes = positions.shape[0]
        row, col = edge_index
        
        # Compute pairwise distances (E(3)-invariant)
        dist = torch.norm(positions[row] - positions[col], dim=1, keepdim=True)  # [num_edges, 1]
        
        # Message passing with proper shape handling
        messages = torch.zeros(num_nodes, self.hidden_dim, device=features.device)
        
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            # Concatenate feature vector with scalar distance
            # features[src] is [hidden_dim], dist[i] is [1]
            edge_feat = torch.cat([features[src], dist[i]], dim=0)  # [hidden_dim + 1]
            messages[dst] += self.mlp(edge_feat)
        
        return features + messages


class NonEquivariantGNN(nn.Module):
    """
    Non-equivariant GNN that uses raw coordinates (violates equivariance)
    This should have HIGH equivariance loss
    """
    def __init__(self, hidden_dim=64, spatial_dim=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.spatial_dim = spatial_dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + spatial_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, positions, features, edge_index, batch):
        num_nodes = positions.shape[0]
        row, col = edge_index
        
        messages = torch.zeros(num_nodes, self.hidden_dim, device=features.device)
        
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            # Concatenate features with raw positions (NOT equivariant!)
            edge_feat = torch.cat([features[src], positions[src]], dim=0)  # [hidden_dim + 3]
            messages[dst] += self.mlp(edge_feat)
        
        return features + messages


class PerfectlyInvariantGNN(nn.Module):
    """
    Perfectly invariant GNN that ignores positions entirely
    Should have ZERO equivariance loss for all geometric groups
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, positions, features, edge_index, batch):
        num_nodes = positions.shape[0]
        row, col = edge_index
        
        # Ignore positions entirely (perfectly invariant)
        messages = torch.zeros(num_nodes, self.hidden_dim, device=features.device)
        
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            messages[dst] += self.mlp(features[src])
        
        return features + messages


# ========== Comprehensive Test Suite ==========

if __name__ == "__main__":
    print("=" * 80)
    print("EQUIVARIANCE LOSS TEST SUITE")
    print("=" * 80)
    
    # Test setup
    torch.manual_seed(42)
    num_nodes = 15
    hidden_dim = 64
    num_edges = 50
    
    positions = torch.randn(num_nodes, 3)
    features = torch.randn(num_nodes, hidden_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    batch = torch.zeros(num_nodes, dtype=torch.long)
    
    # Test networks
    networks = {
        'Distance-based GNN (E(3)-invariant)': DistanceGNN(hidden_dim),
        'Non-equivariant GNN (uses raw coords)': NonEquivariantGNN(hidden_dim),
        'Perfectly invariant GNN (ignores pos)': PerfectlyInvariantGNN(hidden_dim)
    }
    
    groups = ['permutation', 'SO3', 'translation', 'SE3', 'E3', 'O3', 'reflection', 'scaling']
    
    for network_name, network in networks.items():
        print(f"\n{network_name}")
        print("-" * 80)
        
        for group in groups:
            try:
                loss_fn = EquivarianceLoss(
                    group_type=group,
                    num_samples=5,
                    feature_type='invariant',
                    normalize=True
                )
                
                def network_fn(pos, feat, edges, b):
                    return network(pos, feat, edges, b)
                
                loss = loss_fn(
                    network_fn=network_fn,
                    positions=positions,
                    features=features,
                    edge_index=edge_index,
                    batch=batch
                )
                
                # Classify loss magnitude
                loss_val = loss.item()
                if loss_val < 1e-6:
                    status = "✓ PERFECT"
                elif loss_val < 1e-3:
                    status = "✓ Good"
                elif loss_val < 1e-1:
                    status = "⚠ Moderate violation"
                else:
                    status = "✗ Strong violation"
                
                print(f"  {group:15} loss: {loss_val:.6f}  {status}")
                
            except Exception as e:
                print(f"  {group:15} ERROR: {str(e)}")
    
    print("\n" + "=" * 80)
    print("Expected behavior:")
    print("  - Distance-based GNN: Low loss for geometric groups (SO3, SE3, E3, translation)")
    print("  - Non-equivariant GNN: HIGH loss for all groups")
    print("  - Perfectly invariant GNN: ZERO loss for geometric groups")
    print("=" * 80)
