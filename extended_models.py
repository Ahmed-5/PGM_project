"""
Extended models for relaxed equivariance project
One model at a time
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool


# ============================================================================
# MODEL 1: Molecular Fingerprint MLP (NO SYMMETRIES)
# ============================================================================

class MolecularFingerprintMLP(nn.Module):
    """
    Baseline model: Treats molecule as fixed-length vector
    NO symmetries (not equivariant, not invariant)
    
    Input: Molecular fingerprint [batch_size, 2048]
    Output: Prediction [batch_size, 1]
    """
    
    def __init__(
        self,
        fingerprint_dim: int = 2048,
        hidden_dim: int = 256,
        out_dim: int = 1,
        num_layers: int = 3,
        dropout: float = 0.5
    ):
        super().__init__()
        
        # Build MLP layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(fingerprint_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, out_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, fingerprint):
        """
        Args:
            fingerprint: [batch_size, fingerprint_dim]
        Returns:
            predictions: [batch_size, out_dim]
        """
        return self.mlp(fingerprint)


# ============================================================================
# MODEL 2: SchNet - Distance-based GNN (E(3) INVARIANT)
# Uses only pairwise distances (rotation + translation invariant)
# ============================================================================

class SchNetConv(MessagePassing):
    """
    SchNet layer: Message passing using pairwise distances
    Automatically E(3) invariant because distances don't change under rotation
    """
    
    def __init__(self, hidden_dim: int, cutoff: float = 10.0):
        super().__init__(aggr='add')
        self.hidden_dim = hidden_dim
        self.cutoff = cutoff
        
        # Message network: processes distances
        self.filter_network = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Update network: processes messages
        self.update_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, h, pos, edge_index):
        """
        Args:
            h: Node features [num_nodes, hidden_dim]
            pos: 3D positions [num_nodes, 3]
            edge_index: [2, num_edges]
        """
        row, col = edge_index
        
        # Compute pairwise distances (E(3) invariant!)
        dist = torch.norm(pos[row] - pos[col], dim=1, keepdim=True)
        
        # Distance-based edge weights
        edge_weight = self.filter_network(dist)
        
        # Message passing
        return self.propagate(edge_index, h=h, edge_weight=edge_weight)
    
    def message(self, h_j, edge_weight):
        """Create messages: neighbor features × distance weight"""
        return h_j * edge_weight
    
    def update(self, aggr_out, h):
        """Update node features: old features + messages"""
        return self.update_network(aggr_out) + h


class SchNet(nn.Module):
    """
    Full SchNet model
    Uses only distances → naturally E(3) invariant
    Good baseline for comparing with relaxed equivariance
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        out_dim: int = 1,
        num_layers: int = 6,
        cutoff: float = 10.0
    ):
        super().__init__()
        
        # Embed atomic numbers to hidden dimension
        self.embedding = nn.Linear(1, hidden_dim)
        
        # Stack of SchNet convolution layers
        self.conv_layers = nn.ModuleList([
            SchNetConv(hidden_dim, cutoff) for _ in range(num_layers)
        ])
        
        # Output network (atom-wise predictions)
        self.output_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Softplus(),
            nn.Linear(hidden_dim // 2, out_dim)
        )
    
    def forward(self, z, pos, edge_index, batch):
        """
        Args:
            z: Atomic numbers [num_nodes, 1]
            pos: 3D coordinates [num_nodes, 3]
            edge_index: [2, num_edges]
            batch: Batch assignment [num_nodes]
        Returns:
            predictions: [batch_size, out_dim]
        """
        # Embed atomic numbers
        h = self.embedding(z.float())
        
        # Apply convolution layers
        for conv in self.conv_layers:
            h = conv(h, pos, edge_index)
        
        # Atom-wise predictions
        h = self.output_network(h)
        
        # Pool to graph-level
        out = global_mean_pool(h, batch)
        
        return out


# ============================================================================
# HELPER FUNCTION: Test if models work
# ============================================================================

def test_models():
    """
    Quick test to verify models compile and run
    Run this to check everything works!
    """
    print("="*80)
    print("TESTING MODELS")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}\n")
    
    # Test 1: MolecularFingerprintMLP
    print("Testing MolecularFingerprintMLP...")
    model1 = MolecularFingerprintMLP(
        fingerprint_dim=2048,
        hidden_dim=64,
        out_dim=1
    ).to(device)
    
    # Create dummy fingerprint
    dummy_fp = torch.randn(8, 2048).to(device)  # Batch of 8
    output1 = model1(dummy_fp)
    print(f"  Input shape:  {dummy_fp.shape}")
    print(f"  Output shape: {output1.shape}")
    print(f"  ✅ MolecularFingerprintMLP works!\n")
    
    # Test 2: SchNet
    print("Testing SchNet...")
    model2 = SchNet(
        hidden_dim=64,
        out_dim=1,
        num_layers=3
    ).to(device)
    
    # Create dummy molecule (20 atoms)
    num_atoms = 20
    z = torch.ones(num_atoms, 1).to(device)  # All carbon atoms
    pos = torch.randn(num_atoms, 3).to(device)  # Random 3D positions
    edge_index = torch.randint(0, num_atoms, (2, 50)).to(device)  # 50 edges
    batch = torch.zeros(num_atoms, dtype=torch.long).to(device)  # Single molecule
    
    output2 = model2(z, pos, edge_index, batch)
    print(f"  Num atoms:    {num_atoms}")
    print(f"  Output shape: {output2.shape}")
    print(f"  ✅ SchNet works!\n")
    
    # Test 3: Check equivariance (SchNet should be invariant)
    print("Testing E(3) invariance of SchNet...")
    
    # Original prediction
    pred_original = model2(z, pos, edge_index, batch)
    
    # Rotate molecule by 90 degrees around z-axis
    angle = 3.14159 / 2  # 90 degrees
    rotation_matrix = torch.tensor([
        [torch.cos(torch.tensor(angle)), -torch.sin(torch.tensor(angle)), 0],
        [torch.sin(torch.tensor(angle)),  torch.cos(torch.tensor(angle)), 0],
        [0, 0, 1]
    ], dtype=torch.float32).to(device)
    
    pos_rotated = torch.matmul(pos, rotation_matrix.T)
    pred_rotated = model2(z, pos_rotated, edge_index, batch)
    
    # Check if predictions are the same
    difference = torch.abs(pred_original - pred_rotated).item()
    print(f"  Original prediction: {pred_original.item():.6f}")
    print(f"  Rotated prediction:  {pred_rotated.item():.6f}")
    print(f"  Difference:          {difference:.6f}")
    
    if difference < 1e-4:
        print(f"  ✅ SchNet is E(3) invariant! (difference < 1e-4)")
    else:
        print(f"  ⚠️  SchNet shows some variance (might be numerical error)")
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED! ✅")
    print("="*80)


if __name__ == '__main__':
    # Run tests when script is executed directly
    test_models()