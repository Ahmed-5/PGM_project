import torch
import traceback
from equivariant_gnn import BaseGNN

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
