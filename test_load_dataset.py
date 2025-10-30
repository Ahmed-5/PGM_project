from load_dataset import load_dataset
import traceback

if __name__ == '__main__':
    # Test dataset loading
    from config import get_config
    
    datasets_to_test = [
        'ZINC', 'QM9', 'QM7b', 'AQSOL',  # Original
        'MD17', 'MD22', 'rMD17', 'OC20',  # Molecular dynamics
        'ISO17', 'Molecule3D', 'ATOM3D',  # Special molecular
        'ModelNet40', 'ShapeNet', 'PartNet'  # Point clouds
    ]
    
    for dataset_name in datasets_to_test:
        print(f"\n{'='*80}")
        print(f"Testing: {dataset_name}")
        print('='*80)
        
        config = get_config('default')
        config.data.dataset_name = dataset_name
        config.data.use_positions = (dataset_name != 'ZINC')
        
        try:
            train, val, test = load_dataset(config)
            print("✓ Successfully loaded")
            
            # Print sample
            sample = train[0] if hasattr(train, '__getitem__') else next(iter(train))
            print(f"\nSample data:")
            print(f"  x shape: {sample.x.shape if hasattr(sample, 'x') and sample.x is not None else 'N/A'}")
            print(f"  pos shape: {sample.pos.shape if hasattr(sample, 'pos') and sample.pos is not None else 'N/A'}")
            print(f"  edge_index shape: {sample.edge_index.shape if hasattr(sample, 'edge_index') and sample.edge_index is not None else 'N/A'}")
            print(f"  y shape: {sample.y.shape if hasattr(sample, 'y') and sample.y is not None else 'N/A'}")
            
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            print(traceback.format_exc())
