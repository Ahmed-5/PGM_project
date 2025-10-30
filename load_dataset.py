"""
Comprehensive dataset loading for equivariant GNN benchmarks
Supports molecular, point cloud, and specialized 3D geometry datasets
"""

import os
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import ZINC, QM9, ModelNet, QM7b
from torch_geometric.transforms import Compose, NormalizeScale
import numpy as np
from typing import Tuple, Optional
import warnings
from dataclasses import dataclass
from typing import Literal
import traceback


# ========== OC20 Data Conversion Utilities ==========

def create_oc20_aselmdb(atoms_list, output_path: str):
    """
    Convert list of ASE Atoms objects to ASE LMDB format for fairchem
    
    Args:
        atoms_list: List of ASE Atoms objects
        output_path: Path to output .aselmdb file
    """
    try:
        from fairchem.core.datasets import LMDBDatabase
        from tqdm import tqdm
        
        with LMDBDatabase(output_path, readonly=False) as db:
            for i, atoms in enumerate(tqdm(atoms_list, desc="Writing to LMDB")):
                db.write(atoms, id=i)
        
        print(f"Successfully created LMDB at: {output_path}")
        
    except ImportError:
        raise ImportError("Creating LMDB requires: pip install fairchem-core")


def read_oc20_sample(dataset_path: str, index: int = 0):
    """
    Read a single sample from OC20 dataset
    
    Args:
        dataset_path: Path to OC20 directory
        index: Index of sample to read
    
    Returns:
        ASE Atoms object with energy, forces, etc.
    """
    from fairchem.core.datasets import AseDBDataset
    
    config = {
        'src': dataset_path,
        'a2g_args': {
            'r_energy': True,
            'r_forces': True,
            'r_distances': True,
            'r_edges': True,
        }
    }
    
    dataset = AseDBDataset(config=config)
    atoms = dataset.get_atoms(index)
    
    print(f"Sample {index}:")
    print(f"  Number of atoms: {len(atoms)}")
    print(f"  Energy: {atoms.get_potential_energy():.4f} eV")
    print(f"  Forces shape: {atoms.get_forces().shape}")
    print(f"  Cell: {atoms.get_cell()}")
    
    return atoms


# ========== OC20-specific Config Updates ==========

@dataclass
class OC20Config:
    """OC20-specific configuration"""
    task: Literal['s2ef', 'is2re', 'is2rs'] = 's2ef'
    
    # S2EF: Structure to Energy and Forces (most common)
    # IS2RE: Initial Structure to Relaxed Energy
    # IS2RS: Initial Structure to Relaxed Structure
    
    # Data split
    split: Literal['train', 'val_id', 'val_ood_ads', 'val_ood_cat', 'val_ood_both', 'test'] = 'train'
    
    # Subset sizes (for debugging)
    train_size: Optional[int] = None  # None = use all
    val_size: Optional[int] = None
    test_size: Optional[int] = None
    
    # Energy/force prediction settings
    predict_forces: bool = True
    normalize_labels: bool = True
    
    # Filtering
    max_neighbors: int = 50
    cutoff: float = 6.0


# Dataset symmetry requirements (for validation)
DATASET_SYMMETRIES = {
    'ZINC': ['permutation'],
    'QM9': ['permutation', 'e3'],
    'QM7b': ['permutation', 'e3'],
    'AQSOL': ['permutation'],
    'MD17': ['permutation', 'e3'],  # Forces require equivariance
    'MD22': ['permutation', 'e3'],
    'rMD17': ['permutation', 'e3'],
    'OC20': ['permutation', 'se3'],  # Periodic boundary conditions
    'ISO17': ['permutation', 'e3'],
    'Molecule3D': ['permutation', 'e3'],
    'ATOM3D': ['permutation', 'e3'],
    'ModelNet40': ['so3'],  # Rigid body rotations
    'ShapeNet': ['so3', 'reflection'],
    'PartNet': ['permutation', 'so3'],
}


class AtomDegreeOneHot:
    """One-hot encode atom types and degrees"""
    def __init__(self, num_atom_types=28, max_degree=10):
        self.num_atom_types = num_atom_types
        self.max_degree = max_degree
    
    def __call__(self, data):
        # One-hot encode degrees
        degree = torch.zeros(data.x.shape[0], self.max_degree + 1)
        deg = torch.bincount(data.edge_index[0], minlength=data.x.shape[0])
        deg = deg.clamp(max=self.max_degree)
        degree.scatter_(1, deg.unsqueeze(1), 1)
        
        # Concatenate with atom features
        data.x = torch.cat([data.x, degree], dim=1)
        return data


def load_dataset(config) -> Tuple:
    """
    Load dataset based on configuration
    
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    dataset_name = config.data.dataset_name
    
    print(f"Loading dataset: {dataset_name}")
    
    # ========== MOLECULAR GRAPH DATASETS (2D) ==========
    
    if dataset_name == 'ZINC':
        """
        ZINC: Molecular property prediction
        - 250K molecules (12K subset available)
        - Task: Predict constrained solubility
        - Symmetries: Permutation only (no 3D coords)
        """
        transform = AtomDegreeOneHot(num_atom_types=28, max_degree=10)
        
        train_dataset = ZINC(
            root=config.data.root,
            subset=config.data.subset,
            split='train',
            transform=transform
        )
        
        val_dataset = ZINC(
            root=config.data.root,
            subset=config.data.subset,
            split='val',
            transform=transform
        )
        
        test_dataset = ZINC(
            root=config.data.root,
            subset=config.data.subset,
            split='test',
            transform=transform
        )
    
    elif dataset_name == 'AQSOL':
        """
        AQSOL: Aqueous solubility prediction
        - 9,982 molecules
        - Task: Predict solubility in water
        - Symmetries: Permutation only
        
        Note: Has bug in PyG 2.x with string handling. Using workaround.
        """
        try:
            from torch_geometric.datasets import AQSOL
            
            # Workaround for PyG AQSOL bug
            transform = Compose([
                AtomDegreeOneHot(num_atom_types=28, max_degree=10),
            ])
            
            # Try loading with error handling
            try:
                dataset = AQSOL(root=config.data.root, transform=transform)
            except TypeError as e:
                if "expected np.ndarray (got str)" in str(e):
                    warnings.warn(
                        "AQSOL has known bug in torch_geometric. "
                        "Skipping or use alternative solubility dataset (ESOL). "
                        "Issue: https://github.com/pyg-team/pytorch_geometric/issues/7845"
                    )
                    # Fallback: Use QM9 or ZINC instead
                    raise NotImplementedError(
                        "AQSOL currently broken in PyG. Use ESOL or ZINC instead."
                    )
                else:
                    raise e
            
            # Split 80/10/10
            train_size = int(0.8 * len(dataset))
            val_size = int(0.1 * len(dataset))
            test_size = len(dataset) - train_size - val_size
            
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(config.seed)
            )
            
        except Exception as e:
            raise RuntimeError(f"AQSOL loading failed: {e}")
    
    # ========== QUANTUM CHEMISTRY DATASETS (3D) ==========
    
    elif dataset_name == 'QM9':
        """
        QM9: Quantum chemistry properties
        - 134K molecules with 3D geometries
        - Task: Predict 13 quantum properties
        - Symmetries: Permutation + E(3)
        - Properties: HOMO, LUMO, gap, dipole moment, etc.
        """
        dataset = QM9(root=config.data.root)
        
        # Select target property (default: HOMO-LUMO gap)
        target_idx = config.data.qm9_target if hasattr(config.data, 'qm9_target') else 7
        
        # Filter out molecules with NaN targets
        valid_indices = []
        for i, data in enumerate(dataset):
            if not torch.isnan(data.y[0, target_idx]):
                valid_indices.append(i)
        
        dataset = dataset[valid_indices]
        
        # Update target to single property
        for data in dataset:
            data.y = data.y[0, target_idx].unsqueeze(0)

        targets = torch.cat([data.y for data in dataset])
        mean, std = targets.mean(), targets.std()
        for data in dataset:
            data.y = (data.y - mean) / std
        
        # Split 80/10/10
        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(config.seed)
        )
    
    elif dataset_name == 'QM7b':
        """
        QM7b: Atomization energies
        - 7,165 molecules
        - Task: Predict atomization energy
        - Symmetries: E(3)
        """
        dataset = QM7b(root=config.data.root)
        
        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(config.seed)
        )
    
    # ========== MOLECULAR DYNAMICS DATASETS (3D) ==========
    
    elif dataset_name in ['MD17', 'rMD17']:
        """
        MD17 / rMD17: Molecular dynamics trajectories
        - 150K-1M conformations per molecule
        - Task: Predict energy and forces
        - Symmetries: E(3) (forces are vector fields!)
        - Molecules: aspirin, benzene, ethanol, maleic_acid, naphthalene, 
                    salicylic_acid, toluene, uracil
        """
        try:
            from torch_geometric.datasets import MD17
            
            molecule = config.data.md17_molecule if hasattr(config.data, 'md17_molecule') else 'aspirin'
            
            dataset = MD17(
                root=config.data.root,
                name=molecule,
                # revised=(dataset_name == 'rMD17')
            )
            
            # Standard split: 1000 train, rest val/test
            train_size = 1000
            val_size = min(1000, (len(dataset) - train_size) // 2)
            test_size = len(dataset) - train_size - val_size
            
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(config.seed)
            )
            
            print(f"  Molecule: {molecule}")
            print(f"  Total conformations: {len(dataset)}")
            
        except ImportError:
            raise ImportError("MD17 requires torch_geometric>=2.0. Install with: pip install torch_geometric")
    
    elif dataset_name == 'MD22':
        """
        MD22: Extended MD17 with larger molecules
        - Molecules: DHA, Stachyose, AT-AT, etc.
        - Task: Energy and force prediction
        - Symmetries: E(3)
        """
        try:
            from torch_geometric.datasets import MD17
            
            molecule = config.data.md17_molecule if hasattr(config.data, 'md17_molecule') else 'Ac-Ala3-NHMe'
            
            # MD22 uses same loader as MD17
            dataset = MD17(
                root=config.data.root,
                name=molecule
            )
            
            train_size = min(1000, int(0.1 * len(dataset)))
            val_size = min(1000, int(0.05 * len(dataset)))
            test_size = len(dataset) - train_size - val_size
            
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(config.seed)
            )
            
        except ImportError:
            raise ImportError("MD22 requires torch_geometric>=2.0")
    
    elif dataset_name == 'ISO17':
        """
        ISO17: Constitutional isomers
        - C7O2H10 isomers
        - Task: Distinguish isomers, predict energies
        - Symmetries: E(3) + chemical structure
        """
        warnings.warn("ISO17 not directly available in PyG. Please download manually from quantum-machine.org")
        raise NotImplementedError("ISO17 requires manual download")
    
    # ========== CATALYST / MATERIALS DATASETS (3D) ==========
    
    elif dataset_name == 'OC20':
        """
        OC20 (Open Catalyst 2020): Catalyst surface adsorption
        - 1.3M+ structures
        - Task: Predict adsorption energies and forces
        - Symmetries: SE(3) with periodic boundary conditions
        """
        try:
            from fairchem.core.datasets import AseDBDataset
            
            task = config.data.oc20_task if hasattr(config.data, 'oc20_task') else 's2ef'
            data_root = os.path.join(config.data.root, 'oc20', task)
            
            # Check if data exists
            if not os.path.exists(data_root):
                raise FileNotFoundError(
                    f"OC20 data not found at {data_root}\n\n"
                    "Download instructions:\n"
                    "1. Visit: https://fair-chem.github.io/core/datasets/oc20.html\n"
                    "2. Download desired split (e.g., S2EF train/val/test)\n"
                    "3. Extract to {data_root}\n\n"
                    "Quick download (S2EF-2M):\n"
                    "  wget https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_train_2M.tar\n"
                    "  tar -xvf s2ef_train_2M.tar -C {data_root}\n\n"
                    "For testing, you can skip OC20 for now."
                )
            
            
            train_config = {
                'src': os.path.join(data_root, 'train'),
                'a2g_args': {
                    'r_energy': True,
                    'r_forces': True,
                    'r_distances': True,
                    'r_edges': True,
                }
            }
            
            val_config = {
                'src': os.path.join(data_root, 'val_id'),
                'a2g_args': {
                    'r_energy': True,
                    'r_forces': True,
                    'r_distances': True,
                    'r_edges': True,
                }
            }
            
            test_config = {
                'src': os.path.join(data_root, 'test_id'),
                'a2g_args': {
                    'r_energy': True,
                    'r_forces': True,
                    'r_distances': True,
                    'r_edges': True,
                }
            }
            
            train_dataset = AseDBDataset(config=train_config)
            val_dataset = AseDBDataset(config=val_config)
            test_dataset = AseDBDataset(config=test_config)
            
            print(f"  Task: {task}")
            print(f"  Using fairchem-core")
            
        except ImportError:
            raise ImportError(
                "OC20 requires FAIRChem. Install:\n"
                "  pip install fairchem-core"
            )
        except FileNotFoundError as e:
            warnings.warn(str(e))
            raise NotImplementedError("OC20 data not downloaded. See error message above.")


    
    # ========== 3D MOLECULE GENERATION DATASETS ==========
    
    elif dataset_name == 'Molecule3D':
        """
        Molecule3D: 3D geometry prediction
        - 3.9M molecules
        - Task: Predict 3D geometry from 2D graph
        - Symmetries: Tests if model learns E(3) constraints
        """
        warnings.warn("Molecule3D requires custom loader. Download from moleculenet.org")
        raise NotImplementedError("Molecule3D requires custom dataset implementation")
    
    elif dataset_name == 'ATOM3D':
        """
        ATOM3D: 3D biomolecular tasks
        - Multiple tasks: ligand binding, protein interface, etc.
        - Symmetries: Biological system symmetries
        """
        try:
            import atom3d.datasets as da
            
            # Specify task and proper data path
            atom3d_task = config.data.atom3d_task if hasattr(config.data, 'atom3d_task') else 'smp'
            atom3d_root = os.path.join(config.data.root, 'atom3d', atom3d_task)
            
            if not os.path.exists(atom3d_root):
                raise FileNotFoundError(
                    f"ATOM3D data not found at {atom3d_root}. "
                    "Download from: https://www.atom3d.ai/\n"
                    "Example: wget https://zenodo.org/record/4962515/files/SMP-train.tar.gz"
                )
            
            # Load dataset with proper format
            dataset = da.load_dataset(atom3d_root, 'lmdb')  # Specify format
            
            if isinstance(dataset, dict):
                train_dataset = dataset['train']
                val_dataset = dataset['val']
                test_dataset = dataset['test']
            else:
                # Manual split
                train_size = int(0.8 * len(dataset))
                val_size = int(0.1 * len(dataset))
                test_size = len(dataset) - train_size - val_size
                
                train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
                    dataset, [train_size, val_size, test_size],
                    generator=torch.Generator().manual_seed(config.seed)
                )
            
            print(f"  Task: {atom3d_task}")
            
        except ImportError:
            raise ImportError(
                "ATOM3D requires: pip install atom3d\n"
                "Also download data from: https://www.atom3d.ai/"
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(str(e))
    
    # ========== POINT CLOUD DATASETS (3D) ==========
    
    elif dataset_name == 'ModelNet40':
        """
        ModelNet40: 3D CAD model classification
        - 12,311 models, 40 categories
        - Task: Shape classification
        - Symmetries: SO(3) + discrete object symmetries
        """
        transform = Compose([NormalizeScale()])
        
        try:
            train_dataset = ModelNet(
                root=config.data.root,
                name='40',
                train=True,
                transform=transform
            )
            
            test_dataset = ModelNet(
                root=config.data.root,
                name='40',
                train=False,
                transform=transform
            )
            
            # Split train into train/val (90/10)
            train_size = int(0.9 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(config.seed)
            )
            
            print(f"  Note: ModelNet uses point clouds (pos only, no node features)")
            
        except Exception as e:
            if "No connection" in str(e) or "refused" in str(e):
                raise ConnectionError(
                    "ModelNet40 download failed due to network/proxy issue.\n\n"
                    "Manual download:\n"
                    "1. Download from: https://modelnet.cs.princeton.edu/ModelNet40.zip\n"
                    "2. Extract to: {config.data.root}/ModelNet40/\n"
                    "3. Re-run the script\n\n"
                    "Alternative: Use a different dataset for point cloud experiments.\n"
                    "For now, you can test with molecular datasets (ZINC, QM9, MD17)."
                )
            else:
                raise e
    
    elif dataset_name == 'ShapeNet':
        """
        ShapeNet: Large-scale 3D shape dataset
        - 51,300 3D models, 55 categories
        - Symmetries: Object-specific (bilateral, rotational)
        """
        try:
            from torch_geometric.datasets import ShapeNet
            
            dataset = ShapeNet(
                root=config.data.root,
                categories=None,  # All categories
                transform=NormalizeScale()
            )
            
            train_size = int(0.8 * len(dataset))
            val_size = int(0.1 * len(dataset))
            test_size = len(dataset) - train_size - val_size
            
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(config.seed)
            )
            
        except ImportError:
            raise ImportError("ShapeNet requires torch_geometric>=2.0")
        except Exception as e:
            if "No connection" in str(e) or "refused" in str(e):
                warnings.warn(
                    "ShapeNet download failed. Network/proxy issue.\n"
                    "Manual download: https://shapenet.org/\n"
                    "Skipping ShapeNet for now."
                )
                raise NotImplementedError("ShapeNet unavailable due to network issue")
            else:
                raise e
            
    
    elif dataset_name == 'PartNet':
        """
        PartNet: Part-based 3D objects
        - Part-level annotations
        - Symmetries: Part-level + assembly
        """
        warnings.warn(
            "PartNet is not directly available in PyG.\n"
            "Download from: https://www.shapenet.org/download/parts\n"
            "Requires custom dataset implementation."
        )
        raise NotImplementedError(
            "PartNet requires manual download and custom dataset class.\n"
            "Use ModelNet40 or ShapeNet for point cloud experiments instead."
        )
    
    # ========== VALIDATE DATASET & CONFIG ==========
    
    # Check if positions are available
    sample_data = train_dataset[0] if hasattr(train_dataset, '__getitem__') else next(iter(train_dataset))
    has_pos = hasattr(sample_data, 'pos') and sample_data.pos is not None
    
    if config.data.use_positions and not has_pos:
        warnings.warn(
            f"Config specifies use_positions=True but {dataset_name} has no 3D coordinates. "
            "Setting use_positions=False."
        )
        config.data.use_positions = False
    
    if not config.data.use_positions and has_pos:
        print(f"  Note: Dataset has 3D coordinates but use_positions=False")
    
    # Validate symmetry groups match dataset
    required_symmetries = DATASET_SYMMETRIES.get(dataset_name, ['permutation'])
    requested_symmetries = config.equivariance.symmetry_groups
    
    geometric_groups = {'so3', 'o3', 'se3', 'e3', 'translation', 'reflection', 'scaling'}
    
    if any(g in requested_symmetries for g in geometric_groups) and not has_pos:
        warnings.warn(
            f"Geometric symmetry groups {requested_symmetries} require 3D coordinates, "
            f"but {dataset_name} has none. Consider using permutation-only models."
        )
    
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val: {len(val_dataset)}")
    print(f"  Test: {len(test_dataset)}")
    print(f"  Has 3D coordinates: {has_pos}")
    print(f"  Recommended symmetries: {required_symmetries}")
    
    return train_dataset, val_dataset, test_dataset


# ========== HELPER FUNCTIONS ==========

def get_dataset_info(dataset_name: str) -> dict:
    """Get metadata about a dataset"""
    info = {
        'ZINC': {
            'size': '250K (12K subset)',
            'task': 'Regression',
            'properties': 'Constrained solubility',
            'dimensions': '2D graph',
            'symmetries': ['permutation'],
            'recommended_models': ['gcn', 'gin', 'graphsage']
        },
        'QM9': {
            'size': '134K',
            'task': 'Regression',
            'properties': '13 quantum properties',
            'dimensions': '3D',
            'symmetries': ['permutation', 'e3'],
            'recommended_models': ['schnet', 'dimenet', 'painn']
        },
        'MD17': {
            'size': '150K-1M per molecule',
            'task': 'Force prediction',
            'properties': 'Energy, forces',
            'dimensions': '3D',
            'symmetries': ['permutation', 'e3'],
            'recommended_models': ['egnn', 'painn', 'nequip'],
            'note': 'Forces require E(3) equivariance'
        },
        'OC20': {
            'size': '1.3M+',
            'task': 'Catalyst adsorption',
            'properties': 'Energy, forces',
            'dimensions': '3D periodic',
            'symmetries': ['permutation', 'se3'],
            'recommended_models': ['gemnet', 'escn', 'painn']
        },
        'ModelNet40': {
            'size': '12K',
            'task': 'Classification',
            'properties': 'Shape category',
            'dimensions': '3D point cloud',
            'symmetries': ['so3'],
            'recommended_models': ['vector_neuron', 'se3_transformer']
        }
    }
    
    return info.get(dataset_name, {})


# if __name__ == '__main__':
#     # Test dataset loading
#     from config import get_config
    
#     datasets_to_test = [
#         'ZINC', 'QM9', 'QM7b', 'AQSOL',  # Original
#         'MD17', 'MD22', 'rMD17', 'OC20',  # Molecular dynamics
#         'ISO17', 'Molecule3D', 'ATOM3D',  # Special molecular
#         'ModelNet40', 'ShapeNet', 'PartNet'  # Point clouds
#     ]
    
#     for dataset_name in datasets_to_test:
#         print(f"\n{'='*80}")
#         print(f"Testing: {dataset_name}")
#         print('='*80)
        
#         config = get_config('default')
#         config.data.dataset_name = dataset_name
#         config.data.use_positions = (dataset_name != 'ZINC')
        
#         try:
#             train, val, test = load_dataset(config)
#             print("✓ Successfully loaded")
            
#             # Print sample
#             sample = train[0] if hasattr(train, '__getitem__') else next(iter(train))
#             print(f"\nSample data:")
#             print(f"  x shape: {sample.x.shape if hasattr(sample, 'x') and sample.x is not None else 'N/A'}")
#             print(f"  pos shape: {sample.pos.shape if hasattr(sample, 'pos') and sample.pos is not None else 'N/A'}")
#             print(f"  edge_index shape: {sample.edge_index.shape if hasattr(sample, 'edge_index') and sample.edge_index is not None else 'N/A'}")
#             print(f"  y shape: {sample.y.shape if hasattr(sample, 'y') and sample.y is not None else 'N/A'}")
            
#         except Exception as e:
#             print(f"✗ Error: {str(e)}")
#             print(traceback.format_exc())
