"""
Configuration file for Relaxed Equivariance GNN experiments
Updated to support multiple model architectures and symmetry groups
"""

import torch
from dataclasses import dataclass, field
from typing import Literal, Optional, List
import warnings
from datetime import datetime
from pydantic.warnings import UnsupportedFieldAttributeWarning

warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Basic architecture
    in_channels: int = 28 + 11  # degree one-hot for ZINC
    hidden_channels: int = 64
    out_channels: int = 1  # Regression task
    num_layers: int = 4
    dropout: float = 0.5
    spatial_dim: int = 3
    
    # Model type selection (now supports all architectures)
    model_type: Literal[
        'raw_mlp', 'transformer', 'gcn', 'gin', 'graphsage',
        'schnet', 'dimenet', 'egnn', 'painn',
        'vector_neuron', 'se3_transformer', 'nequip', 'clofnet'
    ] = 'gcn'
    
    # Model-specific parameters
    num_heads: int = 8  # For Transformer, SE3Transformer
    num_gaussians: int = 50  # For SchNet, DimeNet, NequIP
    num_spherical: int = 7  # For DimeNet
    cutoff: float = 10.0  # For distance-based models
    update_coords: bool = False  # For EGNN
    max_ell: int = 2  # For NequIP (angular momentum)
    num_degrees: int = 2  # For SE3Transformer


@dataclass
class EquivarianceLossConfig:
    """Equivariance loss configuration"""
    # Which symmetry groups to enforce
    symmetry_groups: List[str] = field(default_factory=lambda: ['permutation'])
    
    # Weights for each symmetry group
    group_weights: dict = field(default_factory=lambda: {
        'permutation': 0.1,
        'so3': 0.1,
        'o3': 0.1,
        'se3': 0.1,
        'e3': 0.1,
        'translation': 0.1,
        'reflection': 0.1,
        'scaling': 0.05
    })
    
    # Loss computation settings
    num_samples: int = 3  # Number of random transformations per group
    normalize: bool = True  # Normalize by feature magnitude
    feature_type: Literal['invariant', 'equivariant'] = 'invariant'
    max_translation: float = 5.0
    scale_range: tuple = (0.5, 2.0)


@dataclass
class SchedulerConfig:
    """Depth-adaptive scheduler configuration"""
    schedule_type: Literal['constant', 'exponential', 'linear', 'learnable'] = 'exponential'
    alpha_0: float = 1.0  # Initial equivariance weight
    beta: float = 0.1  # Exponential decay rate
    gamma: float = 0.1  # Linear decay rate


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    scheduler_lr: Literal['step', 'cosine', 'none'] = 'step'
    patience: int = 20  # Early stopping
    grad_clip: float = 1.0  # Gradient clipping


@dataclass
class DataConfig:
    """Dataset configuration"""
    dataset_name: Literal[
        'ZINC', 'QM9', 'QM7', 'AQSOL',  # Original
        'MD17', 'MD22', 'rMD17', 'OC20',  # Molecular dynamics
        'ISO17', 'Molecule3D', 'ATOM3D',  # Special molecular
        'ModelNet40', 'ShapeNet', 'PartNet'  # Point clouds
    ] = 'ZINC'
    
    subset: bool = True
    root: str = './data'
    num_workers: int = 4
    use_positions: bool = False
    
    # MD17/MD22 specific
    md17_molecule: Literal[
        'aspirin', 'benzene', 'ethanol', 'maleic_acid',
        'naphthalene', 'salicylic_acid', 'toluene', 'uracil'
    ] = 'aspirin'
    
    # OC20 specific
    oc20_task: Literal['s2ef', 'is2re', 'is2rs'] = 's2ef'
    
    # QM9 target property
    qm9_target: int = 7  # HOMO-LUMO gap
    
    # Point cloud specific
    modelnet_num_points: int = 1024
    use_normals: bool = False


# Dataset symmetry requirements
DATASET_SYMMETRIES = {
    'ZINC': ['permutation'],
    'QM9': ['permutation', 'e3'],
    'QM7': ['permutation', 'e3'],
    'MD17': ['permutation', 'e3'],  # Forces require equivariance
    'MD22': ['permutation', 'e3'],
    'rMD17': ['permutation', 'e3'],
    'OC20': ['permutation', 'se3'],  # Periodic systems
    'ISO17': ['permutation', 'e3'],
    'ModelNet40': ['so3'],  # Rigid body rotations
    'ShapeNet': ['so3', 'reflection'],  # Objects can have mirrors
    'PartNet': ['permutation', 'so3'],  # Part permutations + rotations
}

# Recommended model types per dataset
DATASET_MODEL_RECOMMENDATIONS = {
    'ZINC': ['gcn', 'gin', 'graphsage'],  # Graph only
    'QM9': ['schnet', 'dimenet', 'painn'],  # 3D geometry
    'MD17': ['egnn', 'painn', 'nequip'],  # Force prediction
    'MD22': ['painn', 'nequip'],  # Larger molecules
    'OC20': ['gemnet', 'escn'],  # Specialized for catalysts
    'ModelNet40': ['vector_neuron', 'se3_transformer'],  # Point clouds
    'ShapeNet': ['vector_neuron', 'clofnet'],  # 3D shapes
}

@dataclass
class LoggingConfig:
    """Logging configuration"""
    logger_type: Literal['wandb', 'tensorboard', 'none'] = 'none'
    
    # Weights & Biases settings
    wandb_project: str = 'relaxed-equivariance-gnn'
    wandb_entity: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_tags: list = None
    wandb_notes: Optional[str] = None
    
    # TensorBoard settings
    tensorboard_dir: str = './runs'
    
    # General logging settings
    log_interval: int = 10
    log_gradients: bool = False
    log_layer_outputs: bool = True
    log_equivariance_metrics: bool = True  # Log per-group equivariance violations
    save_model_artifact: bool = True
    
    def __post_init__(self):
        if self.wandb_tags is None:
            self.wandb_tags = []


@dataclass
class ExperimentConfig:
    """Full experiment configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    equivariance: EquivarianceLossConfig = field(default_factory=EquivarianceLossConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    seed: int = 42
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Experiment metadata
    experiment_name: str = 'default'
    timestamp: str = field(default_factory=lambda: datetime.now().strftime('%Y%m%d_%H%M%S'))
    checkpoint_dir: str = ''
    
    def __post_init__(self):
        if not self.checkpoint_dir:
            self.checkpoint_dir = f'./checkpoints/{self.experiment_name}_{self.timestamp}'
    
    def to_dict(self):
        """Convert to dictionary for logging"""
        return {
            'model': vars(self.model),
            'equivariance': {
                'symmetry_groups': self.equivariance.symmetry_groups,
                'group_weights': self.equivariance.group_weights,
                'num_samples': self.equivariance.num_samples,
                'normalize': self.equivariance.normalize,
                'feature_type': self.equivariance.feature_type,
            },
            'scheduler': vars(self.scheduler),
            'training': vars(self.training),
            'data': vars(self.data),
            'logging': vars(self.logging),
            'seed': self.seed,
            'device': self.device,
            'experiment_name': self.experiment_name,
            'timestamp': self.timestamp,
            'checkpoint_dir': self.checkpoint_dir,
        }


def get_config(config_name: str = 'default') -> ExperimentConfig:
    """Factory function for different experiment configurations"""
    
    if config_name == 'default':
        config = ExperimentConfig()
        config.experiment_name = 'default'
        return config
    
    elif config_name == 'baseline':
        """Standard GNN without equivariance loss"""
        config = ExperimentConfig()
        config.model.model_type = 'gcn'
        config.scheduler.alpha_0 = 0.0
        config.experiment_name = 'baseline_gcn'
        config.logging.wandb_tags = ['baseline', 'no-equivariance', 'gcn']
        return config
    
    elif config_name == 'permutation_only':
        """Standard GNN with only permutation equivariance loss"""
        config = ExperimentConfig()
        config.model.model_type = 'gin'
        config.equivariance.symmetry_groups = ['permutation']
        config.equivariance.group_weights = {'permutation': 0.1}
        config.scheduler.schedule_type = 'exponential'
        config.experiment_name = 'permutation_only'
        config.logging.wandb_tags = ['permutation', 'gin']
        return config
    
    elif config_name == 'e3_invariant':
        """E(3)-invariant model (SchNet)"""
        config = ExperimentConfig()
        config.model.model_type = 'schnet'
        config.data.use_positions = True
        config.equivariance.symmetry_groups = ['so3', 'translation', 'reflection']
        config.equivariance.group_weights = {
            'so3': 0.1,
            'translation': 0.1,
            'reflection': 0.05
        }
        config.experiment_name = 'e3_invariant_schnet'
        config.logging.wandb_tags = ['e3-invariant', 'schnet', 'geometric']
        return config
    
    elif config_name == 'e3_equivariant':
        """E(3)-equivariant model (EGNN)"""
        config = ExperimentConfig()
        config.model.model_type = 'egnn'
        config.model.update_coords = True
        config.data.use_positions = True
        config.equivariance.symmetry_groups = ['e3']
        config.equivariance.group_weights = {'e3': 0.1}
        config.experiment_name = 'e3_equivariant_egnn'
        config.logging.wandb_tags = ['e3-equivariant', 'egnn', 'coordinate-update']
        return config
    
    elif config_name == 'so3_equivariant':
        """SO(3)-equivariant model (Vector Neurons)"""
        config = ExperimentConfig()
        config.model.model_type = 'vector_neuron'
        config.data.use_positions = True
        config.equivariance.symmetry_groups = ['so3', 'translation']
        config.equivariance.group_weights = {
            'so3': 0.15,
            'translation': 0.1
        }
        config.experiment_name = 'so3_equivariant_vn'
        config.logging.wandb_tags = ['so3-equivariant', 'vector-neurons']
        return config
    
    elif config_name == 'se3_transformer':
        """SE(3)-equivariant transformer"""
        config = ExperimentConfig()
        config.model.model_type = 'se3_transformer'
        config.model.num_heads = 8
        config.data.use_positions = True
        config.equivariance.symmetry_groups = ['se3']
        config.equivariance.group_weights = {'se3': 0.1}
        config.training.learning_rate = 0.0005  # Lower LR for attention
        config.experiment_name = 'se3_transformer'
        config.logging.wandb_tags = ['se3-equivariant', 'attention', 'transformer']
        return config
    
    elif config_name == 'multi_symmetry':
        """Multiple symmetry groups with different weights"""
        config = ExperimentConfig()
        config.model.model_type = 'egnn'
        config.data.use_positions = True
        config.equivariance.symmetry_groups = ['permutation', 'so3', 'translation', 'scaling']
        config.equivariance.group_weights = {
            'permutation': 0.05,
            'so3': 0.1,
            'translation': 0.1,
            'scaling': 0.03
        }
        config.scheduler.schedule_type = 'exponential'
        config.scheduler.alpha_0 = 1.5
        config.experiment_name = 'multi_symmetry_egnn'
        config.logging.wandb_tags = ['multi-symmetry', 'combined-loss']
        return config
    
    elif config_name == 'depth_adaptive_e3':
        """Depth-adaptive equivariance for E(3) model"""
        config = ExperimentConfig()
        config.model.model_type = 'painn'
        config.model.num_layers = 8
        config.data.use_positions = True
        config.equivariance.symmetry_groups = ['e3']
        config.scheduler.schedule_type = 'exponential'
        config.scheduler.alpha_0 = 2.0
        config.scheduler.beta = 0.15
        config.experiment_name = 'depth_adaptive_painn'
        config.logging.wandb_tags = ['depth-adaptive', 'painn', 'e3']
        return config
    
    elif config_name == 'learnable_weights':
        """Learnable equivariance weights"""
        config = ExperimentConfig()
        config.model.model_type = 'nequip'
        config.data.use_positions = True
        config.equivariance.symmetry_groups = ['e3']
        config.scheduler.schedule_type = 'learnable'
        config.training.num_epochs = 150
        config.training.learning_rate = 0.0005
        config.experiment_name = 'learnable_weights_nequip'
        config.logging.wandb_tags = ['learnable', 'nequip', 'meta-learning']
        config.logging.log_gradients = True
        return config
    
    elif config_name == 'ablation_no_symmetry':
        """Ablation: No geometric symmetry (uses raw coords)"""
        config = ExperimentConfig()
        config.model.model_type = 'transformer'
        config.data.use_positions = True
        config.scheduler.alpha_0 = 0.0
        config.experiment_name = 'ablation_no_symmetry'
        config.logging.wandb_tags = ['ablation', 'no-symmetry', 'baseline']
        return config
    
    elif config_name == 'comparison_all_models':
        """For systematic comparison across all models"""
        configs = []
        model_types = ['gcn', 'schnet', 'egnn', 'vector_neuron', 'se3_transformer', 'painn', 'nequip']
        
        for model_type in model_types:
            config = ExperimentConfig()
            config.model.model_type = model_type
            
            # Set appropriate symmetry groups for each model
            if model_type in ['gcn', 'gin', 'graphsage']:
                config.equivariance.symmetry_groups = ['permutation']
                config.data.use_positions = False
            elif model_type in ['schnet', 'dimenet']:
                config.equivariance.symmetry_groups = ['so3', 'translation']
                config.data.use_positions = True
            elif model_type in ['egnn', 'painn', 'nequip']:
                config.equivariance.symmetry_groups = ['e3']
                config.data.use_positions = True
            elif model_type == 'vector_neuron':
                config.equivariance.symmetry_groups = ['so3', 'translation']
                config.data.use_positions = True
            elif model_type == 'se3_transformer':
                config.equivariance.symmetry_groups = ['se3']
                config.data.use_positions = True
            
            config.experiment_name = f'comparison_{model_type}'
            config.logging.wandb_tags = ['comparison', model_type]
            configs.append(config)
        
        return configs
    
    else:
        raise ValueError(f"Unknown config name: {config_name}. Available configs: "
                        f"default, baseline, permutation_only, e3_invariant, e3_equivariant, "
                        f"so3_equivariant, se3_transformer, multi_symmetry, depth_adaptive_e3, "
                        f"learnable_weights, ablation_no_symmetry, comparison_all_models")


# Quick access to common configurations
CONFIGS = {
    'baseline': get_config('baseline'),
    'gcn': get_config('permutation_only'),
    'schnet': get_config('e3_invariant'),
    'egnn': get_config('e3_equivariant'),
    'painn': get_config('depth_adaptive_e3'),
    'vector_neuron': get_config('so3_equivariant'),
    'se3_transformer': get_config('se3_transformer'),
}


if __name__ == "__main__":
    # Test configurations
    print("Testing configuration system...\n")
    
    for name in ['baseline', 'e3_equivariant', 'multi_symmetry']:
        config = get_config(name)
        print(f"Config: {name}")
        print(f"  Model: {config.model.model_type}")
        print(f"  Symmetries: {config.equivariance.symmetry_groups}")
        print(f"  Weights: {config.equivariance.group_weights}")
        print(f"  Use positions: {config.data.use_positions}")
        print()
