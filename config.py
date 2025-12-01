"""
Configuration system for Equivariant GNN experiments
Enhanced with validation, type safety, and flexible presets
"""

import os
import torch
from dataclasses import dataclass, field, asdict
from typing import Literal, Optional, List, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime
import json

@dataclass
class ModelConfig:
    """Model architecture configuration with validation"""
    
    in_channels: int = 12
    hidden_channels: int = 64
    out_channels: int = 1
    num_layers: int = 4
    dropout: float = 0.5
    spatial_dim: int = 3
    
    model_type: Literal[
        'raw_mlp', 'transformer', 'gcn', 'gin', 'graphsage',
        'schnet', 'dimenet', 'egnn', 'painn',
        'vector_neuron', 'se3_transformer', 'nequip', 'clofnet'
    ] = 'gcn'
    
    # Architecture-specific parameters
    num_heads: int = 8
    num_gaussians: int = 50
    num_spherical: int = 7
    cutoff: float = 10.0
    update_coords: bool = False
    max_ell: int = 2
    num_degrees: int = 2
    
    # Standard GNN extensions
    use_pos: bool = False  # Concatenate positions with node features for standard GNNs
    
    use_layer_norm: bool = False
    use_batch_norm: bool = True
    
    def __post_init__(self):
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")
        if not 0 <= self.dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")


@dataclass
class EquivarianceLossConfig:
    """Equivariance loss with adaptive weighting and stochastic support"""
    symmetry_groups: List[str] = field(default_factory=lambda: [])
    group_weights: Dict[str, float] = field(default_factory=lambda: {
        'permutation': 0.1, 'so3': 0.1, 'o3': 0.1, 'se3': 0.1,
        'e3': 0.1, 'translation': 0.1, 'reflection': 0.1, 'scaling': 0.05
    })
    num_samples: int = 2  # Reduced default for speed
    normalize: bool = True
    feature_type: Literal['invariant', 'equivariant'] = 'invariant'
    max_translation: float = 5.0
    scale_range: Tuple[float, float] = (0.5, 2.0)
    use_adaptive_weighting: bool = False
    
    # [NEW] Stochastic Regularization for Ablation Efficiency
    stochastic_probability: float = 0.25  # Only apply loss on 25% of batches

    layer_weight_strategy: Literal['constant', 'linear_decay', 'linear_inc', 
                                   'exp_decay', 'exp_inc', 'u_shaped', 'learnable'] = 'constant'
    layer_decay_rate: float = 0.5  # For exponential strategies
    
    def __post_init__(self):
        valid_groups = {
            'permutation', 'so3', 'o3', 'se3', 'e3',
            'translation', 'reflection', 'scaling'
        }
        for group in self.symmetry_groups:
            if group not in valid_groups:
                raise ValueError(f"Invalid symmetry group '{group}'")


@dataclass
class SchedulerConfig:
    """Scheduling for both learning rate and equivariance weights"""
    
    schedule_type: Literal[
        'constant', 'exponential', 'linear', 'inverse', 'u_shaped', 'learnable'
    ] = 'exponential'
    alpha_0: float = 1.0
    beta: float = 0.1
    gamma: float = 0.1
    
    lr_schedule: Literal['step', 'cosine', 'plateau', 'exponential', 'none'] = 'cosine'
    lr_step_size: int = 50
    lr_gamma: float = 0.5
    lr_warmup_epochs: int = 5

    plateau_patience: int = 5  # For 'plateau' lr_schedule
    plateau_factor: float = 0.5  # For 'plateau' lr_schedule

    exponential_decay_rate: float = 0.95  # For 'exponential' lr_schedule


@dataclass
class TrainingConfig:
    """Training hyperparameters with modern best practices"""
    
    batch_size: int = 32
    num_epochs: int = 1
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    
    optimizer: Literal['adam', 'adamw', 'sgd', 'rmsprop'] = 'adamw'
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    
    patience: int = 20
    min_delta: float = 1e-4
    use_amp: bool = True  # Mixed precision training (Enabled by default for speed)
    accumulation_steps: int = 1


@dataclass
class DataConfig:
    """Dataset configuration"""
    
    dataset_name: Literal[
        'ZINC', 'QM9', 'QM7b', 'AQSOL',  # Original
        'MD17', 'MD22', 'rMD17', 'OC20',  # Molecular dynamics
        'ISO17', 'Molecule3D', 'ATOM3D',  # Special molecular
        'ModelNet40', 'ShapeNet', 'PartNet'  # Point clouds
    ] = 'ZINC'

    subset: bool = True
    root: str = './data'
    num_workers: int = 4
    use_positions: bool = False
    
    # [OPTIMIZED] Data loading defaults
    persistent_workers: bool = True
    prefetch_factor: int = 2
    
    md17_molecule: str = 'aspirin'
    qm9_target: int = 7
    
    use_augmentation: bool = False
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1

    rewiring: str = 'none' # 'none', 'spectral', 'geometric'
    rewiring_k: int = 2    # hops or neighbors
    rewiring_threshold: float = 5.0 # distance threshold


@dataclass
class LoggingConfig:
    """Comprehensive logging configuration"""
    
    logger_type: Literal['wandb', 'tensorboard', 'both', 'none'] = 'none'
    
    # wandb_project: str = 'PGM_Project_wandb'
    # wandb_entity: Optional[str] = "PGM"
    wandb_project: str = 'PGM'
    wandb_entity: Optional[str] = "PGM_Project_wandb"
    wandb_name: Optional[str] = None
    wandb_notes: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    wandb_mode: Literal['online', 'offline', 'disabled'] = 'online'
    
    tensorboard_dir: str = './runs'
    
    log_interval: int = 10
    log_gradients: bool = False
    log_equivariance_metrics: bool = True
    save_checkpoint: bool = True
    checkpoint_interval: int = 10
    save_best_only: bool = True
    verbose: bool = True
    save_model_artifact: bool = True


@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    equivariance: EquivarianceLossConfig = field(default_factory=EquivarianceLossConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    experiment_name: str = 'default'
    seed: int = 42
    device: str = field(default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu')
    # device: str = 'cpu'
    deterministic: bool = True
    
    checkpoint_dir: str = ''
    output_dir: str = ''
    timestamp: str = field(default_factory=lambda: datetime.now().strftime('%Y%m%d_%H%M%S'))
    
    def __post_init__(self):
        if not self.checkpoint_dir:
            self.checkpoint_dir = f'./checkpoints/{self.experiment_name}_{self.timestamp}'
        if not self.output_dir:
            self.output_dir = f'./outputs/{self.experiment_name}_{self.timestamp}'
        
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Validate model-data compatibility
        position_models = {
            'schnet', 'dimenet', 'egnn', 'painn',
            'vector_neuron', 'se3_transformer', 'nequip', 'clofnet'
        }
        if self.model.model_type in position_models and not self.data.use_positions:
            raise ValueError(
                f"Model '{self.model.model_type}' requires 3D positions. "
                "Set data.use_positions=True"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model': asdict(self.model),
            'equivariance': asdict(self.equivariance),
            'scheduler': asdict(self.scheduler),
            'training': asdict(self.training),
            'data': asdict(self.data),
            'logging': asdict(self.logging),
            'experiment_name': self.experiment_name,
            'seed': self.seed,
            'device': self.device,
        }
    
    def save(self, path: Optional[str] = None):
        if path is None:
            path = Path(self.output_dir) / 'config.json'
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'ExperimentConfig':
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(
            model=ModelConfig(**config_dict['model']),
            equivariance=EquivarianceLossConfig(**config_dict['equivariance']),
            scheduler=SchedulerConfig(**config_dict['scheduler']),
            training=TrainingConfig(**config_dict['training']),
            data=DataConfig(**config_dict['data']),
            logging=LoggingConfig(**config_dict['logging']),
            experiment_name=config_dict['experiment_name'],
            seed=config_dict['seed'],
            device=config_dict['device'],
        )


def get_config(preset: str = 'default') -> ExperimentConfig:
    """Factory for preset configurations"""
    
    presets = {
        'baseline': lambda: ExperimentConfig(
            experiment_name='baseline_gcn',
            model=ModelConfig(model_type='gcn'),
            scheduler=SchedulerConfig(alpha_0=0.0),
        ),
        'e3_equivariant': lambda: ExperimentConfig(
            experiment_name='e3_egnn',
            model=ModelConfig(model_type='egnn', update_coords=True),
            equivariance=EquivarianceLossConfig(
                symmetry_groups=['e3'],
                group_weights={'e3': 0.1}
            ),
            data=DataConfig(use_positions=True),
        ),
        'multi_symmetry': lambda: ExperimentConfig(
            experiment_name='multi_sym',
            model=ModelConfig(model_type='egnn'),
            equivariance=EquivarianceLossConfig(
                symmetry_groups=['permutation', 'so3', 'translation'],
                group_weights={'permutation': 0.05, 'so3': 0.1, 'translation': 0.1},
                use_adaptive_weighting=True,
            ),
            data=DataConfig(use_positions=True),
        ),
    }
    
    if preset == 'default':
        return ExperimentConfig()
    elif preset in presets:
        return presets[preset]()
    else:
        raise ValueError(f"Unknown preset '{preset}'")
