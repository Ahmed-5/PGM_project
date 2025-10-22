"""
Configuration file for Relaxed Equivariance GNN experiments
"""

import torch
from dataclasses import dataclass
from typing import Literal, Optional
import warnings
from datetime import datetime
from pydantic.warnings import UnsupportedFieldAttributeWarning

warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    in_channels: int = 28  # ZINC dataset default
    # in_channels: int = 28+11  # degree one-hot for ZINC
    hidden_channels: int = 64
    out_channels: int = 1  # Regression task
    num_layers: int = 4
    dropout: float = 0.5
    gnn_type: Literal['GCN', 'GIN', 'GraphSAGE'] = 'GCN'
    
@dataclass
class SchedulerConfig:
    """Depth-adaptive scheduler configuration"""
    schedule_type: Literal['constant', 'exponential', 'linear', 'learnable'] = 'exponential'
    alpha_0: float = 1.0  # Initial equivariance weight
    beta: float = 0.1     # Exponential decay rate
    gamma: float = 0.1    # Linear decay rate
    
@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    scheduler_lr: Literal['step', 'cosine', 'none'] = 'step'
    patience: int = 20  # Early stopping
    
@dataclass
class DataConfig:
    """Dataset configuration"""
    dataset_name: Literal['ZINC', 'QM9', 'AQSOL'] = 'ZINC'
    subset: bool = True  # Use ZINC-12k
    root: str = './data'
    num_workers: int = 4
    
@dataclass
class LoggingConfig:
    """Logging configuration"""
    logger_type: Literal['wandb', 'tensorboard', 'none'] = 'tensorboard'
    
    # Weights & Biases settings
    wandb_project: str = 'relaxed-equivariance-gnn'
    wandb_entity: Optional[str] = None  # Your wandb username/team
    wandb_name: Optional[str] = None    # Run name (auto-generated if None)
    wandb_tags: list = None             # Tags for organizing runs
    wandb_notes: Optional[str] = None   # Additional notes
    
    # TensorBoard settings
    tensorboard_dir: str = './runs'
    
    # General logging settings
    log_interval: int = 10  # Log every N batches
    log_gradients: bool = False  # Log gradient histograms
    log_layer_outputs: bool = True  # Log layer-wise metrics
    save_model_artifact: bool = True  # Save model as artifact (wandb only)
    
    def __post_init__(self):
        if self.wandb_tags is None:
            self.wandb_tags = []
    
@dataclass
class ExperimentConfig:
    """Full experiment configuration"""
    model: ModelConfig = ModelConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()
    logging: LoggingConfig = LoggingConfig()
    
    seed: int = 42
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    # checkpoint directory for saving models and logs add the date-time stamp
    experiment_name: str = 'default'
    timestamp: str = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir: str = f'./checkpoints/{experiment_name}_{timestamp}'

    def dict(self):
        """Convert the ExperimentConfig to a dictionary"""
        return {
            'model': vars(self.model),
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
        # No equivariance loss
        config = ExperimentConfig()
        config.scheduler.alpha_0 = 0.0
        config.experiment_name = 'baseline'
        config.logging.wandb_tags = ['baseline', 'no-equivariance']
        return config
    
    elif config_name == 'constant_alpha':
        # Constant equivariance weight
        config = ExperimentConfig()
        config.scheduler.schedule_type = 'constant'
        config.scheduler.alpha_0 = 1.0
        config.experiment_name = 'constant_alpha'
        config.logging.wandb_tags = ['constant', 'alpha=1.0']
        return config
    
    elif config_name == 'exponential_decay':
        # Strong early layers, weak late layers
        config = ExperimentConfig()
        config.scheduler.schedule_type = 'exponential'
        config.scheduler.alpha_0 = 2.0
        config.scheduler.beta = 0.2
        config.experiment_name = 'exponential_decay'
        config.logging.wandb_tags = ['exponential', 'depth-adaptive']
        return config
    
    elif config_name == 'linear_decay':
        config = ExperimentConfig()
        config.scheduler.schedule_type = 'linear'
        config.scheduler.alpha_0 = 2.0
        config.scheduler.gamma = 0.15
        config.experiment_name = 'linear_decay'
        config.logging.wandb_tags = ['linear', 'depth-adaptive']
        return config
    
    elif config_name == 'learnable':
        config = ExperimentConfig()
        config.scheduler.schedule_type = 'learnable'
        config.training.num_epochs = 150  # Need more epochs to learn alphas
        config.experiment_name = 'learnable_alphas'
        config.logging.wandb_tags = ['learnable', 'bayesian']
        config.logging.log_gradients = True
        return config
    
    elif config_name == 'deep_network':
        # Test depth-adaptive on deeper networks
        config = ExperimentConfig()
        config.model.num_layers = 16
        config.scheduler.schedule_type = 'exponential'
        config.scheduler.beta = 0.1
        config.experiment_name = 'deep_network_16layers'
        config.logging.wandb_tags = ['deep', '16-layers']
        return config
    
    else:
        raise ValueError(f"Unknown config name: {config_name}")
