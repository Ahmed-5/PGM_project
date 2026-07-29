"""
Unified logging interface for Weights & Biases, TensorBoard, and no logging
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
from pydantic.warnings import UnsupportedFieldAttributeWarning

warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: tensorboard not available. Install with: pip install tensorboard")


class BaseLogger:
    """Base logger interface"""
    
    def __init__(self, config):
        self.config = config
        self.step = 0
        
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log scalar metrics"""
        pass
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters"""
        pass
    
    def log_model_graph(self, model, input_data):
        """Log model architecture graph"""
        pass
    
    def log_histogram(self, name: str, values: torch.Tensor, step: Optional[int] = None):
        """Log histogram of values"""
        pass
    
    def log_image(self, name: str, image, step: Optional[int] = None):
        """Log image/plot"""
        pass
    
    def watch_model(self, model, log_freq: int = 100):
        """Watch model gradients and parameters"""
        pass
    
    def finish(self):
        """Finish logging"""
        pass


class WandbLogger(BaseLogger):
    """Weights & Biases logger"""
    
    def __init__(self, config):
        super().__init__(config)
        
        if not WANDB_AVAILABLE:
            raise ImportError("wandb is not installed. Install with: pip install wandb")
        
        # Generate run name if not provided
        run_name = config.logging.wandb_name
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{config.experiment_name}_{timestamp}"
        
        # Initialize wandb
        self.run = wandb.init(
            project=config.logging.wandb_project,
            entity=config.logging.wandb_entity,
            name=run_name,
            tags=config.logging.wandb_tags,
            notes=config.logging.wandb_notes,
            config=self._flatten_config(config)
        )
        
        print(f"✓ WandB initialized: {self.run.url}")
    
    def _flatten_config(self, config) -> Dict[str, Any]:
        """Flatten nested config for wandb"""
        flat_config = {}
        
        # Model config
        for key, value in config.model.__dict__.items():
            flat_config[f'model/{key}'] = value
        
        # Scheduler config
        for key, value in config.scheduler.__dict__.items():
            flat_config[f'scheduler/{key}'] = value
        
        # Training config
        for key, value in config.training.__dict__.items():
            flat_config[f'training/{key}'] = value
        
        # Data config
        for key, value in config.data.__dict__.items():
            flat_config[f'data/{key}'] = value
        
        # Other config
        flat_config['seed'] = config.seed
        flat_config['device'] = config.device
        
        return flat_config
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log scalar metrics"""
        if step is None:
            step = self.step
        
        # Filter out non-scalar values for main logging
        scalar_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.number)):
                scalar_metrics[key] = value
            elif isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    scalar_metrics[key] = value.item()
        
        wandb.log(scalar_metrics, step=step)
        self.step = step + 1
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters"""
        wandb.config.update(params, allow_val_change=True)
    
    def log_histogram(self, name: str, values: torch.Tensor, step: Optional[int] = None):
        """Log histogram of values"""
        if step is None:
            step = self.step
        
        wandb.log({name: wandb.Histogram(values.detach().cpu().numpy())}, step=step)
    
    def log_image(self, name: str, image, step: Optional[int] = None):
        """Log matplotlib figure or image array"""
        if step is None:
            step = self.step
        
        if isinstance(image, plt.Figure):
            wandb.log({name: wandb.Image(image)}, step=step)
        else:
            wandb.log({name: wandb.Image(image)}, step=step)
    
    def watch_model(self, model, log_freq: int = 100):
        """Watch model gradients and parameters"""
        wandb.watch(model, log='all', log_freq=log_freq)
    
    def save_model_artifact(self, model_path: str, name: str = 'model'):
        """Save model as wandb artifact"""
        if self.config.logging.save_model_artifact:
            artifact = wandb.Artifact(name, type='model')
            artifact.add_file(model_path)
            self.run.log_artifact(artifact)
    
    def finish(self):
        """Finish wandb run"""
        wandb.finish()
        print("✓ WandB run finished")


class TensorBoardLogger(BaseLogger):
    """TensorBoard logger"""
    
    def __init__(self, config):
        super().__init__(config)
        
        if not TENSORBOARD_AVAILABLE:
            raise ImportError("TensorBoard is not available. Install with: pip install tensorboard")
        
        # Create log directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(
            config.logging.tensorboard_dir,
            f"{config.experiment_name}_{timestamp}"
        )
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=log_dir)
        self.log_dir = log_dir
        
        print(f"✓ TensorBoard initialized: {log_dir}")
        print(f"  View with: tensorboard --logdir={config.logging.tensorboard_dir}")
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log scalar metrics"""
        if step is None:
            step = self.step
        
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.number)):
                self.writer.add_scalar(key, value, step)
            elif isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    self.writer.add_scalar(key, value.item(), step)
        
        self.step = step + 1
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters as text"""
        # Convert to string format
        hparams_text = "\n".join([f"{k}: {v}" for k, v in params.items()])
        self.writer.add_text('hyperparameters', hparams_text, 0)
    
    def log_model_graph(self, model, input_data):
        """Log model architecture graph"""
        try:
            self.writer.add_graph(model, input_data)
        except Exception as e:
            print(f"Warning: Could not log model graph: {e}")
    
    def log_histogram(self, name: str, values: torch.Tensor, step: Optional[int] = None):
        """Log histogram of values"""
        if step is None:
            step = self.step
        
        self.writer.add_histogram(name, values.detach().cpu(), step)
    
    def log_image(self, name: str, image, step: Optional[int] = None):
        """Log matplotlib figure"""
        if step is None:
            step = self.step
        
        if isinstance(image, plt.Figure):
            self.writer.add_figure(name, image, step)
        else:
            # Assume it's an array
            self.writer.add_image(name, image, step)
    
    def finish(self):
        """Close tensorboard writer"""
        self.writer.close()
        print("✓ TensorBoard writer closed")


class NoLogger(BaseLogger):
    """Dummy logger that does nothing"""
    
    def __init__(self, config):
        super().__init__(config)
        print("✓ No logging enabled")
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        pass
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        pass
    
    def finish(self):
        pass


class CompositeLogger(BaseLogger):
    """Forwards every logging call to multiple child loggers (e.g. wandb + tensorboard)."""

    def __init__(self, config, loggers: List[BaseLogger]):
        super().__init__(config)
        self.loggers = loggers
        print(f"✓ Composite logger over {[type(l).__name__ for l in loggers]}")

    def _forward(self, method: str, *args, **kwargs):
        for logger in self.loggers:
            fn = getattr(logger, method, None)
            if callable(fn):
                fn(*args, **kwargs)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        self._forward('log_metrics', metrics, step)

    def log_hyperparameters(self, params: Dict[str, Any]):
        self._forward('log_hyperparameters', params)

    def log_model_graph(self, model, input_data):
        self._forward('log_model_graph', model, input_data)

    def log_histogram(self, name: str, values: torch.Tensor, step: Optional[int] = None):
        self._forward('log_histogram', name, values, step)

    def log_image(self, name: str, image, step: Optional[int] = None):
        self._forward('log_image', name, image, step)

    def watch_model(self, model, log_freq: int = 100):
        self._forward('watch_model', model, log_freq)

    def save_model_artifact(self, model_path: str, name: str = 'model'):
        self._forward('save_model_artifact', model_path, name)

    def finish(self):
        self._forward('finish')


def get_logger(config) -> BaseLogger:
    """Factory function to get appropriate logger"""

    logger_type = config.logging.logger_type.lower()

    if logger_type == 'wandb':
        return WandbLogger(config)
    elif logger_type == 'tensorboard':
        return TensorBoardLogger(config)
    elif logger_type == 'both':
        return CompositeLogger(config, [WandbLogger(config), TensorBoardLogger(config)])
    elif logger_type == 'none':
        return NoLogger(config)
    else:
        raise ValueError(f"Unknown logger type: {logger_type}. Choose from: wandb, tensorboard, both, none")


class MetricsTracker:
    """Helper class to track and aggregate metrics"""
    
    def __init__(self):
        self.metrics = {}
    
    def update(self, metrics: Dict[str, float]):
        """Update metrics"""
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
    
    def get_averages(self) -> Dict[str, float]:
        """Get average of all tracked metrics"""
        return {key: np.mean(values) for key, values in self.metrics.items()}
    
    def reset(self):
        """Reset all metrics"""
        self.metrics = {}
