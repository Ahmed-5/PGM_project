"""Unified logging: Weights & Biases, TensorBoard, composite, or no-op.

Absorbs the legacy ``logger.py`` factory and the remul hparam flattening.
Interface: ``log_metrics``, ``log_hyperparameters``, ``log_histogram``,
``log_image``, ``watch_model``, ``save_model_artifact``, ``finish``.
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


def flatten_hparams(config_dict: dict) -> dict[str, Any]:
    """Flat hparams with only types TensorBoard/wandb accept."""
    out: dict[str, Any] = {}
    for section, values in config_dict.items():
        items = values.items() if isinstance(values, dict) else [(section, values)]
        for key, value in items:
            k = f"{section}/{key}" if isinstance(values, dict) else section
            if value is None:
                out[k] = "none"
            elif isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                out[k] = str(value)
            elif isinstance(value, (str, int, float, bool)):
                out[k] = value
            else:
                out[k] = str(value)
    return out


class BaseLogger:
    def __init__(self, config):
        self.config = config

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        pass

    def log_hyperparameters(self, params: Dict[str, Any]):
        pass

    def log_histogram(self, name: str, values: torch.Tensor, step: Optional[int] = None):
        pass

    def log_image(self, name: str, image, step: Optional[int] = None):
        pass

    def watch_model(self, model, log_freq: int = 100):
        pass

    def save_model_artifact(self, model_path: str, name: str = "model"):
        pass

    def finish(self):
        pass


class WandbLogger(BaseLogger):
    def __init__(self, config):
        super().__init__(config)
        if not WANDB_AVAILABLE:
            raise ImportError("wandb is not installed. Install with: pip install wandb")
        run_name = config.log.wandb_name or (
            f"{config.run.experiment_name}_{datetime.now():%Y%m%d_%H%M%S}")
        self.run = wandb.init(
            project=config.log.wandb_project,
            entity=config.log.wandb_entity,
            name=run_name,
            mode=config.log.wandb_mode,
            config=config.to_dict() if hasattr(config, "to_dict") else {},
        )

    def log_metrics(self, metrics, step=None):
        wandb.log(metrics, step=step)

    def log_hyperparameters(self, params):
        wandb.config.update(params, allow_val_change=True)

    def watch_model(self, model, log_freq=100):
        wandb.watch(model, log="all", log_freq=log_freq)

    def save_model_artifact(self, model_path, name="model"):
        artifact = wandb.Artifact(name, type="model")
        artifact.add_file(model_path)
        self.run.log_artifact(artifact)

    def finish(self):
        wandb.finish()


class TensorBoardLogger(BaseLogger):
    def __init__(self, config):
        super().__init__(config)
        if not TENSORBOARD_AVAILABLE:
            raise ImportError("tensorboard not available. Install with: pip install tensorboard")
        log_dir = f"{config.log.tensorboard_dir}/{config.run.experiment_name}_{datetime.now():%Y%m%d_%H%M%S}"
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_metrics(self, metrics, step=None):
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, step)

    def log_hyperparameters(self, params):
        flat = flatten_hparams(params) if any(isinstance(v, dict) for v in params.values()) else params
        self.writer.add_hparams(flat, {"hparam/placeholder": 0.0})

    def log_histogram(self, name, values, step=None):
        self.writer.add_histogram(name, values.detach().cpu(), step)

    def log_image(self, name, image, step=None):
        self.writer.add_figure(name, image, step)

    def finish(self):
        self.writer.close()


class NoLogger(BaseLogger):
    pass


class CompositeLogger(BaseLogger):
    """Forward every call to multiple child loggers (wandb + tensorboard)."""

    def __init__(self, config, loggers: List[BaseLogger]):
        super().__init__(config)
        self.loggers = loggers

    def _forward(self, method: str, *args, **kwargs):
        for logger in self.loggers:
            fn = getattr(logger, method, None)
            if callable(fn):
                fn(*args, **kwargs)

    def log_metrics(self, metrics, step=None):
        self._forward("log_metrics", metrics, step)

    def log_hyperparameters(self, params):
        self._forward("log_hyperparameters", params)

    def log_histogram(self, name, values, step=None):
        self._forward("log_histogram", name, values, step)

    def log_image(self, name, image, step=None):
        self._forward("log_image", name, image, step)

    def watch_model(self, model, log_freq=100):
        self._forward("watch_model", model, log_freq)

    def save_model_artifact(self, model_path, name="model"):
        self._forward("save_model_artifact", model_path, name)

    def finish(self):
        self._forward("finish")


def get_logger(config) -> BaseLogger:
    logger_type = config.log.logger_type.lower()
    if logger_type == "wandb":
        return WandbLogger(config)
    if logger_type == "tensorboard":
        return TensorBoardLogger(config)
    if logger_type == "both":
        return CompositeLogger(config, [WandbLogger(config), TensorBoardLogger(config)])
    if logger_type == "none":
        return NoLogger(config)
    raise ValueError(f"Unknown logger type: {logger_type}. Choose: wandb, tensorboard, both, none")
