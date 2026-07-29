"""REMUL: Relaxed Equivariance via Multitask Learning — dynamics experiments.

Self-contained reproduction of the datasets, models and training procedure from
Elhag et al. (arXiv:2410.17878), kept separate from the repository's ZINC/QM9
graph-regression pipeline.
"""
from .config import ExperimentConfig, DataConfig, ModelConfig, TrainConfig, LogConfig

__all__ = ["ExperimentConfig", "DataConfig", "ModelConfig", "TrainConfig", "LogConfig"]
