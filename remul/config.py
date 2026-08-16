"""Configuration for the REMUL dynamics experiments (arXiv:2410.17878).

These dataclasses are intentionally separate from the top-level ``config.py``
(which configures the ZINC/QM9 graph-regression pipeline). The paper's tasks are
3D trajectory / position prediction, so they need their own knobs.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal, Optional

DatasetName = Literal["nbody", "nbody_egnn", "md17", "motion_capture"]
ModelName = Literal[
    "transformer", "mlp", "gnn", "mpnn",
    "egnn", "se3_transformer", "tfn", "gatr", "egno", "hegnn", "gmn",
    "emlp", "rpp", "per",
]
TrainMode = Literal["standard", "remul", "da"]
Penalty = Literal["constant", "gradual"]


@dataclass
class DataConfig:
    name: DatasetName = "nbody"
    root: str = "data/remul"
    # N-body
    n_bodies: int = 4                 # 4 (GATr-style) or 5 (EGNN-style)
    num_steps: int = 100              # Euler integration steps to predict ahead
    dt: float = 0.001
    distribution: Literal["in", "ood"] = "in"
    field_strength: float = 0.0       # uniform external accel; 0 => SO(3)-symmetric,
                                      # >0 breaks SO(3)->SO(2) about field_axis (symmetry-breaking knob)
    field_axis: int = 2               # 0=x, 1=y, 2=z; residual symmetry is SO(2) about this axis
    iso_input: bool = False           # isotropize inputs (random SO(3) rot before integration) →
                                      # SO(3)-symmetric input distribution but SO(2) task (label-breaks-symmetry)
    n_train: int = 100
    n_val: int = 5000
    n_test: int = 5000
    # MD17
    molecule: str = "aspirin"
    delta_t: int = 5000               # frame gap (MD17=5000, MoCap=30)
    md17_n_train: int = 500
    md17_n_val: int = 2000
    md17_n_test: int = 2000
    # Motion capture
    mocap_subject: int = 35           # 35 = walking, 9 = running
    # Common
    seed: int = 0


@dataclass
class ModelConfig:
    name: ModelName = "transformer"
    hidden_dim: int = 128
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.0
    # Transformer channel count (paper uses 384 for N-body/MoCap Transformer)
    channels: int = 384
    # SE(3)/steerable
    num_degrees: int = 4
    se3_channels: int = 8
    # GATr / EGNO
    num_multivectors: int = 16
    num_fourier_modes: int = 3
    # EMLP / RPP / PER
    mlp_hidden: int = 680


@dataclass
class TrainConfig:
    mode: TrainMode = "remul"         # standard | remul | da
    penalty: Penalty = "constant"     # constant | gradual (GradNorm)
    group: str = "so3"
    alpha: float = 1.0                # objective weight (paper: initial alpha=1)
    beta: float = 1.0                 # equivariance weight (the REMUL lever)
    num_group_samples: int = 1
    metric: Literal["l1", "l2"] = "l2"
    # GradNorm
    gradnorm_alpha: float = 1.5
    gradnorm_lr: float = 0.025
    # Optimization
    epochs: int = 500
    max_steps: Optional[int] = None   # if set, overrides epochs (N-body uses steps)
    batch_size: int = 100
    lr: float = 5e-4
    weight_decay: float = 0.0
    # Evaluation
    eval_group_samples: int = 20
    eval_equiv_batches: int = 16   # cap batches for per-epoch E/E' averaging (final test uses the full split)
    seed: int = 0
    device: str = "cpu"


@dataclass
class LogConfig:
    logger_type: Literal["none", "tensorboard", "wandb"] = "none"
    log_dir: str = "outputs/remul"
    run_name: str = ""                # empty → auto slug from dataset/model/mode
    log_every: int = 50
    eval_every: int = 1               # epochs


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    log: LogConfig = field(default_factory=LogConfig)

    def to_dict(self) -> dict:
        return asdict(self)
