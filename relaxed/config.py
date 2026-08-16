"""Unified configuration for the relaxed-equivariance framework.

A single ``ExperimentConfig`` covers both tasks:

* ``task='dynamics'`` — the REMUL paper setup (N-body, CMU MoCap, MD17
  trajectories; dense ``(B, N, *)`` tensors; modes standard/da/remul with the
  ground-truth-anchored equivariance objective).
* ``task='graph'`` — molecular graph property regression (ZINC, QM9, QM7b,
  MD17-graph, ModelNet40; PyG batches; the layer-wise functional equivariance
  objective with depth-scheduled per-layer weights).

The task is inferred from the dataset name (``TASK_BY_DATASET``); every
behavior of the two legacy frameworks is reachable via flags.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import torch

DYNAMICS_DATASETS = {"nbody", "nbody_egnn", "md17_dyn", "mocap"}
GRAPH_DATASETS = {"ZINC", "QM9", "QM7b", "MD17", "ModelNet40"}
TASK_BY_DATASET = {**{d: "dynamics" for d in DYNAMICS_DATASETS},
                   **{d: "graph" for d in GRAPH_DATASETS}}

DYN_MODELS = {"transformer", "mlp", "gnn", "mpnn", "egnn", "se3_transformer",
              "tfn", "gatr", "egno", "hegnn", "gmn", "emlp", "rpp", "per"}
GRAPH_MODELS = {"raw_mlp", "transformer", "gcn", "gin", "graphsage", "schnet",
                "dimenet", "egnn", "painn", "vector_neuron", "se3_transformer",
                "nequip", "clofnet"}
GROUPS = {"permutation", "so3", "o3", "se3", "e3", "translation", "reflection",
          "scaling", "so2_x", "so2_y", "so2_z"}


@dataclass
class DataConfig:
    name: str = "nbody"
    root: str = "data"
    # dynamics: nbody
    n_bodies: int = 4
    num_steps: int = 100
    dt: float = 0.001
    distribution: Literal["in", "ood"] = "in"
    field_strength: float = 0.0   # external accel; 0=SO(3)-symmetric, >0 breaks to SO(2) about field_axis
    field_axis: int = 2           # 0=x,1=y,2=z
    iso_input: bool = False       # SO(3)-symmetric inputs but SO(2) task (label-breaks-symmetry wedge)
    n_train: int = 100
    n_val: int = 5000
    n_test: int = 5000
    # dynamics: md17_dyn
    molecule: str = "aspirin"
    delta_t: int = 5000
    md17_n_train: int = 500
    md17_n_val: int = 2000
    md17_n_test: int = 2000
    # dynamics: mocap
    mocap_subject: int = 35
    mocap_delta_t: int = 30  # explicit per-dataset default (no cross-dataset hacks)
    # graph
    subset: bool = True
    num_workers: int = 4
    use_positions: bool = False
    feature_type: Literal["atomic", "combined"] = "combined"
    persistent_workers: bool = True
    prefetch_factor: int = 2
    md17_molecule: str = "aspirin"
    qm9_target: int = 7
    use_augmentation: bool = False
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    rewiring: str = "none"
    rewiring_k: int = 2
    rewiring_threshold: float = 5.0


@dataclass
class ModelConfig:
    name: str = "transformer"
    # shared
    num_layers: int = 4
    dropout: float = 0.5
    # dynamics zoo
    channels: int = 384
    num_heads: int = 8
    hidden_dim: int = 64
    mlp_hidden: int = 680
    num_degrees: int = 4
    se3_channels: int = 8
    num_multivectors: int = 16
    num_fourier_modes: int = 3
    # graph zoo
    in_channels: int = 12
    hidden_channels: int = 64
    out_channels: int = 1
    spatial_dim: int = 3
    num_gaussians: int = 50
    num_spherical: int = 7
    cutoff: float = 10.0
    update_coords: bool = False
    max_ell: int = 2
    use_pos: bool = False
    use_batch_norm: bool = True
    use_layer_norm: bool = False

    def __post_init__(self):
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")


@dataclass
class LossConfig:
    """Equivariance objective: which formulation, which groups, how weighted."""
    formulation: Literal["remul", "layerwise"] = "remul"
    # remul formulation (ground-truth anchored) + E/E' eval
    group: str = "so3"
    metric: Literal["l1", "l2"] = "l2"
    num_group_samples: int = 1
    # layerwise formulation (functional, per-layer)
    symmetry_groups: List[str] = field(default_factory=list)
    group_weights: Dict[str, float] = field(default_factory=lambda: {
        "permutation": 0.1, "so3": 0.1, "o3": 0.1, "se3": 0.1,
        "e3": 0.1, "translation": 0.1, "reflection": 0.1, "scaling": 0.05})
    num_samples: int = 2
    normalize: bool = True
    feature_type: Literal["invariant", "equivariant"] = "invariant"
    max_translation: float = 5.0
    scale_range: Tuple[float, float] = (0.5, 2.0)
    stochastic_probability: float = 0.25
    layer_weight_strategy: str = "constant"
    layer_decay_rate: float = 0.5
    # Normalize per-group weights to a fixed total so group-set arms are
    # strength-matched (fairness for "do groups help?"). See
    # train._resolve_group_weights.
    normalize_group_weights: bool = True
    total_equivariance_strength: float = 1.0

    def __post_init__(self):
        for g in self.symmetry_groups:
            if g not in GROUPS:
                raise ValueError(f"Invalid symmetry group '{g}'")


@dataclass
class ScheduleConfig:
    """Global equivariance strength + LR scheduling (single source of truth)."""
    alpha_0: float = 1.0        # global eq strength, applied ONCE via layer weights
    beta: float = 0.1           # exponential decay rate parameter (DepthScheduler)
    gamma: float = 0.1          # linear decay parameter (DepthScheduler)
    lr_schedule: Literal["none", "step", "cosine", "plateau", "exponential"] = "none"
    lr_step_size: int = 50
    lr_gamma: float = 0.5
    lr_warmup_epochs: int = 5
    plateau_patience: int = 5
    plateau_factor: float = 0.5
    plateau_mode: Literal["min", "max"] = "min"
    exponential_decay_rate: float = 0.95


@dataclass
class TrainConfig:
    task: Literal["dynamics", "graph"] = "dynamics"   # auto-set from dataset
    # REMUL multitask (dynamics)
    mode: Literal["standard", "remul", "da"] = "remul"
    penalty: Literal["constant", "gradual"] = "constant"
    alpha: float = 1.0
    beta: float = 1.0
    gradnorm_alpha: float = 1.5
    gradnorm_lr: float = 0.025
    # optimization
    epochs: int = 500
    max_steps: Optional[int] = None
    batch_size: int = 100
    lr: float = 5e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0          # >0 enables clipping; mandatory for gatr/egnn/egno/gmn
                                    # which NaN/diverge at the paper lr without it (harmless to stable models)
    accumulation_steps: int = 1
    use_amp: bool = False
    patience: int = 20
    min_delta: float = 1e-4
    # evaluation
    eval_group_samples: int = 20
    eval_equiv_batches: int = 16
    per_axis_eval: bool = False   # also report per-axis E' (so2_x/y/z; paper App. D.5)
    # OOD / rotational-robustness eval for the GRAPH task: rotated-test MAE and
    # label-free functional equivariance error E/E' at the graph-prediction level.
    ood_eval: bool = False
    ood_num_rotations: int = 8
    ood_groups: List[str] = field(default_factory=lambda: ["so3"])
    seed: int = 0
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class LogConfig:
    logger_type: Literal["none", "tensorboard", "wandb", "both"] = "none"
    log_dir: str = "outputs/relaxed"
    run_name: str = ""
    log_every: int = 50
    eval_every: int = 1
    wandb_project: str = "PGM_Project"  # was swapped with entity
    wandb_entity: Optional[str] = None  # None -> wandb uses the user's default entity
    wandb_name: Optional[str] = None
    wandb_mode: Literal["online", "offline", "disabled"] = "online"
    tensorboard_dir: str = "./runs"
    save_history: bool = True      # write per-epoch metrics to history.jsonl


@dataclass
class RunConfig:
    experiment_name: str = "default"
    seeds: List[int] = field(default_factory=lambda: [0])
    checkpoint_dir: str = ""
    output_dir: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    log: LogConfig = field(default_factory=LogConfig)
    run: RunConfig = field(default_factory=RunConfig)

    def __post_init__(self):
        self.finalize()

    def finalize(self):
        """(Re)derive task, run dirs and validate; safe to call after overrides."""
        # Task follows the dataset family.
        if self.data.name in TASK_BY_DATASET:
            self.train.task = TASK_BY_DATASET[self.data.name]
        else:
            raise ValueError(
                f"Unknown dataset '{self.data.name}'. Dynamics: {sorted(DYNAMICS_DATASETS)}; "
                f"graph: {sorted(GRAPH_DATASETS)}")

        # Formulation default follows the task unless explicitly overridden.
        canonical = "remul" if self.train.task == "dynamics" else "layerwise"
        if self.loss.formulation != canonical:
            import warnings
            warnings.warn(
                f"loss.formulation='{self.loss.formulation}' is not implemented for "
                f"task '{self.train.task}'; the canonical '{canonical}' formulation "
                "is used instead.", UserWarning)

        # Run directories.
        if not self.run.checkpoint_dir:
            self.run.checkpoint_dir = f"./checkpoints/{self.run.experiment_name}_{self.run.timestamp}"
        if not self.run.output_dir:
            self.run.output_dir = f"./outputs/{self.run.experiment_name}_{self.run.timestamp}"

        # Validation.
        if self.train.task == "graph":
            position_models = {"schnet", "dimenet", "egnn", "painn", "vector_neuron",
                               "se3_transformer", "nequip", "clofnet"}
            if self.model.name in position_models and not self.data.use_positions:
                raise ValueError(
                    f"Model '{self.model.name}' requires 3D positions. "
                    "Set data.use_positions=True")
        else:
            valid = DYN_MODELS
            if self.model.name not in valid:
                raise ValueError(f"Unknown dynamics model '{self.model.name}'")
        if self.train.task == "graph" and self.model.name not in GRAPH_MODELS:
            raise ValueError(f"Unknown graph model '{self.model.name}'")
        if self.loss.formulation == "layerwise" and self.train.task == "graph":
            geometric = {"so3", "o3", "se3", "e3", "translation", "reflection", "scaling"}
            if geometric & set(self.loss.symmetry_groups) and not self.data.use_positions:
                import warnings
                warnings.warn(
                    "Geometric groups with data.use_positions=False: positions are "
                    "zeroed and the geometric equivariance loss is trivially ~0.",
                    UserWarning)

    def to_dict(self) -> dict:
        return {section: asdict(getattr(self, section))
                for section in ("data", "model", "loss", "schedule", "train", "log", "run")}

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        return cls(
            data=DataConfig(**d.get("data", {})),
            model=ModelConfig(**d.get("model", {})),
            loss=LossConfig(**d.get("loss", {})),
            schedule=ScheduleConfig(**d.get("schedule", {})),
            train=TrainConfig(**d.get("train", {})),
            log=LogConfig(**d.get("log", {})),
            run=RunConfig(**d.get("run", {})),
        )
