"""Adapters: unified ``relaxed`` config -> legacy family configs.

Each engine family keeps its own legacy defaults (e.g. dynamics uses
``dropout=0.0``/``hidden_dim=128``, graph uses ``dropout=0.5``/
``hidden_channels=64``). To guarantee parity, the adapter starts from the
legacy family's default config and applies ONLY the fields the user actually
set (tracked by the CLI in ``config._overridden``). If no override set is
present (programmatic construction), all unified values are applied.

Field-name mappings between the unified sections and the legacy sections are
explicit below; fields without a mapping keep the legacy default.
"""
from __future__ import annotations

from typing import Any, Optional

# unified dotted key -> legacy section.attribute
_TO_REMUL = {
    # data
    "data.name": ("data", "name"), "data.root": ("data", "root"),
    "data.n_bodies": ("data", "n_bodies"), "data.num_steps": ("data", "num_steps"),
    "data.dt": ("data", "dt"), "data.distribution": ("data", "distribution"),
    "data.field_strength": ("data", "field_strength"),
    "data.field_axis": ("data", "field_axis"),
    "data.n_train": ("data", "n_train"), "data.n_val": ("data", "n_val"),
    "data.n_test": ("data", "n_test"), "data.molecule": ("data", "molecule"),
    "data.delta_t": ("data", "delta_t"),
    "data.md17_n_train": ("data", "md17_n_train"),
    "data.md17_n_val": ("data", "md17_n_val"),
    "data.md17_n_test": ("data", "md17_n_test"),
    "data.mocap_subject": ("data", "mocap_subject"),
    "data.mocap_delta_t": ("data", "delta_t"),  # explicit, replaces the old hack
    # model
    "model.name": ("model", "name"), "model.channels": ("model", "channels"),
    "model.num_layers": ("model", "num_layers"),
    "model.num_heads": ("model", "num_heads"),
    "model.hidden_dim": ("model", "hidden_dim"),
    "model.mlp_hidden": ("model", "mlp_hidden"),
    "model.num_degrees": ("model", "num_degrees"),
    "model.se3_channels": ("model", "se3_channels"),
    "model.num_multivectors": ("model", "num_multivectors"),
    "model.num_fourier_modes": ("model", "num_fourier_modes"),
    "model.dropout": ("model", "dropout"),
    # train
    "train.mode": ("train", "mode"), "train.penalty": ("train", "penalty"),
    "train.alpha": ("train", "alpha"), "train.beta": ("train", "beta"),
    "train.gradnorm_alpha": ("train", "gradnorm_alpha"),
    "train.gradnorm_lr": ("train", "gradnorm_lr"),
    "train.epochs": ("train", "epochs"), "train.max_steps": ("train", "max_steps"),
    "train.batch_size": ("train", "batch_size"), "train.lr": ("train", "lr"),
    "train.weight_decay": ("train", "weight_decay"),
    "train.eval_group_samples": ("train", "eval_group_samples"),
    "train.eval_equiv_batches": ("train", "eval_equiv_batches"),
    "train.seed": ("train", "seed"), "train.device": ("train", "device"),
    # loss -> remul train section
    "loss.group": ("train", "group"), "loss.metric": ("train", "metric"),
    "loss.num_group_samples": ("train", "num_group_samples"),
    # log
    "log.logger_type": ("log", "logger_type"), "log.log_dir": ("log", "log_dir"),
    "log.run_name": ("log", "run_name"), "log.log_every": ("log", "log_every"),
    "log.eval_every": ("log", "eval_every"),
}

_TO_GRAPH = {
    # data
    "data.name": ("data", "dataset_name"), "data.root": ("data", "root"),
    "data.subset": ("data", "subset"), "data.num_workers": ("data", "num_workers"),
    "data.use_positions": ("data", "use_positions"),
    "data.feature_type": ("data", "feature_type"),
    "data.persistent_workers": ("data", "persistent_workers"),
    "data.prefetch_factor": ("data", "prefetch_factor"),
    "data.md17_molecule": ("data", "md17_molecule"),
    "data.qm9_target": ("data", "qm9_target"),
    "data.use_augmentation": ("data", "use_augmentation"),
    "data.train_split": ("data", "train_split"), "data.val_split": ("data", "val_split"),
    "data.test_split": ("data", "test_split"), "data.rewiring": ("data", "rewiring"),
    "data.rewiring_k": ("data", "rewiring_k"),
    "data.rewiring_threshold": ("data", "rewiring_threshold"),
    # model
    "model.name": ("model", "model_type"),
    "model.in_channels": ("model", "in_channels"),
    "model.hidden_channels": ("model", "hidden_channels"),
    "model.out_channels": ("model", "out_channels"),
    "model.num_layers": ("model", "num_layers"),
    "model.dropout": ("model", "dropout"),
    "model.spatial_dim": ("model", "spatial_dim"),
    "model.num_heads": ("model", "num_heads"),
    "model.num_gaussians": ("model", "num_gaussians"),
    "model.num_spherical": ("model", "num_spherical"),
    "model.cutoff": ("model", "cutoff"),
    "model.update_coords": ("model", "update_coords"),
    "model.max_ell": ("model", "max_ell"),
    "model.num_degrees": ("model", "num_degrees"),
    "model.use_pos": ("model", "use_pos"),
    "model.use_layer_norm": ("model", "use_layer_norm"),
    "model.use_batch_norm": ("model", "use_batch_norm"),
    # loss -> equivariance
    "loss.symmetry_groups": ("equivariance", "symmetry_groups"),
    "loss.group_weights": ("equivariance", "group_weights"),
    "loss.num_samples": ("equivariance", "num_samples"),
    "loss.normalize": ("equivariance", "normalize"),
    "loss.feature_type": ("equivariance", "feature_type"),
    "loss.max_translation": ("equivariance", "max_translation"),
    "loss.scale_range": ("equivariance", "scale_range"),
    "loss.stochastic_probability": ("equivariance", "stochastic_probability"),
    "loss.layer_weight_strategy": ("equivariance", "layer_weight_strategy"),
    "loss.layer_decay_rate": ("equivariance", "layer_decay_rate"),
    "loss.normalize_group_weights": ("equivariance", "normalize_group_weights"),
    "loss.total_equivariance_strength": ("equivariance", "total_equivariance_strength"),
    # schedule
    "schedule.alpha_0": ("scheduler", "alpha_0"),
    "schedule.beta": ("scheduler", "beta"), "schedule.gamma": ("scheduler", "gamma"),
    "schedule.lr_schedule": ("scheduler", "lr_schedule"),
    "schedule.lr_step_size": ("scheduler", "lr_step_size"),
    "schedule.lr_gamma": ("scheduler", "lr_gamma"),
    "schedule.lr_warmup_epochs": ("scheduler", "lr_warmup_epochs"),
    "schedule.plateau_patience": ("scheduler", "plateau_patience"),
    "schedule.plateau_factor": ("scheduler", "plateau_factor"),
    "schedule.plateau_mode": ("scheduler", "plateau_mode"),
    "schedule.exponential_decay_rate": ("scheduler", "exponential_decay_rate"),
    # train -> training
    "train.epochs": ("training", "num_epochs"),
    "train.batch_size": ("training", "batch_size"),
    "train.lr": ("training", "learning_rate"),
    "train.weight_decay": ("training", "weight_decay"),
    "train.grad_clip": ("training", "grad_clip"),
    "train.accumulation_steps": ("training", "accumulation_steps"),
    "train.use_amp": ("training", "use_amp"),
    "train.patience": ("training", "patience"),
    "train.min_delta": ("training", "min_delta"),
    "train.seed": ("seed", None), "train.device": ("device", None),
    # log -> logging
    "log.logger_type": ("logging", "logger_type"),
    "log.wandb_project": ("logging", "wandb_project"),
    "log.wandb_entity": ("logging", "wandb_entity"),
    "log.wandb_name": ("logging", "wandb_name"),
    "log.wandb_mode": ("logging", "wandb_mode"),
    "log.tensorboard_dir": ("logging", "tensorboard_dir"),
    # run
    "run.experiment_name": ("experiment_name", None),
    "run.checkpoint_dir": ("checkpoint_dir", None),
    "run.output_dir": ("output_dir", None),
    "run.timestamp": ("timestamp", None),
}


def _apply(unified, legacy, mapping: dict, overridden: Optional[set]) -> None:
    apply_all = overridden is None
    for ukey, (section, attr) in mapping.items():
        if not apply_all and ukey not in overridden:
            continue
        parts = ukey.split(".")
        value = getattr(getattr(unified, parts[0]), parts[1])
        target = legacy if attr is None else getattr(legacy, section)
        name = section if attr is None else attr
        current = getattr(target, name, None)
        if current is not None and value is not None and not isinstance(value, type(current)):
            try:
                value = type(current)(value)
            except (ValueError, TypeError):
                pass
        setattr(target, name, value)


def to_remul_config(unified):
    """Unified config -> legacy ``remul.config.ExperimentConfig`` (defaults preserved)."""
    from remul.config import ExperimentConfig as RemulExperimentConfig
    legacy = RemulExperimentConfig()
    _apply(unified, legacy, _TO_REMUL, getattr(unified, "_overridden", None))
    # The unified device default (cuda-if-available) is deliberate — always forward.
    legacy.train.device = unified.train.device
    return legacy


def to_graph_config(unified):
    """Unified config -> legacy top-level ``config.ExperimentConfig`` (defaults preserved)."""
    import config as graph_config_module
    # Bypass __post_init__ side effects until fields are applied.
    legacy = graph_config_module.ExperimentConfig.__new__(graph_config_module.ExperimentConfig)
    for section, cls in (("model", graph_config_module.ModelConfig),
                         ("equivariance", graph_config_module.EquivarianceLossConfig),
                         ("scheduler", graph_config_module.SchedulerConfig),
                         ("training", graph_config_module.TrainingConfig),
                         ("data", graph_config_module.DataConfig),
                         ("logging", graph_config_module.LoggingConfig)):
        setattr(legacy, section, cls())
    legacy.experiment_name = "default"
    legacy.seed = 42
    legacy.device = unified.train.device
    legacy.deterministic = True
    legacy.checkpoint_dir = ""
    legacy.output_dir = ""
    from datetime import datetime
    legacy.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _apply(unified, legacy, _TO_GRAPH, getattr(unified, "_overridden", None))
    legacy.finalize()
    return legacy
