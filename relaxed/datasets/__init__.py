"""Unified dataset registry over both families.

* dynamics — N-body (synthetic), MD17 trajectories, CMU MoCap; returns a dict
  ``{train, val, test, ood?, meta}`` of dense ``DynamicsDataset`` splits.
* graph — ZINC, QM9, QM7b, MD17-graph, ModelNet40; returns
  ``(train, val, test)`` PyG datasets with seeded splits and train-split-only
  target normalization.

Both are imported from the legacy loaders (single source of truth) with the
config adapted so each family keeps its legacy defaults.
"""
from __future__ import annotations

from ..adapt import to_graph_config, to_remul_config

DYNAMICS_DATASETS = {"nbody", "nbody_egnn", "md17_dyn", "mocap"}
GRAPH_DATASETS = {"ZINC", "QM9", "QM7b", "MD17", "ModelNet40"}

# unified name -> legacy remul dataset name
_DYN_NAME_MAP = {"nbody": "nbody", "nbody_egnn": "nbody_egnn",
                 "md17_dyn": "md17", "mocap": "motion_capture"}


def build_datasets(cfg):
    """Build dataset splits from the unified config."""
    if cfg.train.task == "dynamics":
        from remul.datasets import build_datasets as _build_dyn
        legacy = to_remul_config(cfg)
        legacy.data.name = _DYN_NAME_MAP[cfg.data.name]
        return _build_dyn(legacy.data)

    from load_dataset import load_dataset
    legacy = to_graph_config(cfg)
    return load_dataset(legacy)
