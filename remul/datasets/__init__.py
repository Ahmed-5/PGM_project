"""Datasets for the REMUL dynamics experiments."""
from .common import DynamicsDataset, make_loader, collate_dynamics
from .nbody import build_nbody_datasets
from .md17 import build_md17_datasets
from .motion_capture import build_motion_capture_datasets

__all__ = [
    "DynamicsDataset",
    "make_loader",
    "collate_dynamics",
    "build_nbody_datasets",
    "build_md17_datasets",
    "build_motion_capture_datasets",
    "build_datasets",
]


def build_datasets(cfg):
    """Dispatch to the right dataset builder given a DataConfig.

    Returns a dict with keys ``train``/``val``/``test`` (and optionally ``ood``)
    mapping to :class:`DynamicsDataset` instances, plus a ``meta`` dict with
    ``num_node_features`` and ``num_nodes``.
    """
    name = cfg.name
    if name in ("nbody", "nbody_egnn"):
        return build_nbody_datasets(cfg)
    if name == "md17":
        return build_md17_datasets(cfg)
    if name == "motion_capture":
        return build_motion_capture_datasets(cfg)
    raise ValueError(f"Unknown dataset: {name}")
