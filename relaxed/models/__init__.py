"""Unified model registry over both zoos.

* ``dyn.*``   — the dense dynamics models (REMUL paper zoo: transformer, mlp,
  gnn + equivariant baselines), imported from the legacy ``remul.models``.
* ``graph.*`` — the PyG graph models (BaseGNN's 13 types), imported from the
  legacy ``equivariant_gnn``.

``build_model`` dispatches on the task and adapts config defaults so each zoo
keeps its legacy hyperparameter defaults unless explicitly overridden.
"""
from __future__ import annotations

from ..adapt import to_graph_config, to_remul_config

DYN_MODELS = {"transformer", "mlp", "gnn", "mpnn", "egnn", "se3_transformer",
              "tfn", "gatr", "egno", "hegnn", "gmn", "emlp", "rpp", "per"}
GRAPH_MODELS = {"raw_mlp", "transformer", "gcn", "gin", "graphsage", "schnet",
                "dimenet", "egnn", "painn", "vector_neuron", "se3_transformer",
                "nequip", "clofnet"}

# Models without architectural equivariance constraints (the REMUL subjects).
_UNCONSTRAINED_DYN = {"transformer", "mlp", "gnn"}
_UNCONSTRAINED_GRAPH = {"raw_mlp", "transformer", "gcn", "gin", "graphsage"}

# Placeholder architectures that do NOT implement their implied geometry.
_PLACEHOLDER_GRAPH = {"se3_transformer", "nequip", "dimenet"}


def build_model(cfg, meta: dict | None = None):
    """Build a model from the unified config + dataset metadata.

    ``meta`` (with ``num_node_features``/``num_nodes``) is required only for
    the dynamics zoo.
    """
    if cfg.train.task == "dynamics":
        if meta is None:
            raise ValueError("dynamics models require dataset metadata")
        from remul.models import build_model as _build_dyn
        legacy = to_remul_config(cfg)
        return _build_dyn(legacy.model, meta["num_node_features"], meta["num_nodes"])

    from equivariant_gnn import BaseGNN
    legacy = to_graph_config(cfg)
    m = legacy.model
    return BaseGNN(
        in_channels=m.in_channels,
        hidden_channels=m.hidden_channels,
        out_channels=m.out_channels,
        num_layers=m.num_layers,
        dropout=m.dropout,
        model_type=m.model_type,
        spatial_dim=m.spatial_dim,
        num_heads=m.num_heads,
        num_gaussians=m.num_gaussians,
        num_spherical=m.num_spherical,
        cutoff=m.cutoff,
        update_coords=m.update_coords,
        max_ell=m.max_ell,
        num_degrees=m.num_degrees,
        use_pos=m.use_pos,
    )


def is_unconstrained(task: str, name: str) -> bool:
    return name in (_UNCONSTRAINED_DYN if task == "dynamics" else _UNCONSTRAINED_GRAPH)


def is_placeholder(task: str, name: str) -> bool:
    return task == "graph" and name in _PLACEHOLDER_GRAPH
