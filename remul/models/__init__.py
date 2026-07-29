"""Model registry for the REMUL dynamics experiments."""
from __future__ import annotations

from .unconstrained import Transformer, MLP, GNN
from .egnn import EGNN
from .equivariant import EGNO, HEGNN, GMN
from .se3 import SE3Transformer
from .gatr import GATr
from .mlp_baselines import EMLP, RPP, PER
from .mpnn import MPNN

__all__ = ["build_model", "MODEL_REGISTRY"]


def _se3(nf, nn_, cfg):
    return SE3Transformer(nf, nn_, cfg, attention=True)


def _tfn(nf, nn_, cfg):
    return SE3Transformer(nf, nn_, cfg, attention=False)


MODEL_REGISTRY = {
    # unconstrained (REMUL subjects)
    "transformer": Transformer,
    "mlp": MLP,
    "gnn": GNN,
    # equivariant baselines
    "egnn": EGNN,
    "se3_transformer": _se3,
    "tfn": _tfn,
    "gatr": GATr,
    "egno": EGNO,
    "hegnn": HEGNN,
    "gmn": GMN,
    # MLP-family (approximate / equivariant)
    "emlp": EMLP,
    "rpp": RPP,
    "per": PER,
    "mpnn": MPNN,
}


def build_model(cfg_model, num_node_features: int, num_nodes: int):
    name = cfg_model.name
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Options: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](num_node_features, num_nodes, cfg_model)


def is_unconstrained(name: str) -> bool:
    return name in ("transformer", "mlp", "gnn")
