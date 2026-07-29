"""MD17 molecular-dynamics dataset (Section 6.3 / Appendix C.4).

We load a single-molecule MD17 trajectory via ``torch_geometric.datasets.MD17``
(downloaded from quantum-machine.org), then form (initial frame -> future frame)
prediction pairs separated by ``delta_t`` steps (paper uses ΔT=5000). The model
input is the initial position + atom type (one-hot); the target is the future
position. Velocity is estimated as the displacement to the previous frame.

Split follows the paper: 500 train / 2000 val / 2000 test pairs.
"""
from __future__ import annotations

import torch

from .common import DynamicsDataset

# paper name -> torch_geometric MD17 name
_MOLECULE_MAP = {
    "aspirin": "aspirin",
    "benzene": "benzene",
    "ethanol": "ethanol",
    "malonaldehyde": "malonaldehyde",
    "naphthalene": "naphthalene",
    "salicylic": "salicylic acid",
    "salicylic acid": "salicylic acid",
    "toluene": "toluene",
    "uracil": "uracil",
}

# common atomic numbers in MD17 molecules -> contiguous index for one-hot
_ATOM_TYPES = [1, 6, 7, 8]  # H, C, N, O


def _one_hot_atoms(z: torch.Tensor) -> torch.Tensor:
    table = {a: i for i, a in enumerate(_ATOM_TYPES)}
    idx = torch.tensor([table.get(int(a), len(_ATOM_TYPES)) for a in z])
    oh = torch.zeros(len(z), len(_ATOM_TYPES) + 1)
    oh[torch.arange(len(z)), idx] = 1.0
    return oh


def build_md17_datasets(cfg):
    from torch_geometric.datasets import MD17

    name = _MOLECULE_MAP.get(cfg.molecule.lower())
    if name is None:
        raise ValueError(
            f"Unknown MD17 molecule '{cfg.molecule}'. Options: {sorted(_MOLECULE_MAP)}"
        )
    ds = MD17(root=f"{cfg.root}/md17/{name.replace(' ', '_')}", name=name)

    n_frames = len(ds)
    dt = cfg.delta_t
    z = ds[0].z
    n_atoms = z.shape[0]
    h_static = _one_hot_atoms(z)  # (N, F) invariant atom types

    n_total = cfg.md17_n_train + cfg.md17_n_val + cfg.md17_n_test
    max_start = n_frames - dt - 1
    if max_start <= 0:
        raise ValueError(
            f"Trajectory too short ({n_frames} frames) for delta_t={dt}."
        )

    g = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(max_start, generator=g)[:n_total] + 1  # start >=1 for velocity
    if perm.shape[0] < n_total:
        # fall back to sampling with replacement if the trajectory is short
        perm = torch.randint(1, max_start + 1, (n_total,), generator=g)

    def gather(starts):
        pos0 = torch.stack([ds[int(i)].pos for i in starts])         # (S, N, 3)
        posm1 = torch.stack([ds[int(i) - 1].pos for i in starts])
        target = torch.stack([ds[int(i) + dt].pos for i in starts])
        vel = pos0 - posm1
        com = pos0.mean(dim=1, keepdim=True)
        pos0 = pos0 - com
        target = target - com
        h = h_static.unsqueeze(0).expand(len(starts), -1, -1).clone()
        return DynamicsDataset(pos0, vel, h, target)

    tr = perm[: cfg.md17_n_train]
    va = perm[cfg.md17_n_train: cfg.md17_n_train + cfg.md17_n_val]
    te = perm[cfg.md17_n_train + cfg.md17_n_val:]

    datasets = {"train": gather(tr), "val": gather(va), "test": gather(te)}
    datasets["meta"] = {
        "num_node_features": h_static.shape[-1],
        "num_nodes": n_atoms,
    }
    return datasets
