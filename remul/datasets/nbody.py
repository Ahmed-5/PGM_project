"""Synthetic N-body dynamical systems.

Two variants, matching the two used in the paper:

* ``nbody`` (GATr-style gravity, Section 6.1 / Appendix C.2): 4 bodies, a heavy
  central mass (U[1,10]) and lighter satellites (mass U[0.01,0.1]) placed at a
  radius U[0.1,1.0]. We integrate Newtonian gravity for ``num_steps`` Euler steps
  and predict the final positions. In-distribution samples are rotated within
  ±10°; the OOD split uses rotations in [−180,−90]∪[90,180]°.

* ``nbody_egnn`` (Satorras et al. charged particles, Appendix D.4): 5 particles
  carrying charges ±1 interacting through Coulomb forces; predict positions after
  ``num_steps`` steps.

Both are fully synthetic (no download required). Center of mass is subtracted so
the task is translation invariant, and rotations act on the centered coordinates.
"""
from __future__ import annotations

import math

import torch

from .common import DynamicsDataset
from ..geometry import random_rotation_matrix, random_rotation_in_range, apply_rotation


def _simulate_gravity(pos, vel, mass, num_steps, dt, softening=1e-1, field=None):
    """Leapfrog-ish Euler integration of Newtonian gravity.

    pos/vel: (S, N, 3); mass: (S, N, 1). Returns final positions (S, N, 3).
    ``field``: optional (3,) uniform external acceleration. When non-zero it
    breaks the SO(3) symmetry of the dynamics down to SO(2) about the field
    axis (used for the controlled symmetry-breaking sweep).
    """
    pos = pos.clone()
    vel = vel.clone()
    for _ in range(num_steps):
        # pairwise displacement r_j - r_i : (S, N, N, 3)
        disp = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist2 = (disp ** 2).sum(-1) + softening ** 2  # (S, N, N)
        inv = dist2.pow(-1.5).unsqueeze(-1)           # (S, N, N, 1)
        mj = mass.unsqueeze(1)                          # (S, 1, N, 1)
        acc = (disp * inv * mj).sum(dim=2)             # (S, N, 3)
        if field is not None:
            acc = acc + field                          # broadcast (3,) -> (S, N, 3)
        vel = vel + dt * acc
        pos = pos + dt * vel
    return pos


def _simulate_coulomb(pos, vel, charge, num_steps, dt, softening=1e-1):
    pos = pos.clone()
    vel = vel.clone()
    q = charge  # (S, N, 1)
    for _ in range(num_steps):
        disp = pos.unsqueeze(2) - pos.unsqueeze(1)     # r_i - r_j : (S, N, N, 3)
        dist2 = (disp ** 2).sum(-1) + softening ** 2
        inv = dist2.pow(-1.5).unsqueeze(-1)
        qq = (q.unsqueeze(2) * q.unsqueeze(1))          # (S, N, N, 1)
        force = (disp * inv * qq).sum(dim=2)            # like charges repel
        vel = vel + dt * force
        pos = pos + dt * vel
    return pos


# Small per-sample angular jitter (radians ~ 0.1 => ~6 deg) around the canonical
# template. Kept well below the in-distribution split cone (10 deg) so the
# in/OOD rotation split stays a genuine distribution shift (see RC1 below).
_DIR_JITTER = 0.1


def _canonical_template(n_sat):
    """Fixed (canonical-frame) unit directions for the satellites via a
    deterministic Fibonacci/golden-spiral spread on the sphere.

    Shared across all samples so the *base* configuration has a well-defined
    orientation. Returns (n_sat, 3) unit vectors.
    """
    i = torch.arange(n_sat, dtype=torch.float32)
    golden = math.pi * (3.0 - math.sqrt(5.0))
    z = 1.0 - 2.0 * (i + 0.5) / max(n_sat, 1)
    r = torch.sqrt(torch.clamp(1.0 - z * z, min=1e-9))
    theta = golden * i
    return torch.stack([r * torch.cos(theta), r * torch.sin(theta), z], dim=-1)


def _make_gravity(n_samples, n_bodies, num_steps, dt, generator, field_strength=0.0, field_axis=2):
    n_sat = n_bodies - 1
    center_mass = torch.rand(n_samples, 1, generator=generator) * 9 + 1.0  # U[1,10]
    sat_mass = torch.rand(n_samples, n_sat, generator=generator) * 0.09 + 0.01
    mass = torch.cat([center_mass, sat_mass], dim=1).unsqueeze(-1)  # (S, N, 1)

    pos = torch.zeros(n_samples, n_bodies, 3)
    # satellites on a sphere at radius U[0.1,1.0]
    radius = torch.rand(n_samples, n_sat, generator=generator) * 0.9 + 0.1
    # RC1 FIX: build directions in a CANONICAL frame (fixed template + small
    # jitter) instead of isotropic randn. Previously isotropic directions made
    # the base configuration span all of SO(3), so the in-dist (+/-10 deg) and
    # OOD (90-180 deg) split rotations were distributionally identical and
    # equivariance was "free". Now orientation is concentrated near the
    # template, so the split rotation is a real distribution shift.
    template = _canonical_template(n_sat).unsqueeze(0)                     # (1, n_sat, 3)
    jitter = torch.randn(n_samples, n_sat, 3, generator=generator) * _DIR_JITTER
    dirs = template + jitter
    dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-9)
    pos[:, 1:, :] = dirs * radius.unsqueeze(-1)

    # Canonical tangential initial velocity (fixed orbital sense): t = dir x up.
    # The reference "up" is the field axis, so the whole input distribution is
    # symmetric about the same axis the field breaks — otherwise the velocity
    # structure would bake in a hidden z-preference and make an off-z field task
    # not cleanly SO(2) about its own axis (a confound in the loss-group test).
    vel = torch.zeros(n_samples, n_bodies, 3)
    up_vec = torch.zeros(3); up_vec[int(field_axis)] = 1.0
    up = up_vec.view(1, 1, 3).expand_as(dirs)
    tangent = torch.cross(dirs, up, dim=-1)
    # where dir is ~parallel to up, use a different reference axis
    degenerate = tangent.norm(dim=-1, keepdim=True) < 1e-4
    alt_vec = torch.zeros(3); alt_vec[(int(field_axis) + 1) % 3] = 1.0
    alt = torch.cross(dirs, alt_vec.view(1, 1, 3).expand_as(dirs), dim=-1)
    tangent = torch.where(degenerate, alt, tangent)
    tangent = tangent / (tangent.norm(dim=-1, keepdim=True) + 1e-9)
    speed = (center_mass.unsqueeze(1) / (radius.unsqueeze(-1) + 1e-3)).sqrt() * 0.3
    vel[:, 1:, :] = tangent * speed

    field = None
    if field_strength:
        field = torch.zeros(3)
        field[int(field_axis)] = float(field_strength)  # along field_axis -> residual SO(2) about it
    target = _simulate_gravity(pos, vel, mass, num_steps, dt, field=field)
    h = mass  # scalar feature = mass
    return pos, vel, h, target


def _make_charged(n_samples, n_bodies, num_steps, dt, generator):
    pos = torch.randn(n_samples, n_bodies, 3, generator=generator) * 0.5
    vel = torch.randn(n_samples, n_bodies, 3, generator=generator) * 0.1
    charge = (torch.randint(0, 2, (n_samples, n_bodies, 1), generator=generator).float() * 2 - 1)
    target = _simulate_coulomb(pos, vel, charge, num_steps, dt)
    h = charge  # scalar feature = charge (±1)
    return pos, vel, h, target


def _apply_split_rotation(pos, vel, target, distribution, generator):
    """Rotate each sample: in-dist ±10°, OOD [−180,−90]∪[90,180]°."""
    s = pos.shape[0]
    if distribution == "in":
        rot = random_rotation_matrix(s, max_angle=math.radians(10), generator=generator)
    else:
        rot = random_rotation_in_range(
            s,
            [(-math.pi, -math.pi / 2), (math.pi / 2, math.pi)],
            generator=generator,
        )
    return apply_rotation(pos, rot), apply_rotation(vel, rot), apply_rotation(target, rot)


def _center(pos, target):
    com = pos.mean(dim=1, keepdim=True)
    return pos - com, target - com


def build_nbody_datasets(cfg):
    g = torch.Generator().manual_seed(cfg.seed)
    egnn_style = cfg.name == "nbody_egnn"
    n_bodies = 5 if egnn_style else cfg.n_bodies
    maker = _make_charged if egnn_style else _make_gravity
    fs = float(getattr(cfg, "field_strength", 0.0) or 0.0)
    fa = int(getattr(cfg, "field_axis", 2))
    # With a fixed +z field the task has a preferred axis (SO(2)_z, not SO(3)).
    # Applying the random rotation split would isotropize that axis and undo the
    # symmetry breaking, so we keep a fixed frame (and no OOD-rotation split) when
    # a field is present. field=0 reproduces the standard rotation-OOD benchmark.
    broken = (fs != 0.0)

    def build(n, distribution):
        if egnn_style:
            pos, vel, h, target = maker(n, n_bodies, cfg.num_steps, cfg.dt, g)
        else:
            pos, vel, h, target = maker(n, n_bodies, cfg.num_steps, cfg.dt, g, field_strength=fs, field_axis=fa)
        pos, target = _center(pos, target)
        if not egnn_style and not broken:
            pos, vel, target = _apply_split_rotation(pos, vel, target, distribution, g)
            pos, target = _center(pos, target)
        return DynamicsDataset(pos, vel, h, target)

    datasets = {
        "train": build(cfg.n_train, cfg.distribution),
        "val": build(cfg.n_val, cfg.distribution),
        "test": build(cfg.n_test, "in"),
    }
    if not egnn_style and not broken:
        datasets["ood"] = build(cfg.n_test, "ood")
    datasets["meta"] = {
        "num_node_features": datasets["train"].num_node_features,
        "num_nodes": n_bodies,
    }
    return datasets
