"""Group actions and geometric utilities for the REMUL dynamics experiments.

The paper (Elhag et al., 2024, arXiv:2410.17878) considers the rotation group
SO(3) for the equivariance objective and handles translation by subtracting the
center of mass. This module provides:

* uniform sampling of SO(3) rotation matrices,
* the input action ``phi(g)`` and output action ``rho(g)`` (both are simply a
  3D rotation applied to the geometric channels: positions and velocities on
  the input side, positions on the output side),
* center-of-mass subtraction (the translation handling used throughout).

Everything operates on tensors shaped ``(..., N, 3)`` so it works for both a
single graph and a batched ``(B, N, 3)`` layout.
"""
from __future__ import annotations

import torch


def random_rotation_matrix(
    batch_size: int = 1,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    max_angle: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample rotation matrices.

    Args:
        batch_size: number of independent rotations to draw.
        max_angle: if ``None`` draw a Haar-uniform rotation from SO(3). If a
            float (radians) is given, draw an axis-angle rotation whose angle is
            uniform in ``[-max_angle, max_angle]`` about a uniformly random axis.
            This is used to build the N-body in-distribution / OOD splits, where
            rotations are restricted to specific angular ranges.

    Returns:
        Tensor of shape ``(batch_size, 3, 3)``.
    """
    if max_angle is None:
        return _random_so3(batch_size, device=device, dtype=dtype, generator=generator)
    return _axis_angle_rotation(
        batch_size,
        low=-max_angle,
        high=max_angle,
        device=device,
        dtype=dtype,
        generator=generator,
    )


def random_rotation_in_range(
    batch_size: int,
    angle_ranges: list[tuple[float, float]],
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Rotation about a random axis, angle drawn uniformly from a union of ranges.

    ``angle_ranges`` is a list of ``(low, high)`` tuples in radians. Each sample
    picks one range uniformly, then an angle uniform within it. Used for the
    N-body OOD split (angles in ``[-180,-90]`` or ``[90,180]`` degrees).
    """
    ranges = torch.tensor(angle_ranges, device=device, dtype=dtype)
    idx = torch.randint(0, ranges.shape[0], (batch_size,), device=device, generator=generator)
    chosen = ranges[idx]
    u = torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
    angles = chosen[:, 0] + u * (chosen[:, 1] - chosen[:, 0])
    return _rotation_from_axis_angle(_random_unit_axis(batch_size, device, dtype, generator), angles)


def _random_so3(batch_size, *, device, dtype, generator=None) -> torch.Tensor:
    """Haar-uniform SO(3) via QR of a Gaussian matrix (sign-corrected)."""
    a = torch.randn(batch_size, 3, 3, device=device, dtype=dtype, generator=generator)
    q, r = torch.linalg.qr(a)
    # Make the decomposition unique / uniform: fix signs from diag(R).
    d = torch.diagonal(r, dim1=-2, dim2=-1)
    q = q * torch.sign(d).unsqueeze(-2)
    # Ensure determinant +1 (proper rotation, not reflection).
    det = torch.linalg.det(q)
    q[:, :, 0] = q[:, :, 0] * det.unsqueeze(-1)
    return q


def _random_unit_axis(batch_size, device, dtype, generator=None) -> torch.Tensor:
    axis = torch.randn(batch_size, 3, device=device, dtype=dtype, generator=generator)
    return axis / (axis.norm(dim=-1, keepdim=True) + 1e-12)


def _axis_angle_rotation(batch_size, *, low, high, device, dtype, generator=None) -> torch.Tensor:
    axis = _random_unit_axis(batch_size, device, dtype, generator)
    angles = torch.rand(batch_size, device=device, dtype=dtype, generator=generator) * (high - low) + low
    return _rotation_from_axis_angle(axis, angles)


def _rotation_from_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues' formula. ``axis``: (B,3) unit vectors, ``angle``: (B,)."""
    b = axis.shape[0]
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    zeros = torch.zeros_like(x)
    k = torch.stack(
        [zeros, -z, y, z, zeros, -x, -y, x, zeros], dim=-1
    ).reshape(b, 3, 3)
    eye = torch.eye(3, device=axis.device, dtype=axis.dtype).expand(b, 3, 3)
    sin = torch.sin(angle).view(b, 1, 1)
    cos = torch.cos(angle).view(b, 1, 1)
    return eye + sin * k + (1 - cos) * torch.bmm(k, k)


def apply_rotation(x: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    """Apply rotation ``rot`` to coordinates ``x``.

    Args:
        x: ``(N, 3)`` or ``(B, N, 3)``.
        rot: ``(3, 3)`` or ``(B, 3, 3)``.
    """
    if rot.dim() == 2:
        return x @ rot.transpose(-1, -2)
    if x.dim() == 2:
        # single graph, batched rotation is ambiguous; treat as (3,3)
        raise ValueError("Batched rotation requires batched coordinates (B, N, 3).")
    return torch.bmm(x, rot.transpose(-1, -2))


def subtract_center_of_mass(
    pos: torch.Tensor, mass: torch.Tensor | None = None, dim: int = -2
) -> tuple[torch.Tensor, torch.Tensor]:
    """Subtract the (mass-weighted) center of mass along the node dimension.

    Returns ``(centered_pos, com)`` so the shift can be re-applied to targets.
    """
    if mass is None:
        com = pos.mean(dim=dim, keepdim=True)
    else:
        w = mass / (mass.sum(dim=dim, keepdim=True) + 1e-12)
        com = (pos * w).sum(dim=dim, keepdim=True)
    return pos - com, com
