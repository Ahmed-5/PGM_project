"""Group actions and geometric utilities (single implementation).

Unifies the rotation helpers formerly in ``remul/geometry.py`` (Haar SO(3),
axis-angle, OOD range sampling, COM subtraction) with axis-aligned SO(2)
sampling for the per-axis equivariance analysis (paper Appendix D.5).

Everything operates on tensors shaped ``(..., N, 3)``.
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

    ``max_angle=None`` draws Haar-uniform SO(3); otherwise axis-angle with
    angle uniform in ``[-max_angle, max_angle]`` about a uniform random axis.
    Returns ``(batch_size, 3, 3)``.
    """
    if max_angle is None:
        return _random_so3(batch_size, device=device, dtype=dtype, generator=generator)
    return _axis_angle_rotation(
        batch_size, low=-max_angle, high=max_angle,
        device=device, dtype=dtype, generator=generator)


def random_rotation_in_range(
    batch_size: int,
    angle_ranges: list[tuple[float, float]],
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Rotation about a random axis, angle uniform over a union of ranges (OOD)."""
    ranges = torch.tensor(angle_ranges, device=device, dtype=dtype)
    idx = torch.randint(0, ranges.shape[0], (batch_size,), device=device, generator=generator)
    chosen = ranges[idx]
    u = torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
    angles = chosen[:, 0] + u * (chosen[:, 1] - chosen[:, 0])
    return _rotation_from_axis_angle(_random_unit_axis(batch_size, device, dtype, generator), angles)


def random_so2_batch(
    axis: str,
    batch_size: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Uniform SO(2) rotations about a coordinate axis ('x', 'y', or 'z').

    Used for the per-axis equivariance error analysis (paper Appendix D.5).
    """
    angle = torch.rand(batch_size, device=device, dtype=dtype, generator=generator) * 2 * torch.pi
    cos, sin = torch.cos(angle), torch.sin(angle)
    rot = torch.zeros(batch_size, 3, 3, device=device, dtype=dtype)
    if axis == "z":
        rot[:, 0, 0], rot[:, 0, 1] = cos, -sin
        rot[:, 1, 0], rot[:, 1, 1] = sin, cos
        rot[:, 2, 2] = 1.0
    elif axis == "y":
        rot[:, 0, 0], rot[:, 0, 2] = cos, sin
        rot[:, 2, 0], rot[:, 2, 2] = -sin, cos
        rot[:, 1, 1] = 1.0
    elif axis == "x":
        rot[:, 1, 1], rot[:, 1, 2] = cos, -sin
        rot[:, 2, 1], rot[:, 2, 2] = sin, cos
        rot[:, 0, 0] = 1.0
    else:
        raise ValueError(f"axis must be 'x', 'y' or 'z', got '{axis}'")
    return rot


def sample_rotations(group: str, batch_size: int, device, dtype,
                     generator: torch.Generator | None = None) -> torch.Tensor:
    """Unified rotation sampler for the supported dynamics groups."""
    if group in ("so3", "e3", "se3", "o3"):
        return random_rotation_matrix(batch_size, device=device, dtype=dtype, generator=generator)
    if group in ("so2_x", "so2_y", "so2_z"):
        return random_so2_batch(group[-1], batch_size, device=device, dtype=dtype,
                                generator=generator)
    raise ValueError(f"Unsupported group: {group}")


def _random_so3(batch_size, *, device, dtype, generator=None) -> torch.Tensor:
    """Haar-uniform SO(3) via QR of a Gaussian matrix (sign-corrected)."""
    a = torch.randn(batch_size, 3, 3, device=device, dtype=dtype, generator=generator)
    q, r = torch.linalg.qr(a)
    d = torch.diagonal(r, dim1=-2, dim2=-1)
    q = q * torch.sign(d).unsqueeze(-2)
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
    """Apply rotation ``rot`` to coordinates ``x`` — ``(N,3)`` or ``(B,N,3)``."""
    if rot.dim() == 2:
        return x @ rot.transpose(-1, -2)
    if x.dim() == 2:
        raise ValueError("Batched rotation requires batched coordinates (B, N, 3).")
    return torch.bmm(x, rot.transpose(-1, -2))


def subtract_center_of_mass(
    pos: torch.Tensor, mass: torch.Tensor | None = None, dim: int = -2
) -> tuple[torch.Tensor, torch.Tensor]:
    """Subtract the (mass-weighted) center of mass along the node dimension."""
    if mass is None:
        com = pos.mean(dim=dim, keepdim=True)
    else:
        w = mass / (mass.sum(dim=dim, keepdim=True) + 1e-12)
        com = (pos * w).sum(dim=dim, keepdim=True)
    return pos - com, com
