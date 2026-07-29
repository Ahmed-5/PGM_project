"""Depth-adaptive per-layer equivariance weights and LR scheduler factories.

Absorbed from the legacy ``schedulers.py`` with one fix: the learnable
schedule is initialized through the *inverse* softplus so the effective
initial weights equal the intended ``alpha_0 * exp(-beta * l)`` (previously
the raw values were passed through softplus, shifting the init).
"""
from __future__ import annotations

import math
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _inverse_softplus(y: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(y.clamp_min(1e-8)))


class DepthScheduler(nn.Module):
    """Layer-wise equivariance weight scheduling.

    Strategies: constant, exponential, linear, inverse, u_shaped, learnable,
    linear_inc, exp_inc. Aliases: linear_decay->linear, exp_decay->exponential.
    The global strength alpha_0 is applied once, inside these weights.
    """

    def __init__(self, num_layers: int, schedule_type: str = "exponential",
                 alpha_0: float = 1.0, beta: float = 0.1, gamma: float = 0.1):
        super().__init__()
        self.num_layers = num_layers

        aliases = {"linear_decay": "linear", "exp_decay": "exponential"}
        self.schedule_type = aliases.get(schedule_type.lower(), schedule_type.lower())
        self.alpha_0 = alpha_0
        self.beta = beta
        self.gamma = gamma

        valid_types = {"constant", "exponential", "linear", "inverse", "u_shaped",
                       "learnable", "linear_inc", "exp_inc"}
        if self.schedule_type not in valid_types:
            raise ValueError(f"schedule_type must be one of {valid_types}, got {schedule_type}")

        if self.schedule_type == "learnable":
            target = alpha_0 * np.exp(-beta * np.arange(num_layers))
            # Inverse-softplus so effective initial weights equal `target`.
            self.alpha = nn.Parameter(
                _inverse_softplus(torch.tensor(target, dtype=torch.float32)))
        else:
            self._compute_schedule_vectorized()

    def _compute_schedule_vectorized(self):
        layer_indices = torch.arange(self.num_layers, dtype=torch.float32)

        if self.schedule_type == "constant":
            alpha = torch.full((self.num_layers,), self.alpha_0, dtype=torch.float32)
        elif self.schedule_type == "exponential":
            alpha = self.alpha_0 * torch.exp(-self.beta * layer_indices)
        elif self.schedule_type == "exp_inc":
            reverse_indices = (self.num_layers - 1) - layer_indices
            alpha = self.alpha_0 * torch.exp(-self.beta * reverse_indices)
        elif self.schedule_type == "linear":
            alpha = torch.clamp(self.alpha_0 - self.gamma * layer_indices, min=0.0)
        elif self.schedule_type == "linear_inc":
            if self.num_layers > 1:
                alpha = (layer_indices / (self.num_layers - 1)) * self.alpha_0
            else:
                alpha = torch.tensor([self.alpha_0])
        elif self.schedule_type == "inverse":
            alpha = self.alpha_0 / (1.0 + self.beta * layer_indices)
        elif self.schedule_type == "u_shaped":
            mid = self.num_layers / 2.0
            distance_from_mid = torch.abs(layer_indices - (mid - 0.5)) / max(mid, 1.0)
            alpha = self.alpha_0 * (0.5 + 0.5 * distance_from_mid)

        self.register_buffer("alpha", alpha, persistent=True)

    def get_alpha(self, layer_idx: Union[int, torch.Tensor]) -> torch.Tensor:
        if self.schedule_type == "learnable":
            return F.softplus(self.alpha[layer_idx])
        return self.alpha[layer_idx]

    def get_all_alphas(self) -> torch.Tensor:
        if self.schedule_type == "learnable":
            return F.softplus(self.alpha)
        return self.alpha.clone()

    def forward(self, layer_idx: int) -> torch.Tensor:
        return self.get_alpha(layer_idx)

    def extra_repr(self) -> str:
        return (f"num_layers={self.num_layers}, schedule_type='{self.schedule_type}', "
                f"alpha_0={self.alpha_0}, beta={self.beta}, gamma={self.gamma}")


def get_lr_scheduler(optimizer, config) -> torch.optim.lr_scheduler._LRScheduler | None:
    """Factory for LR schedulers from the unified ScheduleConfig."""
    name = config.lr_schedule
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma)
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=getattr(config, "t_max", 100))
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=config.plateau_patience,
            factor=config.plateau_factor, mode=config.plateau_mode)
    if name == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=config.exponential_decay_rate)
    return None
