"""Unified regression metrics (task) — single implementation."""
from __future__ import annotations

import torch


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.nn.functional.mse_loss(pred.float(), target.float()))


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float((pred.float() - target.float()).abs().mean())


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.sqrt(torch.nn.functional.mse_loss(pred.float(), target.float())))


def r2_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred, target = pred.float(), target.float()
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def regression_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    return {"MAE": mae(pred, target), "RMSE": rmse(pred, target),
            "R2": r2_score(pred, target), "MSE": mse(pred, target)}
