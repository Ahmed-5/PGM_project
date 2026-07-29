"""Unified equivariance objectives — one import surface, both formulations.

1. **REMUL formulation** (paper Eq. 6 / App. C.1), ground-truth anchored:
   ``L_equi = E_g ℓ(f(φ(g)x), ρ(g)y)`` — the training term for the dynamics
   task, plus the fixed GradNorm and the label-free functional equivariance
   errors E (Eq. 8) / E′ (Eq. 9) used for evaluation (incl. per-axis groups).

2. **Layer-wise formulation** (functional): ``‖f_l(g·x) − g·f_l(x)‖²`` over
   intermediate layer outputs for the graph task, with per-layer weights from
   ``relaxed.schedulers.DepthScheduler``. Re-exported from the battle-tested
   v3 implementation (single source of truth — do not copy).
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn

from .geometry import apply_rotation, sample_rotations

# Single source of truth for the layer-wise functional loss (8 groups,
# GPU-vectorized, [N,H,3] vector support). Imported, not duplicated.
from equivariance_loss import EquivarianceLoss as LayerEquivarianceLoss  # noqa: F401


# ---------------------------------------------------------------------------
# REMUL formulation (ground-truth anchored)
# ---------------------------------------------------------------------------


def _metric(pred: torch.Tensor, target: torch.Tensor, kind: str) -> torch.Tensor:
    """Per-sample reduction to a scalar loss. ``pred/target``: (B, N, 3)."""
    diff = pred - target
    if kind == "l1":
        return diff.abs().mean()
    if kind == "l2":
        return (diff ** 2).mean()
    if kind == "l2norm":
        return diff.reshape(diff.shape[0], -1).norm(dim=-1).mean()
    raise ValueError(f"Unknown metric kind: {kind}")


def rotate_batch(batch: dict, rot: torch.Tensor) -> dict:
    """Shallow copy of ``batch`` with geometric channels rotated by ``rot`` (B,3,3).

    Rotates ``pos``, ``vel`` (inputs, phi(g)) and ``target`` (output, rho(g)).
    Scalar features ``h`` and ``mass`` are invariant.
    """
    out = dict(batch)
    if batch.get("pos") is not None:
        out["pos"] = apply_rotation(batch["pos"], rot)
    if batch.get("vel") is not None:
        out["vel"] = apply_rotation(batch["vel"], rot)
    if batch.get("target") is not None:
        out["target"] = apply_rotation(batch["target"], rot)
    return out


class RemulLoss(nn.Module):
    """REMUL objective: task term + ground-truth-anchored equivariance term.

    Args:
        group: symmetry group for the equivariance term (so3 default;
            so2_x/so2_y/so2_z for per-axis analysis).
        metric: l1 or l2.
        num_group_samples: group elements drawn per step (paper default 1).
    """

    def __init__(self, group: str = "so3", metric: str = "l2", num_group_samples: int = 1):
        super().__init__()
        self.group = group
        self.metric = metric
        self.num_group_samples = num_group_samples

    def sample_rotations(self, batch_size: int, device, dtype) -> torch.Tensor:
        return sample_rotations(self.group, batch_size, device, dtype)

    def objective_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return _metric(pred, target, self.metric)

    def equivariance_loss(
        self, model_forward: Callable[[dict], torch.Tensor], batch: dict
    ) -> torch.Tensor:
        """Ground-truth-anchored equivariance loss ℓ(f(φ(g)x), ρ(g)y)."""
        pos = batch["pos"]
        b = pos.shape[0]
        device, dtype = pos.device, pos.dtype
        total = pos.new_zeros(())
        for _ in range(self.num_group_samples):
            rot = self.sample_rotations(b, device, dtype)
            rotated = rotate_batch(batch, rot)
            pred_rot = model_forward(rotated)
            total = total + _metric(pred_rot, rotated["target"], self.metric)
        return total / self.num_group_samples


class GradNorm(nn.Module):
    """GradNorm adaptive loss weighting (Chen et al., 2018) — the fixed version.

    Task weights are updated ONLY via their own gradient (no leakage into
    model parameters), with plain SGD per Algorithm 1.
    """

    def __init__(self, num_tasks: int = 2, alpha: float = 1.5, lr: float = 0.025,
                 init_weights: Optional[list[float]] = None):
        super().__init__()
        init = init_weights if init_weights is not None else [1.0] * num_tasks
        self.weights = nn.Parameter(torch.tensor(init, dtype=torch.float32))
        self.alpha = alpha
        self.opt = torch.optim.SGD([self.weights], lr=lr)
        self.initial_losses: torch.Tensor | None = None

    def weighted_sum(self, losses: list[torch.Tensor]) -> torch.Tensor:
        return sum(w * l for w, l in zip(self.weights, losses))

    @torch.no_grad()
    def _record_initial(self, losses: list[torch.Tensor]):
        if self.initial_losses is None:
            self.initial_losses = torch.stack([l.detach() for l in losses]).clamp_min(1e-8)

    def update(self, losses: list[torch.Tensor], shared_parameter: torch.Tensor) -> None:
        """One GradNorm step; call after the main backward pass."""
        self._record_initial(losses)
        norms = []
        for i, l in enumerate(losses):
            g = torch.autograd.grad(self.weights[i] * l, shared_parameter,
                                    retain_graph=True, create_graph=True)[0]
            norms.append(g.norm())
        norms = torch.stack(norms)
        mean_norm = norms.mean().detach()
        loss_ratios = torch.stack([l.detach() for l in losses]).clamp_min(1e-8) / self.initial_losses
        inverse_rates = loss_ratios / loss_ratios.mean()
        targets = (mean_norm * inverse_rates ** self.alpha).detach()
        grad_loss = (norms - targets).abs().sum()
        # Update ONLY the task weights (see legacy bug: full backward() leaked
        # second-order gradients into every model parameter).
        self.opt.zero_grad()
        self.weights.grad = torch.autograd.grad(grad_loss, self.weights)[0]
        self.opt.step()
        with torch.no_grad():
            self.weights.clamp_(min=1e-3)
            self.weights.mul_(len(losses) / self.weights.sum())


@torch.no_grad()
def equivariance_error(
    model_forward: Callable[[dict], torch.Tensor],
    batch: dict,
    num_samples: int = 20,
    group: str = "so3",
    metric_variant: str = "E_prime",
) -> torch.Tensor:
    """Functional equivariance error (paper Eq. 8 / Eq. 9), label-free.

    ``metric_variant='E_prime'``: mean over samples of ‖f(φ(g)x) − ρ(g)f(x)‖.
    ``'E'``: norm of the difference of the two group-averaged quantities.
    ``group`` may be so3 or so2_x/so2_y/so2_z for per-axis analysis.
    """
    pos = batch["pos"]
    b = pos.shape[0]
    f_x = model_forward(batch)  # (B, N, 3)

    rho_f, f_phi = [], []
    for _ in range(num_samples):
        rot = sample_rotations(group, b, pos.device, pos.dtype)
        rho_f.append(apply_rotation(f_x, rot))
        rotated = rotate_batch({k: v for k, v in batch.items() if k != "target"}, rot)
        f_phi.append(model_forward(rotated))

    rho_f = torch.stack(rho_f)
    f_phi = torch.stack(f_phi)

    if metric_variant == "E_prime":
        diff = (f_phi - rho_f).reshape(num_samples, b, -1).norm(dim=-1)
        return diff.mean()
    if metric_variant == "E":
        diff = (rho_f.mean(0) - f_phi.mean(0)).reshape(b, -1).norm(dim=-1)
        return diff.mean()
    raise ValueError(f"Unknown metric_variant: {metric_variant}")


@torch.no_grad()
def per_axis_equivariance_errors(
    model_forward: Callable[[dict], torch.Tensor],
    batch: dict,
    num_samples: int = 20,
) -> dict[str, float]:
    """Per-axis E′ for rotations about x/y/z (paper Appendix D.5 / Table 9)."""
    out = {}
    for axis in ("x", "y", "z"):
        out[f"E_prime_so2_{axis}"] = float(
            equivariance_error(model_forward, batch, num_samples,
                               group=f"so2_{axis}", metric_variant="E_prime"))
    return out
