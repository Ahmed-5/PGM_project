"""REMUL training objective, GradNorm penalty adaptation, and equivariance metrics.

Paper reference: Elhag et al., "Relaxed Equivariance via Multitask Learning"
(arXiv:2410.17878).

Two things are deliberately kept separate here:

1. The **training** equivariance loss (Eq. 6 / Appendix C.1) is *ground-truth
   anchored*:  L_equi = (1/n) Σ E_{g~G} ℓ( f(φ(g) x_i), ρ(g) y_i ).
   It compares the prediction on a transformed input against the transformed
   ground truth, NOT against the transformed prediction. This is what REMUL
   actually optimizes.

2. The **evaluation** equivariance error (Eq. 8 / Eq. 9) is *functional*:
   it compares f(φ(g)x) against ρ(g) f(x) and does not use labels. These are the
   E and E' metrics reported in the paper.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from .geometry import apply_rotation, random_rotation_matrix


def _metric(pred: torch.Tensor, target: torch.Tensor, kind: str) -> torch.Tensor:
    """Per-sample reduction to a scalar loss. ``pred/target``: (B, N, 3)."""
    diff = pred - target
    if kind == "l1":
        return diff.abs().mean()
    if kind == "l2":
        return (diff ** 2).mean()
    if kind == "l2norm":  # mean over samples of the L2 norm (matches metric E')
        return diff.reshape(diff.shape[0], -1).norm(dim=-1).mean()
    raise ValueError(f"Unknown metric kind: {kind}")


def rotate_batch(batch: dict, rot: torch.Tensor) -> dict:
    """Return a shallow copy of ``batch`` with geometric channels rotated.

    ``rot`` is ``(B, 3, 3)``. Rotates ``pos``, ``vel`` (inputs, phi(g)) and
    ``target`` (output, rho(g)). Scalar features ``h`` and ``mass`` are invariant.
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
    """Computes the REMUL objective and equivariance losses.

    Args:
        group: symmetry group for the equivariance term. Only ``so3`` is used in
            the paper; ``so2_z`` (rotations about the vertical axis) is provided
            for the Motion-Capture axis analysis.
        metric: ``l1`` or ``l2`` for the equivariance/objective discrepancy.
        num_group_samples: group elements drawn per sample per step (default 1,
            as in the paper; larger values average the equivariance term).
    """

    def __init__(self, group: str = "so3", metric: str = "l2", num_group_samples: int = 1):
        super().__init__()
        self.group = group
        self.metric = metric
        self.num_group_samples = num_group_samples

    def sample_rotations(self, batch_size: int, device, dtype) -> torch.Tensor:
        if self.group in ("so3", "e3", "se3"):
            return random_rotation_matrix(batch_size, device=device, dtype=dtype)
        if self.group == "so2_z":
            # Rotation about the z-axis only.
            angle = torch.rand(batch_size, device=device, dtype=dtype) * 2 * torch.pi
            cos, sin = torch.cos(angle), torch.sin(angle)
            rot = torch.zeros(batch_size, 3, 3, device=device, dtype=dtype)
            rot[:, 0, 0] = cos
            rot[:, 0, 1] = -sin
            rot[:, 1, 0] = sin
            rot[:, 1, 1] = cos
            rot[:, 2, 2] = 1.0
            return rot
        raise ValueError(f"Unsupported group: {self.group}")

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
    """GradNorm adaptive loss weighting (Chen et al., 2018), as used by REMUL's
    "gradual penalty".

    Maintains task weights ``w = [w_obj, w_equi]`` (kept positive), and updates
    them so that each task trains at a similar relative rate. ``shared_parameter``
    should be the weight tensor of the last shared layer (Algorithm 1 uses the
    last layer's weights ``W``).
    """

    def __init__(self, num_tasks: int = 2, alpha: float = 1.5, lr: float = 0.025,
                 init_weights: list[float] | None = None):
        super().__init__()
        init = init_weights if init_weights is not None else [1.0] * num_tasks
        self.weights = nn.Parameter(torch.tensor(init, dtype=torch.float32))
        self.alpha = alpha  # GradNorm's restoring-force exponent (gamma in the paper).
        # Algorithm 1 updates the task weights with plain gradient descent.
        self.opt = torch.optim.SGD([self.weights], lr=lr)
        self.initial_losses: torch.Tensor | None = None

    def weighted_sum(self, losses: list[torch.Tensor]) -> torch.Tensor:
        return sum(w * l for w, l in zip(self.weights, losses))

    @torch.no_grad()
    def _record_initial(self, losses: list[torch.Tensor]):
        if self.initial_losses is None:
            self.initial_losses = torch.stack([l.detach() for l in losses]).clamp_min(1e-8)

    def update(self, losses: list[torch.Tensor], shared_parameter: torch.Tensor) -> None:
        """One GradNorm step. Call after the main backward pass but before the
        model optimizer step (uses gradients w.r.t. ``shared_parameter``)."""
        self._record_initial(losses)
        norms = []
        for i, l in enumerate(losses):
            g = torch.autograd.grad(self.weights[i] * l, shared_parameter, retain_graph=True, create_graph=True)[0]
            norms.append(g.norm())
        norms = torch.stack(norms)
        mean_norm = norms.mean().detach()
        loss_ratios = torch.stack([l.detach() for l in losses]).clamp_min(1e-8) / self.initial_losses
        inverse_rates = loss_ratios / loss_ratios.mean()
        targets = (mean_norm * inverse_rates ** self.alpha).detach()
        grad_loss = (norms - targets).abs().sum()
        # Update ONLY the task weights. Taking the gradient of the meta-loss
        # w.r.t. ``self.weights`` (instead of calling ``grad_loss.backward()``)
        # is essential: ``norms`` was built with ``create_graph=True``, so a
        # full backward() would leak second-order GradNorm gradients into every
        # model parameter's .grad, contaminating the main optimizer step.
        self.opt.zero_grad()
        self.weights.grad = torch.autograd.grad(grad_loss, self.weights)[0]
        self.opt.step()
        with torch.no_grad():
            self.weights.clamp_(min=1e-3)
            # Renormalize weights to sum to num_tasks (standard GradNorm bookkeeping).
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

    ``metric_variant='E_prime'`` returns E' (Eq. 9): mean over samples of
    ||f(φ(g)x) - ρ(g)f(x)||. ``'E'`` returns E (Eq. 8): norm of the difference of
    the two group-averaged quantities.
    """
    loss_helper = RemulLoss(group=group)
    pos = batch["pos"]
    b = pos.shape[0]
    f_x = model_forward(batch)  # (B, N, 3)

    rho_f = []  # rho(g) f(x)
    f_phi = []  # f(phi(g) x)
    for _ in range(num_samples):
        rot = loss_helper.sample_rotations(b, pos.device, pos.dtype)
        rho_f.append(apply_rotation(f_x, rot))
        rotated = rotate_batch({k: v for k, v in batch.items() if k != "target"}, rot)
        f_phi.append(model_forward(rotated))

    rho_f = torch.stack(rho_f)  # (M, B, N, 3)
    f_phi = torch.stack(f_phi)

    if metric_variant == "E_prime":
        diff = (f_phi - rho_f).reshape(num_samples, b, -1).norm(dim=-1)  # (M, B)
        return diff.mean()
    if metric_variant == "E":
        diff = (rho_f.mean(0) - f_phi.mean(0)).reshape(b, -1).norm(dim=-1)  # (B,)
        return diff.mean()
    raise ValueError(f"Unknown metric_variant: {metric_variant}")
