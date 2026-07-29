"""Training loop for the REMUL dynamics experiments.

Supports the three training modes from the paper:

* ``standard`` : task loss only (no symmetry handling).
* ``da``       : data augmentation (task loss on randomly rotated input/target).
* ``remul``    : multitask objective  alpha*L_obj + beta*L_equi, with either a
                 ``constant`` penalty or a ``gradual`` penalty via GradNorm.

Evaluation reports task MSE (test, and OOD for N-body) plus the functional
equivariance errors E and E' (paper Eq. 8 / Eq. 9).
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn

from .config import ExperimentConfig
from .datasets import build_datasets, make_loader
from .models import build_model
from .experiment_log import (
    hparams_for_tensorboard,
    make_log_dir,
    save_run_artifacts,
)
from .losses import RemulLoss, GradNorm, equivariance_error, rotate_batch


def _make_logger(cfg, log_dir: str):
    """Return SummaryWriter or None."""
    if cfg.log.logger_type != "tensorboard":
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("Warning: tensorboard not installed. pip install tensorboard")
        return None
    return SummaryWriter(log_dir=log_dir)


def _get_last_shared_weight(model: nn.Module) -> torch.Tensor:
    last = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last = m.weight
    if last is None:
        last = next(model.parameters())
    return last


def _to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, loss_fn: RemulLoss, device, eval_group_samples=20,
             max_equiv_batches=None):
    """MSE over the full split; E/E' averaged over up to ``max_equiv_batches``
    batches (None = full split). Per-epoch evals use the cap to stay cheap;
    final test/OOD evals always use the full split."""
    model.eval()
    total_mse, n = 0.0, 0
    for batch in loader:
        batch = _to_device(batch, device)
        pred = model(batch)
        total_mse += torch.nn.functional.mse_loss(pred, batch["target"], reduction="sum").item()
        n += batch["target"].numel()
    mse = total_mse / max(n, 1)

    # Functional equivariance errors averaged over data (paper: (1/|D|) Σ_x …).
    e_sum, e_prime_sum, n_samples = 0.0, 0.0, 0
    fwd = lambda b: model(b)
    for i, batch in enumerate(loader):
        if max_equiv_batches is not None and i >= max_equiv_batches:
            break
        batch = _to_device(batch, device)
        b = batch["pos"].shape[0]
        e_sum += equivariance_error(fwd, batch, eval_group_samples, loss_fn.group, "E").item() * b
        e_prime_sum += equivariance_error(fwd, batch, eval_group_samples, loss_fn.group, "E_prime").item() * b
        n_samples += b
    denom = max(n_samples, 1)
    return {"mse": mse, "E": e_sum / denom, "E_prime": e_prime_sum / denom}


def train(cfg: ExperimentConfig, verbose: bool = True):
    torch.manual_seed(cfg.train.seed)
    device = cfg.train.device

    data = build_datasets(cfg.data)
    meta = data["meta"]
    train_loader = make_loader(data["train"], cfg.train.batch_size, shuffle=True)
    val_loader = make_loader(data["val"], cfg.train.batch_size)
    test_loader = make_loader(data["test"], cfg.train.batch_size)
    ood_loader = make_loader(data["ood"], cfg.train.batch_size) if "ood" in data else None

    model = build_model(cfg.model, meta["num_node_features"], meta["num_nodes"]).to(device)
    loss_fn = RemulLoss(cfg.train.group, cfg.train.metric, cfg.train.num_group_samples)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    gradnorm = None
    if cfg.train.mode == "remul" and cfg.train.penalty == "gradual":
        gradnorm = GradNorm(2, cfg.train.gradnorm_alpha, cfg.train.gradnorm_lr,
                            init_weights=[cfg.train.alpha, cfg.train.beta]).to(device)
        shared_w = _get_last_shared_weight(model)

    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"[remul] model={cfg.model.name} params={n_params:,} "
              f"dataset={cfg.data.name} mode={cfg.train.mode} penalty={cfg.train.penalty} "
              f"group={cfg.train.group} nodes={meta['num_nodes']} feat={meta['num_node_features']} "
              f"device={device}")

    log_dir = make_log_dir(cfg)
    save_run_artifacts(cfg, log_dir, step=0)
    writer = _make_logger(cfg, log_dir)
    if verbose:
        print(f"[remul] run log_dir={log_dir}")

    step = 0
    max_steps = cfg.train.max_steps
    # When max_steps is set, derive the epoch count from it (the paper's N-body
    # budget is 50k steps; with tiny epoch sizes a fixed epochs default would
    # silently stop far short of it).
    if max_steps is not None:
        steps_per_epoch = max(1, len(train_loader))
        total_epochs = -(-max_steps // steps_per_epoch)  # ceil
    else:
        total_epochs = cfg.train.epochs
    # Evaluate at most ~50 times over the run regardless of epoch count.
    eval_every = max(cfg.log.eval_every, total_epochs // 50)
    for epoch in range(total_epochs):
        model.train()
        for batch in train_loader:
            batch = _to_device(batch, device)
            opt.zero_grad()

            if cfg.train.mode == "standard":
                pred = model(batch)
                total = loss_fn.objective_loss(pred, batch["target"])
                obj = total.detach()
                equi = torch.tensor(0.0)
                total.backward()
                opt.step()

            elif cfg.train.mode == "da":
                rot = loss_fn.sample_rotations(batch["pos"].shape[0], device, batch["pos"].dtype)
                rb = rotate_batch(batch, rot)
                pred = model(rb)
                total = loss_fn.objective_loss(pred, rb["target"])
                obj = total.detach()
                equi = torch.tensor(0.0)
                total.backward()
                opt.step()

            else:  # remul
                pred = model(batch)
                obj = loss_fn.objective_loss(pred, batch["target"])
                equi = loss_fn.equivariance_loss(lambda b: model(b), batch)
                if gradnorm is not None:
                    weighted = gradnorm.weighted_sum([obj, equi])
                    weighted.backward(retain_graph=True)
                    gradnorm.update([obj, equi], shared_w)
                    opt.step()
                    total = weighted.detach()
                else:
                    total = cfg.train.alpha * obj + cfg.train.beta * equi
                    total.backward()
                    opt.step()
                obj, equi = obj.detach(), equi.detach()

            step += 1
            if verbose and step % cfg.log.log_every == 0:
                w = ("" if gradnorm is None
                     else f" w={gradnorm.weights.detach().cpu().numpy().round(3)}")
                total_v = float(total.detach()) if torch.is_tensor(total) else float(total)
                print(f"  step {step} | total {total_v:.4e} | obj {float(obj):.4e} | equi {float(equi):.4e}{w}")
                if writer:
                    writer.add_scalar("train/total_loss", total_v, step)
                    writer.add_scalar("train/obj_loss", float(obj), step)
                    writer.add_scalar("train/equi_loss", float(equi), step)
                    if gradnorm is not None:
                        for i, wv in enumerate(gradnorm.weights.detach().cpu().numpy()):
                            writer.add_scalar(f"gradnorm/weight_{i}", wv, step)
            if max_steps is not None and step >= max_steps:
                break
        if max_steps is not None and step >= max_steps:
            break

        if verbose and (epoch + 1) % eval_every == 0:
            m = evaluate(model, val_loader, loss_fn, device, cfg.train.eval_group_samples,
                         max_equiv_batches=cfg.train.eval_equiv_batches)
            print(f"[epoch {epoch+1}] val mse {m['mse']:.4e} | E {m['E']:.4e} | E' {m['E_prime']:.4e}")
            if writer:
                writer.add_scalar("val/mse", m["mse"], step)
                writer.add_scalar("val/E", m["E"], step)
                writer.add_scalar("val/E_prime", m["E_prime"], step)

    results = {"test": evaluate(model, test_loader, loss_fn, device, cfg.train.eval_group_samples)}
    if ood_loader is not None:
        results["ood"] = evaluate(model, ood_loader, loss_fn, device, cfg.train.eval_group_samples)
    if log_dir:
        save_run_artifacts(cfg, log_dir, step=step, results=results)
    if writer:
        for split, m in results.items():
            for k, v in m.items():
                writer.add_scalar(f"{split}/{k}", v, step)
        metric_tags = {}
        for split, m in results.items():
            for k, v in m.items():
                metric_tags[f"hparam/{split}_{k}"] = v
        writer.add_hparams(hparams_for_tensorboard(cfg), metric_tags)
        writer.close()
        if verbose:
            print(f"[remul] logs saved to {log_dir}")
    if verbose:
        print(f"[remul] final: {results}")
    return model, results
