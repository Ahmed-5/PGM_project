"""Graph engine (molecular graph property regression) — migrated from the
legacy top-level ``train.py``.

Trains a BaseGNN on ZINC/QM9/QM7b/MD17-graph/ModelNet40 with the layer-wise
functional equivariance loss (stochastic, 1/p-rescaled, per-layer weights from
a DepthScheduler). The equivariance plumbing itself
(``compute_equivariance_losses``) is imported from the legacy module — single
source of truth.

Differences from the legacy loop (the fixes):
* trailing partial gradient-accumulation group is flushed (previously its
  gradients were silently discarded);
* predictions/targets appended once (was twice — 2x metric memory);
* the mislabeled ``layer_weight/..._loss`` logging (which re-logged losses) is
  removed; actual weights are logged under ``layer_weights/layer_i``;
* learnable DepthScheduler init passes through inverse softplus so effective
  initial weights equal the intended schedule (see relaxed/schedulers.py);
* writes schema-v2 ``record.json`` + ``history.jsonl`` + legacy-compatible
  config JSON and ``test_metrics.json``.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from ..adapt import to_graph_config
from ..logging import get_logger
from ..metrics import mae as mae_metric, rmse as rmse_metric, r2_score
from ..reporting import make_record, write_record
from ..schedulers import DepthScheduler

# Legacy single-source plumbing (read-only use).
from equivariance_loss import EquivarianceLoss
from train import (build_optimizer, compute_equivariance_losses,
                   get_batch_predictions_and_targets, initialize_equivariance_losses)
from utils import EarlyStopping, load_checkpoint, save_checkpoint, set_seed


def _forward_pred(model, x, pos, edge_index, batch_idx):
    """Graph-level prediction, squeezed to match the target shape."""
    out = model(x, pos, edge_index, batch_idx)
    if out.dim() == 2 and out.shape[1] == 1:
        out = out.squeeze(1)
    return out


@torch.no_grad()
def evaluate_ood(model, loader, device, legacy, groups, num_rotations):
    """Rotational-robustness metrics on the test split (invariant targets).

    For each group g in ``groups`` and ``num_rotations`` sampled transforms of
    the input positions (only), reports:
      - ``{g}/MAE`` / ``{g}/RMSE``: elementwise error of ``f(g.x)`` vs the
        unchanged (invariant) target — the OOD generalization metric;
      - ``{g}/E_prime``: mean per-graph ``||f(g.x) - f(x)||`` (output
        invariance error, paper Eq. 9 specialized to invariant outputs);
      - ``{g}/E``: mean per-graph ``||mean_g f(g.x) - f(x)||`` (paper Eq. 8).
    Returns {} when positions are unavailable (metric is meaningless).
    """
    model.eval()
    if not legacy.data.use_positions:
        return {}
    samplers = {g: EquivarianceLoss(group_type=g).to(device) for g in groups}
    out = {}
    for g, sampler in samplers.items():
        abs_err, sq_err, n_elems = 0.0, 0.0, 0
        eprime, e_val, n_graphs = 0.0, 0.0, 0
        has_pos = True
        for batch in loader:
            batch = batch.to(device)
            if not hasattr(batch, "pos") or batch.pos is None:
                has_pos = False
                break
            pos = batch.pos
            target = batch.y
            if target.dim() == 2 and target.shape[1] == 1:
                target = target.squeeze(1)
            ng = int(batch.batch.max().item()) + 1
            f_x = _forward_pred(model, batch.x, pos, batch.edge_index, batch.batch)
            f_x_flat = f_x.reshape(ng, -1)
            f_gx_sum = torch.zeros_like(f_x_flat)
            for _ in range(num_rotations):
                R, t, s = sampler.sample_geometric_batch(ng, device)
                pos_g = sampler.apply_geometric_transform(pos, batch.batch, R, t, s)
                f_gx = _forward_pred(model, batch.x, pos_g, batch.edge_index, batch.batch)
                diff_t = f_gx - target
                abs_err += diff_t.abs().sum().item()
                sq_err += (diff_t ** 2).sum().item()
                n_elems += diff_t.numel()
                f_gx_flat = f_gx.reshape(ng, -1)
                eprime += (f_gx_flat - f_x_flat).norm(dim=-1).sum().item()
                f_gx_sum += f_gx_flat
            e_val += (f_gx_sum / max(num_rotations, 1) - f_x_flat).norm(dim=-1).sum().item()
            n_graphs += ng
        if not has_pos:
            return {}
        out[f"{g}/MAE"] = abs_err / max(n_elems, 1)
        out[f"{g}/RMSE"] = (sq_err / max(n_elems, 1)) ** 0.5
        out[f"{g}/E_prime"] = eprime / max(n_graphs, 1)
        out[f"{g}/E"] = e_val / max(n_graphs, 1)
    return out


@torch.no_grad()
def internal_equivariance_error(model, loader, eq_losses, legacy, device):
    """Mean group-weight-free, all-ones-layer-weight functional eq loss on a
    split, plus per-group totals — how equivariant the trained model actually
    is (comparable across arms; folds in recompute_unweighted_eq)."""
    if not eq_losses:
        return {}
    model.eval()
    total, n = 0.0, 0
    per_group = {}
    for batch in loader:
        batch = batch.to(device)
        eq_total, eq_dict = compute_equivariance_losses(
            model, batch, eq_losses, legacy, device,
            layer_weights=None, apply_group_weights=False)
        total += float(eq_total)
        n += 1
        for key, val in eq_dict.items():
            if key.endswith("_total"):
                per_group[key] = per_group.get(key, 0.0) + float(val)
    n = max(n, 1)
    out = {"eq_loss_unweighted": total / n}
    for key, val in per_group.items():
        name = key.replace("eq_loss/", "").replace("_total", "")
        out[f"unweighted/{name}"] = val / n
    return out


def train_epoch(model, loader, optimizer, device, task_loss_fn, eq_losses,
                logger, epoch, legacy, cfg, depth_scheduler, scaler) -> dict:
    model.train()
    epoch_losses = {"task": 0.0, "eq": 0.0, "total": 0.0}
    all_preds, all_targets = [], []

    use_amp = legacy.training.use_amp
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(pbar):
        batch = batch.to(device)
        layer_weights = depth_scheduler.get_all_alphas()
        do_equivariance = (torch.rand(1).item() < legacy.equivariance.stochastic_probability)

        with torch.autocast(enabled=use_amp, device_type=device.split(":")[0]):
            pred, target = get_batch_predictions_and_targets(model, batch, legacy, device)
            task_loss = task_loss_fn(pred, target)

            eq_loss_total = torch.tensor(0.0, device=device)
            eq_loss_dict = {}
            if do_equivariance and len(eq_losses) > 0:
                eq_loss_total, eq_loss_dict = compute_equivariance_losses(
                    model, batch, eq_losses, legacy, device, layer_weights=layer_weights)
                eq_loss_total = eq_loss_total / legacy.equivariance.stochastic_probability

            # alpha_0 is applied once via the DepthScheduler layer weights.
            total_loss = task_loss + eq_loss_total
            total_loss = total_loss / legacy.training.accumulation_steps

            all_preds.append(pred.detach().float().cpu())
            all_targets.append(target.detach().float().cpu())

        scaler.scale(total_loss).backward()

        is_last = (batch_idx + 1) == len(loader)
        # FIX: also flush a trailing partial accumulation group.
        if (batch_idx + 1) % legacy.training.accumulation_steps == 0 or is_last:
            if legacy.training.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), legacy.training.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        log_task = task_loss.item()
        log_eq = eq_loss_total.item() if do_equivariance else 0.0
        log_total = log_task + log_eq
        epoch_losses["task"] += log_task
        epoch_losses["eq"] += log_eq
        epoch_losses["total"] += log_total

        current_step = (epoch - 1) * len(loader) + batch_idx
        step_metrics = {"train/step_loss": log_total,
                        "train/step_task_loss": log_task,
                        "train/step_eq_loss": log_eq}
        if do_equivariance:
            for k, v in eq_loss_dict.items():
                step_metrics[k] = v.item() if isinstance(v, torch.Tensor) else v
        if layer_weights is not None:
            for i, w in enumerate(layer_weights.detach().cpu().numpy()):
                step_metrics[f"layer_weights/layer_{i}"] = w
        logger.log_metrics(step_metrics, step=current_step)

        pbar.set_postfix({"loss": f"{log_total:.4f}",
                          "eq": f"{log_eq:.4f}" if do_equivariance else "-"})

    num_batches = len(loader)
    metrics = {
        "train/loss": epoch_losses["total"] / num_batches,
        "train/task_loss": epoch_losses["task"] / num_batches,
        "train/eq_loss_total": epoch_losses["eq"] / num_batches,
    }
    if all_preds:
        preds_cat = torch.cat(all_preds)
        targets_cat = torch.cat(all_targets)
        metrics.update({
            "train/MAE": mae_metric(preds_cat, targets_cat),
            "train/RMSE": rmse_metric(preds_cat, targets_cat),
            "train/R2": r2_score(preds_cat, targets_cat),
        })
    return metrics


@torch.inference_mode()
def evaluate(model, loader, device, task_loss_fn, eq_losses, legacy,
             depth_scheduler=None) -> dict:
    model.eval()
    layer_weights = depth_scheduler.get_all_alphas() if depth_scheduler is not None else None
    split_losses = {"task": 0.0, "eq": 0.0, "total": 0.0}
    all_preds, all_targets = [], []

    for batch in loader:
        batch = batch.to(device)
        pred, target = get_batch_predictions_and_targets(model, batch, legacy, device)
        task_loss = task_loss_fn(pred, target)
        eq_loss_total, _ = compute_equivariance_losses(
            model, batch, eq_losses, legacy, device, layer_weights=layer_weights)
        total_loss = task_loss + eq_loss_total

        split_losses["task"] += task_loss.item()
        split_losses["eq"] += float(eq_loss_total)
        split_losses["total"] += float(total_loss)
        all_preds.append(pred.detach().cpu())
        all_targets.append(target.detach().cpu())

    num_batches = len(loader)
    metrics = {
        "loss": split_losses["total"] / num_batches,
        "task_loss": split_losses["task"] / num_batches,
        "eq_loss_total": split_losses["eq"] / num_batches,
    }
    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    metrics.update({"MAE": mae_metric(preds, targets),
                    "RMSE": rmse_metric(preds, targets),
                    "R2": r2_score(preds, targets)})
    return metrics


def train(cfg, verbose: bool = True):
    """Train one graph run from the unified config."""
    legacy = to_graph_config(cfg)   # legacy defaults + explicit overrides (finalized)
    legacy.seed = cfg.train.seed    # multi-seed loop sets this per run
    set_seed(legacy.seed)
    torch.set_float32_matmul_precision("high")
    device = legacy.device

    Path(legacy.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(legacy.output_dir).mkdir(parents=True, exist_ok=True)

    # Order matters for RNG parity with the legacy loop: datasets, loaders, model.
    from load_dataset import load_dataset
    train_dataset, val_dataset, test_dataset = load_dataset(legacy)
    loader_kwargs = dict(
        batch_size=legacy.training.batch_size,
        num_workers=legacy.data.num_workers,
        pin_memory=True,
        persistent_workers=legacy.data.persistent_workers if legacy.data.num_workers > 0 else False,
        prefetch_factor=legacy.data.prefetch_factor if legacy.data.num_workers > 0 else None,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    from ..models import build_model
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"[relaxed/graph] model={legacy.model.model_type} params={n_params:,} "
              f"dataset={legacy.data.dataset_name} device={device}")

    eq_losses = {k: v.to(device) for k, v in initialize_equivariance_losses(legacy).items()}

    # One DepthScheduler weight per real model layer (see bug A — the final
    # graph output equals the last captured layer, so no extra +1 slot).
    num_check_layers = legacy.model.num_layers
    decay_rate = getattr(legacy.equivariance, "layer_decay_rate", 0.5)
    import math
    beta_val = -math.log(max(decay_rate, 1e-6))
    depth_scheduler = DepthScheduler(
        num_layers=num_check_layers,
        schedule_type=legacy.equivariance.layer_weight_strategy,
        alpha_0=legacy.scheduler.alpha_0,
        beta=beta_val,
        gamma=legacy.scheduler.gamma,
    ).to(device)

    params = list(model.parameters())
    if legacy.equivariance.layer_weight_strategy == "learnable":
        params += list(depth_scheduler.parameters())
    optimizer = build_optimizer(params, legacy)

    # LR scheduler (legacy behavior preserved)
    from ..schedulers import get_lr_scheduler
    legacy.scheduler.t_max = legacy.training.num_epochs
    lr_scheduler = get_lr_scheduler(optimizer, legacy.scheduler)

    logger = get_logger(cfg)
    logger.log_hyperparameters(cfg.to_dict())

    task_loss_fn = nn.MSELoss()
    early_stopping = EarlyStopping(patience=legacy.training.patience,
                                   min_delta=legacy.training.min_delta)
    scaler = torch.GradScaler(enabled=legacy.training.use_amp)

    history_path = os.path.join(legacy.output_dir, "history.jsonl")
    best_val_loss = float("inf")
    t_start = time.time()
    epoch = 0
    for epoch in range(1, legacy.training.num_epochs + 1):
        epoch_start = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer, device,
                                    task_loss_fn, eq_losses, logger, epoch,
                                    legacy, cfg, depth_scheduler, scaler)
        val_metrics_raw = evaluate(model, val_loader, device, task_loss_fn,
                                   eq_losses, legacy, depth_scheduler=depth_scheduler)
        val_metrics = {f"val/{k}": v for k, v in val_metrics_raw.items()}

        if lr_scheduler is not None:
            if legacy.scheduler.lr_schedule == "plateau":
                lr_scheduler.step(val_metrics["val/loss"])
            else:
                lr_scheduler.step()

        epoch_metrics = {**train_metrics, **val_metrics}
        epoch_metrics["learning_rate"] = optimizer.param_groups[0]["lr"]
        logger.log_metrics(epoch_metrics, step=epoch * len(train_loader))
        if cfg.log.save_history:
            with open(history_path, "a") as f:
                f.write(json.dumps({"epoch": epoch, **epoch_metrics}) + "\n")

        if verbose:
            print(f"Epoch {epoch}/{legacy.training.num_epochs} ({time.time()-epoch_start:.1f}s) "
              f"train {train_metrics['train/loss']:.4f} | val {val_metrics['val/loss']:.4f} "
              f"| val MAE {val_metrics['val/MAE']:.4f} | val eq {val_metrics['val/eq_loss_total']:.2e}")

        if val_metrics["val/loss"] < best_val_loss:
            best_val_loss = val_metrics["val/loss"]
            save_checkpoint(
                model, optimizer, epoch, best_val_loss,
                os.path.join(legacy.checkpoint_dir, f"{legacy.experiment_name}_best.pt"),
                additional_state={"depth_scheduler_state_dict": depth_scheduler.state_dict()},
            )

        early_stopping(val_metrics["val/loss"], epoch)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break

    train_seconds = time.time() - t_start

    # Best-checkpoint -> test protocol
    checkpoint_path = os.path.join(legacy.checkpoint_dir, f"{legacy.experiment_name}_best.pt")
    checkpoint = load_checkpoint(model, optimizer, checkpoint_path)
    if isinstance(checkpoint, dict) and "depth_scheduler_state_dict" in checkpoint:
        try:
            depth_scheduler.load_state_dict(checkpoint["depth_scheduler_state_dict"])
        except (RuntimeError, ValueError) as e:
            # Older checkpoints may hold a differently-sized (num_layers+1)
            # learnable schedule; skip rather than crash (bug A migration).
            print(f"[relaxed/graph] WARN: could not load depth_scheduler state "
                  f"({e}); keeping freshly-initialized schedule.")

    t_eval = time.time()
    test_raw = evaluate(model, test_loader, device, task_loss_fn, eq_losses,
                        legacy, depth_scheduler=depth_scheduler)
    eval_seconds = time.time() - t_eval
    test_metrics = {f"test/{k}": v for k, v in test_raw.items()}

    # Extra evaluation: how equivariant the trained model actually is
    # (group-weight-free functional error) + OOD / rotational robustness.
    test_section = {k.split("/", 1)[1]: v for k, v in test_metrics.items()}
    internal_eq = internal_equivariance_error(model, test_loader, eq_losses, legacy, device)
    test_section.update(internal_eq)
    record_metrics = {"test": test_section}

    ood_metrics = {}
    if getattr(cfg.train, "ood_eval", False):
        ood_metrics = evaluate_ood(
            model, test_loader, device, legacy,
            groups=list(cfg.train.ood_groups),
            num_rotations=cfg.train.ood_num_rotations)
        test_mae = test_section.get("MAE")
        if ood_metrics and test_mae is not None:
            for g in cfg.train.ood_groups:
                if f"{g}/MAE" in ood_metrics:
                    ood_metrics[f"{g}/MAE_gap"] = ood_metrics[f"{g}/MAE"] - test_mae
        if ood_metrics:
            record_metrics["ood"] = ood_metrics

    # Flat metric dict (prefixed) for logger + legacy test_metrics.json.
    flat_metrics = dict(test_metrics)
    flat_metrics.update({f"test/{k}": v for k, v in internal_eq.items()})
    flat_metrics.update({f"ood/{k}": v for k, v in ood_metrics.items()})

    # Legacy-compatible artifacts
    with open(os.path.join(legacy.checkpoint_dir, f"{legacy.experiment_name}_config.json"), "w") as f:
        json.dump(legacy.to_dict(), f, indent=4)
    with open(os.path.join(legacy.output_dir, "test_metrics.json"), "w") as f:
        json.dump({k: float(v) for k, v in flat_metrics.items()}, f, indent=2)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    record = make_record(
        slug=legacy.experiment_name,
        timestamp=legacy.timestamp or ts,
        framework="graph",
        config=cfg.to_dict(),
        metrics=record_metrics,
        seed=legacy.seed,
        status="completed",
        timing={"train_s": round(train_seconds, 2), "eval_s": round(eval_seconds, 2),
                "epochs": epoch},
        run_dir=legacy.output_dir,
        checkpoint=checkpoint_path,
    )
    write_record(record, path=os.path.join(legacy.output_dir, "record.json"))

    logger.log_metrics(flat_metrics, step=epoch * len(train_loader))
    logger.finish()

    if verbose:
        print(f"[relaxed/graph] test: {test_metrics}")
    return model, test_metrics
