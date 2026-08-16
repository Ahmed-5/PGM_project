"""Dynamics engine (REMUL paper task) — migrated from legacy ``remul/train.py``.

Training modes: ``standard`` (task loss only), ``da`` (data augmentation with
rotated input/target), ``remul`` (multitask alpha*L_obj + beta*L_equi with
constant or GradNorm-gradual penalty). Evaluation reports task MSE plus the
label-free functional equivariance errors E/E′ (optionally per-axis).

Differences from the legacy loop (the fixes):
* optional gradient clipping (``train.grad_clip > 0``) — stabilizes
  GATr/EGNO-style divergences (NaNs in the full suite);
* optional LR schedule via ``schedule.lr_schedule`` (default ``none`` =
  paper fidelity; cosine/step available for the N-body plateau issue);
* evaluation honors ``loss.metric`` (previously hard-coded MSE);
* writes a schema-v2 ``record.json`` + per-epoch ``history.jsonl`` via
  ``relaxed.reporting`` (model, full config, hyperparams, env, metrics);
* keeps the max_steps->epoch derivation, strided evals, capped per-epoch E/E′
  batches, and full-split final test/OOD evaluation.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime

import torch
import torch.nn as nn

from ..adapt import to_remul_config
from ..datasets import build_datasets
from ..losses import GradNorm, RemulLoss, equivariance_error, rotate_batch, AugerinoAug, AugeredModel
from ..geometry import random_so2_batch
from ..metrics import mse as mse_metric
from ..models import build_model
from ..reporting import make_record, write_record
from ..schedulers import get_lr_scheduler


def _make_logger(cfg, log_dir: str):
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
def evaluate_ood_axis(model, loader, axis_char, device, n_rot=8):
    """OOD robustness to the task's residual symmetry (RQ: does soft equivariance
    pay off off-distribution?).

    For a field-broken task the true dynamics IS SO(2)-equivariant about the field
    axis, yet the training set is in a fixed frame (no rotation augmentation). We
    evaluate test MSE on inputs+targets rotated by random SO(2) rotations about
    that axis — orientations unseen in training. A model that respects the residual
    symmetry is unchanged; a non-equivariant one degrades. Returns (mse, degrade)
    where degrade = ood_mse / canonical_mse (1.0 = perfectly robust)."""
    model.eval()
    tot, n_elem, tot_can, n_can = 0.0, 0, 0.0, 0
    for batch in loader:
        batch = _to_device(batch, device)
        b = batch["pos"].shape[0]
        tot_can += torch.nn.functional.mse_loss(model(batch), batch["target"], reduction="sum").item()
        n_can += batch["target"].numel()
        for _ in range(n_rot):
            rot = random_so2_batch(axis_char, b, device=device, dtype=batch["pos"].dtype)
            rb = rotate_batch(batch, rot)
            tot += torch.nn.functional.mse_loss(model(rb), rb["target"], reduction="sum").item()
            n_elem += rb["target"].numel()
    mse = tot / max(n_elem, 1)
    can = tot_can / max(n_can, 1)
    return {"mse": mse, "canonical_mse": can, "degrade": mse / (can + 1e-12)}


def _task_metric(pred, target, kind: str) -> float:
    if kind == "l1":
        return float((pred - target).abs().mean())
    return mse_metric(pred, target)


@torch.no_grad()
def evaluate(model, loader, loss_fn: RemulLoss, device, eval_group_samples=20,
             max_equiv_batches=None, compute_equiv=True):
    """Task metric over the full split; E/E' averaged over up to
    ``max_equiv_batches`` batches (None = full split). ``compute_equiv=False``
    skips the (expensive) E/E' pass — used for cheap best-checkpoint selection."""
    model.eval()
    total_l1, n_samples_l = 0.0, 0
    total_mse, n_elem = 0.0, 0
    total_copy_mse, total_delta = 0.0, 0.0  # RC3: no-motion baseline + learned-delta scale
    for batch in loader:
        batch = _to_device(batch, device)
        pred = model(batch)
        total_l1 += _task_metric(pred, batch["target"], loss_fn.metric) * batch["target"].shape[0]
        n_samples_l += batch["target"].shape[0]
        total_mse += torch.nn.functional.mse_loss(
            pred, batch["target"], reduction="sum").item()
        n_elem += batch["target"].numel()
        # RC3: copy/persistence baseline (pred := pos) and the RMS of the
        # learned displacement, so a collapsed pred≈pos model is detectable.
        total_copy_mse += torch.nn.functional.mse_loss(
            batch["pos"], batch["target"], reduction="sum").item()
        total_delta += (pred - batch["pos"]).reshape(pred.shape[0], -1).norm(dim=-1).sum().item()

    e_sum, e_prime_sum, n_samples = 0.0, 0.0, 0
    if compute_equiv:
        fwd = lambda b: model(b)
        for i, batch in enumerate(loader):
            if max_equiv_batches is not None and i >= max_equiv_batches:
                break
            batch = _to_device(batch, device)
            b = batch["pos"].shape[0]
            e_sum += equivariance_error(fwd, batch, eval_group_samples, loss_fn.group, "E").item() * b
            e_prime_sum += equivariance_error(fwd, batch, eval_group_samples, loss_fn.group, "E_prime").item() * b
            n_samples += b
    eq_denom = max(n_samples, 1)
    mse = total_mse / max(n_elem, 1)
    copy_mse = total_copy_mse / max(n_elem, 1)
    delta_rms = total_delta / max(n_samples_l, 1)
    e_prime = e_prime_sum / eq_denom
    out = {"mse": mse, "E": e_sum / eq_denom, "E_prime": e_prime,
           # RC3 scale-relative diagnostics:
           "copy_baseline_mse": copy_mse,
           "mse_rel": mse / (copy_mse + 1e-12),      # <1 => beats no-motion baseline
           "delta_rms": delta_rms,                    # ~0 => collapsed to persistence
           "E_prime_rel": e_prime / (delta_rms + 1e-9)}  # equivariance error relative to learned motion
    if loss_fn.metric == "l1":
        out["l1"] = total_l1 / max(n_samples_l, 1)
    return out


def train(cfg, verbose: bool = True):
    """Train one dynamics run from the unified config."""
    legacy = to_remul_config(cfg)          # legacy defaults + explicit overrides
    legacy.train.seed = cfg.train.seed     # multi-seed loop sets this per run
    tcfg = legacy.train
    torch.manual_seed(tcfg.seed)
    device = tcfg.device

    data = build_datasets(cfg)
    meta = data["meta"]
    from remul.datasets import make_loader  # legacy loader helper (read-only use)
    train_loader = make_loader(data["train"], tcfg.batch_size, shuffle=True)
    val_loader = make_loader(data["val"], tcfg.batch_size)
    test_loader = make_loader(data["test"], tcfg.batch_size)
    ood_loader = make_loader(data["ood"], tcfg.batch_size) if "ood" in data else None

    model = build_model(cfg, meta).to(device)
    loss_fn = RemulLoss(tcfg.group, tcfg.metric, tcfg.num_group_samples).to(device)
    # Augerino baseline: a learnable augmentation distribution + averaged predictor.
    augerino = AugerinoAug().to(device) if tcfg.mode == "augerino" else None
    aug_model = AugeredModel(model, augerino, ncopies=4) if augerino is not None else None
    eval_model = aug_model if aug_model is not None else model
    # Include the loss's own parameters (the learnable so2_learn axis, or the Augerino
    # ranges) so the residual symmetry is discovered jointly with the model — these are
    # a few parameters and need a much higher LR than the model to move.
    extra_params = list(loss_fn.parameters()) + (list(augerino.parameters()) if augerino else [])
    if extra_params:
        opt = torch.optim.Adam(
            [{"params": list(model.parameters()), "lr": tcfg.lr},
             {"params": extra_params, "lr": max(0.03, 100.0 * tcfg.lr)}],
            weight_decay=tcfg.weight_decay)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)

    # NEW: optional LR schedule (off by default for paper fidelity).
    # Created after total_epochs is known so T_max spans the whole run.
    lr_scheduler = None

    gradnorm = None
    if tcfg.mode == "remul" and tcfg.penalty == "gradual":
        gradnorm = GradNorm(2, tcfg.gradnorm_alpha, tcfg.gradnorm_lr,
                            init_weights=[tcfg.alpha, tcfg.beta]).to(device)
        shared_w = _get_last_shared_weight(model)

    grad_clip = float(getattr(cfg.train, "grad_clip", 0.0) or 0.0)

    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"[relaxed/dyn] model={cfg.model.name} params={n_params:,} "
              f"dataset={cfg.data.name} mode={tcfg.mode} penalty={tcfg.penalty} "
              f"group={tcfg.group} nodes={meta['num_nodes']} feat={meta['num_node_features']} "
              f"device={device}")

    # Run directory + record keeping.
    from remul.experiment_log import run_slug
    slug = cfg.log.run_name or run_slug(legacy)
    if len(cfg.run.seeds) > 1:
        slug = f"{slug}_seed{tcfg.seed}"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(cfg.log.log_dir, f"{slug}_{ts}")
    os.makedirs(log_dir, exist_ok=True)
    writer = _make_logger(cfg, log_dir)
    history_path = os.path.join(log_dir, "history.jsonl")
    if verbose:
        print(f"[relaxed/dyn] run log_dir={log_dir}")

    step = 0
    max_steps = tcfg.max_steps
    if max_steps is not None:
        steps_per_epoch = max(1, len(train_loader))
        total_epochs = -(-max_steps // steps_per_epoch)
    else:
        total_epochs = tcfg.epochs
    eval_every = max(cfg.log.eval_every, total_epochs // 50)
    # Learnable-axis warmup (off by default — the cosine LR decay makes a late
    # release counter-productive; more group samples per step is the better lever).
    axis_warmup = 0

    if cfg.schedule.lr_schedule != "none":
        sched_cfg = cfg.schedule
        sched_cfg.t_max = total_epochs
        lr_scheduler = get_lr_scheduler(opt, sched_cfg)

    best_val = float("inf")   # RC2: track best-by-val model to dodge late-step collapse
    best_state = None
    t_start = time.time()
    for epoch in range(total_epochs):
        model.train()
        for batch in train_loader:
            batch = _to_device(batch, device)
            opt.zero_grad()

            if tcfg.mode == "standard":
                pred = model(batch)
                total = loss_fn.objective_loss(pred, batch["target"])
                obj = total.detach()
                equi = torch.tensor(0.0)
                total.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()

            elif tcfg.mode == "da":
                rot = loss_fn.sample_rotations(batch["pos"].shape[0], device, batch["pos"].dtype)
                rb = rotate_batch(batch, rot)
                pred = model(rb)
                total = loss_fn.objective_loss(pred, rb["target"])
                obj = total.detach()
                equi = torch.tensor(0.0)
                total.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()

            elif tcfg.mode == "augerino":
                pred = aug_model(batch)                       # averaged equivariant prediction
                task = loss_fn.objective_loss(pred, batch["target"])
                breadth = -tcfg.beta * augerino.theta.sum()   # reward larger ranges (anti-collapse)
                total = task + breadth
                obj = task.detach(); equi = augerino.theta.detach().mean()
                total.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()

            else:  # remul
                pred = model(batch)
                obj = loss_fn.objective_loss(pred, batch["target"])
                # so2_learn warmup: no equivariance term yet (axis stays put until
                # the model has learned the dynamics).
                equi = (obj.new_zeros(()) if (axis_warmup and step < axis_warmup)
                        else loss_fn.equivariance_loss(lambda b: model(b), batch))
                if gradnorm is not None:
                    weighted = gradnorm.weighted_sum([obj, equi])
                    weighted.backward(retain_graph=True)
                    gradnorm.update([obj, equi], shared_w)
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    opt.step()
                    total = weighted.detach()
                else:
                    total = tcfg.alpha * obj + tcfg.beta * equi
                    total.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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

        if lr_scheduler is not None and cfg.schedule.lr_schedule != "plateau":
            lr_scheduler.step()

        if (epoch + 1) % eval_every == 0:
            # Cheap MSE-only eval for best-checkpoint selection (E/E' is measured
            # only at the final eval — it is the expensive part).
            m = evaluate(eval_model, val_loader, loss_fn, device, tcfg.eval_group_samples,
                         compute_equiv=False)
            if m["mse"] < best_val:      # RC2: snapshot best-by-val model
                best_val = m["mse"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if verbose:
                print(f"[epoch {epoch+1}] val mse {m['mse']:.4e} | mse_rel {m['mse_rel']:.3f} "
                      f"| delta_rms {m['delta_rms']:.3e}")
            if writer:
                writer.add_scalar("val/mse", m["mse"], step)
                writer.add_scalar("val/E", m["E"], step)
                writer.add_scalar("val/E_prime", m["E_prime"], step)
            if cfg.log.save_history:
                with open(history_path, "a") as f:
                    f.write(json.dumps({"epoch": epoch + 1, "step": step,
                                        **{f"val/{k}": v for k, v in m.items()}}) + "\n")

    train_seconds = time.time() - t_start

    # RC2: evaluate the best-by-val checkpoint, not the (possibly collapsed)
    # final model. Falls back to the final model if no val eval ran.
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save the (best) model so the run can be re-evaluated later (e.g. per-axis
    # equivariance analysis) without retraining. Skippable via REMUL_NO_CKPT=1 to
    # avoid filling disk during large sweeps (metrics live in record.json anyway).
    model_path = os.path.join(log_dir, "model_final.pt")
    if os.environ.get("REMUL_NO_CKPT"):
        model_path = None
    else:
        torch.save(model.state_dict(), model_path)

    t_eval = time.time()
    # MSE over the full split; E/E' estimated over a capped number of batches
    # (a subset gives a stable functional-equivariance estimate at far lower cost
    # than the full 5000-sample x eval_group_samples rotation sweep).
    eq_cap = max(tcfg.eval_equiv_batches, 48)
    results = {"test": evaluate(eval_model, test_loader, loss_fn, device,
                                tcfg.eval_group_samples, max_equiv_batches=eq_cap)}
    if cfg.train.per_axis_eval:
        from ..losses import per_axis_equivariance_errors
        batch = _to_device(next(iter(test_loader)), device)
        results["test"].update(per_axis_equivariance_errors(
            lambda b: model(b), batch, tcfg.eval_group_samples))
    if ood_loader is not None:
        results["ood"] = evaluate(model, ood_loader, loss_fn, device,
                                  tcfg.eval_group_samples, max_equiv_batches=eq_cap)
    # Residual-symmetry OOD robustness: for a field-broken task the true dynamics
    # is SO(2)-equivariant about the field axis; test on rotations about it.
    fs = float(getattr(cfg.data, "field_strength", 0.0) or 0.0)
    # Field-broken N-body OR MoCap (whose dynamics is SO(2)-equivariant about the
    # vertical/heading axis — physically arbitrary heading): test robustness to
    # rotations about that residual axis (set via --data.field_axis).
    if fs != 0.0 or getattr(cfg.data, "name", "") == "mocap":
        axis_char = {0: "x", 1: "y", 2: "z"}.get(int(getattr(cfg.data, "field_axis", 2)), "z")
        results["ood_axis"] = evaluate_ood_axis(eval_model, test_loader, axis_char, device)
    # Surface the best validation mse so model/beta selection is on VAL, not test
    # (avoids the test-set selection bias when picking the reported REMUL beta).
    if best_val != float("inf"):
        results["val"] = {"best_mse": float(best_val)}
    if augerino is not None:
        th = augerino.theta.detach().float().cpu()
        results["augerino_theta"] = {"x": float(th[0]), "y": float(th[1]), "z": float(th[2])}
    # Learnable-axis discovery: report the recovered axis and its alignment with
    # the true residual symmetry (the field axis).
    if getattr(loss_fn, "group", None) == "so2_learn":
        ax = loss_fn.learn_axis.detach().float().cpu()
        ax = ax / (ax.norm() + 1e-9)
        fa = int(getattr(cfg.data, "field_axis", 2))
        true = torch.zeros(3); true[fa] = 1.0
        cos = float(abs(torch.dot(ax, true)))
        results["learned_axis"] = {
            "vector": [round(v, 4) for v in ax.tolist()],
            "cosine_to_field_axis": cos,
            "angle_deg": float(torch.acos(torch.clamp(torch.tensor(cos), 0.0, 1.0)) * 180.0 / torch.pi),
        }
    eval_seconds = time.time() - t_eval

    if writer:
        for split, m in results.items():
            for k, v in m.items():
                writer.add_scalar(f"{split}/{k}", v, step)
        writer.close()

    record = make_record(
        slug=slug,
        timestamp=ts,
        framework="remul",
        config=cfg.to_dict(),
        metrics=results,
        seed=tcfg.seed,
        status="completed" if all(
            v == v for m in results.values() for v in m.values()) else "nan",
        timing={"train_s": round(train_seconds, 2), "eval_s": round(eval_seconds, 2),
                "epochs": total_epochs, "steps": step},
        run_dir=log_dir,
        checkpoint=model_path,
    )
    write_record(record)
    # Legacy-compatible config.json (so older tooling keeps working).
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump({"slug": slug, "step": step, "config": legacy.to_dict(),
                   "results": results}, f, indent=2, sort_keys=True)

    if verbose:
        print(f"[relaxed/dyn] final: {results}")
    return model, results
