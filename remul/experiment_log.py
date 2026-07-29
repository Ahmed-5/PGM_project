"""Run metadata helpers for TensorBoard logs and result aggregation."""
from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .config import ExperimentConfig


def run_slug(cfg: ExperimentConfig) -> str:
    """Filesystem-safe identifier for a run (no timestamp)."""
    parts = [cfg.data.name, cfg.model.name, cfg.train.mode]
    if cfg.data.name == "md17":
        parts.append(cfg.data.molecule)
    elif cfg.data.name == "motion_capture":
        parts.append(f"subj{cfg.data.mocap_subject}")
    if cfg.train.mode == "remul":
        parts.append(cfg.train.penalty)
        parts.append(f"beta{cfg.train.beta:g}")
    elif cfg.train.mode == "da":
        parts.append("da")
    return "_".join(str(p) for p in parts)


def make_log_dir(cfg: ExperimentConfig) -> str:
    slug = cfg.log.run_name or run_slug(cfg)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(cfg.log.log_dir, f"{slug}_{ts}")


def flatten_config(cfg: ExperimentConfig) -> dict[str, Any]:
    d = cfg.to_dict()
    flat: dict[str, Any] = {}
    for section, values in d.items():
        if isinstance(values, dict):
            for key, value in values.items():
                flat[f"{section}/{key}"] = value
        else:
            flat[section] = values
    return flat


def hparams_for_tensorboard(cfg: ExperimentConfig) -> dict[str, Any]:
    """Flat hparams with only types TensorBoard accepts."""
    out: dict[str, Any] = {}
    for key, value in flatten_config(cfg).items():
        if value is None:
            out[key] = "none"
        elif isinstance(value, (str, int, float, bool)):
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                out[key] = str(value)
            else:
                out[key] = value
        else:
            out[key] = str(value)
    return out


def save_run_artifacts(
    cfg: ExperimentConfig,
    log_dir: str,
    *,
    step: int,
    results: Optional[dict] = None,
) -> None:
    os.makedirs(log_dir, exist_ok=True)
    payload = {
        "slug": run_slug(cfg),
        "step": step,
        "config": cfg.to_dict(),
        "results": results or {},
    }
    path = os.path.join(log_dir, "config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def load_run_record(run_dir: Path) -> Optional[dict[str, Any]]:
    """Load metadata + final metrics for one run directory."""
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return None
    with open(cfg_path, encoding="utf-8") as f:
        meta = json.load(f)
    config = meta.get("config") or {}
    data = config.get("data") or {}
    model = config.get("model") or {}
    train = config.get("train") or {}
    results = meta.get("results") or {}

    record = {
        "run_dir": run_dir.name,
        "path": str(run_dir),
        "dataset": data.get("name"),
        "model": model.get("name"),
        "mode": train.get("mode"),
        "penalty": train.get("penalty"),
        "beta": train.get("beta"),
        "molecule": data.get("molecule"),
        "mocap_subject": data.get("mocap_subject"),
        "train_steps": meta.get("step"),
        "test_mse": _metric(results, "test", "mse"),
        "test_E": _metric(results, "test", "E"),
        "test_E_prime": _metric(results, "test", "E_prime"),
        "ood_mse": _metric(results, "ood", "mse"),
        "ood_E": _metric(results, "ood", "E"),
        "ood_E_prime": _metric(results, "ood", "E_prime"),
    }
    return record


def _metric(results: dict, split: str, key: str) -> Optional[float]:
    split_metrics = results.get(split)
    if not isinstance(split_metrics, dict):
        return None
    value = split_metrics.get(key)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return float(value)
