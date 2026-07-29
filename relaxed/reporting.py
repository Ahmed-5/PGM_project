"""Unified run-record schema (v2) and readers for all historical formats.

One ``RunRecord`` (a plain dict) describes any training run in this repo,
whether produced by the legacy ``remul`` package, the legacy top-level graph
pipeline, or the unified ``relaxed`` package. The schema:

    schema       : int (2)
    slug         : filesystem-safe run identifier (no timestamp)
    timestamp    : YYYYMMDD_HHMMSS
    status       : completed | nan | stale | smoke | in_progress | failed
    framework    : remul | graph
    git_hash     : str | None
    env          : {torch, cuda, gpu}
    seed         : int | None
    config       : full nested config dict
    hyperparams  : flattened {"section/key": value} dict
    metrics      : {split: {metric: float}} — task + equivariance numbers
    timing       : {train_s, epochs, steps}
    run_dir      : directory holding the artifacts
    checkpoint   : path to best checkpoint | None

Writers produce ``record.json`` inside the run dir; readers lift every
historical format into the same dict so a single aggregator can consume all
runs ever made here.
"""
from __future__ import annotations

import glob
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Optional

SCHEMA_VERSION = 2

# ---------------------------------------------------------------------------
# Record construction
# ---------------------------------------------------------------------------


def _git_hash() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def _env() -> dict[str, Any]:
    env: dict[str, Any] = {"torch": None, "cuda": None, "gpu": None}
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda"] = torch.version.cuda if torch.cuda.is_available() else None
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    return env


def flatten_config(config: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested config to {"section/key": value} hyperparameters."""
    flat: dict[str, Any] = {}
    for section, values in config.items():
        if isinstance(values, dict):
            for key, value in values.items():
                flat[f"{section}/{key}"] = value
        else:
            flat[section] = values
    return flat


def make_record(*, slug: str, timestamp: str, framework: str, config: dict,
                metrics: dict, seed: Optional[int] = None,
                status: str = "completed", timing: Optional[dict] = None,
                run_dir: str = "", checkpoint: Optional[str] = None) -> dict:
    return {
        "schema": SCHEMA_VERSION,
        "slug": slug,
        "timestamp": timestamp,
        "status": status,
        "framework": framework,
        "git_hash": _git_hash(),
        "env": _env(),
        "seed": seed,
        "config": config,
        "hyperparams": flatten_config(config),
        "metrics": metrics,
        "timing": timing or {},
        "run_dir": run_dir,
        "checkpoint": checkpoint,
    }


def write_record(record: dict, path: Optional[str] = None) -> str:
    path = path or os.path.join(record["run_dir"], "record.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, sort_keys=True, default=str)
    return path


def load_record(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _finite(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _any_nan(metrics: dict) -> bool:
    for split in metrics.values():
        if isinstance(split, dict):
            for v in split.values():
                if isinstance(v, float) and math.isnan(v):
                    return True
    return False


# ---------------------------------------------------------------------------
# Legacy readers
# ---------------------------------------------------------------------------


def read_remul_run(run_dir: str) -> Optional[dict]:
    """Lift a legacy remul run (outputs/remul/<slug>_<ts>/config.json)."""
    cfg_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(cfg_path):
        return None
    with open(cfg_path, encoding="utf-8") as f:
        meta = json.load(f)
    config = meta.get("config") or {}
    results = meta.get("results") or {}
    step = meta.get("step")
    slug = meta.get("slug") or os.path.basename(run_dir)
    train = config.get("train") or {}
    data = config.get("data") or {}

    # Status classification
    smoke = (train.get("max_steps") == 3 or
             (data.get("n_train") is not None and data.get("n_train", 100) <= 32) or
             (data.get("md17_n_train") is not None and data.get("md17_n_train", 500) <= 16))
    max_steps = train.get("max_steps")
    if not results:
        status = "in_progress"
    elif smoke:
        status = "smoke"
    elif _any_nan(results):
        status = "nan"
    elif max_steps and step is not None and step < max_steps:
        status = "stale"  # truncated before reaching the step budget (pre-fix)
    else:
        status = "completed"

    metrics = {split: {k: _finite(v) for k, v in m.items()}
               for split, m in results.items() if isinstance(m, dict)}
    return make_record(
        slug=slug,
        timestamp=os.path.basename(run_dir).rsplit("_", 2)[-2] + "_" +
                  os.path.basename(run_dir).rsplit("_", 1)[-1],
        framework="remul",
        config=config,
        metrics=metrics,
        seed=train.get("seed"),
        status=status,
        timing={"steps": step, "epochs": train.get("epochs")},
        run_dir=run_dir,
        checkpoint=None,
    )


def read_graph_run(ckpt_dir: str, outputs_root: str = "outputs") -> Optional[dict]:
    """Lift a legacy top-level run (checkpoints/<exp>_<ts>/ + outputs/<exp>_<ts>/)."""
    run_name = os.path.basename(ckpt_dir)
    config_files = glob.glob(os.path.join(ckpt_dir, "*_config.json"))
    if not config_files:
        return None
    with open(config_files[0], encoding="utf-8") as f:
        config = json.load(f)
    exp_name = config.get("experiment_name", run_name)
    metrics_file = os.path.join(outputs_root, run_name, "test_metrics.json")
    metrics: dict[str, dict[str, float]] = {}
    status = "in_progress"
    if os.path.exists(metrics_file):
        with open(metrics_file, encoding="utf-8") as f:
            raw = json.load(f)
        test = {k.split("/", 1)[1]: _finite(v) for k, v in raw.items()
                if k.startswith("test/")}
        metrics = {"test": {k: v for k, v in test.items() if v is not None}}
        status = "completed"
        epochs = (config.get("training") or {}).get("num_epochs", 100)
        if epochs <= 3:
            status = "smoke"

    best = glob.glob(os.path.join(ckpt_dir, "*_best.pt"))
    ts = run_name.rsplit("_", 2)[-2] + "_" + run_name.rsplit("_", 1)[-1]
    return make_record(
        slug=exp_name,
        timestamp=ts,
        framework="graph",
        config=config,
        metrics=metrics,
        seed=config.get("seed"),
        status=status,
        timing={"epochs": (config.get("training") or {}).get("num_epochs")},
        run_dir=ckpt_dir,
        checkpoint=best[0] if best else None,
    )


def iter_all_runs(remul_root: str = "outputs/remul",
                  checkpoints_root: str = "checkpoints",
                  outputs_root: str = "outputs") -> list[dict]:
    """Read every run ever produced by either framework (top-level dirs only)."""
    records = []
    for run_dir in sorted(glob.glob(os.path.join(remul_root, "*_*"))):
        if not os.path.isdir(run_dir):
            continue
        rec = read_remul_run(run_dir)
        if rec:
            records.append(rec)
    for ckpt_dir in sorted(glob.glob(os.path.join(checkpoints_root, "*_*"))):
        if not os.path.isdir(ckpt_dir):
            continue
        rec = read_graph_run(ckpt_dir, outputs_root)
        if rec:
            records.append(rec)
    # v2 records written by the unified package (outputs/relaxed/<slug>_<ts>/
    # and outputs/<exp>_<ts>/)
    for rec_path in sorted(glob.glob(os.path.join(outputs_root, "relaxed", "*_*", "record.json")) +
                           glob.glob(os.path.join(outputs_root, "*_*", "record.json"))):
        try:
            records.append(load_record(rec_path))
        except (json.JSONDecodeError, OSError):
            continue
    # De-duplicate by (run_dir) preferring v2 records
    seen: dict[str, dict] = {}
    for rec in records:
        key = rec.get("run_dir", rec["slug"] + rec["timestamp"])
        if key not in seen or rec.get("schema") == SCHEMA_VERSION and "record.json" in str(rec):
            seen[key] = rec
    return list(seen.values())
