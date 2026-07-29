"""Aggregate REMUL TensorBoard runs into paper-style comparison tables.

Reads ``outputs/remul/*/config.json`` (written by ``remul.train``) and falls
back to scalar tags in event files when results are missing from the JSON.

Usage:
    python -m remul.collect_results
    python -m remul.collect_results --logdir outputs/remul --csv results/remul_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Optional

from .experiment_log import load_run_record

# Paper reference values (MSE in table units: N-body ×10^-3, MoCap/MD17 ×10^-2).
PAPER_REFS = {
    "nbody": {
        "SE(3)-Tr": (5.16, 4.85),
        "GATr": (1.49, 1.41),
        "Transformer": (8.99, 27.06),
        "DA-Tr": (4.20, 4.21),
        "REMUL-Tr": (1.94, 1.83),
    },
    "motion_capture_walk": {
        "SE(3)-Tr": (10.85, None),
        "GATr": (10.06, None),
        "Transformer": (5.21, None),
        "REMUL-Tr": (4.95, None),
    },
    "motion_capture_run": {
        "Transformer": (20.78, None),
        "REMUL-Tr": (18.5, None),
    },
    "md17_aspirin": {
        "EGNN": (14.41, None),
        "GNN": (9.26, None),
        "REMUL-GNN": (9.28, None),
    },
}


def _load_scalars(run_dir: Path) -> dict[str, float]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as exc:
        raise SystemExit("tensorboard is required: pip install tensorboard") from exc

    if not list(run_dir.glob("events.out.tfevents.*")):
        return {}
    ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()
    out: dict[str, float] = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        if events:
            out[tag.replace("/", "_")] = events[-1].value
    train = ea.Scalars("train/total_loss") if "train/total_loss" in ea.Tags().get("scalars", []) else []
    if train:
        out["train_steps"] = train[-1].step
    return out


def _finite(value: Optional[float]) -> bool:
    return value is not None and not math.isnan(value) and not math.isinf(value)


def _paper_scale(dataset: Optional[str]) -> float:
    if dataset == "nbody":
        return 1e3
    if dataset == "nbody_egnn":
        return 1.0
    return 1e2


def _label(record: dict[str, Any]) -> str:
    model = record.get("model") or "?"
    mode = record.get("mode") or "?"
    penalty = record.get("penalty")
    beta = record.get("beta")
    if mode == "remul":
        return f"{model}+REMUL({penalty}, β={beta:g})"
    if mode == "da":
        return f"{model}+DA"
    return model


def _collect_runs(logdir: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for run_dir in sorted(logdir.iterdir()):
        if not run_dir.is_dir():
            continue
        record = load_run_record(run_dir)
        scalars = _load_scalars(run_dir)
        if record is None:
            if not scalars:
                continue
            record = {
                "run_dir": run_dir.name,
                "path": str(run_dir),
                "dataset": None,
                "model": None,
                "mode": None,
                "penalty": None,
                "beta": None,
                "molecule": None,
                "mocap_subject": None,
                "train_steps": scalars.get("train_steps"),
                "test_mse": scalars.get("test_mse"),
                "test_E": scalars.get("test_E"),
                "test_E_prime": scalars.get("test_E_prime"),
                "ood_mse": scalars.get("ood_mse"),
                "ood_E": scalars.get("ood_E"),
                "ood_E_prime": scalars.get("ood_E_prime"),
                "labeled": False,
            }
        else:
            record["labeled"] = True
            if not _finite(record.get("test_mse")) and scalars.get("test_mse") is not None:
                record["test_mse"] = scalars["test_mse"]
                record["test_E"] = scalars.get("test_E")
                record["test_E_prime"] = scalars.get("test_E_prime")
                record["ood_mse"] = scalars.get("ood_mse")
                record["train_steps"] = scalars.get("train_steps", record.get("train_steps"))
        runs.append(record)
    return runs


def _fmt(value: Optional[float], scale: float) -> str:
    if not _finite(value):
        return "—"
    return f"{value * scale:.3f}"


def _print_table(title: str, rows: list[tuple[str, str, str, str]]) -> None:
    print(f"\n{title}")
    print(f"{'Method':<36} {'In-dist MSE':>12} {'OOD MSE':>12} {'E′':>12}")
    print("-" * 76)
    for method, ind, ood, ep in rows:
        print(f"{method:<36} {ind:>12} {ood:>12} {ep:>12}")


def _best_by_label(runs: list[dict[str, Any]], key_fn) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for run in runs:
        if not _finite(run.get("test_mse")):
            continue
        label = key_fn(run)
        prev = grouped.get(label)
        if prev is None or run["test_mse"] < prev["test_mse"]:
            grouped[label] = run
    return grouped


def _print_legacy_summary(runs: list[dict[str, Any]]) -> None:
    legacy = [r for r in runs if not r.get("labeled") and _finite(r.get("test_mse"))]
    if not legacy:
        return
    from collections import defaultdict
    import statistics

    by_steps: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for run in legacy:
        steps = run.get("train_steps")
        if steps is not None:
            by_steps[int(steps)].append(run)

    print("\nLegacy runs (no config.json) — grouped by training steps:")
    for steps in sorted(by_steps):
        bucket = by_steps[steps]
        has_ood = any(_finite(r.get("ood_mse")) for r in bucket)
        dataset_hint = "N-body" if has_ood else ("MD17 (~500 epochs)" if steps == 1500 else "MoCap (~2000 epochs)" if steps == 34000 else "unknown")
        mses = [r["test_mse"] for r in bucket]
        eps = [r["test_E_prime"] for r in bucket if _finite(r.get("test_E_prime"))]
        scale = 1e3 if has_ood else 1e2
        print(
            f"  {steps} steps ({len(bucket)} runs, ~{dataset_hint}): "
            f"MSE×10^{'-3' if has_ood else '2'} med={statistics.median(mses) * scale:.3f} "
            f"(min={min(mses) * scale:.3f}, max={max(mses) * scale:.3f})"
        )
        if eps:
            print(f"    E′ med={statistics.median(eps):.3e}")


def _print_paper_comparison(runs: list[dict[str, Any]]) -> None:
    labeled = [r for r in runs if r.get("labeled")]
    unlabeled = len(runs) - len(labeled)
    print(f"Runs found: {len(runs)} ({len(labeled)} labeled, {unlabeled} legacy/unlabeled)")

    nbody = [r for r in labeled if r.get("dataset") == "nbody"]
    if nbody:
        scale = _paper_scale("nbody")
        grouped = _best_by_label(nbody, _label)
        rows = []
        for label, run in sorted(grouped.items()):
            rows.append((
                label,
                _fmt(run.get("test_mse"), scale),
                _fmt(run.get("ood_mse"), scale),
                _fmt(run.get("test_E_prime"), 1.0),
            ))
        _print_table("Table 1 — N-body (MSE ×10⁻³; lower is better)", rows)
        refs = PAPER_REFS["nbody"]
        print("Paper refs (in / OOD):", ", ".join(f"{k} {v[0]:.2f}/{v[1]:.2f}" for k, v in refs.items()))

    for subject, title_key, title in [
        (35, "motion_capture_walk", "Table 2 — MoCap walking, subject 35 (MSE ×10⁻²)"),
        (9, "motion_capture_run", "Table 2 — MoCap running, subject 9 (MSE ×10⁻²)"),
    ]:
        subset = [r for r in labeled if r.get("dataset") == "motion_capture" and r.get("mocap_subject") == subject]
        if not subset:
            continue
        scale = _paper_scale("motion_capture")
        grouped = _best_by_label(subset, _label)
        rows = [(label, _fmt(run.get("test_mse"), scale), "—", _fmt(run.get("test_E_prime"), 1.0))
                for label, run in sorted(grouped.items())]
        _print_table(title, rows)
        refs = PAPER_REFS.get(title_key, {})
        if refs:
            print("Paper refs:", ", ".join(f"{k} {v[0]:.2f}" for k, v in refs.items()))

    md17 = [r for r in labeled if r.get("dataset") == "md17"]
    if md17:
        scale = _paper_scale("md17")
        molecules = sorted({r.get("molecule") for r in md17 if r.get("molecule")})
        print(f"\nTable 3 — MD17 (MSE ×10⁻²), molecules: {', '.join(molecules)}")
        for mol in molecules:
            subset = [r for r in md17 if r.get("molecule") == mol and _finite(r.get("test_mse")) and r["test_mse"] < 10]
            if not subset:
                continue
            grouped = _best_by_label(subset, _label)
            best = min(subset, key=lambda r: r["test_mse"])
            print(f"  {mol}: best {_label(best)} → {_fmt(best.get('test_mse'), scale)}  "
                  f"(n={len(subset)} finite runs)")

    nbody8 = [r for r in labeled if r.get("dataset") == "nbody_egnn"]
    if nbody8:
        scale = _paper_scale("nbody_egnn")
        grouped = _best_by_label(nbody8, _label)
        rows = [(label, _fmt(run.get("test_mse"), scale), "—", _fmt(run.get("test_E_prime"), 1.0))
                for label, run in sorted(grouped.items())]
        _print_table("Table 8 — Charged N-body (raw MSE; lower is better)", rows)

    _print_legacy_summary(runs)


def _write_csv(path: Path, runs: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_dir", "labeled", "dataset", "model", "mode", "penalty", "beta",
        "molecule", "mocap_subject", "train_steps",
        "test_mse", "test_mse_x1e3", "test_mse_x1e2",
        "ood_mse", "test_E", "test_E_prime", "path",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            scale = _paper_scale(run.get("dataset"))
            row = {k: run.get(k) for k in fields if k not in ("test_mse_x1e3", "test_mse_x1e2", "labeled")}
            row["labeled"] = run.get("labeled", False)
            mse = run.get("test_mse")
            row["test_mse_x1e3"] = mse * 1e3 if _finite(mse) else ""
            row["test_mse_x1e2"] = mse * 1e2 if _finite(mse) else ""
            writer.writerow(row)
    print(f"\nWrote CSV: {path}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate REMUL TensorBoard runs.")
    parser.add_argument("--logdir", default="outputs/remul", help="TensorBoard log root")
    parser.add_argument("--csv", default="", help="Optional CSV output path")
    args = parser.parse_args(argv)

    logdir = Path(args.logdir)
    if not logdir.is_dir():
        print(f"No log directory: {logdir}", file=sys.stderr)
        return 1

    runs = _collect_runs(logdir)
    if not runs:
        print(f"No runs found under {logdir}")
        return 1

    _print_paper_comparison(runs)
    if args.csv:
        _write_csv(Path(args.csv), runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
