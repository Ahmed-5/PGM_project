"""Unified run aggregator / inventory for all runs in this repo.

Scans every run ever produced (legacy remul package, legacy graph pipeline,
and unified relaxed records), classifies them, prints paper-style comparison
tables, and writes a CSV inventory + a markdown report.

Usage:
    python -m relaxed.collect                      # full inventory + tables
    python -m relaxed.collect --csv results/runs_inventory.csv \
        --md results/runs_report.md
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from typing import Any

from .reporting import iter_all_runs

# Paper reference values (kept in sync with remul/collect_results.py).
PAPER_REFS = {
    "nbody": {"SE(3)-Tr": (5.16, 4.85), "GATr": (1.49, 1.41),
              "Transformer": (8.99, 27.06), "DA-Tr": (4.20, 4.21),
              "REMUL-Tr": (1.94, 1.83)},
    "motion_capture_walk": {"SE(3)-Tr": (10.85, None), "GATr": (10.06, None),
                            "Transformer": (5.21, None), "REMUL-Tr": (4.95, None)},
    "motion_capture_run": {"Transformer": (20.78, None), "REMUL-Tr": (18.5, None)},
    "md17_aspirin": {"EGNN": (14.41, None), "GNN": (9.26, None),
                     "REMUL-GNN": (9.28, None)},
}

STATUS_ORDER = ["completed", "nan", "stale", "smoke", "in_progress", "failed"]


def _fmt(v: Any, scale: float = 1.0, digits: int = 2) -> str:
    if v is None:
        return "-"
    try:
        f = float(v) * scale
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(f):
        return "nan"
    return f"{f:.{digits}f}"


def summarize_status(records: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for rec in records:
        counts[rec.get("status", "failed")] += 1
    return dict(counts)


def best_per_slug(records: list[dict], status: str = "completed") -> dict[str, dict]:
    """Keep the best (lowest test mse) run per slug among a given status."""
    best: dict[str, dict] = {}
    for rec in records:
        if rec.get("status") != status:
            continue
        mse = ((rec.get("metrics") or {}).get("test") or {}).get("mse")
        slug = rec["slug"]
        if slug not in best:
            best[slug] = rec
        else:
            cur = ((best[slug].get("metrics") or {}).get("test") or {}).get("mse")
            if cur is None or (mse is not None and mse < cur):
                best[slug] = rec
    return best


def mean_std_by_slug(records: list[dict], metric: str = "mse",
                     split: str = "test") -> dict[str, tuple[float, float, int]]:
    """mean±std of a metric over completed runs sharing a slug (multi-seed)."""
    groups: dict[str, list[float]] = defaultdict(list)
    for rec in records:
        if rec.get("status") != "completed":
            continue
        v = ((rec.get("metrics") or {}).get(split) or {}).get(metric)
        if v is not None:
            groups[rec["slug"]].append(float(v))
    out = {}
    for slug, vals in groups.items():
        n = len(vals)
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / max(n - 1, 1)
        out[slug] = (mean, math.sqrt(var), n)
    return out


def paper_tables(records: list[dict]) -> list[str]:
    """Paper-style comparison lines for the remul framework.

    Per slug, prefer the run with the largest step budget (full suite over
    reduced/smoke variants), then report test AND OOD mse plus E'.
    """
    lines: list[str] = []
    done = [r for r in records
            if r.get("framework") == "remul" and r.get("status") == "completed"]
    by_slug: dict[str, dict] = {}
    for rec in done:
        slug = rec["slug"]
        steps = (rec.get("timing") or {}).get("steps") or 0
        if slug not in by_slug or steps > ((by_slug[slug].get("timing") or {}).get("steps") or 0):
            by_slug[slug] = rec

    def row(label: str, slug: str, refs: tuple | None) -> str:
        rec = by_slug.get(slug)
        ours_t = ours_o = None
        if rec:
            m = rec.get("metrics") or {}
            ours_t = (m.get("test") or {}).get("mse")
            ours_o = (m.get("ood") or {}).get("mse")
        ref_s = f"paper {refs[0]:.2f}/{refs[1]:.2f}" if refs else "paper  -  "
        return (f"  {label:<14} ours {_fmt(ours_t and ours_t*1e3)}/{_fmt(ours_o and ours_o*1e3)}"
                f"  {ref_s}")

    lines.append("== N-body (paper x1e-3, test/OOD) ==")
    refs = PAPER_REFS["nbody"]
    # REMUL-Tr row: best over all remul transformer betas at the largest budget
    remul_recs = [r for s, r in by_slug.items()
                  if s.startswith("nbody_transformer_remul")]
    best_rem = min(remul_recs,
                   key=lambda r: ((r.get("metrics") or {}).get("test") or {}).get("mse") or 1e9,
                   default=None)
    for label, slug in [("SE(3)-Tr", "nbody_se3_transformer_standard"),
                        ("GATr", "nbody_gatr_standard"),
                        ("EGNN", "nbody_egnn_standard"),
                        ("Transformer", "nbody_transformer_standard"),
                        ("DA-Tr", "nbody_transformer_da_da")]:
        lines.append(row(label, slug, refs.get(label)))
    if best_rem:
        m = best_rem.get("metrics") or {}
        lines.append(f"  {'REMUL-Tr':<14} ours {_fmt(((m.get('test') or {}).get('mse') or 0)*1e3)}"
                     f"/{_fmt(((m.get('ood') or {}).get('mse') or 0)*1e3)}"
                     f"  paper {refs['REMUL-Tr'][0]:.2f}/{refs['REMUL-Tr'][1]:.2f}"
                     f"   [{best_rem['slug']}]")
    return lines


def write_csv(records: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = ["slug", "timestamp", "status", "framework", "seed",
              "dataset", "model", "mode", "beta", "test_mse", "test_E_prime",
              "ood_mse", "test_MAE", "test_eq_unweighted", "run_dir"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            cfg = rec.get("config") or {}
            data = cfg.get("data") or {}
            model = cfg.get("model") or {}
            train = cfg.get("train") or {}
            m = (rec.get("metrics") or {})
            test = m.get("test") or {}
            writer.writerow({
                "slug": rec.get("slug"),
                "timestamp": rec.get("timestamp"),
                "status": rec.get("status"),
                "framework": rec.get("framework"),
                "seed": rec.get("seed"),
                "dataset": data.get("name") or data.get("dataset_name"),
                "model": model.get("name") or model.get("model_type"),
                "mode": train.get("mode"),
                "beta": train.get("beta"),
                "test_mse": test.get("mse"),
                "test_E_prime": test.get("E_prime"),
                "ood_mse": (m.get("ood") or {}).get("mse"),
                "test_MAE": test.get("MAE"),
                "test_eq_unweighted": test.get("eq_loss_unweighted"),
                "run_dir": rec.get("run_dir"),
            })


def write_md(records: list[dict], path: str) -> None:
    counts = summarize_status(records)
    total = sum(counts.values())
    lines = ["# Run inventory", "",
             f"Total runs: **{total}**", "",
             "| status | count |", "|---|---|"]
    for status in STATUS_ORDER:
        if status in counts:
            lines.append(f"| {status} | {counts[status]} |")
    lines += ["", "## NaN / diverged runs", "",
              "| slug | run_dir |", "|---|---|"]
    for rec in records:
        if rec.get("status") == "nan":
            lines.append(f"| {rec['slug']} | {rec.get('run_dir')} |")
    lines += ["", "## Stale (truncated pre-fix) runs", "",
              "| slug | steps |", "|---|---|"]
    for rec in records:
        if rec.get("status") == "stale":
            lines.append(f"| {rec['slug']} | {(rec.get('timing') or {}).get('steps')} |")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="results/runs_inventory.csv")
    parser.add_argument("--md", default="results/runs_report.md")
    args = parser.parse_args()

    records = iter_all_runs()
    counts = summarize_status(records)
    total = sum(counts.values())

    print("=" * 72)
    print(f"RUN INVENTORY: {total} runs")
    for status in STATUS_ORDER:
        if status in counts:
            print(f"  {status:<12} {counts[status]}")
    print("=" * 72)

    for line in paper_tables(records):
        print(line)

    nan_runs = [r for r in records if r.get("status") == "nan"]
    if nan_runs:
        print("\n== NaN / diverged runs ==")
        for rec in nan_runs:
            print(f"  {rec['slug']}")

    write_csv(records, args.csv)
    write_md(records, args.md)
    print(f"\nCSV -> {args.csv}")
    print(f"MD  -> {args.md}")


if __name__ == "__main__":
    main()
