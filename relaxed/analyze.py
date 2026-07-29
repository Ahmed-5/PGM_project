"""Multi-seed aggregation + significance analysis for the graph ablations.

Answers the two research questions defensibly (the repo's first significance
testing):

* **RQ1 (layerwise scheduling):** for a fixed cell (dataset, model, group set,
  alpha_0), compare each schedule against ``constant`` and against ``baseline``
  (no equivariance loss).
* **RQ2 (equivariance groups):** for a fixed cell (dataset, model, schedule,
  alpha_0), compare each group set against a single-group reference (``so3``)
  and against ``baseline``.

Metrics are judged on BOTH families the user asked for:
  - in-distribution accuracy: ``test.MAE`` (lower better);
  - OOD / robustness: ``ood.<g>.MAE`` and functional equivariance ``ood.<g>.E_prime``
    plus internal ``test.eq_loss_unweighted`` (all lower better).

Per arm we report mean±std over seeds, a 95% percentile-bootstrap CI, and — vs
the reference arm — a paired difference with a paired t-test p-value (paired
over shared seeds), Holm-corrected across the family, plus a paired bootstrap
CI of the difference. A Pareto front (accuracy vs equivariance error) is drawn.

Usage:
    python -m relaxed.analyze
    python -m relaxed.analyze --dataset QM9 --model gcn --ood-group so3 \
        --csv results/arms.csv --pareto results/pareto_qm9_gcn.png
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from typing import Any, Optional

import numpy as np

try:
    from scipy import stats as _scipy_stats
except Exception:  # noqa: BLE001
    _scipy_stats = None

from .reporting import iter_all_runs

# Reproducible bootstrap/permutation RNG (analysis is a plain CLI, not a
# workflow script, so numpy RNG is fine here).
_RNG = np.random.default_rng(0)


# --------------------------------------------------------------------------- #
# Config / metric access (handles both unified and legacy record schemas)
# --------------------------------------------------------------------------- #
def _get(rec: dict, *dotted_candidates: str, default: Any = None) -> Any:
    """Return the first present value among several dotted config paths."""
    cfg = rec.get("config") or {}
    for dotted in dotted_candidates:
        cur: Any = cfg
        ok = True
        for part in dotted.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok and cur is not None:
            return cur
    return default


def arm_signature(rec: dict) -> dict:
    groups = tuple(sorted(_get(rec, "loss.symmetry_groups",
                                "equivariance.symmetry_groups", default=[]) or []))
    alpha0 = _get(rec, "schedule.alpha_0", "scheduler.alpha_0", default=0.0)
    try:
        alpha0 = float(alpha0)
    except (TypeError, ValueError):
        alpha0 = 0.0
    baseline = (len(groups) == 0) or (alpha0 == 0.0)
    return {
        "dataset": _get(rec, "data.name", "data.dataset_name", default="?"),
        "model": _get(rec, "model.name", "model.model_type", default="?"),
        "groups": "baseline" if baseline else "+".join(groups),
        "schedule": "none" if baseline else _get(
            rec, "loss.layer_weight_strategy",
            "equivariance.layer_weight_strategy", default="constant"),
        "alpha_0": 0.0 if baseline else alpha0,
        "prob": _get(rec, "loss.stochastic_probability",
                     "equivariance.stochastic_probability", default=None),
        "num_samples": _get(rec, "loss.num_samples",
                            "equivariance.num_samples", default=None),
        "baseline": baseline,
    }


def get_metric(rec: dict, key: str) -> Optional[float]:
    """key = 'split.metric'; metric may itself contain '/' (e.g. 'so3/E_prime')."""
    split, _, metric = key.partition(".")
    val = ((rec.get("metrics") or {}).get(split) or {}).get(metric)
    try:
        v = float(val)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def bootstrap_ci(values: list[float], n: int = 10000, ci: float = 0.95):
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return (float("nan"), float("nan"))
    if a.size == 1:
        return (float(a[0]), float(a[0]))
    idx = _RNG.integers(0, a.size, size=(n, a.size))
    means = a[idx].mean(axis=1)
    lo, hi = (1 - ci) / 2, 1 - (1 - ci) / 2
    return (float(np.quantile(means, lo)), float(np.quantile(means, hi)))


def paired_test(a_map: dict, b_map: dict, n_boot: int = 10000):
    """Paired difference (arm A - reference B) over shared seeds.

    Returns dict {n, diff, ci_low, ci_high, p} where p is a paired t-test
    (scipy) if available, else a two-sided sign-flip permutation p-value.
    """
    shared = sorted(set(a_map) & set(b_map))
    if len(shared) < 2:
        return {"n": len(shared), "diff": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"), "p": float("nan")}
    a = np.array([a_map[s] for s in shared], dtype=float)
    b = np.array([b_map[s] for s in shared], dtype=float)
    d = a - b
    diff = float(d.mean())

    # Paired bootstrap CI of the mean difference.
    idx = _RNG.integers(0, d.size, size=(n_boot, d.size))
    boot = d[idx].mean(axis=1)
    ci_low, ci_high = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))

    if _scipy_stats is not None:
        try:
            p = float(_scipy_stats.ttest_rel(a, b).pvalue)
        except Exception:  # noqa: BLE001
            p = _sign_flip_p(d)
    else:
        p = _sign_flip_p(d)
    return {"n": len(shared), "diff": diff, "ci_low": ci_low, "ci_high": ci_high, "p": p}


def _sign_flip_p(d: np.ndarray, n_perm: int = 20000) -> float:
    """Two-sided sign-flip permutation p-value for a paired difference."""
    obs = abs(d.mean())
    n = d.size
    if n <= 20:  # exact enumeration
        count = total = 0
        for mask in range(1 << n):
            signs = np.array([1 if (mask >> i) & 1 else -1 for i in range(n)])
            total += 1
            if abs((d * signs).mean()) >= obs - 1e-12:
                count += 1
        return count / total
    signs = _RNG.choice([-1.0, 1.0], size=(n_perm, n))
    perm_means = np.abs((signs * d).mean(axis=1))
    return float((perm_means >= obs - 1e-12).mean())


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down correction; NaNs pass through unchanged."""
    idx = [i for i, p in enumerate(pvals) if p == p]  # non-NaN
    order = sorted(idx, key=lambda i: pvals[i])
    m = len(order)
    out = list(pvals)
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * pvals[i])
        running = max(running, adj)  # enforce monotonicity
        out[i] = running
    return out


def _mean_std(vals: list[float]):
    a = np.asarray(vals, dtype=float)
    if a.size == 0:
        return (float("nan"), float("nan"))
    return (float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0)


# --------------------------------------------------------------------------- #
# Arm aggregation
# --------------------------------------------------------------------------- #
def graph_records(records: list[dict]) -> list[dict]:
    return [r for r in records
            if r.get("framework") == "graph" and r.get("status") == "completed"]


def by_seed(recs: list[dict], metric: str) -> dict:
    """{seed: value} for one arm's runs (last write wins per seed)."""
    out = {}
    for r in recs:
        v = get_metric(r, metric)
        if v is not None:
            out[r.get("seed")] = v
    return out


def group_by_arm(records: list[dict]):
    """arm_label -> (signature, [records]) restricted to graph/completed."""
    arms: dict[str, list[dict]] = defaultdict(list)
    sigs: dict[str, dict] = {}
    for r in graph_records(records):
        sig = arm_signature(r)
        label = f"{sig['dataset']}/{sig['model']} | {sig['groups']} | {sig['schedule']} | a0={sig['alpha_0']:g}"
        arms[label].append(r)
        sigs[label] = sig
    return arms, sigs


# --------------------------------------------------------------------------- #
# Verdict tables
# --------------------------------------------------------------------------- #
def _verdict(diff: float, p: float, lower_is_better: bool = True, alpha: float = 0.05) -> str:
    if not (p == p) or not (diff == diff):
        return "n/a"
    if p >= alpha:
        return "n.s."
    improved = diff < 0 if lower_is_better else diff > 0
    return "HELPS" if improved else "HURTS"


def comparison_table(records, *, vary: str, fixed: dict, reference_value: str,
                     baseline_arm: Optional[list] = None,
                     metrics: Optional[list] = None):
    """One RQ table: vary a single axis, compare each value vs the reference
    value and vs baseline, on several metrics.

    vary: 'schedule' (RQ1) or 'groups' (RQ2).
    fixed: dict of the held-fixed axes and required values (dataset, model, ...).
    reference_value: the value of `vary` used as the within-family reference.
    """
    metrics = metrics or ["test.MAE"]
    recs = graph_records(records)

    # Select the cell (fixed axes) and, for the varied axis, the non-baseline arms.
    def matches_fixed(sig):
        for k, v in fixed.items():
            if v is not None and sig.get(k) != v:
                return False
        return True

    cell = defaultdict(list)  # vary_value -> records
    baseline_recs = list(baseline_arm) if baseline_arm else []
    for r in recs:
        sig = arm_signature(r)
        if sig["baseline"]:
            # a baseline for this dataset/model
            if (fixed.get("dataset") in (None, sig["dataset"])
                    and fixed.get("model") in (None, sig["model"])):
                baseline_recs.append(r)
            continue
        if matches_fixed(sig):
            cell[sig[vary]].append(r)

    rows = []
    values = sorted(cell.keys())
    for metric in metrics:
        ref_map = by_seed(cell.get(reference_value, []), metric)
        base_map = by_seed(baseline_recs, metric)
        # collect p-values for Holm correction within this metric/family
        pending = []
        for val in values:
            arm_recs = cell[val]
            arm_map = by_seed(arm_recs, metric)
            mean, std = _mean_std(list(arm_map.values()))
            vs_ref = paired_test(arm_map, ref_map) if val != reference_value else {
                "n": len(arm_map), "diff": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p": float("nan")}
            vs_base = paired_test(arm_map, base_map)
            pending.append({"metric": metric, "value": val, "n": len(arm_map),
                            "mean": mean, "std": std,
                            "vs_ref": vs_ref, "vs_base": vs_base})
        # Holm across arms (vs reference) for this metric
        corr = holm([row["vs_ref"]["p"] for row in pending])
        for row, pc in zip(pending, corr):
            row["p_ref_holm"] = pc
        rows.extend(pending)
    return rows


def print_table(rows, vary: str, reference_value: str):
    if not rows:
        print("  (no runs for this cell)")
        return
    hdr = (f"  {'metric':24s}{vary:18s}{'n':>3s}  {'mean':>10s}{'±std':>9s}"
           f"  {'Δ vs ref':>10s}{'p(Holm)':>9s}  {'verdict':8s}{'Δ vs base':>10s}{'p':>8s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        lower = True  # all our metrics are lower-is-better
        v_ref = _verdict(r["vs_ref"]["diff"], r.get("p_ref_holm", float("nan")), lower)
        base = r["vs_base"]
        print(f"  {r['metric']:24s}{str(r['value']):18s}{r['n']:>3d}  "
              f"{r['mean']:>10.4f}{r['std']:>9.4f}  "
              f"{r['vs_ref']['diff']:>10.4f}{r.get('p_ref_holm', float('nan')):>9.3f}  "
              f"{v_ref:8s}{base['diff']:>10.4f}{base['p']:>8.3f}")


# --------------------------------------------------------------------------- #
# Pareto front (accuracy vs equivariance error)
# --------------------------------------------------------------------------- #
def pareto_points(records, x_metric="test.MAE", y_metric="test.eq_loss_unweighted"):
    """Per-arm mean (x, y) points with labels (both minimized)."""
    arms, sigs = group_by_arm(records)
    pts = []
    for label, recs in arms.items():
        xs = list(by_seed(recs, x_metric).values())
        ys = list(by_seed(recs, y_metric).values())
        if xs and ys:
            pts.append({"label": label, "x": float(np.mean(xs)),
                        "y": float(np.mean(ys)), "sig": sigs[label],
                        "x_std": float(np.std(xs)) if len(xs) > 1 else 0.0,
                        "y_std": float(np.std(ys)) if len(ys) > 1 else 0.0})
    return pts


def pareto_front(points) -> list[dict]:
    """Non-dominated set (minimize both x and y)."""
    front = []
    for p in points:
        dominated = any(q["x"] <= p["x"] and q["y"] <= p["y"]
                        and (q["x"] < p["x"] or q["y"] < p["y"]) for q in points)
        if not dominated:
            front.append(p)
    return sorted(front, key=lambda p: p["x"])


def plot_pareto(points, path: str, x_metric: str, y_metric: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    front = pareto_front(points)
    front_set = {p["label"] for p in front}
    fig, ax = plt.subplots(figsize=(7, 5))
    for p in points:
        on = p["label"] in front_set
        ax.errorbar(p["x"], p["y"], xerr=p["x_std"], yerr=p["y_std"], fmt="o",
                    ms=7 if on else 4, color="#d62728" if on else "#9aa0a6",
                    ecolor="#cccccc", capsize=2, zorder=3 if on else 1)
    if front:
        ax.plot([p["x"] for p in front], [p["y"] for p in front],
                "-", color="#d62728", lw=1.5, zorder=2, label="Pareto front")
    ax.set_xlabel(x_metric + "  (lower better)")
    ax.set_ylabel(y_metric + "  (lower better)")
    ax.set_title("Accuracy vs equivariance error (per arm, mean ± std over seeds)")
    ax.legend()
    fig.tight_layout()
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def write_arms_csv(records, path: str, metrics: list[str]):
    import csv
    import os
    arms, sigs = group_by_arm(records)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = ["arm", "dataset", "model", "groups", "schedule", "alpha_0",
              "prob", "num_samples", "n_seeds"]
    for m in metrics:
        fields += [f"{m}_mean", f"{m}_std", f"{m}_ci_low", f"{m}_ci_high"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for label, recs in sorted(arms.items()):
            sig = sigs[label]
            row = {"arm": label, "dataset": sig["dataset"], "model": sig["model"],
                   "groups": sig["groups"], "schedule": sig["schedule"],
                   "alpha_0": sig["alpha_0"], "prob": sig["prob"],
                   "num_samples": sig["num_samples"]}
            n_seeds = 0
            for m in metrics:
                vals = list(by_seed(recs, m).values())
                n_seeds = max(n_seeds, len(vals))
                mean, std = _mean_std(vals)
                lo, hi = bootstrap_ci(vals)
                row[f"{m}_mean"] = mean
                row[f"{m}_std"] = std
                row[f"{m}_ci_low"] = lo
                row[f"{m}_ci_high"] = hi
            row["n_seeds"] = n_seeds
            w.writerow(row)
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="QM9")
    ap.add_argument("--model", default="gcn")
    ap.add_argument("--ood-group", default="so3",
                    help="group name whose OOD metrics to report (ood.<g>.MAE / E_prime)")
    ap.add_argument("--ref-schedule", default="constant")
    ap.add_argument("--ref-groups", default="so3")
    ap.add_argument("--alpha0", type=float, default=None,
                    help="fix alpha_0 for the RQ cells (default: any)")
    ap.add_argument("--csv", default="results/arms.csv")
    ap.add_argument("--pareto", default="results/pareto.png")
    args = ap.parse_args()

    records = iter_all_runs()
    gr = graph_records(records)
    print("=" * 84)
    print(f"GRAPH ABLATION ANALYSIS — {len(gr)} completed graph runs")
    print("=" * 84)
    if not gr:
        print("No completed graph runs found yet. Run the grid, then re-run analyze.")
        return

    g = args.ood_group
    metrics = ["test.MAE", f"ood.{g}/MAE", f"ood.{g}/E_prime", "test.eq_loss_unweighted"]

    print(f"\n### RQ1 — does LAYERWISE SCHEDULING help?  "
          f"(cell: {args.dataset}/{args.model}, groups={args.ref_groups}, "
          f"vs schedule='{args.ref_schedule}' and baseline)\n")
    rows1 = comparison_table(
        records, vary="schedule",
        fixed={"dataset": args.dataset, "model": args.model,
               "groups": args.ref_groups, "alpha_0": args.alpha0},
        reference_value=args.ref_schedule, metrics=metrics)
    print_table(rows1, "schedule", args.ref_schedule)

    print(f"\n### RQ2 — do equivariance GROUPS help?  "
          f"(cell: {args.dataset}/{args.model}, schedule={args.ref_schedule}, "
          f"vs groups='{args.ref_groups}' and baseline)\n")
    rows2 = comparison_table(
        records, vary="groups",
        fixed={"dataset": args.dataset, "model": args.model,
               "schedule": args.ref_schedule, "alpha_0": args.alpha0},
        reference_value=args.ref_groups, metrics=metrics)
    print_table(rows2, "groups", args.ref_groups)

    pts = pareto_points(records, "test.MAE", "test.eq_loss_unweighted")
    if pts:
        write_arms_csv(records, args.csv, metrics)
        plot_pareto(pts, args.pareto, "test.MAE", "test.eq_loss_unweighted")
        front = pareto_front(pts)
        print(f"\nPareto front (test.MAE vs test.eq_loss_unweighted), {len(front)} arms:")
        for p in front:
            print(f"  MAE={p['x']:.4f}  eq={p['y']:.4e}  [{p['label']}]")
        print(f"\nArms CSV  -> {args.csv}")
        print(f"Pareto    -> {args.pareto}")


if __name__ == "__main__":
    main()
