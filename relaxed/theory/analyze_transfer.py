"""Analyze the deep-net transfer experiments (E1 crossover slope, E3/E4 selection).

Reads every outputs/e1e3/**/record.json, keys each run by its config fields
(model, mode, group, field, axis, n_train, seed) and metrics (test.mse,
val.best_mse), then:

  E1  for each n_train, seed-average test.mse for strict (egnn) and unconstrained
      (mlp); locate the crossover field s_cross where egnn_mse == mlp_mse; fit
      log(s_cross) vs log(n) and report the slope (theory: -1/2).

  E3/E4  for each (field, axis), select per seed the loss group with the lowest
      *validation* MSE (non-oracle) and report the recovery rate of the true
      residual axis; field=0 is the isotropic R=0 control.

Writes results/transfer_e1.json and results/transfer_e3.json and prints a report.
"""
from __future__ import annotations

import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

ROOT = "outputs/e1e3"


def load_records():
    rows = []
    for path in glob.glob(os.path.join(ROOT, "**", "record.json"), recursive=True):
        try:
            r = json.load(open(path))
        except Exception:
            continue
        if r.get("status") != "completed":
            continue
        c = r.get("config", {})
        m = r.get("metrics", {})
        exp = c.get("run", {}).get("experiment_name", "")
        rows.append({
            "exp": exp,
            "model": c.get("model", {}).get("name"),
            "mode": c.get("train", {}).get("mode"),
            "group": c.get("loss", {}).get("group"),
            "field": float(c.get("data", {}).get("field_strength", 0.0)),
            "axis": int(c.get("data", {}).get("field_axis", 2)),
            "n": int(c.get("data", {}).get("n_train", 0)),
            "seed": int(r.get("seed", 0)),
            "test_mse": float(m.get("test", {}).get("mse", float("nan"))),
            "val_mse": float(m.get("val", {}).get("best_mse", float("nan"))),
        })
    return rows


def _interp_cross(fields, egnn, mlp):
    """First field where egnn_mse crosses above mlp_mse; linear interp in field."""
    order = np.argsort(fields)
    f = np.array(fields)[order]; e = np.array(egnn)[order]; mm = np.array(mlp)[order]
    d = e - mm                                   # <0: egnn wins ; >0: mlp wins
    for i in range(len(f) - 1):
        if d[i] <= 0 < d[i + 1]:
            # linear interpolation of the zero crossing in field
            return float(f[i] + (0 - d[i]) * (f[i + 1] - f[i]) / (d[i + 1] - d[i]))
    return None


def analyze_e1(rows, prefix="e1_"):
    e1 = [r for r in rows if r["exp"].startswith(prefix)]
    # seed-average test_mse by (model, n, field)
    agg = defaultdict(list)
    for r in e1:
        agg[(r["model"], r["n"], r["field"])].append(r["test_mse"])
    mean = {k: float(np.mean(v)) for k, v in agg.items()}
    ns = sorted({k[1] for k in mean})
    fields = sorted({k[2] for k in mean})
    crossings = []
    curves = {}
    for n in ns:
        egnn = [mean.get(("egnn", n, f), float("nan")) for f in fields]
        mlp = [mean.get(("mlp", n, f), float("nan")) for f in fields]
        curves[n] = {"fields": fields, "egnn": egnn, "mlp": mlp}
        sc = _interp_cross(fields, egnn, mlp)
        if sc and sc > 0:
            crossings.append((n, sc))
    slope = intercept = r2 = None
    if len(crossings) >= 2:
        logn = np.array([math.log(n) for n, _ in crossings])
        logs = np.array([math.log(s) for _, s in crossings])
        A = np.vstack([logn, np.ones_like(logn)]).T
        (slope, intercept), *_ = np.linalg.lstsq(A, logs, rcond=None)
        pred = A @ np.array([slope, intercept])
        ss_res = float(((logs - pred) ** 2).sum())
        ss_tot = float(((logs - logs.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    out = {"n_train": ns, "fields": fields, "crossings": crossings,
           "slope": None if slope is None else float(slope),
           "intercept": None if intercept is None else float(intercept),
           "r2": None if r2 is None else float(r2),
           "theory_slope": -0.5, "curves": curves,
           "n_records": len(e1)}
    return out


def analyze_e3(rows):
    e3 = [r for r in rows if r["exp"].startswith("e3_")]
    groups = ["so2_x", "so2_y", "so2_z", "so3"]
    true_axis_group = {2: "so2_z", 0: "so2_x", 1: "so2_y"}
    cells = sorted({(r["field"], r["axis"]) for r in e3})
    results = {}
    for field, axis in cells:
        cell = [r for r in e3 if r["field"] == field and r["axis"] == axis]
        seeds = sorted({r["seed"] for r in cell})
        picks, per_seed = [], []
        val_by_group = defaultdict(list); test_by_group = defaultdict(list)
        for s in seeds:
            sub = {r["group"]: r for r in cell if r["seed"] == s and r["mode"] == "remul"}
            if not all(g in sub for g in groups):
                continue
            valerrs = {g: sub[g]["val_mse"] for g in groups}
            pick = min(valerrs, key=valerrs.get)
            picks.append(pick)
            per_seed.append({"seed": s, "pick": pick, "val": valerrs})
            for g in groups:
                val_by_group[g].append(sub[g]["val_mse"])
                test_by_group[g].append(sub[g]["test_mse"])
        std_test = [r["test_mse"] for r in cell if r["mode"] == "standard"]
        true_g = true_axis_group.get(axis) if field != 0 else "so3"
        recov = picks.count(true_g) / len(picks) if picks else float("nan")
        results[f"field{int(field)}_axis{axis}"] = {
            "field": field, "axis": axis, "true_group": true_g,
            "recovery_rate": recov, "n_seeds": len(picks),
            "selected_mode": max(set(picks), key=picks.count) if picks else None,
            "mean_val_mse": {g: float(np.mean(v)) for g, v in val_by_group.items()},
            "mean_test_mse": {g: float(np.mean(v)) for g, v in test_by_group.items()},
            "no_loss_test_mse": float(np.mean(std_test)) if std_test else None,
            "per_seed": per_seed,
        }
    return {"cells": results, "n_records": len(e3)}


def _print_e1(e1, title):
    print("=" * 78)
    print(title)
    print("=" * 78)
    for n, sc in e1["crossings"]:
        print(f"  n_train={n:>5}   s_cross={sc:.3f}")
    if e1["slope"] is not None:
        print(f"\n  fitted log-log slope = {e1['slope']:.3f}   (theory -0.500)   R^2={e1['r2']:.3f}")
        verdict = "MATCHES -1/2" if abs(e1["slope"] + 0.5) < 0.12 else "does NOT match -1/2"
        print(f"  VERDICT: crossover scaling {verdict}")
    else:
        print("  (not enough crossings)")


def main():
    rows = load_records()
    print(f"loaded {len(rows)} completed records from {ROOT}\n")
    e1 = analyze_e1(rows, "e1_")
    e1v2 = analyze_e1(rows, "e1v2_")
    e3 = analyze_e3(rows)

    if e1v2["crossings"]:
        _print_e1(e1v2, "E1v2 · CROSSOVER SCALING — variance-limited (small-n) regime [HEADLINE]")
        print()
    _print_e1(e1, "E1 · CROSSOVER SCALING — original sweep (spans into capacity-limited regime)")

    print("\n" + "=" * 78)
    print("E3/E4 · NON-ORACLE GROUP SELECTION (select by validation MSE)")
    print("=" * 78)
    for key, c in e3["cells"].items():
        tag = "isotropic R=0 control" if c["field"] == 0 else f"field={c['field']:.0f} axis={c['axis']}"
        print(f"\n  [{tag}]  true group = {c['true_group']}  ({c['n_seeds']} seeds)")
        print(f"    recovery rate = {c['recovery_rate']:.0%}   (selected mode: {c['selected_mode']})")
        vals = c["mean_val_mse"]
        best = min(vals, key=vals.get)
        line = "    val MSE: " + "  ".join(f"{g}={vals[g]:.2e}{'*' if g == best else ''}" for g in vals)
        print(line)
        if c["no_loss_test_mse"] is not None:
            print(f"    no-loss baseline test MSE = {c['no_loss_test_mse']:.2e}")

    os.makedirs("results", exist_ok=True)
    json.dump(e1, open("results/transfer_e1.json", "w"), indent=2)
    json.dump(e1v2, open("results/transfer_e1v2.json", "w"), indent=2)
    json.dump(e3, open("results/transfer_e3.json", "w"), indent=2)
    print("\nwrote results/transfer_e1.json, results/transfer_e1v2.json, results/transfer_e3.json")


if __name__ == "__main__":
    main()
