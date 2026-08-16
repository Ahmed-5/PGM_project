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
            "beta": float(c.get("train", {}).get("beta", float("nan"))),
            "seed": int(r.get("seed", 0)),
            "test_mse": float(m.get("test", {}).get("mse", float("nan"))),
            "val_mse": float(m.get("val", {}).get("best_mse", float("nan"))),
            "ood_mse": float(m.get("ood_axis", {}).get("mse", float("nan"))),
            "ood_degrade": float(m.get("ood_axis", {}).get("degrade", float("nan"))),
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


def analyze_e6(rows):
    """E6 MoCap real-data OOD robustness about the vertical (Y) axis. Matched group
    = so2_y. Compare OOD-MSE / degradation across groups + no-loss, with paired tests."""
    from scipy import stats
    e6 = [r for r in rows if r["exp"].startswith("e6_")]
    if not e6:
        return {"n_records": 0}
    groups = ["so2_y", "so2_x", "so2_z", "so3"]
    seeds = sorted({r["seed"] for r in e6})
    ood = {g: [] for g in groups}; degr = {g: [] for g in groups}; std_ood = []; test = {g: [] for g in groups}
    for s in seeds:
        sub = {r["group"]: r for r in e6 if r["seed"] == s and r["mode"] == "remul"}
        stdr = [r for r in e6 if r["seed"] == s and r["mode"] == "standard"]
        if not all(g in sub for g in groups):
            continue
        for g in groups:
            ood[g].append(sub[g]["ood_mse"]); degr[g].append(sub[g]["ood_degrade"]); test[g].append(sub[g]["test_mse"])
        if stdr:
            std_ood.append(stdr[0]["ood_mse"])
    tg = "so2_y"
    wrong = ["so2_x", "so2_z"]
    def pair_p(a, b):
        return float(stats.ttest_rel(a, b).pvalue) if len(a) == len(b) and len(a) > 1 else float("nan")
    best_wrong = list(np.min(np.stack([ood[g] for g in wrong]), axis=0)) if ood[tg] else []
    out = {"n_records": len(e6), "n_seeds": len(ood[tg]),
           "mean_ood_mse": {g: float(np.mean(ood[g])) for g in groups},
           "mean_degrade": {g: float(np.mean(degr[g])) for g in groups},
           "mean_test_mse": {g: float(np.mean(test[g])) for g in groups},
           "no_loss_ood_mse": float(np.mean(std_ood)) if std_ood else None,
           "matched_vs_wrong_p": pair_p(best_wrong, ood[tg]) if best_wrong else float("nan"),
           "matched_vs_so3_p": pair_p(ood["so3"], ood[tg]),
           "matched_vs_noloss_p": pair_p(std_ood, ood[tg]) if len(std_ood) == len(ood[tg]) else float("nan")}
    return out


def analyze_e2(rows):
    """Interior optimum / β* transfer (Thm 1). For each group, ER vs β on both the
    in-distribution and OOD (residual-symmetry) metrics; report the argmin β."""
    from collections import defaultdict
    e2 = [r for r in rows if r["exp"].startswith("e2_")]
    if not e2:
        return {"n_records": 0, "groups": {}}
    std = [r for r in e2 if r["mode"] == "standard"]
    out = {"n_records": len(e2), "groups": {},
           "no_loss": {"test_mse": float(np.mean([r["test_mse"] for r in std])) if std else None,
                       "ood_mse": float(np.mean([r["ood_mse"] for r in std])) if std else None}}
    for g in sorted({r["group"] for r in e2 if r["mode"] == "remul"}):
        cell = [r for r in e2 if r["group"] == g and r["mode"] == "remul"]
        betas = sorted({r["beta"] for r in cell})
        test = {b: float(np.mean([r["test_mse"] for r in cell if r["beta"] == b])) for b in betas}
        ood = {b: float(np.mean([r["ood_mse"] for r in cell if r["beta"] == b])) for b in betas}
        b_test = min(test, key=test.get); b_ood = min(ood, key=ood.get)
        out["groups"][g] = {"betas": betas, "test_mse": test, "ood_mse": ood,
                            "argmin_beta_test": b_test, "argmin_beta_ood": b_ood,
                            "interior_ood": betas[0] < b_ood < betas[-1]}
    return out


def analyze_e5(rows):
    """OOD robustness to the residual symmetry: for each field axis, compare the
    OOD-MSE (test rotated about the field axis) and degradation across loss groups.
    Prediction: the matched SO(2) group is most robust (lowest OOD-MSE / degrade)."""
    from scipy import stats
    e5 = [r for r in rows if r["exp"].startswith("e5_")]
    groups = ["so2_x", "so2_y", "so2_z", "so3"]
    true_g = {2: "so2_z", 0: "so2_x", 1: "so2_y"}
    out = {}
    for axis in sorted({r["axis"] for r in e5}):
        cell = [r for r in e5 if r["axis"] == axis]
        seeds = sorted({r["seed"] for r in cell})
        ood = {g: [] for g in groups}; degr = {g: [] for g in groups}; std_ood = []
        paired = {g: [] for g in groups}
        for s in seeds:
            sub = {r["group"]: r for r in cell if r["seed"] == s and r["mode"] == "remul"}
            stdr = [r for r in cell if r["seed"] == s and r["mode"] == "standard"]
            if not all(g in sub for g in groups):
                continue
            for g in groups:
                ood[g].append(sub[g]["ood_mse"]); degr[g].append(sub[g]["ood_degrade"])
                paired[g].append(sub[g]["ood_mse"])
            if stdr:
                std_ood.append(stdr[0]["ood_mse"])
        tg = true_g[axis]
        wrong = [g for g in ["so2_x", "so2_y", "so2_z"] if g != tg]
        best_wrong = np.min(np.stack([paired[g] for g in wrong]), axis=0) if paired[tg] else np.array([])
        p_wrong = float(stats.ttest_rel(best_wrong, paired[tg]).pvalue) if len(paired[tg]) > 1 else float("nan")
        p_so3 = float(stats.ttest_rel(paired["so3"], paired[tg]).pvalue) if len(paired[tg]) > 1 else float("nan")
        p_std = (float(stats.ttest_rel(std_ood, paired[tg]).pvalue)
                 if len(std_ood) == len(paired[tg]) and len(std_ood) > 1 else float("nan"))
        out[f"axis{axis}"] = {
            "axis": axis, "true_group": tg,
            "mean_ood_mse": {g: float(np.mean(ood[g])) for g in groups},
            "mean_degrade": {g: float(np.mean(degr[g])) for g in groups},
            "no_loss_ood_mse": float(np.mean(std_ood)) if std_ood else None,
            "matched_beats_wrong_p": p_wrong, "matched_beats_so3_p": p_so3,
            "matched_beats_noloss_p": p_std, "n_seeds": len(paired[tg]),
        }
    return {"cells": out, "n_records": len(e5)}


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

    e6 = analyze_e6(rows)
    if e6.get("n_records"):
        print("\n" + "=" * 78)
        print("E6 · MoCap real-data OOD robustness about vertical (Y) — matched = so2_y")
        print("=" * 78)
        om = e6["mean_ood_mse"]; best = min(om, key=om.get)
        print("  OOD-MSE: " + "  ".join(f"{g}={om[g]:.2e}{'*' if g == best else ''}" for g in om)
              + (f"  | no-loss={e6['no_loss_ood_mse']:.2e}" if e6["no_loss_ood_mse"] else ""))
        print("  degrade(×): " + "  ".join(f"{g}={e6['mean_degrade'][g]:.2f}" for g in e6["mean_degrade"]))
        print(f"  matched vs wrong-axis p={e6['matched_vs_wrong_p']:.3f} | vs so3 p={e6['matched_vs_so3_p']:.3f} | "
              f"vs no-loss p={e6['matched_vs_noloss_p']:.3f}  ({e6['n_seeds']} seeds)")
        json.dump(e6, open("results/transfer_e6.json", "w"), indent=2)

    e2 = analyze_e2(rows)
    if e2["n_records"]:
        print("\n" + "=" * 78)
        print("E2 · INTERIOR OPTIMUM / β* (Thm 1 transfer)  [field=6, z-axis]")
        print("=" * 78)
        nl = e2["no_loss"]
        print(f"  no-loss (β=0): in-dist MSE={nl['test_mse']:.2e}  OOD-MSE={nl['ood_mse']:.2e}")
        for g, c in e2["groups"].items():
            print(f"\n  [{g}]  argmin β: in-dist={c['argmin_beta_test']:g}  OOD={c['argmin_beta_ood']:g}"
                  f"  (interior-OOD optimum: {c['interior_ood']})")
            print("    β:        " + "  ".join(f"{b:>7g}" for b in c["betas"]))
            print("    in-dist:  " + "  ".join(f"{c['test_mse'][b]:.1e}" for b in c["betas"]))
            print("    OOD:      " + "  ".join(f"{c['ood_mse'][b]:.1e}" for b in c["betas"]))

    e5 = analyze_e5(rows)
    if e5["n_records"]:
        print("\n" + "=" * 78)
        print("E5 · OOD ROBUSTNESS to the residual symmetry (test rotated about field axis)")
        print("=" * 78)
        for key, c in e5["cells"].items():
            print(f"\n  [field axis={c['axis']}  true group = {c['true_group']}]  ({c['n_seeds']} seeds)")
            om = c["mean_ood_mse"]; best = min(om, key=om.get)
            print("    OOD-MSE: " + "  ".join(f"{g}={om[g]:.2e}{'*' if g == best else ''}" for g in om)
                  + (f"  | no-loss={c['no_loss_ood_mse']:.2e}" if c["no_loss_ood_mse"] else ""))
            dg = c["mean_degrade"]
            print("    degrade(×): " + "  ".join(f"{g}={dg[g]:.2f}" for g in dg))
            print(f"    matched vs best-wrong-SO2 p={c['matched_beats_wrong_p']:.3f} | "
                  f"vs so3 p={c['matched_beats_so3_p']:.4f} | vs no-loss p={c['matched_beats_noloss_p']:.4f}")

    os.makedirs("results", exist_ok=True)
    json.dump(e2, open("results/transfer_e2.json", "w"), indent=2)
    json.dump(e5, open("results/transfer_e5.json", "w"), indent=2)
    json.dump(e1, open("results/transfer_e1.json", "w"), indent=2)
    json.dump(e1v2, open("results/transfer_e1v2.json", "w"), indent=2)
    json.dump(e3, open("results/transfer_e3.json", "w"), indent=2)
    print("\nwrote results/transfer_e1.json, results/transfer_e1v2.json, results/transfer_e3.json")


if __name__ == "__main__":
    main()
