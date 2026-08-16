"""Aggregate the model-zoo × dataset grid into a table + per-dataset plots.

Reads outputs/e1e3/zoo_*/**/record.json, aggregates by (model, dataset) over
seeds, classifies each model by its equivariance support, and emits:
  * results/zoo_grid.json   — consumed by build_dashboard.py
  * results/zoo_<dataset>.png — MSE-by-model bar chart (coloured by equiv class)
  * a printed summary table.

Central question: strict-equivariant models should win when their symmetry
matches the task (SO(3) datasets) and lose when it is broken/partial.
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASS = {
    "egnn": "strict", "se3_transformer": "strict", "gatr": "strict", "tfn": "strict", "emlp": "strict",
    "gmn": "strict", "egno": "strict", "hegnn": "strict",
    "rpp": "relaxed-arch",
    "mlp": "unconstrained", "transformer": "unconstrained", "gnn": "unconstrained", "mpnn": "unconstrained",
    "mlp_remul_so3": "soft-loss",
}
CLASS_COLOR = {"strict": "#d08a2c", "relaxed-arch": "#9457c4",
               "unconstrained": "#8a94a0", "soft-loss": "#1aa0aa"}
DATASET_SYM = {"nbody_so3": "SO(3) full", "nbody_broken": "SO(3)→SO(2) broken",
               "nbody_charged": "SO(3) charged", "md17": "E(3) molecular", "mocap": "SO(2) vertical (real)"}
MUTED = "#7a8690"


def load():
    rows = []
    for p in glob.glob("outputs/e1e3/zoo_*/**/record.json", recursive=True):
        try:
            r = json.load(open(p))
        except Exception:
            continue
        if r.get("status") != "completed":
            continue
        exp = r.get("config", {}).get("run", {}).get("experiment_name", "")
        if not exp.startswith("zoo_"):
            continue
        # tag: zoo_{dataset}__{model}__s{seed}
        rest = exp[4:]
        try:
            dname, mname, seed = rest.split("__")
        except ValueError:
            continue
        m = r.get("metrics", {})
        rows.append({"dataset": dname, "model": mname, "seed": seed,
                     "mse": float(m.get("test", {}).get("mse", float("nan"))),
                     "mse_rel": float(m.get("test", {}).get("mse_rel", float("nan"))),
                     "e_prime": float(m.get("test", {}).get("E_prime", float("nan"))),
                     "ood": float(m.get("ood", {}).get("mse", float("nan"))) if "ood" in m else
                            float(m.get("ood_axis", {}).get("mse", float("nan"))) if "ood_axis" in m else float("nan")})
    return rows


def main():
    rows = load()
    if not rows:
        print("no zoo records yet"); return
    datasets = [d for d in ["nbody_so3", "nbody_broken", "nbody_charged", "md17", "mocap"]
                if any(r["dataset"] == d for r in rows)]
    models = [m for m in ["egnn", "se3_transformer", "gatr", "emlp", "rpp", "mlp", "transformer", "mlp_remul_so3"]
              if any(r["model"] == m for r in rows)]

    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        agg[r["model"]][r["dataset"]].append(r)
    table = {}
    for m in models:
        table[m] = {}
        for d in datasets:
            rs = agg[m].get(d, [])
            if rs:
                mrel = float(np.mean([x["mse_rel"] for x in rs]))
                ood_vals = [x["ood"] for x in rs if x["ood"] == x["ood"]]  # drop NaN
                table[m][d] = {"mse": float(np.mean([x["mse"] for x in rs])),
                               "mse_rel": mrel,
                               "e_prime": float(np.mean([x["e_prime"] for x in rs])),
                               "ood": float(np.mean(ood_vals)) if ood_vals else None,
                               "diverged": bool(mrel > 2.0)}   # >2 => worse than persistence => failed

    out = {"datasets": datasets, "models": models, "classes": {m: CLASS.get(m, "?") for m in models},
           "dataset_sym": {d: DATASET_SYM.get(d, "") for d in datasets}, "table": table,
           "n_records": len(rows)}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/zoo_grid.json", "w"), indent=2)

    # per-dataset bar charts (MSE by model, coloured by equivariance class).
    # Diverged models (mse_rel>2) are capped at the top with a hatch so a single
    # blow-up (e.g. gatr on mocap) doesn't crush the scale.
    for d in datasets:
        vals = [(m, table[m][d]) for m in models if d in table[m]]
        if not vals:
            continue
        conv = [c["mse"] for _, c in vals if not c["diverged"]]
        ymax = max(conv) * 4 if conv else max(c["mse"] for _, c in vals)
        ymin = min(c["mse"] for _, c in vals if c["mse"] > 0) * 0.5
        fig, ax = plt.subplots(figsize=(7.4, 4.0), dpi=150)
        fig.patch.set_alpha(0); ax.patch.set_alpha(0)
        names = [m for m, _ in vals]
        for i, (m, c) in enumerate(vals):
            col = CLASS_COLOR[CLASS.get(m, "unconstrained")]
            h = min(c["mse"], ymax) if c["diverged"] else c["mse"]
            ax.bar(i, h, color=col, edgecolor=("#cc3b57" if c["diverged"] else "none"),
                   linewidth=1.5, hatch="///" if c["diverged"] else None)
            if c["diverged"]:
                ax.text(i, ymax * 0.98, "⚠", ha="center", va="top", color="#cc3b57", fontsize=11)
        ax.set_yscale("log"); ax.set_ylim(ymin, ymax)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right", color=MUTED, fontsize=8.5)
        ax.set_ylabel("test MSE (↓)", color=MUTED, fontsize=9.5)
        ax.set_title(f"{d}  —  {DATASET_SYM.get(d,'')}   (⚠ = diverged)", color=MUTED, fontsize=10.5)
        ax.tick_params(colors=MUTED); [ax.spines[s].set_color("#b8c0c8") for s in ax.spines]
        ax.grid(True, axis="y", alpha=.14, color=MUTED)
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CLASS_COLOR.values()]
        leg = ax.legend(handles, CLASS_COLOR.keys(), fontsize=7.5, framealpha=0, loc="best")
        for t in leg.get_texts():
            t.set_color(MUTED)
        plt.tight_layout(); plt.savefig(f"results/zoo_{d}.png", transparent=True, dpi=150); plt.close()

    # equivariance-spectrum figure: E' by model (strict ~1e-7 ... unconstrained ~1)
    eprime = {m: np.nanmean([table[m][d]["e_prime"] for d in datasets if d in table[m]]) for m in models}
    order = sorted(models, key=lambda m: eprime[m])
    fig, ax = plt.subplots(figsize=(7.4, 3.6), dpi=150); fig.patch.set_alpha(0); ax.patch.set_alpha(0)
    ax.bar(range(len(order)), [eprime[m] for m in order],
           color=[CLASS_COLOR[CLASS.get(m, "unconstrained")] for m in order], edgecolor="none")
    ax.set_yscale("log")
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=30, ha="right", color=MUTED, fontsize=8.5)
    ax.set_ylabel("equivariance error E' (↓ = more equivariant)", color=MUTED, fontsize=9)
    ax.set_title("The equivariance spectrum: strict ≈1e-7  →  soft-loss  →  unconstrained ≈1", color=MUTED, fontsize=10.5)
    ax.tick_params(colors=MUTED); [ax.spines[s].set_color("#b8c0c8") for s in ax.spines]
    ax.grid(True, axis="y", alpha=.14, color=MUTED)
    plt.tight_layout(); plt.savefig("results/zoo_spectrum.png", transparent=True, dpi=150); plt.close()
    out["eprime_by_model"] = {m: float(eprime[m]) for m in models}
    json.dump(out, open("results/zoo_grid.json", "w"), indent=2)   # re-dump with spectrum + diverged flags

    # printed summary
    print(f"=== MODEL-ZOO × DATASET (mean test MSE over seeds, {len(rows)} records) ===")
    w = max(len(m) for m in models) + 2
    print(" " * w + "".join(f"{d[:13]:>15}" for d in datasets))
    for m in models:
        line = f"{m:<{w}}"
        for d in datasets:
            c = table[m].get(d)
            cell = f"{c['mse']:.2e}" if c else "—"
            line += f"{cell:>15}"
        print(line + f"   [{CLASS.get(m,'?')}]")
    # who wins each dataset
    print("\nBest model per dataset:")
    for d in datasets:
        cand = [(m, table[m][d]["mse"]) for m in models if d in table[m]]
        if cand:
            bm = min(cand, key=lambda x: x[1])
            print(f"  {d:<14} ({DATASET_SYM.get(d,'')}): {bm[0]} [{CLASS.get(bm[0],'?')}]  mse={bm[1]:.2e}")
    print("\nwrote results/zoo_grid.json + results/zoo_<dataset>.png")


if __name__ == "__main__":
    main()
