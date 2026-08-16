"""Build a self-contained LOCAL results dashboard + tables from all result files.

Reads whatever exists under results/ (theory_gate.json, transfer_e*.json,
rq1_matched.json, transfer_e3learn.json, transfer_e6.json, zoo aggregation) and
emits:
  * results/dashboard.html  — one self-contained page (plots base64-embedded), theme-aware
  * results/RESULTS.md      — markdown tables for every experiment
  * results/tables/*.csv    — one CSV per table

Re-run any time (e.g. after new experiments land):  python -m relaxed.theory.build_dashboard
"""
from __future__ import annotations

import base64
import csv
import glob
import json
import os

R = "results"
os.makedirs(os.path.join(R, "tables"), exist_ok=True)


def _load(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else None


def _img(name):
    p = os.path.join(R, name)
    if not os.path.exists(p):
        return ""
    b = base64.b64encode(open(p, "rb").read()).decode()
    return (f'<div class="fig"><img alt="{name}" src="data:image/png;base64,{b}"></div>')


def _csv(name, header, rows):
    with open(os.path.join(R, "tables", name), "w", newline="") as f:
        w = csv.writer(f); w.writerow(header)
        for r in rows:
            w.writerow(r)


def _htable(header, rows, hi=None):
    """HTML table; hi(row)->css class for emphasis."""
    h = "<div class='scroll'><table><thead><tr>" + "".join(f"<th>{c}</th>" for c in header) + "</tr></thead><tbody>"
    for r in rows:
        cls = f" class='{hi(r)}'" if hi else ""
        h += f"<tr{cls}>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
    return h + "</tbody></table></div>"


def _mdtable(header, rows):
    out = "| " + " | ".join(map(str, header)) + " |\n| " + " | ".join("---" for _ in header) + " |\n"
    for r in rows:
        out += "| " + " | ".join(map(str, r)) + " |\n"
    return out


SECTIONS = []   # (title, subtitle, html, md)


def add(title, subtitle, html, md):
    SECTIONS.append((title, subtitle, html, md))


# ---------------- Theory gate ----------------
g = _load("theory_gate.json")
if g:
    header = ["check", "result", "detail"]
    rows = [[c["name"].split(":")[0], "PASS" if c["passed"] else "FAIL", c["detail"][:150]] for c in g["checks"]]
    _csv("theory_gate.csv", header, rows)
    verdict = "GATE PASSED (6/6)" if g["all_pass"] else "GATE FAILED"
    add("Theory — exact-regime gate", verdict,
        _htable(header, rows, hi=lambda r: "ok" if r[1] == "PASS" else "bad"),
        _mdtable(header, rows))

# ---------------- E1 crossover ----------------
e1 = _load("transfer_e1v2.json"); e1o = _load("transfer_e1.json")
if e1:
    rows = [[n, f"{s:.3f}"] for n, s in e1["crossings"]]
    _csv("e1_crossover.csv", ["n_train", "s_cross"], rows)
    sub = f"variance-limited slope = {e1['slope']:.3f} (theory −0.500), R²={e1['r2']:.2f}"
    if e1o and e1o.get("slope") is not None:
        sub += f"  ·  full-range slope {e1o['slope']:.3f} (saturates)"
    add("E1 — crossover scaling law", sub,
        _htable(["n_train", "s_cross"], rows) + _img("crossover.png"),
        f"**{sub}**\n\n" + _mdtable(["n_train", "s_cross"], rows))

# ---------------- E3 selection ----------------
e3 = _load("transfer_e3.json")
if e3:
    header = ["condition", "true group", "recovery", "so2_x", "so2_y", "so2_z", "so3", "no-loss"]
    rows = []
    for k, c in e3["cells"].items():
        v = c["mean_val_mse"]
        rows.append([("isotropic R=0" if c["field"] == 0 else f"field ax{c['axis']}"), c["true_group"],
                     f"{c['recovery_rate']:.0%}"] + [f"{v[g]:.2e}" for g in ["so2_x", "so2_y", "so2_z", "so3"]]
                    + [f"{c['no_loss_test_mse']:.2e}" if c["no_loss_test_mse"] else "—"])
    _csv("e3_selection.csv", header, rows)
    add("E3 — non-oracle group selection (validation MSE)",
        "recovers the axis (flips with field); over-constraining SO(3) worst everywhere",
        _htable(header, rows), _mdtable(header, rows))

# ---------------- E5 OOD ----------------
e5 = _load("transfer_e5.json")
if e5:
    header = ["field axis", "true group", "so2_x", "so2_y", "so2_z", "so3", "no-loss", "vs wrong p", "vs so3 p"]
    rows = []
    for k, c in e5["cells"].items():
        om = c["mean_ood_mse"]
        rows.append([c["axis"], c["true_group"]] + [f"{om[g]:.2e}" for g in ["so2_x", "so2_y", "so2_z", "so3"]]
                    + [f"{c['no_loss_ood_mse']:.2e}", f"{c['matched_beats_wrong_p']:.3f}", f"{c['matched_beats_so3_p']:.4f}"])
    _csv("e5_ood.csv", header, rows)
    add("E5 — OOD robustness (the payoff)",
        "matched residual symmetry wins 100–1000×; axis sharply significant OOD (p<0.005)",
        _htable(header, rows) + _img("ood_robustness.png"), _mdtable(header, rows))

# ---------------- E2 interior beta* ----------------
e2 = _load("transfer_e2.json")
if e2 and e2.get("groups"):
    header = ["group", "argmin β (in-dist)", "argmin β (OOD)", "interior OOD optimum"]
    rows = [[gname, f"{c['argmin_beta_test']:g}", f"{c['argmin_beta_ood']:g}", "yes" if c["interior_ood"] else "no"]
            for gname, c in e2["groups"].items()]
    _csv("e2_beta.csv", header, rows)
    add("E2 — interior optimum β* (Thm 1 transfer)",
        "OOD wants β*>0; over-constraining SO(3) wants LESS (0.3<1.0) — exactly λ*=V/R",
        _htable(header, rows) + _img("beta_optimum.png"), _mdtable(header, rows))

# ---------------- Learnable axis ----------------
la = _load("transfer_e3learn.json")
if la:
    import numpy as np
    header = ["field axis", "median angle°", "min°", "max°", "dominant-axis-correct"]
    rows = []
    for ax, recs in la["by_axis"].items():
        angs = [r["angle"] for r in recs]
        dom = sum(1 for r in recs if int(np.argmax(np.abs(r["vec"]))) == int(ax)) / len(recs)
        rows.append([f"{'xyz'[int(ax)]}", f"{np.median(angs):.1f}", f"{min(angs):.1f}", f"{max(angs):.1f}", f"{dom:.0%}"])
    _csv("learnable_axis.csv", header, rows)
    add("Learnable continuous axis (deep net)",
        "identifies the correct dominant axis 100% of seeds; ~38° residual (vs 0.4° in sandbox)",
        _htable(header, rows), _mdtable(header, rows))

# ---------------- R1 null ----------------
r1 = _load("rq1_matched.json")
if r1:
    header = ["statistic", "value"]
    rows = [["Spearman(MAE, achieved E')", f"{r1['spearman_mae_eq']:.3f} (p={r1['spearman_p']:.1e})"],
            ["MAE ~ log(E') R²", f"{r1['r2_mae_logEq']:.3f}"],
            ["schedule-shape residual (MAE)", f"{r1['sched_residual_mean']:+.4f} ± {r1['sched_residual_std']:.4f}"],
            ["no-loss baseline MAE", f"{r1['baseline_mae']:.4f} (lowest)"]]
    _csv("r1_matched.csv", header, rows)
    add("R1 — matched-enforcement RQ1 null",
        "enforcement explains 80% of MAE; schedule shape adds ~0; 'scheduling helps' was a confound",
        _htable(header, rows) + _img("rq1_matched.png"), _mdtable(header, rows))

# ---------------- E6 MoCap ----------------
e6 = _load("transfer_e6.json")
if e6:
    om = e6["mean_ood_mse"]
    header = ["group", "OOD-MSE", "degrade×"]
    rows = [[gname, f"{om[gname]:.2e}", f"{e6['mean_degrade'][gname]:.2f}"] for gname in om]
    rows.append(["no-loss", f"{e6['no_loss_ood_mse']:.2e}", "—"])
    _csv("e6_mocap.csv", header, rows)
    sub = (f"matched SO(2)_y most robust · vs wrong-axis p={e6['matched_vs_wrong_p']:.3f}, "
           f"vs SO(3) p={e6['matched_vs_so3_p']:.3f}, vs no-loss p={e6['matched_vs_noloss_p']:.3f}")
    add("E6 — CMU MoCap (real data) OOD robustness", sub,
        _htable(header, rows) + _img("mocap_ood.png"), sub + "\n\n" + _mdtable(header, rows))

# ---------------- Model-zoo × dataset grid ----------------
# ---------------- DA vs REMUL Pareto (#1) ----------------
par = _load("pareto_da_remul.json")
if par:
    order = ["nl", "da_so2_z", "remul_so2_z", "da_so3", "remul_so3", "da_so2_x", "remul_so2_x"]
    plab = {"nl": "no-loss", "da_so2_z": "DA · matched", "remul_so2_z": "REMUL · matched",
            "da_so3": "DA · over SO(3)", "remul_so3": "REMUL · over SO(3)",
            "da_so2_x": "DA · wrong axis", "remul_so2_x": "REMUL · wrong axis"}
    header = ["method", "in-dist MSE", "OOD MSE"]
    rows = [[plab[k], f"{par[k]['in']:.2e}", f"{par[k]['ood']:.2e}"] for k in order if k in par]
    _csv("pareto_da_remul.csv", header, rows)
    add("DA vs REMUL — compute-matched premise test",
        "Known symmetry: augmentation WINS OOD (3.7×). REMUL's niche is graceful degradation under a wrong/over group "
        "(2.1× better in-dist than DA-SO(3)) + it is the vehicle for group selection. Don't claim 'soft loss beats augmentation'.",
        _htable(header, rows) + _img("pareto_da_remul.png"), _mdtable(header, rows))

# ---------------- Cross-backbone replication (#3) ----------------
rep = _load("replicate.json")
if rep:
    header = ["backbone", "matched OOD", "wrong-axis OOD", "over-SO3 OOD", "no-loss OOD", "matched<wrong p", "matched<over p"]
    rows = []
    for bb in ["mlp", "transformer", "gnn"]:
        if bb in rep:
            c = rep[bb]
            rows.append([bb, f"{c['so2_z']['ood']:.2e}", f"{c['so2_x']['ood']:.2e}", f"{c['so3']['ood']:.2e}",
                         f"{c['nl']['ood']:.2e}", f"{c['p_matched_vs_wrong']:.3f}", f"{c['p_matched_vs_over']:.3f}"])
    _csv("replicate.csv", header, rows)
    add("Cross-backbone replication (OOD robustness)",
        "matched residual symmetry is the most OOD-robust on mlp, transformer AND gnn (matched ≪ wrong-axis, p≤0.001 everywhere; 5 seeds each)",
        _htable(header, rows) + _img("replicate_ood.png"), _mdtable(header, rows))

# ---------------- DA-selection vs REMUL-selection ----------------
sel = _load("selcompare.json")
if sel:
    header = ["selection method", "z-field", "x-field", "overall recovery"]
    rows = [["DA-selection", f"{sel['da']['axis_z']:.0%}", f"{sel['da']['axis_x']:.0%}", f"{sel['da']['overall']:.0%} (≈chance)"],
            ["REMUL-selection", f"{sel['remul']['axis_z']:.0%}", f"{sel['remul']['axis_x']:.0%}", f"{sel['remul']['overall']:.0%}"]]
    _csv("selcompare.csv", header, rows)
    add("Group selection: REMUL vs augmentation",
        "You cannot 'select an augmentation' — a wrong DA still fits validation (38%, ≈chance). "
        "REMUL-selection recovers the true residual axis 88% — a wrong penalty conflicts with the task and cleanly raises val error.",
        _htable(header, rows) + _img("selcompare.png"), _mdtable(header, rows))

# ---------------- Data-scaling (#7) ----------------
ds = _load("datascale.json")
if ds:
    ns = ["200", "500", "1000", "2000", "4000"]
    header = ["arm"] + [f"n={n}" for n in ns]
    lab = {"nl": "no-loss", "so2_z": "matched", "so2_x": "wrong-axis", "so3": "over-SO3"}
    rows = [[lab[a]] + [f"{ds[a][n]:.2e}" for n in ns] for a in ["nl", "so2_z", "so2_x", "so3"]]
    _csv("datascale.csv", header, rows)
    add("Data-scaling — is the OOD advantage a small-data artifact?",
        "No. The matched-group OOD advantage is ~1500-2200x vs no-loss and FLAT across a 20x range of n_train "
        "(no-loss never sees the OOD orientations regardless of data size).",
        _htable(header, rows) + _img("datascale.png"), _mdtable(header, rows))

# ---------------- Baselines (#2) ----------------
bl = _load("baselines.json")
if bl:
    header = ["method", "recovers residual axis?", "OOD MSE", "needs known group?"]
    rows = [[k, v["recovers_axis"], (f"{v['ood']:.2e}" if isinstance(v["ood"], (int, float)) else str(v["ood"])), v["needs_known_group"]] for k, v in bl.items()]
    _csv("baselines.csv", header, rows)
    add("Baselines — who can recover WHICH symmetry?",
        "REMUL recovers the residual axis (88% select / 100% learnable) via its ground-truth-anchored loss. Augerino CANNOT (isotropic at every λ) "
        "because augmentation-based discovery conflates task-symmetry with input-distribution symmetry. DA is strongest IF the group is known.",
        _htable(header, rows) + _img("augerino_theta.png"), _mdtable(header, rows))

# ---------------- Label-breaks-symmetry wedge (LieGAN) ----------------
wj = _load("wedge.json"); mat = _load("baselines_matrix.json")
if wj and mat:
    gh = ["group", "LieGAN input-MMD", "task-equiv error", "is task symmetry?"]
    gr = [[g, f"{wj['liegan_input_mmd'][g]:.4f}", f"{wj['task_equiv_err'][g]:.4f}", "YES" if wj['task_equiv_err'][g] < 0.05 else "no"] for g in ["so2_z", "so2_x", "so3"]]
    _csv("wedge.csv", gh, gr)
    mh = ["method (basis)", "fixed-frame inputs", "isotropic inputs"]
    mr = [[f"{k}", v["fixed_frame"], v["isotropic"]] for k, v in mat.items()]
    _csv("baselines_matrix.csv", mh, mr)
    add("Label-breaks-symmetry wedge — REMUL vs LieGAN vs Augerino",
        "Isotropic inputs (SO(3)-symmetric p(x)) + z-field (SO(2)_z task). LieGAN discovers the INPUT symmetry SO(3) and misses the label-breaking; "
        "REMUL anchors on the TASK (‖f(Rx)−Ry‖, so2_z err=0.0000). REMUL is the ONLY method that recovers the task symmetry on BOTH fixed-frame AND isotropic inputs.",
        _htable(gh, gr) + _img("wedge.png") + _htable(mh, mr),
        _mdtable(gh, gr) + "\n\n" + _mdtable(mh, mr))

# ---------------- #4/#5 Graph OOD (scheduling null + equivariance robustness) ----------------
so=_load("sched_ood.json"); go=_load("group_ood_graph.json")
if so and go:
    header=["question","result"]
    rows=[["#4 scheduling null on OOD (QM9)", f"Spearman(OOD-MAE,E')={so['spearman_oodmae_eq']:.2f} (p={so['spearman_p']:.0e}); schedule-shape residual {so['sched_resid_mean']:+.4f}±{so['sched_resid_std']:.4f} -> null holds on OOD"],
          ["#5 equivariance->rotation robustness (QM9)", f"OOD-gap vs E_prime Spearman={go['spearman_gap_eprime']:.2f} (p={go['p']:.0e}); gap {go['gap_no_loss']:.3f} (no-loss) -> {go['gap_most_equiv']:.4f} (enforced), ~27x"],
          ["#5 scope (honest)","group-SELECTION story is dynamics-specific: QM9 is fully SO(3)-symmetric (no partial symmetry to select); 2nd graph dataset blocked (pyg-lib/MD17-graph)"]]
    _csv("graph_ood.csv",header,rows)
    add("#4/#5 — graphs: scheduling null on OOD + equivariance robustness",
        "The scheduling null extends to OOD (not in-distribution-only). Enforcing the symmetry reduces the rotation-OOD gap ~27x on graphs (payoff direction transfers). Group-selection is dynamics-specific (QM9 fully symmetric).",
        _htable(header,rows)+_img("graph_ood.png"), _mdtable(header,rows))

zoo = _load("zoo_grid.json")
if zoo:
    dsym = zoo.get("dataset_sym", {})
    header = ["model (equiv class)"] + [f"{d}" for d in zoo["datasets"]]
    rows = []
    for m in zoo["models"]:
        row = [f"{m} · {zoo['classes'].get(m,'')}"]
        for d in zoo["datasets"]:
            cell = zoo["table"].get(m, {}).get(d)
            if not cell:
                row.append("—")
            else:
                row.append(f"{cell['mse']:.2e}" + (" ⚠" if cell.get("diverged") else ""))
        rows.append(row)
    _csv("zoo_grid.csv", header, rows)
    # equivariance-error spectrum table
    eh = ["model", "equiv class", "equivariance error E'"]
    er = [[m, zoo["classes"].get(m, ""), f"{zoo['eprime_by_model'][m]:.2e}"]
          for m in sorted(zoo["models"], key=lambda x: zoo["eprime_by_model"][x])]
    _csv("zoo_eprime.csv", eh, er)
    figs = _img("zoo_spectrum.png") + "".join(_img(f"zoo_{d}.png") for d in zoo["datasets"])
    symnote = " · ".join(f"{d}={dsym.get(d,'')}" for d in zoo["datasets"])
    add("Model-zoo × dataset grid",
        "strict-equivariant models are strong on clean SO(3) tasks but DIVERGE (⚠) on real partial-symmetry data (mocap); "
        "unconstrained &amp; relaxed are robust across the spectrum. E' orders the models: strict≈1e-7 → soft → unconstrained≈1.",
        f"<p class='sub'>datasets: {symnote}</p>"
        + _htable(header, rows) + _htable(eh, er) + figs,
        _mdtable(header, rows) + "\n\n**Equivariance spectrum (E')**\n\n" + _mdtable(eh, er))


# ---------------- write RESULTS.md ----------------
md = "# REMUL soft-equivariance — results\n\n"
for title, sub, _, mdt in SECTIONS:
    md += f"## {title}\n\n*{sub}*\n\n{mdt}\n\n"
open(os.path.join(R, "RESULTS.md"), "w").write(md)

# ---------------- write dashboard.html ----------------
body = ""
for i, (title, sub, html, _) in enumerate(SECTIONS, 1):
    body += (f"<section><div class='sh'><span class='n'>{i:02d}</span><h2>{title}</h2></div>"
             f"<p class='sub'>{sub}</p>{html}</section>")

CSS = """
:root{--bg:#eef1f4;--panel:#fff;--panel2:#f6f9fb;--ink:#141a21;--muted:#54606c;--faint:#87929f;
--line:#e0e5ea;--line2:#ccd4dc;--accent:#0e7c86;--accent2:#0a5b63;--good:#12855f;--goodb:#e3f4ec;--bad:#cc3b57;--badb:#fbe8ec;
--mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;--sans:system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif;}
@media(prefers-color-scheme:dark){:root{--bg:#0b0f14;--panel:#121820;--panel2:#0e141b;--ink:#e7ecf2;--muted:#9aa6b2;--faint:#6b7682;
--line:#212a34;--line2:#2f3a46;--accent:#3fb6bf;--accent2:#7ad4da;--good:#3fc088;--goodb:#10241b;--bad:#ef6f89;--badb:#2a1720;}}
:root[data-theme=light]{--bg:#eef1f4;--panel:#fff;--panel2:#f6f9fb;--ink:#141a21;--muted:#54606c;--line:#e0e5ea;--line2:#ccd4dc;--accent:#0e7c86;--good:#12855f;--goodb:#e3f4ec;--bad:#cc3b57;--badb:#fbe8ec;}
:root[data-theme=dark]{--bg:#0b0f14;--panel:#121820;--panel2:#0e141b;--ink:#e7ecf2;--muted:#9aa6b2;--line:#212a34;--line2:#2f3a46;--accent:#3fb6bf;--good:#3fc088;--goodb:#10241b;--bad:#ef6f89;--badb:#2a1720;}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.6;font-size:15.5px}
.wrap{max-width:1000px;margin:0 auto;padding:0 24px 60px}
header.top{border-bottom:1px solid var(--line);padding:40px 24px;background:linear-gradient(180deg,color-mix(in oklab,var(--accent) 8%,transparent),transparent 60%)}
.top-in{max-width:1000px;margin:0 auto}
h1{font-size:clamp(24px,4vw,34px);margin:0;font-weight:750;letter-spacing:-.02em}
.lead{color:var(--muted);max-width:70ch;margin:12px 0 0}
section{padding:34px 0 6px;border-bottom:1px solid var(--line)}
.sh{display:flex;align-items:baseline;gap:12px}.sh h2{font-size:clamp(18px,2.6vw,23px);margin:0;font-weight:700}
.sh .n{font-family:var(--mono);color:var(--faint);font-size:13px;font-weight:600}
.sub{color:var(--muted);margin:4px 0 14px;font-size:14px}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--panel);margin:6px 0 14px}
table{border-collapse:collapse;width:100%;font-family:var(--mono);font-variant-numeric:tabular-nums;font-size:12.5px;min-width:520px}
thead th{position:sticky;top:0;background:var(--panel2);text-align:left;color:var(--muted);font-weight:600;font-size:10.5px;
text-transform:uppercase;letter-spacing:.05em;padding:9px 12px;border-bottom:1px solid var(--line2);white-space:nowrap}
tbody td{padding:8px 12px;border-bottom:1px solid var(--line);white-space:nowrap}tbody tr:last-child td{border-bottom:none}
tbody tr.ok td{background:var(--goodb)}tbody tr.bad td{background:var(--badb)}
.fig{margin:10px 0;text-align:center}.fig img{max-width:100%;height:auto;border-radius:8px}
.chip{display:inline-block;font-family:var(--mono);font-size:11px;padding:3px 9px;border-radius:999px;background:var(--goodb);color:var(--good);font-weight:700}
"""
HTML = (f"<title>REMUL — Local Results Dashboard</title>"
        f"<meta name='viewport' content='width=device-width,initial-scale=1'><style>{CSS}</style>"
        f"<header class='top'><div class='top-in'><h1>Soft Equivariance — Results Dashboard</h1>"
        f"<p class='lead'>Local, self-contained summary of the theory gate and all deep-net experiments "
        f"(bias–variance theory of soft equivariance; loss-side group selection). "
        f"<span class='chip'>{len(SECTIONS)} experiments</span></p></div></header>"
        f"<div class='wrap'>{body}</div>")
open(os.path.join(R, "dashboard.html"), "w").write(HTML)

print(f"wrote {R}/dashboard.html ({len(SECTIONS)} sections), {R}/RESULTS.md, {R}/tables/*.csv")
for t, *_ in SECTIONS:
    print("  •", t)
