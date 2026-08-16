"""Deep-net transfer experiments for the soft-equivariance theory (Phase 1).

Runs the two highest-priority experiments from the pre-registration on the
N-body dynamics task, distributing jobs 2-wide across the available GPUs:

  E1  crossover scaling law  — strict EGNN vs unconstrained MLP, sweep field x
      n_train; locate the crossover field s_cross(n) and fit its log-log slope
      (theory: -1/2). This is the transfer kill-gate.

  E3/E4  non-oracle group selection + isotropic control — unconstrained MLP +
      REMUL with each candidate loss group; select the group with the best
      *validation* MSE (never told the axis) and check it recovers the true
      residual symmetry. field=0 is the R=0 isotropic control (no wrong axis
      should be preferred).

Each (config, seed) is a separate job with a unique --log.log_dir and an
--run.experiment_name tag, so parallel runs never collide. Results are read back
from the per-run record.json by relaxed/theory/analyze_transfer.py.

Usage:  python -m relaxed.theory.run_transfer e1 e3   [--steps 6000] [--gpus 0,1]
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = "outputs/e1e3"
PYEXE = sys.executable

MLP = ["--model.name", "mlp", "--model.mlp_hidden", "680", "--model.num_layers", "3"]
EGNN = ["--model.name", "egnn", "--model.hidden_dim", "64", "--model.num_layers", "4"]

COMMON = [
    "--data.name", "nbody", "--data.n_val", "2000", "--data.n_test", "2000",
    "--data.num_steps", "100", "--train.batch_size", "64", "--train.lr", "3e-4",
    "--train.grad_clip", "1.0", "--schedule.lr_schedule", "cosine",
    "--train.device", "cuda", "--log.logger_type", "none",
]


def job(tag, arch, mode_args, field, axis, n_train, seed, steps):
    log_dir = os.path.join(ROOT, tag)
    args = ([PYEXE, "-m", "relaxed.cli"] + COMMON + arch + mode_args +
            ["--data.field_strength", str(field), "--data.field_axis", str(axis),
             "--data.n_train", str(n_train), "--train.max_steps", str(steps),
             "--run.seeds", str(seed), "--log.log_dir", log_dir,
             "--run.experiment_name", tag])
    return {"tag": tag, "args": args, "log_dir": log_dir}


def build_e1(steps):
    """Crossover: strict EGNN vs unconstrained MLP, field x n_train."""
    jobs = []
    ns = [150, 300, 600, 1200, 2400]
    fields = [0, 1, 2, 3, 4, 6]
    seeds = [0, 1]
    for n in ns:
        for f in fields:
            for s in seeds:
                jobs.append(job(f"e1_egnn_n{n}_f{f}_s{s}", EGNN, ["--train.mode", "standard"], f, 2, n, s, steps))
                jobs.append(job(f"e1_mlp_n{n}_f{f}_s{s}", MLP, ["--train.mode", "standard"], f, 2, n, s, steps))
    return jobs


def build_e1v2(steps):
    """Refined crossover in the VARIANCE-LIMITED (small-n) regime, with an extended
    field grid to resolve the higher crossovers. This is where the -1/2 law should
    hold; the original e1 mostly sampled the capacity-limited (saturated) regime."""
    jobs = []
    ns = [80, 120, 180, 270, 400, 600]
    fields = [3, 4, 5, 6, 7, 8, 10, 12]
    seeds = [0, 1]
    for n in ns:
        for f in fields:
            for s in seeds:
                jobs.append(job(f"e1v2_egnn_n{n}_f{f}_s{s}", EGNN, ["--train.mode", "standard"], f, 2, n, s, steps))
                jobs.append(job(f"e1v2_mlp_n{n}_f{f}_s{s}", MLP, ["--train.mode", "standard"], f, 2, n, s, steps))
    return jobs


def build_e3ext(steps):
    """More seeds for the field-broken selection cells (tighten recovery rate)."""
    jobs = []
    groups = ["so2_x", "so2_y", "so2_z", "so3"]
    seeds = [3, 4, 5, 6, 7]
    n = 600
    for field, axis in [(6, 2), (6, 0)]:
        for s in seeds:
            jobs.append(job(f"e3_f{field}_ax{axis}_std_s{s}", MLP, ["--train.mode", "standard"], field, axis, n, s, steps))
            for g in groups:
                mode = ["--train.mode", "remul", "--train.penalty", "constant", "--train.beta", "1.0", "--loss.group", g]
                jobs.append(job(f"e3_f{field}_ax{axis}_{g}_s{s}", MLP, mode, field, axis, n, s, steps))
    return jobs


def build_e3learn(steps):
    """Deep-net continuous learnable-axis discovery: a learnable SO(2) axis trained
    jointly with the model (so2_learn). Both field axes, several seeds. The record
    stores the recovered axis + its angle to the true field axis."""
    jobs = []
    seeds = [0, 1, 2, 3, 4]
    n = 600
    for axis in [2, 0]:
        for s in seeds:
            mode = ["--train.mode", "remul", "--train.penalty", "constant", "--train.beta", "1.0",
                    "--loss.group", "so2_learn", "--loss.num_group_samples", "3"]
            jobs.append(job(f"e3l_ax{axis}_s{s}", MLP, mode, 6, axis, n, s, steps))
    return jobs


def build_zoo(steps):
    """Model-zoo × dataset grid spanning the equivariance spectrum.

    Models: strict-equivariant (egnn/se3_transformer/gatr), approximate/relaxed
    architectures (emlp/rpp), unconstrained (mlp/transformer), and soft-via-loss
    (mlp+REMUL SO(3)). Datasets by inherent symmetry: N-body SO(3) (field 0),
    N-body broken→SO(2) (field 6), charged N-body (SO(3)), MD17 (E(3) molecular),
    and CMU MoCap (real, SO(2) about vertical). Tests the central thesis — strict
    models win when their symmetry matches, lose when it is broken/partial."""
    NB = ["--data.n_train", "500", "--data.n_val", "1000", "--data.n_test", "1000", "--data.num_steps", "100"]
    datasets = [
        ("nbody_so3",     ["--data.name", "nbody", "--data.field_strength", "0"] + NB),
        ("nbody_broken",  ["--data.name", "nbody", "--data.field_strength", "6", "--data.field_axis", "2"] + NB),
        ("nbody_charged", ["--data.name", "nbody_egnn"] + NB),
        ("md17",          ["--data.name", "md17_dyn", "--data.md17_molecule", "aspirin"]),
        ("mocap",         ["--data.name", "mocap", "--data.mocap_subject", "35", "--data.field_axis", "1"]),
    ]
    TF = ["--model.name", "transformer", "--model.channels", "256", "--model.num_layers", "6", "--model.num_heads", "8"]
    models = [
        ("egnn", MLP[:0] + ["--model.name", "egnn", "--model.hidden_dim", "64", "--model.num_layers", "4"], ["--train.mode", "standard"]),
        ("se3_transformer", ["--model.name", "se3_transformer"], ["--train.mode", "standard"]),
        ("gatr", ["--model.name", "gatr"], ["--train.mode", "standard"]),
        ("emlp", ["--model.name", "emlp"], ["--train.mode", "standard"]),
        ("rpp", ["--model.name", "rpp"], ["--train.mode", "standard"]),
        ("mlp", MLP, ["--train.mode", "standard"]),
        ("transformer", TF, ["--train.mode", "standard"]),
        ("mlp_remul_so3", MLP, ["--train.mode", "remul", "--train.penalty", "constant", "--train.beta", "1.0", "--loss.group", "so3"]),
    ]
    seeds = [0, 1]
    jobs = []
    for dname, dargs in datasets:
        for mname, margs, mode in models:
            for s in seeds:
                tag = f"zoo_{dname}__{mname}__s{s}"
                ld = os.path.join(ROOT, tag)
                args = ([PYEXE, "-m", "relaxed.cli"] + dargs + margs + mode +
                        ["--train.max_steps", str(steps), "--train.batch_size", "64", "--train.lr", "3e-4",
                         "--train.grad_clip", "1.0", "--schedule.lr_schedule", "cosine", "--train.device", "cuda",
                         "--log.logger_type", "none", "--run.seeds", str(s),
                         "--log.log_dir", ld, "--run.experiment_name", tag])
                jobs.append({"tag": tag, "args": args, "log_dir": ld})
    return jobs


ZOO_MODELS = [
    ("egnn", ["--model.name", "egnn", "--model.hidden_dim", "64", "--model.num_layers", "4"], ["--train.mode", "standard"]),
    ("se3_transformer", ["--model.name", "se3_transformer"], ["--train.mode", "standard"]),
    ("gatr", ["--model.name", "gatr"], ["--train.mode", "standard"]),
    ("emlp", ["--model.name", "emlp"], ["--train.mode", "standard"]),
    ("rpp", ["--model.name", "rpp"], ["--train.mode", "standard"]),
    ("mlp", MLP, ["--train.mode", "standard"]),
    ("transformer", ["--model.name", "transformer", "--model.channels", "256", "--model.num_layers", "6", "--model.num_heads", "8"], ["--train.mode", "standard"]),
    ("mlp_remul_so3", MLP, ["--train.mode", "remul", "--train.penalty", "constant", "--train.beta", "1.0", "--loss.group", "so3"]),
]


def _gjob(tag, dargs, margs, mode, steps, lr):
    ld = os.path.join(ROOT, tag)
    args = ([PYEXE, "-m", "relaxed.cli"] + dargs + margs + mode +
            ["--train.max_steps", str(steps), "--train.batch_size", "64", "--train.lr", str(lr),
             "--train.grad_clip", "1.0", "--schedule.lr_schedule", "cosine", "--train.device", "cuda",
             "--log.logger_type", "none", "--run.seeds", tag.split("__s")[-1],
             "--log.log_dir", ld, "--run.experiment_name", tag])
    return {"tag": tag, "args": args, "log_dir": ld}


def build_refit(_steps):
    """Follow-up: (1) refit MoCap+MD17 with adequate training + stable LR (the 4k-step
    divergences/under-fitting were artifacts), 3 seeds; (2) a 3rd seed on the nbody
    cells (borderline rankings). Same model configs as the zoo, so results replace/
    augment it directly (delete old zoo_mocap__/zoo_md17__ dirs before running)."""
    NB = ["--data.n_train", "500", "--data.n_val", "1000", "--data.n_test", "1000", "--data.num_steps", "100"]
    jobs = []
    # (1) refit the two under-fit datasets: 15k steps, lr 1e-4 (stable), 3 seeds
    refit = [("mocap", ["--data.name", "mocap", "--data.mocap_subject", "35", "--data.field_axis", "1"]),
             ("md17", ["--data.name", "md17_dyn", "--data.md17_molecule", "aspirin"])]
    for dname, dargs in refit:
        for mname, margs, mode in ZOO_MODELS:
            for s in [0, 1, 2]:
                jobs.append(_gjob(f"zoo_{dname}__{mname}__s{s}", dargs, margs, mode, 15000, 1e-4))
    # (2) 3rd seed for the nbody cells (4k steps, lr 3e-4, matching the original grid)
    nb = [("nbody_so3", ["--data.name", "nbody", "--data.field_strength", "0"] + NB),
          ("nbody_broken", ["--data.name", "nbody", "--data.field_strength", "6", "--data.field_axis", "2"] + NB),
          ("nbody_charged", ["--data.name", "nbody_egnn"] + NB)]
    for dname, dargs in nb:
        for mname, margs, mode in ZOO_MODELS:
            jobs.append(_gjob(f"zoo_{dname}__{mname}__s2", dargs, margs, mode, 4000, 3e-4))
    return jobs






def build_datascale(_steps):
    """#7: does the matched-group OOD advantage survive as n_train grows, or is it a
    small-data artifact? Sweep n_train for no-loss vs matched vs wrong vs over (z-field)."""
    ns = [200, 500, 1000, 2000, 4000]
    arms = [("nl", ["--train.mode", "standard"])]
    for g in ["so2_z", "so2_x", "so3"]:
        arms.append((g, ["--train.mode", "remul", "--train.penalty", "constant", "--train.beta", "1.0", "--loss.group", g]))
    jobs = []
    for n in ns:
        D = ["--data.name", "nbody", "--data.field_strength", "6", "--data.field_axis", "2",
             "--data.n_train", str(n), "--data.n_val", "1000", "--data.n_test", "1000", "--data.num_steps", "100"]
        for aname, amode in arms:
            for s in [0, 1, 2]:
                jobs.append(_gjob(f"ds_n{n}_{aname}__s{s}", D, MLP, amode, 6000, 3e-4))
    return jobs


def build_selcompare(_steps):
    """DA-selection vs REMUL-selection: select the loss group by validation MSE and
    measure recovery of the true residual axis. Tests REMUL's claim that it is the
    vehicle for selection where augmentation is not (a wrong DA corrupts training)."""
    groups = ["so2_x", "so2_y", "so2_z", "so3"]
    jobs = []
    for axis in [2, 0]:
        D = ["--data.name", "nbody", "--data.field_strength", "6", "--data.field_axis", str(axis),
             "--data.n_train", "600", "--data.n_val", "1000", "--data.n_test", "1000", "--data.num_steps", "100"]
        for s in [0, 1, 2, 3]:
            for g in groups:
                jobs.append(_gjob(f"sel_da_ax{axis}_{g}__s{s}", D, MLP, ["--train.mode", "da", "--loss.group", g], 12000, 3e-4))
                jobs.append(_gjob(f"sel_remul_ax{axis}_{g}__s{s}", D, MLP,
                                  ["--train.mode", "remul", "--train.penalty", "constant", "--train.beta", "1.0", "--loss.group", g], 6000, 3e-4))
    return jobs


def build_replicate(_steps):
    """#3: replicate the OOD group-selection headline across 3 backbones x 5 seeds.
    Field-broken (z). Arms: no-loss, REMUL matched (so2_z), wrong-axis (so2_x), over (so3)."""
    D = ["--data.name", "nbody", "--data.field_strength", "6", "--data.field_axis", "2",
         "--data.n_train", "600", "--data.n_val", "1000", "--data.n_test", "1000", "--data.num_steps", "100"]
    backbones = {
        "mlp": ["--model.name", "mlp", "--model.mlp_hidden", "680", "--model.num_layers", "3"],
        "transformer": ["--model.name", "transformer", "--model.channels", "256", "--model.num_layers", "6", "--model.num_heads", "8"],
        "gnn": ["--model.name", "gnn"],
    }
    arms = [("nl", ["--train.mode", "standard"])]
    for g in ["so2_z", "so2_x", "so3"]:
        arms.append((g, ["--train.mode", "remul", "--train.penalty", "constant", "--train.beta", "1.0", "--loss.group", g]))
    jobs = []
    for bname, bargs in backbones.items():
        for aname, amode in arms:
            for s in [0, 1, 2, 3, 4]:
                jobs.append(_gjob(f"rep_{bname}_{aname}__s{s}", D, bargs, amode, 6000, 3e-4))
    return jobs




def build_augtune2(_steps):
    """Map the Augerino transition between 3e-4 and 3e-3 to confirm no anisotropic (axis-recovering) window."""
    D = ["--data.name", "nbody", "--data.field_strength", "6", "--data.field_axis", "2",
         "--data.n_train", "600", "--data.n_val", "1000", "--data.n_test", "1000", "--data.num_steps", "100"]
    jobs = []
    for lam in ["5e-4", "8e-4", "1.5e-3"]:
        for sd in [0, 1]:
            jobs.append(_gjob(f"augt_l{lam}__s{sd}", D, MLP, ["--train.mode", "augerino", "--train.beta", lam], 4000, 3e-4))
    return jobs



def build_wedge(_steps):
    """Label-breaks-symmetry wedge: isotropic inputs (SO(3)-symmetric p(x)) + z-field
    (SO(2)_z task). REMUL per group + Augerino lambda-sweep (task-based methods should
    find SO(2)_z); the LieGAN input-symmetry proxy is computed separately (finds SO(3))."""
    D = ["--data.name", "nbody", "--data.field_strength", "6", "--data.field_axis", "2", "--data.iso_input", "true",
         "--data.n_train", "600", "--data.n_val", "1000", "--data.n_test", "1000", "--data.num_steps", "100"]
    jobs = []
    for s in [0, 1, 2]:
        jobs.append(_gjob(f"wedge_nl__s{s}", D, MLP, ["--train.mode", "standard"], 6000, 3e-4))
        for g in ["so2_x", "so2_y", "so2_z", "so3"]:
            jobs.append(_gjob(f"wedge_remul_{g}__s{s}", D, MLP,
                              ["--train.mode", "remul", "--train.penalty", "constant", "--train.beta", "1.0", "--loss.group", g], 6000, 3e-4))
    for lam in ["1e-5", "3e-5", "1e-4", "3e-4"]:
        for s in [0, 1]:
            jobs.append(_gjob(f"wedge_aug_l{lam}__s{s}", D, MLP, ["--train.mode", "augerino", "--train.beta", lam], 4000, 3e-4))
    return jobs


def build_augtune(_steps):
    """Tune Augerino's breadth lambda (its known sensitivity). z-field task; find the
    window where theta_z stays high but theta_x/theta_y collapse (axis recovered)."""
    D = ["--data.name", "nbody", "--data.field_strength", "6", "--data.field_axis", "2",
         "--data.n_train", "600", "--data.n_val", "1000", "--data.n_test", "1000", "--data.num_steps", "100"]
    jobs = []
    for lam in ["3e-6", "1e-5", "3e-5", "1e-4", "3e-4"]:
        for sd in [0, 1]:
            jobs.append(_gjob(f"augt_l{lam}__s{sd}", D, MLP, ["--train.mode", "augerino", "--train.beta", lam], 4000, 3e-4))
    return jobs


def build_pareto(_steps):
    """Compute-matched DA-vs-REMUL (the premise test): does the soft loss beat plain
    data augmentation? Field-broken N-body (residual SO(2)_z). Compute is matched by
    FORWARD PASSES — REMUL (1 group sample) does 2 forwards/step, so DA/no-loss get 2x
    the steps (6k vs 12k → 12k forwards each). We test the matched group (so2_z, where
    DA directly trains on the OOD orbit and should be strong) AND wrong/over groups
    (so3, so2_x, where hard augmentation on a false symmetry should be catastrophic but
    the soft REMUL penalty should degrade gracefully). OOD = rotations about z."""
    D = ["--data.name", "nbody", "--data.field_strength", "6", "--data.field_axis", "2",
         "--data.n_train", "600", "--data.n_val", "1000", "--data.n_test", "1000", "--data.num_steps", "100"]
    groups = ["so2_z", "so3", "so2_x"]
    jobs = []
    for s in [0, 1, 2]:
        jobs.append(_gjob(f"pareto_nl__s{s}", D, MLP, ["--train.mode", "standard"], 12000, 3e-4))
        for g in groups:
            jobs.append(_gjob(f"pareto_da_{g}__s{s}", D, MLP, ["--train.mode", "da", "--loss.group", g], 12000, 3e-4))
            jobs.append(_gjob(f"pareto_remul_{g}__s{s}", D, MLP,
                              ["--train.mode", "remul", "--train.penalty", "constant", "--train.beta", "1.0", "--loss.group", g], 6000, 3e-4))
    return jobs


def build_mocapfix(_steps):
    """Fair MoCap numbers at the STABLE config (lr 3e-4, 4k steps): the 15k/1e-4
    refit overfit the 200-sample set and destabilised even unconstrained models.
    Divergence of strict rotation-equivariant models persists across budgets."""
    dargs = ["--data.name", "mocap", "--data.mocap_subject", "35", "--data.field_axis", "1"]
    jobs = []
    for mname, margs, mode in ZOO_MODELS:
        for s in [0, 1, 2]:
            jobs.append(_gjob(f"zoo_mocap__{mname}__s{s}", dargs, margs, mode, 4000, 3e-4))
    return jobs


def build_e6mocap(steps):
    """E6 real-data: CMU MoCap (subject 35). The dynamics is SO(2)-equivariant about
    the vertical (Y) axis (heading is physically arbitrary), so we test robustness to
    Y-rotations. Compare no-loss vs matched SO(2)_y vs wrong-axis SO(2) vs SO(3).
    Transformer backbone (the paper's MoCap model; MLP under-fits mocap)."""
    jobs = []
    groups = ["so2_y", "so2_x", "so2_z", "so3"]
    seeds = [0, 1, 2]
    tf = ["--model.name", "transformer", "--model.channels", "256", "--model.num_layers", "6",
          "--model.num_heads", "8"]
    base = [PYEXE, "-m", "relaxed.cli", "--data.name", "mocap", "--data.mocap_subject", "35",
            "--data.field_axis", "1", "--train.max_steps", str(steps), "--train.batch_size", "64",
            "--train.lr", "3e-4", "--train.grad_clip", "1.0", "--schedule.lr_schedule", "cosine",
            "--train.device", "cuda", "--log.logger_type", "none"] + tf

    def mk(tag, mode):
        ld = os.path.join(ROOT, tag)
        return {"tag": tag, "args": base + mode + ["--run.seeds", str(tag[-1]),
                "--log.log_dir", ld, "--run.experiment_name", tag], "log_dir": ld}

    for s in seeds:
        jobs.append(mk(f"e6_std_s{s}", ["--train.mode", "standard"]))
        for g in groups:
            jobs.append(mk(f"e6_{g}_s{s}", ["--train.mode", "remul", "--train.penalty", "constant",
                                            "--train.beta", "1.0", "--loss.group", g]))
    return jobs


def build_e2beta(steps):
    """Interior optimum / closed-form β* (Thm 1 transfer). Sweep β for the MATCHED
    group (so2_z, R=0 → enforcement should keep helping OOD) and the OVER-
    CONSTRAINING group (so3, R>0 → expect an interior β*). field=6, z-axis."""
    jobs = []
    betas = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
    seeds = [0, 1, 2]
    n, field, axis = 600, 6, 2
    for s in seeds:
        jobs.append(job(f"e2_std_s{s}", MLP, ["--train.mode", "standard"], field, axis, n, s, steps))
        for g in ["so2_z", "so3"]:
            for b in betas:
                bs = f"{b:g}".replace(".", "p")
                mode = ["--train.mode", "remul", "--train.penalty", "constant", "--train.beta", str(b), "--loss.group", g]
                jobs.append(job(f"e2_{g}_b{bs}_s{s}", MLP, mode, field, axis, n, s, steps))
    return jobs


def build_e5ood(steps):
    """OOD robustness (#3): field-broken task, all loss groups + no-loss, evaluated
    on test data rotated about the field axis (the residual symmetry, unseen in the
    fixed-frame training set). Does enforcing the matched group pay off off-distribution?"""
    jobs = []
    groups = ["so2_x", "so2_y", "so2_z", "so3"]
    seeds = [0, 1, 2, 3, 4, 5]
    n = 600
    for field, axis in [(6, 2), (6, 0)]:
        for s in seeds:
            jobs.append(job(f"e5_f{field}_ax{axis}_std_s{s}", MLP, ["--train.mode", "standard"], field, axis, n, s, steps))
            for g in groups:
                mode = ["--train.mode", "remul", "--train.penalty", "constant", "--train.beta", "1.0", "--loss.group", g]
                jobs.append(job(f"e5_f{field}_ax{axis}_{g}_s{s}", MLP, mode, field, axis, n, s, steps))
    return jobs


def build_e3(steps):
    """Non-oracle selection (field=6, axis z & x) + isotropic control (field=0)."""
    jobs = []
    groups = ["so2_x", "so2_y", "so2_z", "so3"]
    seeds = [0, 1, 2]
    n = 600
    cells = [(6, 2), (6, 0), (0, 2)]   # (field, axis); field 0 -> isotropic control
    for field, axis in cells:
        for s in seeds:
            jobs.append(job(f"e3_f{field}_ax{axis}_std_s{s}", MLP, ["--train.mode", "standard"], field, axis, n, s, steps))
            for g in groups:
                mode = ["--train.mode", "remul", "--train.penalty", "constant", "--train.beta", "1.0", "--loss.group", g]
                jobs.append(job(f"e3_f{field}_ax{axis}_{g}_s{s}", MLP, mode, field, axis, n, s, steps))
    return jobs


def run_one(spec, gpu_q, done, total, lock):
    gpu = gpu_q.get()
    try:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), REMUL_NO_CKPT="1")
        os.makedirs(spec["log_dir"], exist_ok=True)
        t0 = time.time()
        with open(os.path.join(spec["log_dir"], "run.log"), "w") as lg:
            r = subprocess.run(spec["args"], env=env, stdout=lg, stderr=subprocess.STDOUT)
        dt = time.time() - t0
        with lock:
            done[0] += 1
            print(f"[{done[0]}/{total}] gpu{gpu} {spec['tag']}  rc={r.returncode}  {dt:.0f}s", flush=True)
        return {"tag": spec["tag"], "rc": r.returncode, "seconds": round(dt, 1)}
    finally:
        gpu_q.put(gpu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments", nargs="+", choices=["e1", "e3", "e1v2", "e3ext", "e5ood", "e2beta", "e3learn", "e6mocap", "zoo", "refit", "mocapfix", "pareto", "replicate", "selcompare", "datascale", "augtune", "augtune2", "wedge"])
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--gpus", default="0,1")
    a = ap.parse_args()

    jobs = []
    if "e1" in a.experiments:
        jobs += build_e1(a.steps)
    if "e1v2" in a.experiments:
        jobs += build_e1v2(a.steps)
    if "e3" in a.experiments:
        jobs += build_e3(a.steps)
    if "e3ext" in a.experiments:
        jobs += build_e3ext(a.steps)
    if "e5ood" in a.experiments:
        jobs += build_e5ood(a.steps)
    if "e2beta" in a.experiments:
        jobs += build_e2beta(a.steps)
    if "e3learn" in a.experiments:
        jobs += build_e3learn(a.steps)
    if "e6mocap" in a.experiments:
        jobs += build_e6mocap(a.steps)
    if "zoo" in a.experiments:
        jobs += build_zoo(a.steps)
    if "refit" in a.experiments:
        jobs += build_refit(a.steps)
    if "mocapfix" in a.experiments:
        jobs += build_mocapfix(a.steps)
    if "pareto" in a.experiments:
        jobs += build_pareto(a.steps)
    if "replicate" in a.experiments:
        jobs += build_replicate(a.steps)
    if "selcompare" in a.experiments:
        jobs += build_selcompare(a.steps)
    if "datascale" in a.experiments:
        jobs += build_datascale(a.steps)
    if "augtune" in a.experiments:
        jobs += build_augtune(a.steps)
    if "wedge" in a.experiments:
        jobs += build_wedge(a.steps)
    if "augtune2" in a.experiments:
        jobs += build_augtune2(a.steps)

    gpus = [int(x) for x in a.gpus.split(",")]
    gpu_q = queue.Queue()
    for g in gpus:
        gpu_q.put(g)
    done, lock = [0], threading.Lock()
    total = len(jobs)
    print(f"launching {total} jobs on GPUs {gpus} @ {a.steps} steps", flush=True)
    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        futs = [ex.submit(run_one, s, gpu_q, done, total, lock) for s in jobs]
        for f in futs:
            results.append(f.result())
    os.makedirs(ROOT, exist_ok=True)
    with open(os.path.join(ROOT, "manifest.json"), "w") as f:
        json.dump({"results": results, "total_seconds": round(time.time() - t0, 1)}, f, indent=2)
    bad = [r for r in results if r["rc"] != 0]
    print(f"\nDONE {total} jobs in {(time.time()-t0)/60:.1f} min · failures: {len(bad)}", flush=True)
    for r in bad:
        print("  FAIL", r["tag"], flush=True)


if __name__ == "__main__":
    main()
