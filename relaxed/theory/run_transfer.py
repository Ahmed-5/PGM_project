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
    ap.add_argument("experiments", nargs="+", choices=["e1", "e3", "e1v2", "e3ext"])
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
