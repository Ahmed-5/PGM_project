# REMUL dynamics experiments

Self-contained reproduction of the datasets, models and training procedure from
**"Relaxed Equivariance via Multitask Learning"** (Elhag, Rusch, Di Giovanni,
Bronstein — [arXiv:2410.17878](https://arxiv.org/abs/2410.17878)).

This package is intentionally separate from the repository's ZINC/QM9
graph-property pipeline: the paper's tasks are **3D dynamics / position
prediction**, which need their own data loaders, models and training loop.

## What REMUL is

REMUL trains an *unconstrained* network with a multitask objective

```
L_total = alpha * L_obj(f(x), y)  +  beta * L_equi(f, x, y, G)
```

where the **training** equivariance term is *ground-truth anchored*
(Appendix C.1):

```
L_equi = E_{g~G} || f(phi(g) x) - rho(g) y ||     (NOT || f(g x) - g f(x) ||)
```

`beta` (the lever) controls how equivariant the model becomes. The penalty can be
**constant** or **gradual** (GradNorm, Algorithm 1). The **functional**
equivariance errors `E` (Eq. 8) and `E'` (Eq. 9) are reported at evaluation only.

## Datasets (Section 6 / Appendix C)

| Dataset          | Task                                   | Split (train/val/test)        | Notes |
|------------------|----------------------------------------|-------------------------------|-------|
| `nbody`          | 4-body gravity, predict 100-step ahead | 100 / 5000 / 5000 (+OOD 5000) | synthetic; in-dist ±10°, OOD [90,180]° |
| `nbody_egnn`     | 5 charged particles (Table 8)          | 100 / 5000 / 5000             | synthetic |
| `md17`           | MD17 future 3D structure, ΔT=5000      | 500 / 2000 / 2000             | 8 molecules, downloaded from quantum-machine.org |
| `motion_capture` | CMU MoCap future trajectory, ΔT=30     | 200/600/600 (walk, subj 35), 200/240/240 (run, subj 9) | ASF/AMC downloaded + forward kinematics |

Download everything up front:

```bash
python -m remul.download                       # all 8 MD17 molecules + MoCap 35 & 9
python -m remul.download --md17-molecules ethanol aspirin
```

N-body is generated on the fly (nothing to download).

## Models

| Key                | Type            | Fidelity |
|--------------------|-----------------|----------|
| `transformer`      | unconstrained (REMUL subject) | faithful |
| `mlp`              | unconstrained | faithful |
| `gnn`              | unconstrained (EGNN's non-equivariant counterpart) | faithful |
| `egnn`             | E(3)-equivariant | faithful (Satorras et al. 2021, with velocity) |
| `se3_transformer`  | SE(3)-equivariant attention (e3nn) | compact reimplementation |
| `tfn`              | Tensor Field Network (e3nn) | compact reimplementation |
| `gatr`             | Geometric-algebra transformer | compact GA-inspired reimplementation |
| `egno`             | EGNN + Fourier operator | compact reimplementation |
| `hegnn`            | high-degree steerable EGNN | compact reimplementation |
| `gmn`              | equivariant mechanics (2nd order) | compact reimplementation |
| `mpnn`             | invariant message passing (Gilmer et al.) | compact reimplementation |
| `emlp`             | SO(3)-equivariant MLP | compact reimplementation |
| `rpp`              | Residual Pathway Priors | compact reimplementation |
| `per`              | MLP + equivariance regularizer | compact reimplementation |

The equivariant models are verified to have functional equivariance error
`E' ~ 1e-7` (numerically exact), while the unconstrained ones are far from
equivariant until trained with REMUL/DA. The **core REMUL comparison**
(Transformer / MLP / GNN + REMUL/DA vs EGNN) is a faithful reproduction; the
additional equivariant baselines (GATr, SE(3)-Transformer, EGNO, HEGNN, GMN,
EMLP, RPP, PER) are compact self-contained reimplementations that capture each
method's defining idea rather than the original authors' full architectures.

## Running

Single experiment (any `--section.field value` overrides the config):

```bash
# MD17 GNN + REMUL (constant penalty, beta=1) — the paper's MD17 setup
python -m remul.cli --data.name md17 --data.molecule aspirin \
    --model.name gnn --model.hidden_dim 64 --model.num_layers 4 \
    --train.mode remul --train.penalty constant --train.beta 1.0 \
    --train.epochs 500 --train.batch_size 200 --train.lr 5e-4 --train.device cuda

# N-body Transformer + REMUL (gradual / GradNorm)
python -m remul.cli --data.name nbody --model.name transformer \
    --model.channels 384 --model.num_layers 10 --model.num_heads 8 \
    --train.mode remul --train.penalty gradual --train.beta 10.0 \
    --train.max_steps 50000 --train.batch_size 64 --train.lr 3e-4 --train.device cuda
```

Full suite (maps each block to a paper table) — paper hyperparameters, GPU-scale:

```bash
DEVICE=cuda bash remul/run_experiments.sh
```

Quick end-to-end smoke test (a few steps, CPU):

```bash
SMOKE=1 bash remul/run_experiments.sh
```

## Training modes

* `--train.mode standard` — task loss only.
* `--train.mode da` — data augmentation (task loss on rotated input/target).
* `--train.mode remul --train.penalty constant --train.beta B` — fixed weights.
* `--train.mode remul --train.penalty gradual` — GradNorm-adapted weights.

## Compute note

This repo was validated on CPU with short smoke runs. The paper's numbers require
the Appendix-C budgets (N-body 50k steps; MoCap 2000 epochs; MD17 500 epochs) on a
GPU (`--train.device cuda`).

## TensorBoard and result aggregation

Each run writes `config.json` (full experiment config + final metrics) under
`outputs/remul/<dataset>_<model>_<mode>_…/`. Enable TensorBoard scalars/hparams with:

```bash
LOG_FLAGS="--log.logger_type tensorboard --log.log_every 50" \
  DEVICE=cuda bash remul/run_experiments.sh
tensorboard --logdir outputs/remul
```

Summarize completed runs into paper-style tables (and optional CSV):

```bash
python -m remul.collect_results
python -m remul.collect_results --csv results/remul_summary.csv
```

Legacy runs logged as `run_<timestamp>/` without `config.json` still appear in the
CSV via TensorBoard event tags, but are grouped as unlabeled until re-run.
