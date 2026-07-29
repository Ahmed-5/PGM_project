# AGENTS.md — Relaxed Equivariant Graph Neural Networks

Guidance for AI coding agents working in this repository. It assumes no prior
knowledge of the project.

## Project overview

This repository contains **one unified package (`relaxed/`)** plus the two
legacy code paths it merges, all around the theme of *relaxed / approximate
equivariance* in graph neural networks:

0. **`relaxed/` package (NEW — preferred entry point)** — unifies paths 1 and 2
   behind a single config, dotted-flag CLI (`python -m relaxed.cli`), loss
   library (both the ground-truth-anchored REMUL term *and* the layer-wise
   functional term), model/dataset registries, and a schema-v2 run-reporting
   system (`record.json` + `history.jsonl` per run; `python -m relaxed.collect`
   aggregates every run ever made, legacy included). Two engines behind
   `--data.name`: `dynamics` (N-body/MoCap/MD17 trajectories) and `graph`
   (ZINC/QM9/QM7b/MD17/ModelNet40). New experiment logic: `--run.seeds "0 1 2"`
   multi-seed loops, `--train.grad_clip`, `--schedule.lr_schedule`,
   `--train.per_axis_eval` (paper App. D.5), `python -m relaxed.benchmark`
   (paper §6.4 timing). The legacy paths below stay fully functional (a running
   long suite depends on them) and become thin wrappers once it completes;
   **do not edit `remul/` or the top-level modules while the suite runs**.

1. **`remul/` package (legacy)** — a faithful reproduction of the dynamics experiments from
   the paper *"Relaxed Equivariance via Multitask Learning"* (REMUL, Elhag et al.,
   [arXiv:2410.17878](https://arxiv.org/abs/2410.17878)). It trains *unconstrained*
   models (Transformer, MLP, GNN) with the multitask objective
   `L_total = alpha * L_obj + beta * L_equi`, where the training equivariance term is
   **ground-truth anchored**: `L_equi = E_g || f(phi(g) x) - rho(g) y ||` (paper
   Appendix C.1) — *not* `|| f(g x) - g f(x) ||`. The functional equivariance errors
   `E` (Eq. 8) and `E'` (Eq. 9) are label-free and reported at evaluation only.
   Datasets: N-body (synthetic), CMU MoCap (subjects 35 & 9), MD17 (8 molecules).
   **To reproduce the paper's Tables 1–3, use `remul/`** — see `remul/README.md`.

2. **Top-level modules (legacy)** — a separate research extension (the group's own work,
   "REMUL-Extension") on molecular **graph property regression** (ZINC, QM9, QM7b,
   MD17, ModelNet40). It trains a unified `BaseGNN` (13 architectures, from GCN to
   EGNN/NequIP) with a **layer-wise** equivariance loss
   `L_eq = || f(g·x) - g·f(x) ||²` applied to intermediate layer outputs, weighted by
   a **depth-adaptive schedule** (`DepthScheduler`: constant, linear/exponential
   decay/increase, U-shaped, or learnable per-layer weights). This pipeline is *not*
   the paper setup.

## Technology stack

- **Language:** Python 3 (developed on 3.12; conda recipe in `create_env.sh` pins 3.10).
- **Core:** PyTorch + PyTorch Geometric (PyG), `torch-scatter`, `torch-sparse`,
  `torch-cluster`, `e3nn` (steerable models).
- **Logging:** Weights & Biases and/or TensorBoard (both optional).
- **Utilities:** NumPy, Matplotlib, tqdm, pydantic (only used to silence a warning).
- There is **no `pyproject.toml` / `setup.py`**; dependencies live in
  `requirements.txt` (note: it pins `torch==2.8.0`, while the current `.venv`
  actually has `torch 2.6.0+cu124` — treat `requirements.txt` as approximate).
- No packaging, no CI configuration, no linter/formatter configs in the repo.

## Environment setup

- The project runs from the **`.venv/` virtual environment** at the repo root
  (system Python has no torch). Scripts invoke plain `python`, so activate it first:
  `source .venv/bin/activate`.
- Full install + REMUL dataset download: `bash setup_remul.sh`
  (creates `.venv` if missing, installs `requirements.txt`, then runs
  `python -m remul.download`; use `--skip-download` to install only).
- REMUL datasets only: `python -m remul.download`
  (MD17 × 8 molecules ~3.3 GB + CMU MoCap ~18 MB into `data/remul/`; N-body is
  synthetic, nothing to download). ZINC/QM9/etc. are downloaded on first use by PyG
  into `data/` (already present under `data/raw`, `data/subset`).
- Hardware: training is GPU-oriented (the machine has NVIDIA H200s); `config.py`
  auto-selects `cuda` when available. REMUL smoke tests run on CPU.

### torch-scatter (resolved)

`import torch_scatter` previously failed inside `.venv` (ABI mismatch against
torch 2.6.0+cu124), which broke every top-level import. **Fixed** by installing
the matching wheel:
`pip install --force-reinstall --no-deps torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu124.html`
(now `2.1.2+pt26cu124`). Verify with `python test_gnns.py`. If it ever breaks
again after a torch upgrade, reinstall the matching wheel from
`https://data.pyg.org/whl/` or `pip install --no-build-isolation torch_scatter`.

## How to run

### Unified framework (`relaxed/`) — preferred

```bash
# Dynamics (REMUL paper task): single run
python -m relaxed.cli --data.name nbody --model.name transformer \
    --model.channels 384 --model.num_layers 10 --train.mode remul \
    --train.penalty constant --train.beta 1.0 --train.max_steps 50000 \
    --train.batch_size 64 --train.lr 3e-4 --train.device cuda

# Multi-seed (mean±std via relaxed.collect), stability knobs, per-axis eval
python -m relaxed.cli --data.name nbody $TF --train.mode remul --train.beta 10.0 \
    --train.grad_clip 1.0 --schedule.lr_schedule cosine --run.seeds "0 1 2"

# Graph (layer-wise equivariance): QM9 group x schedule ablations
python -m relaxed.cli --data.name QM9 --data.use_positions true \
    --model.name gcn --model.use_pos true --model.in_channels 11 \
    --loss.formulation layerwise --loss.symmetry_groups "['so3','translation']" \
    --loss.layer_weight_strategy constant --schedule.alpha_0 1.0 --train.epochs 50

# Aggregate every run ever made (both legacy formats + record.json)
python -m relaxed.collect                      # inventory + paper tables
python -m relaxed.benchmark                    # paper §6.4 timing -> results/benchmark.csv
```

- **Config**: `relaxed/config.py` — one `ExperimentConfig`
  (`data/model/loss/schedule/train/log/run`); the task engine is inferred from
  `data.name` (`dynamics`: nbody, nbody_egnn, md17_dyn, mocap; `graph`: ZINC,
  QM9, QM7b, MD17, ModelNet40). Dotted flags override any field;
  `relaxed/adapt.py` maps them onto the legacy family configs **preserving each
  family's defaults for anything not explicitly set** (parity: dynamics matches
  the legacy loop bit-for-bit; graph matches within GPU scatter-atomic noise).
- **Losses**: `relaxed/losses.py` — `RemulLoss` (ground-truth anchored),
  fixed `GradNorm`, label-free E/E′ (+ `so2_x/y/z` per-axis), and
  `LayerEquivarianceLoss` re-exported from `equivariance_loss.py` (single
  source of truth).
- **Engines**: `relaxed/engines/dynamics.py` (adds `train.grad_clip`,
  `schedule.lr_schedule`, metric-honoring eval) and `relaxed/engines/graph.py`
  (adds trailing-accumulation flush, single metric accumulation, learnable
  DepthScheduler inverse-softplus init).
- **Reporting**: every run writes `record.json` (schema v2: slug, status,
  git/env, seed, full config, flattened hyperparams, test/OOD metrics incl.
  equivariance numbers, timing) + `history.jsonl` + a final-model checkpoint
  (dynamics) under the run dir; `relaxed/reporting.py` reads all legacy
  formats too.
- **Experiment drivers**: `run_paper_grid.sh` — the exhaustive paper grid
  (Tables 1/2/3/8, all paper models × datasets × modes × β × penalties) with
  `TABLES`/`ONLY`/`BETAS`/`PENALTIES`/`SEEDS`/`SMOKE`/`DEVICE` switches; the
  gap cells not covered by the legacy suite currently run in tmux session
  `paper_grid_gap`.
- **Tests**: `python relaxed/tests/test_relaxed.py` (11 checks: geometry,
  GradNorm purity, loss anchoring, config, reporting).

### REMUL paper reproduction (`remul/`)

```bash
# Quick end-to-end smoke test (a few steps, CPU, ~4 min)
SMOKE=1 bash remul/run_experiments.sh

# Full paper experiments (GPU-scale, paper hyperparameters)
DEVICE=cuda bash remul/run_experiments.sh

# Single experiment; any --section.field value overrides the config
python -m remul.cli --data.name md17 --data.molecule aspirin \
    --model.name gnn --train.mode remul --train.penalty constant --train.beta 1.0 \
    --train.epochs 500 --train.batch_size 200 --train.lr 5e-4 --train.device cuda

# Aggregate finished runs into paper-style tables / CSV
python -m remul.collect_results
python -m remul.collect_results --csv results/remul_summary.csv
```

REMUL training modes: `--train.mode standard` (task loss only), `da` (data
augmentation with rotated input/target), `remul` with `--train.penalty constant`
(fixed `beta`) or `gradual` (GradNorm-adapted weights). The paper sweeps
`--train.beta` in {0.01, 0.1, 1.0, 10.0, 100.0}.

Each run writes `config.json` (full config + final metrics) under
`outputs/remul/<dataset>_<model>_<mode>_…_<timestamp>/`; TensorBoard scalars/hparams
are enabled with `LOG_FLAGS="--log.logger_type tensorboard --log.log_every 50"`.

### Top-level pipeline (ZINC/QM9 graph regression + layer-wise equivariance)

Preferred entry point is the dataclass-driven CLI (`cli.py`); any nested config
field is settable via dotted flags:

```bash
python cli.py --experiment_name "ZINC_GCN_Learnable" \
  --data.dataset_name ZINC --model.model_type gcn \
  --model.hidden_channels 128 --model.num_layers 6 --training.num_epochs 100 \
  --equivariance.stochastic_probability 0.25 \
  --equivariance.symmetry_groups "['so3', 'translation']" \
  --equivariance.group_weights "{'so3': 0.5, 'translation': 0.5}" \
  --equivariance.layer_weight_strategy learnable --scheduler.alpha_0 0.1 \
  --training.accumulation_steps 2 --logging.logger_type wandb
```

Notes on the CLI: list fields accept space-separated tokens or a Python/JSON list
literal; dict fields accept a Python/JSON dict literal (parsed with
`ast.literal_eval`/`json.loads`); booleans accept `true/1/t` **and** `false/0/f`
(`str2bool`). Overrides are applied before directory creation/validation
(`config.finalize()`), so `--experiment_name` names the run dirs. Real experiment
invocations used by the team are collected in `cmds.txt` / `exps.txt`, and
`run_experiments.sh` runs the 5-run scheduling-strategy ablation
(baseline / constant / exponential / linear / learnable) on ZINC+GCN — with the
**permutation** group (the only meaningful one on ZINC, which has no 3D coords).

For the geometric group × schedule comparison (the "which set of groups and
which schedule is better" question) use QM9, which has 3D positions:

```bash
# Group-set × schedule ablation on QM9 (GCN+pos unconstrained + EGNN control)
EPOCHS=50 bash run_group_schedule_ablation.sh        # EPOCHS=3 for a validation pass
python collect_ablation_results.py --pattern 'QM9_' --csv results/group_schedule_ablation.csv
```

Key data fact: ZINC has no `pos`; geometric groups (`so3`, `translation`, …)
only give a non-vacuous signal on datasets with 3D coordinates (QM9/MD17/
ModelNet40) and a position-consuming model (`--model.use_pos true` for
gcn/gin/graphsage, or a position-aware architecture).

Alternative entry point: `python train.py --config {default|baseline|e3_equivariant|multi_symmetry} --logger {wandb|tensorboard|none}`
(these four presets are the only valid ones; `run_tensorboard.sh` / `run_wandb.sh`
are usage examples).

Artifacts: checkpoints go to `checkpoints/<experiment>_<timestamp>/`
(`<experiment>_best.pt` + config JSON + DepthScheduler state), logs/outputs to
`outputs/<experiment>_<timestamp>/` (incl. `test_metrics.json` for aggregation),
TensorBoard runs to `./runs/`.

## Code organization

### Top-level pipeline (flat modules, no package)

- `config.py` — `ExperimentConfig` = nested dataclasses `ModelConfig`,
  `EquivarianceLossConfig`, `SchedulerConfig`, `TrainingConfig`, `DataConfig`,
  `LoggingConfig` with `__post_init__` validation (e.g. position-aware models
  require `data.use_positions=True`); `get_config(preset)` factory; JSON save/load.
- `cli.py` — generic argparse CLI auto-generated from the config dataclasses;
  applies overrides then calls `train(config)`.
- `train.py` — training loop: mixed precision (AMP), gradient accumulation/clipping,
  stochastic equivariance loss (applied on `stochastic_probability` of batches and
  rescaled by `1/p`), per-layer equivariance aggregation via `DepthScheduler`
  weights, early stopping, best-checkpoint-then-test protocol. Task loss is MSE;
  metrics are MAE/RMSE/R². Equivariance passes run under `model.eval()` so dropout/
  BatchNorm are deterministic while gradients still flow. `evaluate()` uses the
  same DepthScheduler layer weights as training (consistent selection objective);
  test metrics are written to `outputs/<run>/test_metrics.json`.
- `equivariant_gnn.py` — `BaseGNN`, a single class implementing all 13 model types
  (`raw_mlp`, `transformer`, `gcn`, `gin`, `graphsage`, `schnet`, `dimenet`, `egnn`,
  `painn`, `vector_neuron`, `se3_transformer`, `nequip`, `clofnet`) via builder and
  forward-method dispatch dicts; `forward(x, pos, edge_index, batch,
  return_layer_outputs=..., return_node_embeddings=...)` exposes intermediate layer
  outputs for the layer-wise loss **with gradients attached** (required for the
  layer-wise loss to train). `use_pos=True` concatenates positions to the input
  features of gcn/gin/graphsage. NOTE: `se3_transformer`/`nequip`/`dimenet` are
  non-equivariant placeholder implementations (global attention / unused geometry
  / zeroed angles) — `get_symmetry_info` says so and `__init__` warns; do not draw
  equivariance conclusions from them. (`graphormer`/`equiformer` entries exist in
  the dispatch tables but are not registered model types.)
- `equivariance_loss.py` — `EquivarianceLoss` for 8 groups (`permutation`, `so3`,
  `o3`, `se3`, `e3`, `translation`, `reflection`, `scaling`); GPU-vectorized group
  sampling; optional `torch.compile` (off by default, needs Triton). Semantics:
  with the default `feature_type='invariant'` the loss is a layer-wise
  **invariance** penalty on hidden scalar features `||f_l(g·x) − f_l(x)||²`
  (correct for invariant regression targets); permutation always compares against
  the permuted node reference; vector features `[N,H,3]` (PaiNN/vector_neuron)
  are rotated by `R` under `feature_type='equivariant'`.
- `schedulers.py` — `DepthScheduler` (layer-wise equivariance weights; aliases
  `linear_decay`→`linear`, `exp_decay`→`exponential`), `WarmupScheduler`,
  `CompositeScheduler`, LR-scheduler factories.
- `load_dataset.py` — dataset registry/dispatch (`ZINC`, `QM9`, `QM7b`, `MD17`,
  `ModelNet40` have real loaders; other names listed in `DataConfig` are **not**
  implemented), target normalization (z-score statistics computed on the **train
  split only** for QM9/MD17, avoiding val/test leakage), transforms
  (`AtomDegreeOneHot`, `AtomicNumberToOneHot`), `DATASET_SYMMETRIES` metadata.
- `collect_ablation_results.py` — aggregates `checkpoints/<exp>_*/` configs +
  `outputs/<exp>_*/test_metrics.json` into a comparison table / CSV
  (`--pattern 'QM9_'`); `run_group_schedule_ablation.sh` produces those runs.
- `rewiring.py` — `GraphRewiring` (`spectral`, `geometric` strategies) used as a
  dataset transform for the rewiring ablations.
- `utils.py` — seeding, checkpoint save/load, `EarlyStopping`, metrics (JIT-compiled
  MAE/RMSE/R²), misc helpers. `logger.py` — `get_logger` factory over
  WandB/TensorBoard/composite('both')/no-op loggers + `MetricsTracker`.
- `baseline_gnn.py` — older standalone GCN/GIN/SAGE class with layer-output
  tracking; superseded by `equivariant_gnn.BaseGNN` (kept for reference, not wired
  into `train.py`). `relaxed_gnn.py` was deleted.
- `get_time_analysis.py` — profiling variant of the training loop (function-level
  timing, GPU memory, HTML/CSV/JSON reports).

### `remul/` package (paper reproduction)

- `remul/config.py` — own `ExperimentConfig` (sections `data`, `model`, `train`,
  `log`); deliberately separate from top-level `config.py`.
- `remul/cli.py` — `python -m remul.cli --section.field value` overrides.
- `remul/train.py` — training loop with modes `standard` / `da` / `remul`
  (`constant` or GradNorm `gradual` penalty); reports test/OOD MSE plus `E`/`E'`.
  When `train.max_steps` is set it drives the epoch count (N-body's 50k-step
  budget is actually reached); per-epoch evals are strided (≤50 per run) with
  E/E′ averaged over up to `train.eval_equiv_batches` batches, and the final
  test/OOD eval always uses the full split.
- `remul/losses.py` — `RemulLoss` (ground-truth-anchored equivariance term),
  `GradNorm` (updates task weights only — model gradients are untouched, matching
  Algorithm 1; SGD weight optimizer), label-free `equivariance_error` (E / E').
- `remul/geometry.py` — SO(3) rotation helpers.
- `remul/datasets/` — `nbody.py` (synthetic 4-/5-body), `md17.py`,
  `motion_capture.py` (CMU ASF/AMC + forward kinematics), `common.py`
  (`DynamicsDataset`, loaders); dispatched by `build_datasets`.
- `remul/models/` — registry (`build_model`): unconstrained `transformer`, `mlp`,
  `gnn`; equivariant baselines `egnn`, `se3_transformer`/`tfn` (e3nn), `gatr`,
  `egno`, `hegnn`, `gmn`, `mpnn`, `emlp`, `rpp`, `per`. The Transformer/MLP/GNN +
  EGNN comparison is faithful to the paper; the other equivariant baselines are
  compact reimplementations of each method's defining idea (see `remul/README.md`).
- `remul/download.py` — dataset downloader (`python -m remul.download`).
- `remul/run_experiments.sh` — full paper suite; each block maps to a paper table
  (Tables 1, 2, 3, 8); honors `SMOKE=1`, `DEVICE`, `LOG_FLAGS`.
- `remul/run_reduced_replication.sh` — reduced GPU subset for quick trend checks
  (N-body Transformer/EGNN @ 8k steps, MD17 aspirin GNN @ 100 epochs).
- `remul/collect_results.py` + `remul/experiment_log.py` — run-dir naming
  (`<slug>_<timestamp>`), `config.json` artifacts, result aggregation into tables
  and `results/remul_summary.csv`.

## Testing

There is **no pytest suite and no CI**. Tests are standalone scripts run directly
with the venv Python; they print pass/fail tables rather than asserting:

```bash
python test_gnns.py          # instantiates all 13 BaseGNN model types, checks forward shapes
python test_equiloss.py      # v3-API sanity checks on EquivarianceLoss (toy equivariant/non-equivariant nets, [N,H,3] vector layers, gradient flow); exits non-zero on failure
python test_load_dataset.py  # attempts to load every dataset name (needs downloads; several names are expected to fail — only ZINC/QM9/QM7b/MD17/ModelNet40 have loaders)
```

End-to-end verification for the REMUL path is the smoke run:
`SMOKE=1 bash remul/run_experiments.sh`. When changing training code, run the
relevant script(s) above plus a short training run (e.g. `--training.num_epochs 1`)
before considering the change done.

## Development conventions

- **Style:** plain PyTorch/PyG, type hints at API boundaries, docstrings on public
  classes/functions. Many files carry a "Key Optimizations" header comment —
  performance (vectorized ops, no Python loops in hot paths, GPU-native tensor
  ops, avoiding `.item()` syncs) is an explicit project concern; preserve that.
  Dictionary-based dispatch is preferred over long if/elif chains.
- Comments and docs are in English; match the surrounding style.
- **Config discipline:** all hyperparameters flow through the dataclass configs.
  Add new knobs as dataclass fields (they automatically become CLI flags in both
  CLIs). Layer-wise equivariance scheduling is owned by
  `config.equivariance.layer_weight_strategy` (single source of truth), consumed by
  `DepthScheduler`; global strength is `scheduler.alpha_0` applied through the
  per-layer weights (do not multiply it in twice).
- Datasets: add new datasets as a loader in `load_dataset.py` (top-level) or
  `remul/datasets/` (dynamics), register in the corresponding dispatch dict
  (`DATASET_LOADERS` / `build_datasets`) and metadata (`DATASET_SYMMETRIES`).
- Models: add new top-level architectures via the builder/forward dispatch dicts in
  `BaseGNN` plus the `ModelConfig` `Literal`; REMUL models via
  `remul/models/__init__.py` registry.
- Splits are seeded (`torch.Generator().manual_seed(config.seed)`), ZINC uses its
  official train/val/test splits, targets are normalized (Z-score) for QM9/MD17.
- Experiment tracking: runs are identified by `<experiment_name>_<timestamp>`;
  configs are serialized next to checkpoints. Keep `cmds.txt`/`exps.txt`-style
  command records when adding experiment families.
- Git: recent history uses Conventional-Commit-style messages
  (`feat(train): …`, `fix(train): …`); `data/`, `checkpoints/`, `wandb/` are
  gitignored. Do not commit `outputs/`, `results/`, or `.venv/` (not currently
  ignored — add them to `.gitignore` if you touch it).

## Security & operational considerations

- No secrets or credentials are stored in the repo. WandB logging uses hardcoded
  project/entity placeholders (`PGM`, `PGM_Project_wandb`) in `LoggingConfig` —
  override via `--logging.wandb_project` / `--logging.wandb_entity`, or set
  `--logging.logger_type none` (or tensorboard) when you don't want runs uploaded.
  (`LoggingConfig.wandb_mode` exists but is currently not wired into
  `wandb.init`.) `wandb login` is required for online logging.
- Dataset downloads hit external hosts (PyG mirrors, quantum-machine.org,
  mocap.cs.cmu.edu); `setup_remul.sh` and `remul/download.py` perform network
  access by design.
- Checkpoints are saved with `torch.save` and loaded with `weights_only=False`
  (`utils.load_checkpoint`) — only load checkpoints you trust.
- Full REMUL suites are long GPU jobs (paper budgets: N-body 50k steps, MoCap 2000
  epochs, MD17 500 epochs × 8 molecules); `run_experiments.sh` uses `|| true` so one
  failed run does not abort the suite. Use `SMOKE=1` before launching anything big.
