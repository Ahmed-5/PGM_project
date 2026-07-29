#!/bin/bash

# ==============================================================================
# Exhaustive REMUL paper-grid driver (unified relaxed CLI)
#
# Covers every model x dataset combination from the paper's model zoo and
# datasets, in the paper's table structure, with the paper's sweeps:
#   modes standard/da + REMUL with penalties {constant, gradual}
#   beta in {0.01, 0.1, 1.0, 10.0, 100.0}
#   Table 1: N-body           Table 2: MoCap (subj 35 & 9)
#   Table 3: MD17 x 8         Table 8: charged N-body
#
# Switches (env):
#   DEVICE=cuda|cpu            SMOKE=1 (tiny fast settings)
#   TABLES="1 2 3 8"           BETAS="0.01 0.1 1.0 10.0 100.0"
#   PENALTIES="constant gradual"   SEEDS="0"
#   ONLY='<regex>'             run only cells whose tag matches
#   LOG_DIR=outputs/relaxed    EXTRA="...extra cli flags..."
#
# Examples:
#   bash run_paper_grid.sh                                  # everything (days)
#   TABLES=1 bash run_paper_grid.sh                         # N-body only
#   ONLY='t1_gnn_' bash run_paper_grid.sh                   # one cell family
#   SMOKE=1 bash run_paper_grid.sh                          # end-to-end test
# ==============================================================================

set -u
DEVICE="${DEVICE:-cuda}"
SMOKE="${SMOKE:-0}"
TABLES="${TABLES:-1 2 3 8}"
BETAS="${BETAS:-0.01 0.1 1.0 10.0 100.0}"
PENALTIES="${PENALTIES:-constant gradual}"
SEEDS="${SEEDS:-0}"
ONLY="${ONLY:-}"
LOG_DIR="${LOG_DIR:-outputs/relaxed}"
EXTRA="${EXTRA:-}"

if [ "$SMOKE" = "1" ]; then
  NB="--data.n_train 32 --data.n_val 32 --data.n_test 32 --data.num_steps 20"
  MD="--data.md17_n_train 16 --data.md17_n_val 16 --data.md17_n_test 16 --data.delta_t 100"
  COMMON="--train.epochs 1 --train.max_steps 3 --train.batch_size 8 --log.log_every 1"
  TF="--model.name transformer --model.channels 32 --model.num_layers 2 --model.num_heads 4"
  GATR="--model.name gatr --model.channels 32 --model.num_layers 2 --model.num_heads 4 --model.num_multivectors 4"
  MLPF="--model.mlp_hidden 64 --model.num_layers 2"
  GNNF="--model.hidden_dim 32 --model.num_layers 2"
  BETAS="1.0"
else
  NB="--data.n_train 100 --data.n_val 5000 --data.n_test 5000 --data.num_steps 100"
  MD="--data.md17_n_train 500 --data.md17_n_val 2000 --data.md17_n_test 2000 --data.delta_t 5000"
  TF="--model.name transformer --model.channels 384 --model.num_layers 10 --model.num_heads 8"
  GATR="--model.name gatr --model.channels 128 --model.num_layers 12 --model.num_heads 8 --model.num_multivectors 16"
  MLPF="--model.mlp_hidden 680 --model.num_layers 3"
  GNNF="--model.hidden_dim 64 --model.num_layers 4"
fi
NB_BUDGET="--train.max_steps 50000 --train.batch_size 64 --train.lr 3e-4"
MC_BUDGET="--train.epochs 2000 --train.batch_size 12 --train.lr 3e-4"
MD_BUDGET="--train.epochs 500 --train.batch_size 200 --train.lr 5e-4"

run() {
  local tag="$1"; shift
  if [ -n "$ONLY" ] && ! echo "$tag" | grep -qE "$ONLY"; then
    return 0
  fi
  echo "=============================================================="
  echo "CELL $tag"
  python -m relaxed.cli "$@" --train.device "$DEVICE" \
      --run.seeds "$SEEDS" --log.log_dir "$LOG_DIR" $EXTRA || true
}

# Run the full mode sweep for one unconstrained model on one dataset block.
# Args: tag_prefix, dataset+model flags..., budget flags
sweep() {
  local tag="$1"; shift
  run "${tag}_standard" "$@" --train.mode standard
  run "${tag}_da" "$@" --train.mode da
  for PEN in $PENALTIES; do
    for B in $BETAS; do
      run "${tag}_remul_${PEN}_beta${B}" "$@" --train.mode remul --train.penalty "$PEN" --train.beta "$B"
    done
  done
}

has_table() { echo " $TABLES " | grep -q " $1 "; }

# ---------------- Table 1: N-body ----------------
if has_table 1; then
  echo "################ Table 1: N-body ################"
  run t1_base_se3_transformer --data.name nbody $NB --model.name se3_transformer $GNNF $NB_BUDGET --train.mode standard
  run t1_base_gatr --data.name nbody $NB $GATR $NB_BUDGET --train.mode standard
  run t1_base_egnn --data.name nbody $NB --model.name egnn $GNNF $NB_BUDGET --train.mode standard
  sweep t1_transformer --data.name nbody $NB $TF $NB_BUDGET
  sweep t1_mlp --data.name nbody $NB --model.name mlp $MLPF $NB_BUDGET
  sweep t1_gnn --data.name nbody $NB --model.name gnn $GNNF $NB_BUDGET
fi

# ---------------- Table 8: charged N-body ----------------
if has_table 8; then
  echo "################ Table 8: charged N-body ################"
  run t8_base_egnn --data.name nbody_egnn $NB --model.name egnn $GNNF $NB_BUDGET --train.mode standard
  run t8_base_se3_transformer --data.name nbody_egnn $NB --model.name se3_transformer $GNNF $NB_BUDGET --train.mode standard
  run t8_base_tfn --data.name nbody_egnn $NB --model.name tfn $GNNF $NB_BUDGET --train.mode standard
  run t8_base_mpnn --data.name nbody_egnn $NB --model.name mpnn $GNNF $NB_BUDGET --train.mode standard
  sweep t8_gnn --data.name nbody_egnn $NB --model.name gnn $GNNF $NB_BUDGET
  sweep t8_mlp --data.name nbody_egnn $NB --model.name mlp $MLPF $NB_BUDGET
  sweep t8_transformer --data.name nbody_egnn $NB $TF $NB_BUDGET
fi

# ---------------- Table 2: Motion capture ----------------
if has_table 2; then
  echo "################ Table 2: Motion capture ################"
  for SUBJ in 35 9; do
    MC="--data.name mocap --data.mocap_subject $SUBJ --data.mocap_delta_t 30 --train.per_axis_eval true"
    run t2_s${SUBJ}_base_se3_transformer $MC --model.name se3_transformer $GNNF $MC_BUDGET --train.mode standard
    run t2_s${SUBJ}_base_gatr $MC $GATR $MC_BUDGET --train.mode standard
    run t2_s${SUBJ}_base_egno $MC --model.name egno $GNNF $MC_BUDGET --train.mode standard
    run t2_s${SUBJ}_base_egnn $MC --model.name egnn $GNNF $MC_BUDGET --train.mode standard
    sweep t2_s${SUBJ}_transformer $MC $TF $MC_BUDGET
    sweep t2_s${SUBJ}_mlp $MC --model.name mlp $MLPF $MC_BUDGET
    sweep t2_s${SUBJ}_gnn $MC --model.name gnn $GNNF $MC_BUDGET
    sweep t2_s${SUBJ}_emlp $MC --model.name emlp $MLPF $MC_BUDGET
    sweep t2_s${SUBJ}_rpp $MC --model.name rpp $MLPF $MC_BUDGET
    sweep t2_s${SUBJ}_per $MC --model.name per $MLPF $MC_BUDGET
  done
fi

# ---------------- Table 3: MD17 ----------------
if has_table 3; then
  echo "################ Table 3: MD17 ################"
  for MOL in aspirin benzene ethanol malonaldehyde naphthalene salicylic toluene uracil; do
    MDS="--data.name md17_dyn --data.molecule $MOL $MD"
    run t3_${MOL}_base_egnn $MDS --model.name egnn $GNNF $MD_BUDGET --train.mode standard
    run t3_${MOL}_base_gmn $MDS --model.name gmn $GNNF $MD_BUDGET --train.mode standard
    run t3_${MOL}_base_egno $MDS --model.name egno $GNNF $MD_BUDGET --train.mode standard
    run t3_${MOL}_base_hegnn $MDS --model.name hegnn $GNNF $MD_BUDGET --train.mode standard
    sweep t3_${MOL}_gnn $MDS --model.name gnn $GNNF $MD_BUDGET
    sweep t3_${MOL}_mlp $MDS --model.name mlp $MLPF $MD_BUDGET
    sweep t3_${MOL}_transformer $MDS $TF $MD_BUDGET
  done
fi

echo "=============================================================="
echo "Grid finished. Aggregate with: python -m relaxed.collect"
