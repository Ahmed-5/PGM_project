#!/bin/bash
# ==============================================================================
# Faithful N-body (paper Table 1) reproduction with the RC1-RC4 fixes:
#   RC1: canonical-frame data (remul/datasets/nbody.py) -> genuine in/OOD split
#   RC2: best-by-val checkpoint (relaxed/engines/dynamics.py) + grad_clip + cosine
#        LR -> avoids the Transformer persistence-collapse at high steps
#   RC3: scale-relative metrics (mse_rel, E'_rel) reported in each record
#   RC4: beta swept ONLY under penalty=constant (GradNorm 'gradual' ignores beta)
#
# Compares unconstrained {transformer, mlp, gnn} x {standard, da, remul} against
# equivariant baselines {egnn, se3_transformer}. Multi-seed for significance.
#
# Env: DEVICE=cuda  SEEDS="0 1 2"  STEPS=30000  ONLY='<regex>'  LOGDIR=outputs/relaxed
# ==============================================================================
set -u
DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-0 1 2}"
STEPS="${STEPS:-30000}"          # best-checkpoint captures the pre-collapse optimum
LOGDIR="${LOGDIR:-outputs/relaxed}"
ONLY="${ONLY:-}"

TF="--model.name transformer --model.channels 384 --model.num_layers 10 --model.num_heads 8"
MLP="--model.name mlp --model.mlp_hidden 680 --model.num_layers 3"
GNN="--model.name gnn --model.hidden_dim 64 --model.num_layers 4"
EGNN="--model.name egnn --model.hidden_dim 64 --model.num_layers 4"
SE3="--model.name se3_transformer --model.hidden_dim 64 --model.num_layers 4"

run() {
  local tag="$1"; shift
  if [ -n "$ONLY" ] && ! echo "$tag" | grep -qE "$ONLY"; then return 0; fi
  echo "======================================================"
  echo "REPRO CELL $tag"
  # shellcheck disable=SC2086
  python -m relaxed.cli "$@" \
    --data.name nbody --data.n_train 100 --data.n_val 5000 --data.n_test 5000 --data.num_steps 100 \
    --train.max_steps "$STEPS" --train.batch_size 64 --train.lr 3e-4 \
    --train.grad_clip 1.0 --schedule.lr_schedule cosine \
    --train.device "$DEVICE" --run.seeds "$SEEDS" --log.logger_type none --log.log_dir "$LOGDIR" \
    --run.experiment_name "$tag" || true
}

echo "N-body reproduction: seeds=[$SEEDS] steps=$STEPS device=$DEVICE"

# Unconstrained models x training modes (RQ of the paper: can soft methods match hard equivariance?)
for PAIR in "transformer:$TF" "mlp:$MLP" "gnn:$GNN"; do
  NAME="${PAIR%%:*}"; ARGS="${PAIR#*:}"
  run "repro_nbody_${NAME}_standard"        $ARGS --train.mode standard
  run "repro_nbody_${NAME}_da"              $ARGS --train.mode da
  run "repro_nbody_${NAME}_remul_gradual"   $ARGS --train.mode remul --train.penalty gradual
  run "repro_nbody_${NAME}_remul_const_b1"  $ARGS --train.mode remul --train.penalty constant --train.beta 1.0
done

# beta sweep on the flagship Transformer (constant penalty only — RC4)
run "repro_nbody_transformer_remul_const_b0.1" $TF --train.mode remul --train.penalty constant --train.beta 0.1
run "repro_nbody_transformer_remul_const_b10"  $TF --train.mode remul --train.penalty constant --train.beta 10.0

# equivariant baselines (E' ~ 0 by construction; the target REMUL aims to approach)
run "repro_nbody_egnn_standard" $EGNN --train.mode standard
run "repro_nbody_se3_standard"  $SE3  --train.mode standard

echo "======================================================"
echo "N-body reproduction finished."
