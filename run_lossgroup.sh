#!/bin/bash
set -u
DEVICE="${DEVICE:-cuda}"; SEEDS="${SEEDS:-0 1 2}"; STEPS="${STEPS:-30000}"; LOGDIR="${LOGDIR:-outputs/relaxed}"
FIELD="${FIELD:-8}"; AXES="${AXES:-2 0}"
TF="--model.name transformer --model.channels 384 --model.num_layers 10 --model.num_heads 8"
EGNN="--model.name egnn --model.hidden_dim 64 --model.num_layers 4"
run(){ local tag="$1"; shift; echo "=== LOSSGROUP $tag ==="; python -m relaxed.cli "$@" \
  --data.name nbody --data.n_train ${NTRAIN:-100} --data.n_val 2000 --data.n_test 2000 --data.num_steps 100 --data.field_strength "$FIELD" \
  --train.max_steps "$STEPS" --train.batch_size 64 --train.lr 3e-4 --train.grad_clip 1.0 --schedule.lr_schedule cosine \
  --train.device "$DEVICE" --run.seeds "$SEEDS" --log.logger_type none --log.log_dir "$LOGDIR" --run.experiment_name "$tag" || true; }
for AX in $AXES; do
  D="--data.field_axis $AX"
  run "lg_ax${AX}_egnn"   $D $EGNN --train.mode standard
  run "lg_ax${AX}_tf_std" $D $TF --train.mode standard
  for G in so3 so2_x so2_y so2_z; do
    run "lg_ax${AX}_tf_remul_${G}" $D $TF --train.mode remul --train.penalty constant --train.beta 1.0 --loss.group "$G"
  done
done
echo "LOSSGROUP finished."
