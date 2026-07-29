#!/bin/bash
# Stable re-run of the symmetry experiments using the MLP backbone (the transformer
# collapses ~2/3 of seeds at low n_train, poisoning the means). MLP is smaller/stabler
# and clearly worse than egnn at field=0, so the crossover is still visible.
set -u
DEVICE="${DEVICE:-cuda}"; SEEDS="${SEEDS:-0 1 2}"; STEPS="${STEPS:-20000}"; LOGDIR="${LOGDIR:-outputs/relaxed}"
NTRAIN="${NTRAIN:-500}"; MODE="${MODE:-both}"
MLP="--model.name mlp --model.mlp_hidden 680 --model.num_layers 3"
EGNN="--model.name egnn --model.hidden_dim 64 --model.num_layers 4"
run(){ local tag="$1"; shift; echo "=== $tag ==="; python -m relaxed.cli "$@" \
  --data.name nbody --data.n_train "$NTRAIN" --data.n_val 2000 --data.n_test 2000 --data.num_steps 100 \
  --train.max_steps "$STEPS" --train.batch_size 64 --train.lr 3e-4 --train.grad_clip 1.0 --schedule.lr_schedule cosine \
  --train.device "$DEVICE" --run.seeds "$SEEDS" --log.logger_type none --log.log_dir "$LOGDIR" --run.experiment_name "$tag" || true; }
if [ "$MODE" = "cross" ] || [ "$MODE" = "both" ]; then
  for F in 0 2 4 8; do
    D="--data.field_strength $F --data.field_axis 2"
    run "st_cross_f${F}_egnn"    $D $EGNN --train.mode standard
    run "st_cross_f${F}_mlp_std" $D $MLP --train.mode standard
    run "st_cross_f${F}_mlp_so3" $D $MLP --train.mode remul --train.penalty constant --train.beta 0.1 --loss.group so3
  done
fi
if [ "$MODE" = "lg" ] || [ "$MODE" = "both" ]; then
  for AX in 2 0; do
    D="--data.field_strength 6 --data.field_axis $AX"
    run "st_lg_ax${AX}_mlp_std" $D $MLP --train.mode standard
    for G in so3 so2_x so2_y so2_z; do
      run "st_lg_ax${AX}_mlp_${G}" $D $MLP --train.mode remul --train.penalty constant --train.beta 1.0 --loss.group "$G"
    done
  done
fi
echo "SYMSTABLE finished."
