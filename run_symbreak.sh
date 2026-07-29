#!/bin/bash
# ==============================================================================
# Symmetry-breaking crossover experiment.
#
# Question: does a non-equivariant model + equivariance loss BEAT a strictly-
# equivariant model, and does it depend on the task's true symmetry group?
#
# We add a uniform +z field to N-body gravity, tunable via --data.field_strength.
#   field = 0   -> task is SO(3)-symmetric  (egnn's prior is correct)
#   field > 0   -> symmetry breaks to SO(2)_z (egnn's SO(3) prior is WRONG)
# and sweep it against four models:
#   egnn              : strict SO(3) (architecture) — should win at 0, fail as field grows
#   transformer std   : unconstrained (no symmetry)
#   transformer REMUL so3   : soft SO(3) (loss, tunable beta)
#   transformer REMUL so2_z : soft SO(2)_z = the task's TRUE residual symmetry
#
# Hypothesis: as the field grows, egnn's error grows while unconstrained/REMUL
# stay low -> a crossover; REMUL with the matched (so2_z) group is best at high field.
#
# Env: DEVICE=cuda  SEEDS="0 1 2"  STEPS=30000  FIELDS="0 1 2 4 8"
# ==============================================================================
set -u
DEVICE="${DEVICE:-cuda}"; SEEDS="${SEEDS:-0 1 2}"; STEPS="${STEPS:-30000}"; LOGDIR="${LOGDIR:-outputs/relaxed}"
FIELDS="${FIELDS:-0 1 2 4 8}"
TF="--model.name transformer --model.channels 384 --model.num_layers 10 --model.num_heads 8"
EGNN="--model.name egnn --model.hidden_dim 64 --model.num_layers 4"
run(){ local tag="$1"; shift
  echo "======================================================"; echo "SYMBREAK $tag"
  # shellcheck disable=SC2086
  python -m relaxed.cli "$@" \
    --data.name nbody --data.n_train ${NTRAIN:-100} --data.n_val 2000 --data.n_test 2000 --data.num_steps 100 \
    --train.max_steps "$STEPS" --train.batch_size 64 --train.lr 3e-4 --train.grad_clip 1.0 --schedule.lr_schedule cosine \
    --train.device "$DEVICE" --run.seeds "$SEEDS" --log.logger_type none --log.log_dir "$LOGDIR" \
    --run.experiment_name "$tag" || true; }
for F in $FIELDS; do
  D="--data.field_strength $F"
  run "sym_f${F}_egnn_so3strict"        $D $EGNN --train.mode standard
  run "sym_f${F}_transformer_std"       $D $TF --train.mode standard
  run "sym_f${F}_transformer_remulSO3"  $D $TF --train.mode remul --train.penalty constant --train.beta 0.1 --loss.group so3
  run "sym_f${F}_transformer_remulSO2z" $D $TF --train.mode remul --train.penalty constant --train.beta 1.0 --loss.group so2_z
done
echo "SYMBREAK finished."
