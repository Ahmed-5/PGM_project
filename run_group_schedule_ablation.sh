#!/bin/bash

# ==============================================================================
# Group-set x loss-schedule ablation for the REMUL-Extension pipeline
# (layer-wise equivariance loss, top-level cli.py)
#
# Testbed: QM9 (has 3D positions + invariant scalar targets; ZINC has no
# positions, so geometric groups are meaningless there). Target: config default
# (qm9_target=7, U0), z-scored with TRAIN-split statistics.
#
# Models:
#   - gcn + use_pos  : unconstrained, position-sensitive (the REMUL setting)
#   - egnn           : hard-equivariant control (layer eq losses should be ~0)
#
# Design: one-factor-at-a-time (full cross is 25+ runs/model).
#   A. Schedule sweep on group set {so3, translation}:
#      baseline (no loss), constant, linear, exponential, u_shaped, learnable
#   B. Group-set sweep with the constant schedule:
#      {so3}, {translation}, {so3,translation} (shared with A), {e3},
#      {permutation,so3,translation}
# Group weights within a set are uniform and sum to 1 so the total equivariance
# strength is comparable across sets. alpha_0 (global strength) = 1.0, applied
# once through the DepthScheduler layer weights.
#
# Env: EPOCHS (default 50), MODELS (default "gcn" — the unconstrained subject
# of the study), CONTROL (default 1: also run the 2 EGNN hard-equivariance
# control runs), LOGGER (default tensorboard). EPOCHS=3 is a good validation
# pass.
# ==============================================================================

set -u  # NOTE: no `set -e`; one failed run must not abort the matrix

EPOCHS="${EPOCHS:-50}"
MODELS="${MODELS:-gcn}"
CONTROL="${CONTROL:-1}"
LOGGER="${LOGGER:-tensorboard}"
HIDDEN=128
LAYERS=4

run() {
  echo "=============================================================="
  echo "RUN: $*"
  python cli.py "$@" || true
}

base_args() {
  local model="$1"
  echo "--data.dataset_name QM9 --data.use_positions true \
--model.model_type ${model} --model.in_channels 11 \
--model.hidden_channels ${HIDDEN} --model.num_layers ${LAYERS} \
--training.num_epochs ${EPOCHS} --training.batch_size 128 \
--logging.logger_type ${LOGGER}"
}

for MODEL in $MODELS; do
  ARGS="$(base_args $MODEL)"
  EXTRA=""
  if [ "$MODEL" = "gcn" ]; then
    EXTRA="--model.use_pos true"
  fi

  # ---- A. Schedule sweep on {so3, translation} ----
  GS="['so3','translation']"
  GW="{'so3': 0.5, 'translation': 0.5}"

  run --experiment_name "QM9_${MODEL}_Baseline" $ARGS $EXTRA \
    --equivariance.symmetry_groups "$GS" --equivariance.group_weights "$GW" \
    --equivariance.stochastic_probability 0.0 --scheduler.alpha_0 0.0

  for STRAT in constant linear exponential u_shaped learnable; do
    run --experiment_name "QM9_${MODEL}_Sched_${STRAT}" $ARGS $EXTRA \
      --equivariance.symmetry_groups "$GS" --equivariance.group_weights "$GW" \
      --equivariance.stochastic_probability 0.25 \
      --equivariance.layer_weight_strategy "$STRAT" --scheduler.alpha_0 1.0
  done

  # ---- B. Group-set sweep (constant schedule) ----
  run --experiment_name "QM9_${MODEL}_Groups_so3" $ARGS $EXTRA \
    --equivariance.symmetry_groups "['so3']" --equivariance.group_weights "{'so3': 1.0}" \
    --equivariance.stochastic_probability 0.25 \
    --equivariance.layer_weight_strategy constant --scheduler.alpha_0 1.0

  run --experiment_name "QM9_${MODEL}_Groups_translation" $ARGS $EXTRA \
    --equivariance.symmetry_groups "['translation']" --equivariance.group_weights "{'translation': 1.0}" \
    --equivariance.stochastic_probability 0.25 \
    --equivariance.layer_weight_strategy constant --scheduler.alpha_0 1.0

  run --experiment_name "QM9_${MODEL}_Groups_e3" $ARGS $EXTRA \
    --equivariance.symmetry_groups "['e3']" --equivariance.group_weights "{'e3': 1.0}" \
    --equivariance.stochastic_probability 0.25 \
    --equivariance.layer_weight_strategy constant --scheduler.alpha_0 1.0

  run --experiment_name "QM9_${MODEL}_Groups_perm_so3_trans" $ARGS $EXTRA \
    --equivariance.symmetry_groups "['permutation','so3','translation']" \
    --equivariance.group_weights "{'permutation': 0.333, 'so3': 0.333, 'translation': 0.333}" \
    --equivariance.stochastic_probability 0.25 \
    --equivariance.layer_weight_strategy constant --scheduler.alpha_0 1.0
done

# ---- EGNN hard-equivariance control (eq losses should be ~0 by construction) ----
if [ "$CONTROL" = "1" ]; then
  ARGS="$(base_args egnn)"
  GS="['so3','translation']"
  GW="{'so3': 0.5, 'translation': 0.5}"

  run --experiment_name "QM9_egnn_Baseline" $ARGS \
    --equivariance.symmetry_groups "$GS" --equivariance.group_weights "$GW" \
    --equivariance.stochastic_probability 0.0 --scheduler.alpha_0 0.0

  run --experiment_name "QM9_egnn_Sched_constant" $ARGS \
    --equivariance.symmetry_groups "$GS" --equivariance.group_weights "$GW" \
    --equivariance.stochastic_probability 0.25 \
    --equivariance.layer_weight_strategy constant --scheduler.alpha_0 1.0
fi

echo "=============================================================="
echo "Ablation finished (10 runs per model + 2 EGNN controls). Aggregate with:"
echo "  python collect_ablation_results.py --pattern 'QM9_*'"
