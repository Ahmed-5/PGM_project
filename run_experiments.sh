#!/bin/bash

# ==============================================================================
# Scheduling-strategy ablation for Relaxed Equivariance GNN (REMUL-Extension)
#
# Uses cli.py with explicit overrides (no non-existent config presets).
# The layer-wise equivariance weight schedule is selected via
# --equivariance.layer_weight_strategy (constant | linear | exponential |
# linear_inc | exp_inc | u_shaped | learnable).
#
# NOTE: ZINC has no 3D coordinates, so only the *permutation* group gives a
# meaningful equivariance signal here; it is now passed explicitly (previously
# symmetry_groups defaulted to [] and NO equivariance loss was computed).
# For geometric groups (so3/translation/e3) use the QM9 ablation:
#   bash run_group_schedule_ablation.sh
# ==============================================================================

set -e

# Select logging backend: "wandb", "tensorboard", or "none"
LOGGER="${LOGGER:-tensorboard}"

DATASET="ZINC"
MODEL="gcn"
HIDDEN=128
LAYERS=6
EPOCHS=100
COMMON="--data.dataset_name ${DATASET} --model.model_type ${MODEL} \
--model.hidden_channels ${HIDDEN} --model.num_layers ${LAYERS} \
--training.num_epochs ${EPOCHS} --logging.logger_type ${LOGGER}"

echo "Running scheduling-strategy ablation for Relaxed Equivariance GNN"
echo "Logger: ${LOGGER}"
echo "===================================================="

# 1. Baseline: no equivariance loss (alpha_0 = 0, stochastic_probability = 0)
echo -e "\n[1/5] Baseline (no equivariance loss)..."
python cli.py --experiment_name "ZINC_GCN_Baseline" ${COMMON} \
  --equivariance.stochastic_probability 0.0 --scheduler.alpha_0 0.0 \
  --training.batch_size 128

# 2. Constant schedule: fixed alpha across all layers
echo -e "\n[2/5] Constant schedule..."
python cli.py --experiment_name "ZINC_GCN_Constant" ${COMMON} \
  --equivariance.stochastic_probability 0.25 \
  --equivariance.symmetry_groups "['permutation']" \
  --equivariance.layer_weight_strategy constant --scheduler.alpha_0 1.0 \
  --training.accumulation_steps 2

# 3. Exponential decay: weight decays with depth
echo -e "\n[3/5] Exponential decay schedule..."
python cli.py --experiment_name "ZINC_GCN_Exponential" ${COMMON} \
  --equivariance.stochastic_probability 0.25 \
  --equivariance.symmetry_groups "['permutation']" \
  --equivariance.layer_weight_strategy exponential --scheduler.alpha_0 1.0 \
  --equivariance.layer_decay_rate 0.5 --training.accumulation_steps 2

# 4. Linear decay: weight decreases linearly with depth
echo -e "\n[4/5] Linear decay schedule..."
python cli.py --experiment_name "ZINC_GCN_Linear" ${COMMON} \
  --equivariance.stochastic_probability 0.25 \
  --equivariance.symmetry_groups "['permutation']" \
  --equivariance.layer_weight_strategy linear --scheduler.alpha_0 1.0 \
  --scheduler.gamma 0.1 --training.accumulation_steps 2

# 5. Learnable schedule (proposed): per-layer trainable weights
echo -e "\n[5/5] Learnable schedule..."
python cli.py --experiment_name "ZINC_GCN_Learnable" ${COMMON} \
  --equivariance.stochastic_probability 0.25 \
  --equivariance.symmetry_groups "['permutation']" \
  --equivariance.layer_weight_strategy learnable --scheduler.alpha_0 0.1 \
  --training.accumulation_steps 2

echo -e "\n===================================================="
echo "All experiments completed!"
if [ "$LOGGER" = "tensorboard" ]; then
    echo "View results with: tensorboard --logdir=./runs"
elif [ "$LOGGER" = "wandb" ]; then
    echo "View results at: https://wandb.ai"
fi
