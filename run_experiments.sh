#!/bin/bash

# Select logging backend
LOGGER="tensorboard"  # Change to "wandb" or "none"

echo "Running ablation study for Relaxed Equivariance GNN"
echo "Logger: $LOGGER"
echo "===================================================="

# Baseline (no equivariance)
echo "\n[1/5] Running baseline (no equivariance)..."
python train.py --config baseline --logger $LOGGER

# Constant alpha
echo "\n[2/5] Running constant alpha..."
python train.py --config constant_alpha --logger $LOGGER

# Exponential decay
echo "\n[3/5] Running exponential decay..."
python train.py --config exponential_decay --logger $LOGGER

# Linear decay
echo "\n[4/5] Running linear decay..."
python train.py --config linear_decay --logger $LOGGER

# Learnable alphas
echo "\n[5/5] Running learnable alphas..."
python train.py --config learnable --logger $LOGGER

echo "\n===================================================="
echo "All experiments completed!"
if [ "$LOGGER" = "tensorboard" ]; then
    echo "View results with: tensorboard --logdir=./runs"
elif [ "$LOGGER" = "wandb" ]; then
    echo "View results at: https://wandb.ai"
fi
