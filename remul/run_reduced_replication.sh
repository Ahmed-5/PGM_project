#!/bin/bash

# ==============================================================================
# Reduced replication subset for REMUL (arXiv:2410.17878)
#
# A small but representative slice of remul/run_experiments.sh for validating
# paper *trends* on GPU without the full multi-day suite:
#   - N-body: EGNN baseline + Transformer {standard, da,
#     remul(constant beta in {0.1,1,10}), remul(gradual beta=1)} @ 8k steps
#     (paper: 50k steps, batch 64, lr 3e-4, Transformer 384ch/10L/8H).
#   - MD17 aspirin: GNN {standard, da, remul(constant beta in {0.1,1,10})}
#     @ 100 epochs (paper: 500 epochs, batch 200, lr 5e-4).
#
# Expected trends (paper Tables 1 & 3): test MSE standard > da ~ remul,
# E/E' decreasing in beta; EGNN E' ~ 0 by construction.
# ==============================================================================

DEVICE="${DEVICE:-cuda}"
EXTRA_LOG="${LOG_FLAGS:-}"

run() {
  echo "=============================================================="
  echo "RUN: $*"
  python -m remul.cli "$@" --train.device "$DEVICE" $EXTRA_LOG || true
}

echo "Reduced REMUL replication subset on ${DEVICE}"

# ---------- Table 1 slice: N-body ----------
NB="--data.name nbody --train.max_steps 8000 --train.batch_size 64 --train.lr 3e-4"
TF="--model.name transformer --model.channels 384 --model.num_layers 10 --model.num_heads 8"

run $NB --model.name egnn --train.mode standard
run $NB $TF --train.mode standard
run $NB $TF --train.mode da
for B in 0.1 1.0 10.0; do
  run $NB $TF --train.mode remul --train.penalty constant --train.beta $B
done
run $NB $TF --train.mode remul --train.penalty gradual --train.beta 1.0

# ---------- Table 3 slice: MD17 aspirin ----------
MD="--data.name md17 --data.molecule aspirin --train.epochs 100 --train.batch_size 200 --train.lr 5e-4"
GNN="--model.name gnn"

run $MD $GNN --train.mode standard
run $MD $GNN --train.mode da
for B in 0.1 1.0 10.0; do
  run $MD $GNN --train.mode remul --train.penalty constant --train.beta $B
done

echo "=============================================================="
echo "Reduced replication subset finished. Aggregate with:"
echo "  python -m remul.collect_results"
