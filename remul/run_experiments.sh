#!/usr/bin/env bash
# Reproduce the REMUL experiments (arXiv:2410.17878) with the paper's datasets,
# models and hyperparameters (Appendix C).
#
#   Full (paper) runs:   bash remul/run_experiments.sh            # needs a GPU
#   Quick smoke test:    SMOKE=1 bash remul/run_experiments.sh    # a few steps, CPU
#   Choose device:       DEVICE=cuda bash remul/run_experiments.sh
#   TensorBoard:         LOG_FLAGS="--log.logger_type tensorboard --log.log_every 50" \
#                          DEVICE=cuda bash remul/run_experiments.sh
#
# Each block maps to a table in the paper. Beta is the REMUL lever
# (equivariance weight); the paper sweeps beta in {0.01, 0.1, 1.0, 10.0, 100.0}.
set -e

DEVICE="${DEVICE:-cpu}"
# Optional logging, e.g. LOG_FLAGS="--log.logger_type tensorboard --log.log_every 50"
LOG_FLAGS="${LOG_FLAGS:-}"

# Auto-detect CUDA if requested but only CPU torch was installed
if [ "$DEVICE" = "cuda" ]; then
  if ! python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "Warning: DEVICE=cuda but PyTorch has no CUDA support. Falling back to cpu."
    echo "  Fix: pip install torch --index-url https://download.pytorch.org/whl/cu124"
    DEVICE=cpu
  fi
fi

PY="python -m remul.cli"

# Wrapper: forwards all CLI args plus optional LOG_FLAGS
run() {
  # shellcheck disable=SC2086
  $PY "$@" $LOG_FLAGS || true
}

if [ "${SMOKE:-0}" = "1" ]; then
  # Tiny, fast settings just to confirm everything runs end to end.
  NB="--data.n_train 32 --data.n_val 32 --data.n_test 32 --data.num_steps 20"
  MD="--data.md17_n_train 16 --data.md17_n_val 16 --data.md17_n_test 16 --data.delta_t 100"
  COMMON="--train.epochs 1 --train.max_steps 3 --train.batch_size 8 --log.log_every 1 --train.device $DEVICE"
  SMALL_TF="--model.channels 32 --model.num_layers 2 --model.num_heads 4"
  SMALL_GATR="--model.channels 32 --model.num_layers 2 --model.num_heads 4 --model.num_multivectors 4"
  SMALL_GNN="--model.hidden_dim 32 --model.num_layers 2"
  BETAS="1.0"
else
  # Paper hyperparameters (Appendix C). These are GPU-scale.
  NB="--data.n_train 100 --data.n_val 5000 --data.n_test 5000 --data.num_steps 100"
  MD="--data.md17_n_train 500 --data.md17_n_val 2000 --data.md17_n_test 2000 --data.delta_t 5000"
  SMALL_TF="--model.channels 384 --model.num_layers 10 --model.num_heads 8"
  # Paper uses 12 GATr blocks on MoCap (vs 10 for Transformer)
  SMALL_GATR="--model.channels 128 --model.num_layers 12 --model.num_heads 8 --model.num_multivectors 16"
  SMALL_GNN="--model.hidden_dim 64 --model.num_layers 4"
  BETAS="0.01 0.1 1.0 10.0 100.0"
  NB_COMMON="--train.max_steps 50000 --train.batch_size 64 --train.lr 3e-4 --train.device $DEVICE"
  MOCAP_COMMON="--train.epochs 2000 --train.batch_size 12 --train.lr 3e-4 --train.device $DEVICE"
  MD_COMMON="--train.epochs 500 --train.batch_size 200 --train.lr 5e-4 --train.device $DEVICE"
fi
COMMON="${COMMON:-$NB_COMMON}"

echo "################ Table 1: N-body dynamical system ################"
# Equivariant baselines (SE(3)-Transformer, GATr, EGNN)
for M in se3_transformer gatr egnn; do
  EXTRA=""
  [ "$M" = "gatr" ] && EXTRA="$SMALL_GATR"
  [ "$M" != "gatr" ] && EXTRA="$SMALL_TF"
  run --data.name nbody $NB --model.name $M $EXTRA ${NB_COMMON:-$COMMON} --train.mode standard
done
# Standard / DA / REMUL Transformer
run --data.name nbody $NB --model.name transformer $SMALL_TF ${NB_COMMON:-$COMMON} --train.mode standard
run --data.name nbody $NB --model.name transformer $SMALL_TF ${NB_COMMON:-$COMMON} --train.mode da
for B in $BETAS; do
  run --data.name nbody $NB --model.name transformer $SMALL_TF ${NB_COMMON:-$COMMON} \
      --train.mode remul --train.penalty constant --train.beta $B
  run --data.name nbody $NB --model.name transformer $SMALL_TF ${NB_COMMON:-$COMMON} \
      --train.mode remul --train.penalty gradual --train.beta $B
done

echo "################ Table 2: Motion Capture ################"
# subject 35 = walking, subject 9 = running
for SUBJ in 35 9; do
  # Equivariant baselines (SE(3)-Transformer, GATr with 12 layers, EGNO)
  for M in se3_transformer gatr egno; do
    EXTRA="$SMALL_TF"
    [ "$M" = "gatr" ] && EXTRA="$SMALL_GATR"
    run --data.name motion_capture --data.mocap_subject $SUBJ --data.delta_t 30 \
        --model.name $M $EXTRA ${MOCAP_COMMON:-$COMMON} --train.mode standard
  done
  # Transformer: standard / DA / REMUL (constant + gradual)
  run --data.name motion_capture --data.mocap_subject $SUBJ --data.delta_t 30 \
      --model.name transformer $SMALL_TF ${MOCAP_COMMON:-$COMMON} --train.mode standard
  run --data.name motion_capture --data.mocap_subject $SUBJ --data.delta_t 30 \
      --model.name transformer $SMALL_TF ${MOCAP_COMMON:-$COMMON} --train.mode da
  for B in $BETAS; do
    run --data.name motion_capture --data.mocap_subject $SUBJ --data.delta_t 30 \
        --model.name transformer $SMALL_TF ${MOCAP_COMMON:-$COMMON} \
        --train.mode remul --train.penalty constant --train.beta $B
    run --data.name motion_capture --data.mocap_subject $SUBJ --data.delta_t 30 \
        --model.name transformer $SMALL_TF ${MOCAP_COMMON:-$COMMON} \
        --train.mode remul --train.penalty gradual --train.beta $B
  done
  # MLP-family baselines (EMLP / RPP / PER / MLP) — standard + DA + REMUL
  for M in emlp rpp per mlp; do
    run --data.name motion_capture --data.mocap_subject $SUBJ --data.delta_t 30 \
        --model.name $M --model.mlp_hidden 680 --model.num_layers 3 ${MOCAP_COMMON:-$COMMON} \
        --train.mode standard
    run --data.name motion_capture --data.mocap_subject $SUBJ --data.delta_t 30 \
        --model.name $M --model.mlp_hidden 680 --model.num_layers 3 ${MOCAP_COMMON:-$COMMON} \
        --train.mode da
    for B in $BETAS; do
      run --data.name motion_capture --data.mocap_subject $SUBJ --data.delta_t 30 \
          --model.name $M --model.mlp_hidden 680 --model.num_layers 3 ${MOCAP_COMMON:-$COMMON} \
          --train.mode remul --train.penalty constant --train.beta $B
    done
  done
done

echo "################ Table 3: MD17 molecular dynamics ################"
for MOL in aspirin benzene ethanol malonaldehyde naphthalene salicylic toluene uracil; do
  # Equivariant baselines
  for M in egnn gmn egno hegnn; do
    run --data.name md17 --data.molecule $MOL $MD --model.name $M $SMALL_GNN \
        ${MD_COMMON:-$COMMON} --train.mode standard
  done
  # Non-equivariant GNN: standard / DA / REMUL
  run --data.name md17 --data.molecule $MOL $MD --model.name gnn $SMALL_GNN \
      ${MD_COMMON:-$COMMON} --train.mode standard
  run --data.name md17 --data.molecule $MOL $MD --model.name gnn $SMALL_GNN \
      ${MD_COMMON:-$COMMON} --train.mode da
  for B in $BETAS; do
    run --data.name md17 --data.molecule $MOL $MD --model.name gnn $SMALL_GNN \
        ${MD_COMMON:-$COMMON} --train.mode remul --train.penalty constant --train.beta $B
    run --data.name md17 --data.molecule $MOL $MD --model.name gnn $SMALL_GNN \
        ${MD_COMMON:-$COMMON} --train.mode remul --train.penalty gradual --train.beta $B
  done
done

echo "################ Table 8: Charged N-body (EGNN-style, 5 particles) ################"
# Equivariant baselines
for M in egnn se3_transformer tfn mpnn; do
  run --data.name nbody_egnn $NB --model.name $M $SMALL_GNN ${NB_COMMON:-$COMMON} \
      --train.mode standard
done
# Non-equivariant GNN + REMUL/DA
run --data.name nbody_egnn $NB --model.name gnn $SMALL_GNN ${NB_COMMON:-$COMMON} \
    --train.mode standard
run --data.name nbody_egnn $NB --model.name gnn $SMALL_GNN ${NB_COMMON:-$COMMON} \
    --train.mode da
for B in $BETAS; do
  run --data.name nbody_egnn $NB --model.name gnn $SMALL_GNN ${NB_COMMON:-$COMMON} \
      --train.mode remul --train.penalty constant --train.beta $B
done

echo "All experiments dispatched."
