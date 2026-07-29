#!/bin/bash
# 3-seed rigor for the MD17 headline cells (paper backbone GNN/MLP vs EGNN baseline).
# Focused subset (not the full beta sweep) at seeds 0 1 2 for mean+/-std.
set -u
DEVICE="${DEVICE:-cuda}"; SEEDS="${SEEDS:-0 1 2}"; LOGDIR="${LOGDIR:-outputs/relaxed}"
MOLS="${MOLS:-aspirin benzene ethanol malonaldehyde naphthalene salicylic toluene uracil}"
GNN="--model.name gnn --model.hidden_dim 64 --model.num_layers 4"
MLP="--model.name mlp --model.mlp_hidden 680 --model.num_layers 3"
EGNN="--model.name egnn --model.hidden_dim 64 --model.num_layers 4"
run() {
  local tag="$1"; shift
  echo "==== MD17-HL $tag ===="
  # shellcheck disable=SC2086
  python -m relaxed.cli "$@" --data.name md17_dyn --train.epochs 500 --train.batch_size 200 --train.lr 5e-4 \
    --train.device "$DEVICE" --run.seeds "$SEEDS" --log.logger_type none --log.log_dir "$LOGDIR" \
    --run.experiment_name "$tag" || true
}
for MOL in $MOLS; do
  D="--data.molecule $MOL"
  run "hl_md17_${MOL}_egnn_standard"    $D $EGNN --train.mode standard
  run "hl_md17_${MOL}_gnn_standard"     $D $GNN  --train.mode standard
  run "hl_md17_${MOL}_gnn_da"           $D $GNN  --train.mode da
  run "hl_md17_${MOL}_gnn_remul_b0.1"   $D $GNN  --train.mode remul --train.penalty constant --train.beta 0.1
  run "hl_md17_${MOL}_mlp_standard"     $D $MLP  --train.mode standard
  run "hl_md17_${MOL}_mlp_da"           $D $MLP  --train.mode da
  run "hl_md17_${MOL}_mlp_remul_b0.1"   $D $MLP  --train.mode remul --train.penalty constant --train.beta 0.1
done
echo "MD17 headline 3-seed finished."
