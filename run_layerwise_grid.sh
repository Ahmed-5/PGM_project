#!/bin/bash

# ==============================================================================
# Maximal, rigorous layer-wise equivariance-loss grid (relaxed CLI, graph task).
#
# Answers RQ1 (does layer-wise SCHEDULING help?) and RQ2 (do equivariance
# GROUPS help?) defensibly: multi-seed, OOD + functional-equivariance metrics,
# a finer/lower alpha_0 sweep, and stochastic_probability / num_samples sweeps.
#
# Crosses, for unconstrained position-aware GNNs on datasets with 3D coords:
#   - schedules: baseline (no loss) + constant, exponential, exp_inc, linear,
#     linear_inc, inverse, u_shaped, learnable                             (RQ1)
#   - group sets: 8 single groups + {so3,translation},
#     {permutation,so3,translation}, {permutation,e3}                      (RQ2)
#     (weights normalized to sum 1 -> group-set arms are strength-matched)
#   - alpha_0 strength: {0.003,0.01,0.03,0.1,0.3,1.0} on the reference cell
#   - stochastic_probability {0.1,0.25,0.5,1.0} and num_samples {2,4,8} (OFAT)
#   - hard-equivariant / invariant controls (egnn, schnet, painn)
# Every run also reports rotated-test (OOD) MAE and label-free functional
# equivariance error E/E' (see --train.ood_eval).
#
# RQ columns run at strength $A0 (default 0.03; 1.0 tanks in-dist accuracy). Run
# the alpha_0 sweep first, pick the Pareto knee, then re-run with A0=<best>.
#
# Switches (env):
#   DEVICE=cuda  EPOCHS=100  EPOCHS_MD=30  SEEDS="0 1 2 3 4"  A0=0.03
#   MODELS="gcn gin transformer"  LOGGER=tensorboard  ONLY='<regex on tag>'
#   SCHEDS="..."  SETS="..."  A0S="..."  PROBS="..."  NSAMP="..."
#   OOD_EVAL=true  OOD_ROT=8  OOD_GROUPS="so3"  EXTRA="...extra flags..."
#
# Examples:
#   SEEDS="0 1 2 3 4" bash run_layerwise_grid.sh          # maximal (days)
#   ONLY='qm9_gcn_.*_sched_' bash run_layerwise_grid.sh   # RQ1 schedule cols
#   SMOKE=1 bash run_layerwise_grid.sh                    # end-to-end test
# ==============================================================================

set -u
DEVICE="${DEVICE:-cuda}"
SMOKE="${SMOKE:-0}"
EPOCHS="${EPOCHS:-100}"
EPOCHS_MD="${EPOCHS_MD:-30}"
SEEDS="${SEEDS:-0 1 2 3 4}"
A0="${A0:-0.03}"                       # strength for the RQ schedule/group columns
LOGGER="${LOGGER:-tensorboard}"
ONLY="${ONLY:-}"
SCHEDS="${SCHEDS:-constant exponential exp_inc linear linear_inc inverse u_shaped learnable}"
SETS="${SETS:-permutation so3 o3 se3 e3 translation reflection scaling so3_trans perm_so3_trans perm_e3}"
A0S="${A0S:-0.003 0.01 0.03 0.1 0.3 1.0}"
PROBS="${PROBS:-0.1 0.25 0.5 1.0}"
NSAMP="${NSAMP:-2 4 8}"
MODELS="${MODELS:-gcn gin transformer}"
OOD_EVAL="${OOD_EVAL:-true}"
OOD_ROT="${OOD_ROT:-8}"
OOD_GROUPS="${OOD_GROUPS:-so3}"
REF_SET="${REF_SET:-so3_trans}"
REF_SCHED="${REF_SCHED:-constant}"
EXTRA="${EXTRA:-}"

if [ "$SMOKE" = "1" ]; then
  EPOCHS=3; EPOCHS_MD=3; SETS="so3"; SCHEDS="constant learnable"
  MODELS="gcn"; SEEDS="0 1"; A0S="0.03"; PROBS="0.25"; NSAMP="2"; OOD_ROT=2
fi

# group set tag -> symmetry_groups + group_weights (uniform; total is
# renormalized to a fixed strength by the loss so arms are strength-matched).
set_args() {
  case "$1" in
    permutation)    echo "--loss.symmetry_groups permutation --loss.group_weights {'permutation':1.0}" ;;
    so3)            echo "--loss.symmetry_groups so3 --loss.group_weights {'so3':1.0}" ;;
    o3)             echo "--loss.symmetry_groups o3 --loss.group_weights {'o3':1.0}" ;;
    se3)            echo "--loss.symmetry_groups se3 --loss.group_weights {'se3':1.0}" ;;
    e3)             echo "--loss.symmetry_groups e3 --loss.group_weights {'e3':1.0}" ;;
    translation)    echo "--loss.symmetry_groups translation --loss.group_weights {'translation':1.0}" ;;
    reflection)     echo "--loss.symmetry_groups reflection --loss.group_weights {'reflection':1.0}" ;;
    scaling)        echo "--loss.symmetry_groups scaling --loss.group_weights {'scaling':1.0}" ;;
    so3_trans)      echo "--loss.symmetry_groups so3 translation --loss.group_weights {'so3':0.5,'translation':0.5}" ;;
    perm_so3_trans) echo "--loss.symmetry_groups permutation so3 translation --loss.group_weights {'permutation':0.333,'so3':0.333,'translation':0.333}" ;;
    perm_e3)        echo "--loss.symmetry_groups permutation e3 --loss.group_weights {'permutation':0.5,'e3':0.5}" ;;
  esac
}

# Feature type + position handling per model (equivariant vector features for
# painn/egnn/vector_neuron; invariant scalar features otherwise; position
# concatenation for standard GNNs so geometric losses are non-trivial).
model_args() {
  case "$1" in
    gcn|gin|graphsage) echo "--model.name $1 --model.use_pos true --loss.feature_type invariant" ;;
    painn|egnn|vector_neuron|clofnet) echo "--model.name $1 --loss.feature_type equivariant" ;;
    *)                 echo "--model.name $1 --loss.feature_type invariant" ;;
  esac
}

run() {
  local tag="$1"; shift
  if [ -n "$ONLY" ] && ! echo "$tag" | grep -qE "$ONLY"; then
    return 0
  fi
  echo "=============================================================="
  echo "CELL $tag"
  # shellcheck disable=SC2086
  python -m relaxed.cli "$@" --train.device "$DEVICE" \
      --run.seeds "$SEEDS" --log.logger_type "$LOGGER" \
      --train.ood_eval "$OOD_EVAL" --train.ood_num_rotations "$OOD_ROT" \
      --train.ood_groups $OOD_GROUPS $EXTRA || true
}

# Full schedule sweep for one dataset/model/group-set at strength $A0 (RQ1).
sweep_scheds() {
  local tag="$1" set="$2"; shift 2
  run "${tag}_g_${set}_sched_baseline" "$@" \
      $(set_args "$set") --loss.stochastic_probability 0.0 --schedule.alpha_0 0.0
  for SCHED in $SCHEDS; do
    run "${tag}_g_${set}_sched_${SCHED}" "$@" \
        $(set_args "$set") --loss.stochastic_probability 0.25 \
        --loss.layer_weight_strategy "$SCHED" --schedule.alpha_0 "$A0"
  done
}

echo "Grid: models=[$MODELS] scheds=[$SCHEDS] sets=[$SETS] seeds=[$SEEDS] A0=$A0 OOD=$OOD_EVAL"

# ================= Core: QM9 x models x sets x schedules =================
for M in $MODELS; do
  QM9="--data.name QM9 --data.use_positions true --model.in_channels 11 \
       --model.hidden_channels 128 --model.num_layers 4 \
       --train.epochs $EPOCHS --train.batch_size 128 --loss.formulation layerwise"
  for SET in $SETS; do
    sweep_scheds "qm9_${M}" "$SET" $QM9 $(model_args "$M")
  done

  # --- alpha_0 strength sweep on the reference cell (REF_SET + REF_SCHED) ---
  for A0V in $A0S; do
    run "qm9_${M}_alpha0_${A0V}" $QM9 $(model_args "$M") \
        $(set_args "$REF_SET") --loss.stochastic_probability 0.25 \
        --loss.layer_weight_strategy "$REF_SCHED" --schedule.alpha_0 "$A0V"
  done

  # --- stochastic_probability OFAT on the reference cell ---
  for P in $PROBS; do
    run "qm9_${M}_prob_${P}" $QM9 $(model_args "$M") \
        $(set_args "$REF_SET") --loss.stochastic_probability "$P" \
        --loss.layer_weight_strategy "$REF_SCHED" --schedule.alpha_0 "$A0"
  done

  # --- num_samples OFAT on the reference cell ---
  for NS in $NSAMP; do
    run "qm9_${M}_nsamp_${NS}" $QM9 $(model_args "$M") \
        $(set_args "$REF_SET") --loss.stochastic_probability 0.25 --loss.num_samples "$NS" \
        --loss.layer_weight_strategy "$REF_SCHED" --schedule.alpha_0 "$A0"
  done
done

# ================= Controls on QM9 (hard-eq / invariant) =================
QM9C="--data.name QM9 --data.use_positions true --model.in_channels 11 \
      --model.hidden_channels 128 --model.num_layers 4 \
      --train.epochs $EPOCHS --train.batch_size 128 --loss.formulation layerwise"
for C in egnn schnet painn; do
  run "qm9_${C}_baseline" $QM9C $(model_args "$C") \
      $(set_args "$REF_SET") --loss.stochastic_probability 0.0 --schedule.alpha_0 0.0
  run "qm9_${C}_g_${REF_SET}_sched_${REF_SCHED}" $QM9C $(model_args "$C") \
      $(set_args "$REF_SET") --loss.stochastic_probability 0.25 \
      --loss.layer_weight_strategy "$REF_SCHED" --schedule.alpha_0 "$A0"
done

# ============ Transfer: MD17-graph (aspirin), full schedule x group ==========
MDG="--data.name MD17 --data.md17_molecule aspirin --data.use_positions true \
     --model.in_channels 20 --model.hidden_channels 128 --model.num_layers 4 \
     --train.epochs $EPOCHS_MD --train.batch_size 128 --loss.formulation layerwise"
for SET in so3 translation so3_trans e3; do
  sweep_scheds "md17_gcn" "$SET" $MDG $(model_args gcn)
done

# ============ ZINC (no 3D coords): permutation-only schedule sweep ==========
# OOD geometric eval is auto-skipped (no positions).
ZINC="--data.name ZINC --model.name gcn --model.hidden_channels 128 --model.num_layers 6 \
      --train.epochs $EPOCHS --train.batch_size 128 --loss.formulation layerwise"
sweep_scheds "zinc_gcn" "permutation" $ZINC

# NOTE: QM7b and ModelNet40 are excluded — PyG QM7b ships only targets (no node
# features/positions) and ModelNet40 is classification (this engine regresses).
# Coverage NOT crossed (logged deliberately): schedule x group x alpha_0 is
# OFAT off the reference cell (so3_trans/constant/A0), not a full Cartesian
# product; prob/num_samples are OFAT; MD17 uses only gcn.

echo "=============================================================="
echo "Grid finished. Aggregate + significance:"
echo "  python -m relaxed.collect"
echo "  python -m relaxed.analyze --dataset QM9 --model gcn --ood-group so3 \\"
echo "      --csv results/arms.csv --pareto results/pareto_qm9_gcn.png"
