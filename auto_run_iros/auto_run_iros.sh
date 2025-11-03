#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root = parent of this script (repo root)
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
echo "[INFO] REPO_ROOT=$REPO_ROOT"
echo "[INFO] PYTHONPATH=$PYTHONPATH"

DATA_ROOT="IROS_dataset"
OUT_ROOT="$REPO_ROOT/iros_outputs_auto_run_2"
MODELS_ROOT="$OUT_ROOT/models_2"
EXPTS_ROOT="$OUT_ROOT/experiments_2"

mkdir -p "$MODELS_ROOT" "$EXPTS_ROOT"

# ---- Shapes to run ----
SHAPES=("RShape" "IShape" "OShape" "SShape")

# ---- Common training args ----
CURR_FRACS="0.1,0.1,0.2,0.2,0.2"
WINDOW_MODE="mixed"
STEPS=50000
BATCH=16
WIDTH=128
DEPTH=3
NSAMPLES=10000
NTRAIN=3
BASE_LR=5e-4
PRINT_EVERY=5000
SAVE_EVERY=5000

# echo "==================== TRAIN ALL SHAPES ===================="
# for SH in "${SHAPES[@]}"; do
#   echo ""
#   echo "---------------------------------------------"
#   echo "[TRAIN] Shape=${SH} | steps=${STEPS} | width=${WIDTH} | depth=${DEPTH} | ntrain=${NTRAIN} | nsamples=${NSAMPLES}"
#   echo "---------------------------------------------"
#   python -m src.train.train_node_periodic \
#     --shape "$SH" \
#     --data_root "$DATA_ROOT" \
#     --out_root "$MODELS_ROOT" \
#     --nsamples "$NSAMPLES" \
#     --ntrain "$NTRAIN" \
#     --width "$WIDTH" \
#     --depth "$DEPTH" \
#     --steps "$STEPS" \
#     --base_lr "$BASE_LR" \
#     --batch_size "$BATCH" \
#     --curriculum_fracs "$CURR_FRACS" \
#     --window_mode "$WINDOW_MODE" \
#     --print_every "$PRINT_EVERY" \
#     --save_every "$SAVE_EVERY"
# done

echo ""
echo "==================== EXPERIMENTS ===================="
for SH in "${SHAPES[@]}"; do
  echo ""
  echo "====================================================="
  echo "[FIND MODEL] Shape=${SH}"
  echo "====================================================="

  # Pick the newest run for this shape (regardless of ntr/lr/seed).
  RUN_DIR=$(ls -dt \
      "$MODELS_ROOT/$SH"/segcur_w${WIDTH}_d${DEPTH}_ntr*_ns${NSAMPLES}_lr*_seed*_* 2>/dev/null \
      | head -n1)
  if [[ -z "$RUN_DIR" ]]; then
    RUN_DIR=$(ls -dt "$MODELS_ROOT/$SH"/segcur_w${WIDTH}_d${DEPTH}_ntr*_ns*_lr*_seed*_* 2>/dev/null | head -n1)
  fi
  if [[ -z "$RUN_DIR" ]]; then
    echo "[ERROR] No trained run directory found for $SH under $MODELS_ROOT/$SH"
    exit 2
  fi

  MODEL_PATH="$RUN_DIR/${SH}_NODE_segcur.eqx"
  echo "[FOUND] RUN_DIR=$RUN_DIR"
  echo "[FOUND] MODEL_PATH=$MODEL_PATH"

  if [[ ! -f "$MODEL_PATH" ]]; then
    echo "[ERROR] Model file not found for $SH at $MODEL_PATH"
    exit 2
  fi

  SH_OUT="$EXPTS_ROOT/$SH"
  mkdir -p "$SH_OUT"

  # echo ""
  # echo "----- NO-LLC EXPERIMENTS (${SH}) -----"
  # echo "[RUN] no_llc_zero"
  # python -m src.experiments.run_experiments_periodic \
  #   --shape "$SH" \
  #   --model "$MODEL_PATH" \
  #   --out "$SH_OUT/no_llc_zero" \
  #   --no_llc_zero \
  #   --nsamples "$NSAMPLES" --ntrain 1

  echo "[RUN] no_llc_pulses (direct two mid-pulses default)"
  python -m src.experiments.run_experiments_periodic \
    --shape "$SH" \
    --model "$MODEL_PATH" \
    --out "$SH_OUT/no_llc_pulses" \
    --nsamples "$NSAMPLES" --ntrain 1

  echo ""
  echo "----- WITH-LLC (MATCHED ONLY) (${SH}) -----"
  for KIND in const chirp multisine pulse; do
    echo "[RUN] with_llc | matched=${KIND}"
    python -m src.experiments.run_experiments_periodic \
      --shape "$SH" \
      --model "$MODEL_PATH" \
      --out "$SH_OUT/with_llc_matched_${KIND}" \
      --with_llc --matched --matched_type "$KIND" \
      --nsamples "$NSAMPLES" --ntrain 1
  done

  echo ""
  echo "----- WITH-LLC (UNMATCHED ONLY) (${SH}) -----"
  for KIND in const chirp multisine pulse; do
    echo "[RUN] with_llc | unmatched=${KIND}"
    python -m src.experiments.run_experiments_periodic \
      --shape "$SH" \
      --model "$MODEL_PATH" \
      --out "$SH_OUT/with_llc_unmatched_${KIND}" \
      --with_llc --unmatched --unmatched_type "$KIND" \
      --nsamples "$NSAMPLES" --ntrain 1
  done

  echo ""
  echo "----- WITH-LLC (COMBO) (${SH}) -----"
  echo "[RUN] with_llc | matched=multisine + unmatched=pulse"
  python -m src.experiments.run_experiments_periodic \
    --shape "$SH" \
    --model "$MODEL_PATH" \
    --out "$SH_OUT/with_llc_matched_multisine_unmatched_pulse" \
    --with_llc \
    --matched --matched_type multisine \
    --unmatched --unmatched_type pulse \
    --nsamples "$NSAMPLES" --ntrain 1

  echo "-----------------------------------------------------"
  echo "[DONE] ${SH}"
  echo "-----------------------------------------------------"
done

echo ""
echo "=== ALL DONE ==="
