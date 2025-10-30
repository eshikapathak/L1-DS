#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root = parent of this script (repo root)
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
echo "[INFO] REPO_ROOT=$REPO_ROOT"
echo "[INFO] PYTHONPATH=$PYTHONPATH"

DATA_ROOT="IROS_dataset"
OUT_ROOT="$REPO_ROOT/iros_outputs_auto_run"
MODELS_ROOT="$OUT_ROOT/models"
EXPTS_ROOT="$OUT_ROOT/experiments"

mkdir -p "$MODELS_ROOT" "$EXPTS_ROOT"

# ---- Shapes to run ----
SHAPES=("IShape" "RShape" "OShape" "SShape")

# ---- Common training args ----
CURR_FRACS="0.1,0.1,0.2,0.2,0.2"
WINDOW_MODE="mixed"         # use "sequential" if you want strict sliding order
STEPS=50000                # total step budget (evenly split across stages)
BATCH=16
WIDTH=128
DEPTH=3
NSAMPLES=10000
NTRAIN=3
BASE_LR=5e-4
PRINT_EVERY=5000
SAVE_EVERY=5000

echo "==================== TRAIN ALL SHAPES ===================="
for SH in "${SHAPES[@]}"; do
  echo "[TRAIN] ${SH}"
  python -m src.train.train_node_periodic \
    --shape "$SH" \
    --data_root "$DATA_ROOT" \
    --out_root "$MODELS_ROOT" \
    --nsamples "$NSAMPLES" \
    --ntrain "$NTRAIN" \
    --width "$WIDTH" \
    --depth "$DEPTH" \
    --steps "$STEPS" \
    --base_lr "$BASE_LR" \
    --batch_size "$BATCH" \
    --curriculum_fracs "$CURR_FRACS" \
    --window_mode "$WINDOW_MODE" \
    --print_every "$PRINT_EVERY" \
    --save_every "$SAVE_EVERY"
done

echo "==================== EXPERIMENTS ===================="
for SH in "${SHAPES[@]}"; do
  # Model path pattern matches the saver in train_node_periodic.py
  RUN_DIR=$(ls -dt "$MODELS_ROOT/$SH"/segcur_w${WIDTH}_d${DEPTH}_ntr${NTRAIN}_ns${NSAMPLES}_lr* | head -n1)
  MODEL_PATH="$RUN_DIR/${SH}_NODE_segcur.eqx"
  if [[ ! -f "$MODEL_PATH" ]]; then
    echo "[ERROR] Model not found for $SH at $MODEL_PATH"; exit 2
  fi
  echo "[EXPTS] ${SH} | model: $MODEL_PATH"

  SH_OUT="$EXPTS_ROOT/$SH"
  mkdir -p "$SH_OUT"

  # A) No-LLC, zero disturbance
  python -m src.experiments.run_experiments_periodic \
    --shape "$SH" \
    --model "$MODEL_PATH" \
    --out "$SH_OUT/no_llc_zero" \
    --no_llc_zero \
    --nsamples "$NSAMPLES" --ntrain 1

  # B) No-LLC, pulses (default direct two-pulses)
  python -m src.experiments.run_experiments_periodic \
    --shape "$SH" \
    --model "$MODEL_PATH" \
    --out "$SH_OUT/no_llc_pulses" \
    --nsamples "$NSAMPLES" --ntrain 1

  # C) With LLC, no disturbance
  python -m src.experiments.run_experiments_periodic \
    --shape "$SH" \
    --model "$MODEL_PATH" \
    --out "$SH_OUT/with_llc_none" \
    --with_llc \
    --nsamples "$NSAMPLES" --ntrain 1

  # D) With LLC, matched sine
  python -m src.experiments.run_experiments_periodic \
    --shape "$SH" \
    --model "$MODEL_PATH" \
    --out "$SH_OUT/with_llc_matched_sine" \
    --with_llc --matched --matched_type sine \
    --nsamples "$NSAMPLES" --ntrain 1

  # E) With LLC, matched const
  python -m src.experiments.run_experiments_periodic \
    --shape "$SH" \
    --model "$MODEL_PATH" \
    --out "$SH_OUT/with_llc_matched_const" \
    --with_llc --matched --matched_type const \
    --nsamples "$NSAMPLES" --ntrain 1

  # F) With LLC, unmatched sine
  python -m src.experiments.run_experiments_periodic \
    --shape "$SH" \
    --model "$MODEL_PATH" \
    --out "$SH_OUT/with_llc_unmatched_sine" \
    --with_llc --unmatched --unmatched_type sine \
    --nsamples "$NSAMPLES" --ntrain 1

  # G) With LLC, unmatched const
  python -m src.experiments.run_experiments_periodic \
    --shape "$SH" \
    --model "$MODEL_PATH" \
    --out "$SH_OUT/with_llc_unmatched_const" \
    --with_llc --unmatched --unmatched_type const \
    --nsamples "$NSAMPLES" --ntrain 1
done

echo "=== ALL DONE ==="
