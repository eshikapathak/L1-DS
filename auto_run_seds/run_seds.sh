#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
# Shapes to run (adjust as needed)
# SHAPES=("Angle" "CShape" "DoubleBendedLine" "GShape" "PShape" "Sshape" "Sine" "WShape" "Worm" "Snake")
# SHAPES=("DoubleBendedLine" "GShape" "PShape" "Sshape" "Sine" "WShape" "Worm" "Snake")
SHAPES=("Sshape" "Sine" "Snake")


# Fixed training parameters
K_GAUSSIANS=8 #6
NSAMPLES_TRAIN=1000 # Resample points for training data
NSAMPLES_EXPT=1000  # Number of points for experiment references/time grid (should match resample generally)
# Root directory for all outputs of this script
AUTO_RUN_ROOT="auto_run_seds"

# --- Main Loop ---
for SH in "${SHAPES[@]}"; do
  echo ""
  echo "==================== Processing Shape: ${SH} ===================="

  # --- Directories ---
  SHAPE_OUT_ROOT="${AUTO_RUN_ROOT}/${SH}"
  MODEL_OUT_DIR="${SHAPE_OUT_ROOT}/trained_model"
  EXPTS_OUT_BASE="${SHAPE_OUT_ROOT}/experiments"
  mkdir -p "${MODEL_OUT_DIR}"
  mkdir -p "${EXPTS_OUT_BASE}"

  # --- Training ---
  echo "[TRAIN] Shape: ${SH}, K=${K_GAUSSIANS}, Resample=${NSAMPLES_TRAIN}"
  MODEL_BASENAME="${SH}_k${K_GAUSSIANS}_trainResamp${NSAMPLES_TRAIN}"
  EXPECTED_MODEL_FILE="${MODEL_OUT_DIR}/${MODEL_BASENAME}_model.pkl"

  python -m SEDS.seds_train \
    --shape "${SH}" \
    -k ${K_GAUSSIANS} \
    --nsamples_train ${NSAMPLES_TRAIN} \
    -o "${MODEL_OUT_DIR}" \
    --no_show_plots

  # Check if model training was successful
  if [ ! -f "${EXPECTED_MODEL_FILE}" ]; then
    echo "[ERROR] Model file not found after training: ${EXPECTED_MODEL_FILE}"
    echo "Skipping experiments for ${SH}."
    continue # Skip to the next shape
  fi
  echo "[TRAIN] Model saved to ${EXPECTED_MODEL_FILE}"


  # --- Experiments ---
  echo "[EXPTS] Running experiment suite for ${SH} using ${EXPECTED_MODEL_FILE}"

  # Base command arguments for run_experiments_seds
  BASE_EXPT_ARGS=(
    --model_path "${EXPECTED_MODEL_FILE}"
    --shape "${SH}"
    --nsamples ${NSAMPLES_EXPT} # Used for avg demo calc and reference length in expts
    --verbose
  )

  # 1) no_llc (with default direct disturbance from run_experiments_seds)
  echo "  Running: no_llc_pulse"
  python -m src.experiments.run_experiments_seds \
    "${BASE_EXPT_ARGS[@]}" \
    --out "${EXPTS_OUT_BASE}/no_llc_pulse"

  # 2) with_llc
  echo "  Running: with_llc_none"
  python -m src.experiments.run_experiments_seds \
    "${BASE_EXPT_ARGS[@]}" \
    --out "${EXPTS_OUT_BASE}/with_llc_none" \
    --with_llc

  echo "  Running: with_llc_matched_sine"
  python -m src.experiments.run_experiments_seds \
    "${BASE_EXPT_ARGS[@]}" \
    --out "${EXPTS_OUT_BASE}/with_llc_matched_sine" \
    --with_llc --matched --matched_type sine

  echo "  Running: with_llc_matched_const"
  python -m src.experiments.run_experiments_seds \
    "${BASE_EXPT_ARGS[@]}" \
    --out "${EXPTS_OUT_BASE}/with_llc_matched_const" \
    --with_llc --matched --matched_type const

  echo "  Running: with_llc_unmatched_sine"
  python -m src.experiments.run_experiments_seds \
    "${BASE_EXPT_ARGS[@]}" \
    --out "${EXPTS_OUT_BASE}/with_llc_unmatched_sine" \
    --with_llc --unmatched --unmatched_type sine

  echo "  Running: with_llc_unmatched_const"
  python -m src.experiments.run_experiments_seds \
    "${BASE_EXPT_ARGS[@]}" \
    --out "${EXPTS_OUT_BASE}/with_llc_unmatched_const" \
    --with_llc --unmatched --unmatched_type const

  echo "[EXPTS] Finished experiments for ${SH}"

done

echo ""
echo "==================== ALL SHAPES DONE ===================="
