#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
# Shapes to run (must match those used in run_seds_suite.sh)
SHAPES=("Angle" "CShape" "DoubleBendedLine" "GShape" "PShape" "Sshape" "Sine" "WShape" "Worm" "Snake")
# Training parameters used to find the correct model file
K_GAUSSIANS_DEFAULT=6 # Default K value
NSAMPLES_TRAIN=1000 # Resample points used *during training*
NSAMPLES_EXPT=1000  # Number of points for experiment references/time grid (should match resample)
# Root directory where trained models were saved by run_seds_suite.sh
MODEL_ROOT="auto_run_seds"
# Root directory for all outputs of *this* script
OUT_ROOT="auto_run_seds_least_effort"

# --- Helper Function ---
run_case () {
  local SHAPE="$1"
  local MODEL_PATH="$2"
  local OUTDIR="$3"
  shift 3 # Shift shape, model_path, outdir off the arguments
  # The rest of the arguments ($@) are flags for run_experiments_seds

  echo "[RUN] $SHAPE | Selector: least_effort | Condition: $*"
  python -m src.experiments.run_experiments_seds \
    --model_path "${MODEL_PATH}" \
    --shape "${SHAPE}" \
    --nsamples ${NSAMPLES_EXPT} \
    --out "${OUTDIR}" \
    --selector least_effort \
    "$@" # Pass remaining flags (like --with_llc, --matched, --verbose, etc.)
}

# --- Main Loop ---
for SHAPE in "${SHAPES[@]}"; do
  echo ""
  echo "============== Processing Shape: ${SHAPE} (Least-Effort Selector) =============="

  # --- Determine K value for this shape ---
  # Removed 'local' - k_val will be scoped to the loop iteration
  k_val=${K_GAUSSIANS_DEFAULT} # Start with default
  if [[ "${SHAPE}" == "Sshape" || "${SHAPE}" == "Sine" || "${SHAPE}" == "Snake" ]]; then
    k_val=8
    echo "[INFO] Using K=${k_val} for shape ${SHAPE}"
  else
    echo "[INFO] Using default K=${k_val} for shape ${SHAPE}"
  fi

  # --- Construct path to the pre-trained SEDS model ---
  RESAMPLE_TAG="_trainResamp${NSAMPLES_TRAIN}" # Match tag from training script
  MODEL_BASENAME="${SHAPE}_k${k_val}${RESAMPLE_TAG}" # Use k_val here
  MODEL_FILE="${MODEL_ROOT}/${SHAPE}/trained_model/${MODEL_BASENAME}_model.pkl"

  # --- Check if model exists ---
  if [ ! -f "${MODEL_FILE}" ]; then
    echo "[WARNING] Trained SEDS model not found for ${SHAPE} (K=${k_val}) at: ${MODEL_FILE}"
    echo "Skipping experiments for this shape."
    continue # Skip to the next shape
  fi
  echo "[INFO] Using model: ${MODEL_FILE}"

  # --- Define output directory for this shape ---
  SHAPE_OUTDIR="${OUT_ROOT}/${SHAPE}"
  mkdir -p "${SHAPE_OUTDIR}"

  # --- Run Experiment Cases ---

  # 1) no_llc + direct task-space pulsing (default disturbance in run_experiments_seds)
  run_case "${SHAPE}" "${MODEL_FILE}" "${SHAPE_OUTDIR}/no_llc_pulse" --verbose

  # 2) with_llc cases (plant disturbances)
  run_case "${SHAPE}" "${MODEL_FILE}" "${SHAPE_OUTDIR}/with_llc_none" --with_llc --verbose
  run_case "${SHAPE}" "${MODEL_FILE}" "${SHAPE_OUTDIR}/with_llc_matched_sine" --with_llc --matched --matched_type sine --verbose
  run_case "${SHAPE}" "${MODEL_FILE}" "${SHAPE_OUTDIR}/with_llc_matched_const" --with_llc --matched --matched_type const --verbose
  run_case "${SHAPE}" "${MODEL_FILE}" "${SHAPE_OUTDIR}/with_llc_unmatched_sine" --with_llc --unmatched --unmatched_type sine --verbose
  run_case "${SHAPE}" "${MODEL_FILE}" "${SHAPE_OUTDIR}/with_llc_unmatched_const" --with_llc --unmatched --unmatched_type const --verbose

  echo "[INFO] Finished least-effort experiments for ${SHAPE}"

done

echo ""
echo "==================== LEAST-EFFORT SUITE FINISHED ===================="

