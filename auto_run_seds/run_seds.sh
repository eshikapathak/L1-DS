#!/usr/bin/env bash
set -euo pipefail

# -------- Shapes --------
# Example sets:
SHAPES=("Angle" "CShape" "DoubleBendedLine" "GShape" "PShape" "Sshape" "Sine" "WShape" "Worm" "Snake")
# SHAPES=("DoubleBendedLine" "GShape" "PShape" "Sshape" "Sine" "WShape" "Worm" "Snake")
# SHAPES=("Sshape" "Sine" "Snake")

# -------- Globals --------
NSAMPLES_TRAIN=1000   # training resample count
NSAMPLES_EXPT=1000    # experiment time grid length (keep same as training)
AUTO_RUN_ROOT="auto_run_seds"

# -------- Helper: K per shape --------
get_k_for_shape () {
  case "$1" in
    Sshape|Sine|Snake) echo 8 ;;
    *)                 echo 6 ;;
  esac
}

# -------- Helper: run both selectors (DTW + Least-Effort) --------
run_both_selectors () {
  # Usage: run_both_selectors <base-args...> --out OUTDIR [extra flags...]
  local outdir prev=""
  for arg in "$@"; do
    if [[ "$prev" == "--out" ]]; then outdir="$arg"; break; fi
    prev="$arg"
  done

  echo "    [DTW]   -> ${outdir}"
  python -m src.experiments.run_experiments_seds "$@"

  # echo "    [LEAST] -> ${outdir}_le"
  # python -m src.experiments.run_experiments_seds "$@" --selector least_effort --out "${outdir}_le"
}

# ==================== Main Loop ====================
for SH in "${SHAPES[@]}"; do
  echo ""
  echo "==================== Processing Shape: ${SH} ===================="

  # K per shape
  K_GAUSSIANS=$(get_k_for_shape "${SH}")
  echo "[CFG] Shape=${SH} | K=${K_GAUSSIANS} | train_resample=${NSAMPLES_TRAIN} | expt_resample=${NSAMPLES_EXPT}"

  # --- Directories ---
  SHAPE_OUT_ROOT="${AUTO_RUN_ROOT}/${SH}"
  MODEL_OUT_DIR="${SHAPE_OUT_ROOT}/trained_model"
  EXPTS_OUT_BASE="${SHAPE_OUT_ROOT}/experiments_new_dist_high_freq"
  mkdir -p "${MODEL_OUT_DIR}" "${EXPTS_OUT_BASE}"

  # --- Training ---
  MODEL_BASENAME="${SH}_k${K_GAUSSIANS}_trainResamp${NSAMPLES_TRAIN}"
  EXPECTED_MODEL_FILE="${MODEL_OUT_DIR}/${MODEL_BASENAME}_model.pkl"

  # echo "[TRAIN] ${SH} -> ${EXPECTED_MODEL_FILE}"
  # python -m SEDS.seds_train \
  #   --shape "${SH}" \
  #   -k ${K_GAUSSIANS} \
  #   --nsamples_train ${NSAMPLES_TRAIN} \
  #   -o "${MODEL_OUT_DIR}" \
  #   --no_show_plots

  if [[ ! -f "${EXPECTED_MODEL_FILE}" ]]; then
    echo "[ERROR] Model file not found after training: ${EXPECTED_MODEL_FILE}"
    echo "Skipping experiments for ${SH}."
    continue
  fi
  echo "[OK] Model ready."

  # --- Base args for experiments ---
  BASE_EXPT_ARGS=(
    --model_path "${EXPECTED_MODEL_FILE}"
    --shape "${SH}"
    --nsamples ${NSAMPLES_EXPT}
    --verbose
  )

  # ---------------- No-LLC (direct pulses) ----------------
  echo ""
  echo "----- NO-LLC (DIRECT) -----"
  echo "[RUN] no_llc_pulses"
  run_both_selectors \
    "${BASE_EXPT_ARGS[@]}" \
    --out "${EXPTS_OUT_BASE}/no_llc_pulses"

  # ---------------- WITH-LLC: matched only ----------------
  echo ""
  echo "----- WITH-LLC (MATCHED ONLY) -----"
  # for KIND in const chirp multisine pulse; do
  for KIND in multisine; do

    echo "[RUN] with_llc | matched=${KIND}"
    run_both_selectors \
      "${BASE_EXPT_ARGS[@]}" \
      --out "${EXPTS_OUT_BASE}/with_llc_matched_${KIND}" \
      --with_llc --matched --matched_type "${KIND}"
  done

  # ---------------- WITH-LLC: unmatched only ----------------
  echo ""
  echo "----- WITH-LLC (UNMATCHED ONLY) -----"
  for KIND in const chirp multisine pulse; do
    echo "[RUN] with_llc | unmatched=${KIND}"
    run_both_selectors \
      "${BASE_EXPT_ARGS[@]}" \
      --out "${EXPTS_OUT_BASE}/with_llc_unmatched_${KIND}" \
      --with_llc --unmatched --unmatched_type "${KIND}"
  done

  # ---------------- WITH-LLC: combo ----------------
  echo ""
  echo "----- WITH-LLC (COMBO) -----"
  echo "[RUN] with_llc | matched=multisine + unmatched=pulse"
  run_both_selectors \
    "${BASE_EXPT_ARGS[@]}" \
    --out "${EXPTS_OUT_BASE}/with_llc_matched_multisine_unmatched_pulse" \
    --with_llc \
    --matched --matched_type multisine \
    --unmatched --unmatched_type pulse

  echo "-----------------------------------------------------"
  echo "[DONE] ${SH}"
  echo "-----------------------------------------------------"
done

echo ""
echo "==================== ALL SHAPES DONE ===================="
