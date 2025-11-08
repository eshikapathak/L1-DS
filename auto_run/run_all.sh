#!/usr/bin/env bash
set -euo pipefail

# -------- Config --------
# SHAPES=("Sshape" "Sine" "WShape" "Worm" "Snake")
# e.g. full set:
SHAPES=("Angle" "CShape" "DoubleBendedLine" "GShape" "PShape" "Sshape" "Sine" "WShape" "Worm" "Snake")

# -------- Helpers --------
run_both_selectors () {
  # Usage: run_both_selectors --train_yaml CFG --out OUTDIR [other flags...]
  local outdir prev=""
  for arg in "$@"; do
    if [[ "$prev" == "--out" ]]; then outdir="$arg"; break; fi
    prev="$arg"
  done
  prev=""

  echo "    [DTW]   -> ${outdir}"
  python -m src.experiments.run_experiments "$@"

  # echo "    [LEAST] -> ${outdir}_le"
  # python -m src.experiments.run_experiments "$@" --selector least_effort --out "${outdir}_le"
}

echo "==================== LASA AUTO RUN ===================="

for SH in "${SHAPES[@]}"; do
  echo ""
  echo "======================================================="
  echo "[SHAPE] ${SH}"
  echo "======================================================="

  CFG="configs/${SH}/node_train.yaml"
  OUTDIR="auto_run/outputs_newdist_high_freq/expts/${SH}"
  mkdir -p "${OUTDIR}"

  # echo "[TRAIN] ${SH}"
  # python -m src.train.train_node --config "${CFG}"

  echo ""
  echo "----- NO-LLC (DIRECT) -----"
  echo "[RUN] no_llc_pulses (direct mid-pulses default)"
  run_both_selectors \
    --train_yaml "${CFG}" \
    --out "${OUTDIR}/no_llc_pulses" \
    --verbose

  echo ""
  echo "----- WITH-LLC (MATCHED ONLY) -----"
  for KIND in const chirp multisine pulse; do
    echo "[RUN] with_llc | matched=${KIND}"
    run_both_selectors \
      --train_yaml "${CFG}" \
      --out "${OUTDIR}/with_llc_matched_${KIND}" \
      --with_llc --matched --matched_type "${KIND}" \
      --verbose
  done

  echo ""
  echo "----- WITH-LLC (UNMATCHED ONLY) -----"
  for KIND in const chirp multisine pulse; do
    echo "[RUN] with_llc | unmatched=${KIND}"
    run_both_selectors \
      --train_yaml "${CFG}" \
      --out "${OUTDIR}/with_llc_unmatched_${KIND}" \
      --with_llc --unmatched --unmatched_type "${KIND}" \
      --verbose
  done

  echo ""
  echo "----- WITH-LLC (COMBO) -----"
  echo "[RUN] with_llc | matched=multisine + unmatched=pulse"
  run_both_selectors \
    --train_yaml "${CFG}" \
    --out "${OUTDIR}/with_llc_matched_multisine_unmatched_pulse" \
    --with_llc \
    --matched --matched_type multisine \
    --unmatched --unmatched_type pulse \
    --verbose

  echo "-------------------------------------------------------"
  echo "[DONE] ${SH}"
  echo "-------------------------------------------------------"
done

echo ""
echo "=== ALL DONE ==="
