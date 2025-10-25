#!/usr/bin/env bash
set -euo pipefail

# Shapes to run (adjust as needed)
SHAPES=("Sshape" "Sine" "WShape" "Worm" "Snake") 
#("Angle" "CShape" "DoubleBendedLine" "GShape" "PShape" "Sshape" "Sine" "WShape" "Worm" "Snake")

for SH in "${SHAPES[@]}"; do
  echo "==================== ${SH} ===================="

  CFG="configs/${SH}/node_train.yaml"
  OUTDIR="auto_run/outputs/expts/${SH}"
  mkdir -p "${OUTDIR}"

  echo "[TRAIN] ${SH}"
  python -m src.train.train_node --config "${CFG}"

  echo "[EXPTS] ${SH}"

  # 1) no_llc
  # 1b) pulse disturbance in task space (your run_experiments injects mid/two-pulse by default)
  python -m src.experiments.run_experiments \
    --train_yaml "${CFG}" \
    --out "${OUTDIR}/no_llc_pulse" \
    --verbose

  # 2) with_llc
  # 2a) no disturbance
  python -m src.experiments.run_experiments \
    --train_yaml "${CFG}" \
    --out "${OUTDIR}/with_llc_none" \
    --with_llc \
    --verbose

  # 2b) matched sine
  python -m src.experiments.run_experiments \
    --train_yaml "${CFG}" \
    --out "${OUTDIR}/with_llc_matched_sine" \
    --with_llc --matched --matched_type sine \
    --verbose

  # 2c) matched const
  python -m src.experiments.run_experiments \
    --train_yaml "${CFG}" \
    --out "${OUTDIR}/with_llc_matched_const" \
    --with_llc --matched --matched_type const \
    --verbose

  # 2d) unmatched sine
  python -m src.experiments.run_experiments \
    --train_yaml "${CFG}" \
    --out "${OUTDIR}/with_llc_unmatched_sine" \
    --with_llc --unmatched --unmatched_type sine \
    --verbose

  # 3) unmatched const (with_llc)
  python -m src.experiments.run_experiments \
    --train_yaml "${CFG}" \
    --out "${OUTDIR}/with_llc_unmatched_const" \
    --with_llc --unmatched --unmatched_type const \
    --verbose

done

echo "=== ALL DONE ==="
