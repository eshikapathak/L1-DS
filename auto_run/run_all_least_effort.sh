#!/usr/bin/env bash
set -euo pipefail

# SHAPES=(Worm CShape GShape Leaf_2 Sine)
SHAPES=(Angle CShape DoubleBendedLine GShape PShape Sshape Sine WShape Worm Snake)
CFG_DIR="configs"
OUT_ROOT="auto_run/outputs/expts_least_effort"

run_case () {
  local SHAPE="$1"
  local OUTDIR="$2"
  shift 2
  echo "[RUN] $SHAPE | $*"
  python -m src.experiments.run_experiments \
    --train_yaml "${CFG_DIR}/${SHAPE}/node_train.yaml" \
    --out "${OUTDIR}" \
    --selector least_effort \
    "$@"
}

for SHAPE in "${SHAPES[@]}"; do
  echo "============== ${SHAPE} =============="
  OUTDIR="${OUT_ROOT}/${SHAPE}"

  # no_llc + direct task-space pulsing (from run_experiments)
  run_case "${SHAPE}" "${OUTDIR}" --verbose

  # with_llc cases (plant disturbances)
  run_case "${SHAPE}" "${OUTDIR}" --with_llc --verbose
  run_case "${SHAPE}" "${OUTDIR}" --with_llc --matched --matched_type sine --verbose
  run_case "${SHAPE}" "${OUTDIR}" --with_llc --matched --matched_type const --verbose
  run_case "${SHAPE}" "${OUTDIR}" --with_llc --unmatched --unmatched_type sine --verbose
  run_case "${SHAPE}" "${OUTDIR}" --with_llc --unmatched --unmatched_type const --verbose
done

echo "[DONE] Least-effort selector suite finished."
