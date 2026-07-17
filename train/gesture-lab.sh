#!/usr/bin/env bash
# Gesture lab — multi-source distillation trainer for the TinyNet variant.
# Same shape as ./train/insect-poc.sh: stages + per-source selection, and
# prepared data is cached in train/datasets/ so nothing is ever refetched or
# relabelled by accident (see train/datasets/README.md).
#
# Examples:
#   ./train/gesture-lab.sh list                      # what exists already? START HERE
#   ./train/gesture-lab.sh all                       # fetch + prepare + train + eval
#   ./train/gesture-lab.sh prepare hagrid            # teacher-label ONLY hagrid
#   PER_SOURCE=8000 ./train/gesture-lab.sh prepare hagrid
#   EPOCHS=50 ./train/gesture-lab.sh train           # retrain from cached shards
#   FORCE=1 ./train/gesture-lab.sh prepare crude     # relabel after new captures
#
# Sources: crude (your videos) · custom (drop-ins) · hagrid (HF-cached zip)
#          interhand26m / h2o (manual, registration-gated — README has steps)
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-.venv/bin/python}"
STAGE="${1:-list}"
if (($#)); then shift; fi

ARGS=(train/gesture_lab.py "$STAGE"
  --per-source "${PER_SOURCE:-4000}"
  --teacher-conf "${TEACHER_CONF:-0.85}"
  --epochs "${EPOCHS:-30}"
  --trunk-width "${TRUNK_WIDTH:-1}"
  --data-fraction "${DATA_FRACTION:-1.0}")
[[ "${FORCE:-0}" == "1" ]] && ARGS+=(--force)
for src in "$@"; do
  ARGS+=(--source "$src")
done
exec "$PY" "${ARGS[@]}"
