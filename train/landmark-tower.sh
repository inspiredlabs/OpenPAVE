#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Do not inject a Homebrew Python 3.11 site-packages directory into the repo's
# Python 3.12 venv (it loads an ABI-incompatible OpenCV before the local wheel).
export PYTHONPATH=""
PY="${PYTHON:-.venv/bin/python}"
ARGS=(train/landmark_tower.py "${1:-train}" --epochs "${EPOCHS:-40}" --batch "${BATCH:-64}"
      --patience "${PATIENCE:-8}" --data-fraction "${DATA_FRACTION:-1.0}" --device "${DEVICE:-auto}")
shift $(( $# > 0 ? 1 : 0 ))
for source in "$@"; do ARGS+=(--source "$source"); done
exec "$PY" "${ARGS[@]}"
