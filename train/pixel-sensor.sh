#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="."
PY="${PYTHON:-.venv/bin/python}"
ACTION="${1:-all}"
if [[ $# -gt 0 ]]; then shift; fi
case "$ACTION" in
  test)
    exec "$PY" -m unittest train.pixel_sensor.test_projector \
      train.pixel_sensor.test_runtime train.pixel_sensor.test_palm_decoder -v
    ;;
  train)
    exec "$PY" train/pixel_sensor/train.py "$@"
    ;;
  benchmark)
    exec "$PY" -m train.pixel_sensor.benchmark
    ;;
  palm-inspect)
    exec "$PY" -m train.pixel_sensor.palm_decoder
    ;;
  palm-smoke)
    TF_PY="${OPENPAVE_TF_PYTHON:-/opt/anaconda3/envs/openpave-tf/bin/python}"
    exec /usr/bin/env -u PYTHONPATH TF_CPP_MIN_LOG_LEVEL=2 "$TF_PY" \
      -m train.pixel_sensor.tflite_smoke
    ;;
  all)
    "$PY" -m unittest train.pixel_sensor.test_projector \
      train.pixel_sensor.test_runtime train.pixel_sensor.test_palm_decoder -v
    "$PY" train/pixel_sensor/train.py "$@"
    exec "$PY" -m train.pixel_sensor.benchmark
    ;;
  *)
    echo "usage: train/pixel-sensor.sh {test|train|benchmark|palm-inspect|palm-smoke|all} [train options]" >&2
    exit 2
    ;;
esac
