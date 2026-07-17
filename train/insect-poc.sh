#!/usr/bin/env bash
# Modular GPU-first trainer for the OpenPAVE insect-brain POC.
#
# Examples:
#   ./train/insect-poc.sh all                         # demo data + every specialist + bench
#   ./train/insect-poc.sh train motion               # replace ONLY motion; preserve all others
#   BACKEND=mlx  ./train/insect-poc.sh train color   # require Apple Metal/MLX
#   BACKEND=cuml ./train/insect-poc.sh train color   # future NVIDIA/CUDA fallback
#   ./train/insect-poc.sh assemble                    # rebuild ensemble manifest, no training
#   ./train/insect-poc.sh bench presence motion      # CPU/Pi-style portable inference benchmark
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PYTHON:-.venv/bin/python}"
COMMAND="${1:-list}"
if (($#)); then shift; fi

BACKEND="${BACKEND:-auto}"
CONFIG="${CONFIG:-train/insect-poc/config.json}"
DATA_DIR="${DATA_DIR:-train/insect-poc/data}"
RUNS_DIR="${RUNS_DIR:-train/insect-poc/runs}"
SAMPLES="${SAMPLES:-1200}"
ITERATIONS="${ITERATIONS:-1000}"
SEED="${SEED:-17}"

case "$COMMAND" in
  list|prepare-demo|train|assemble|bench|all|prepare-core-demo|train-core|fetch-ipn|prepare-ipn-core|prepare-ipn-direction) ;;
  *) echo "usage: $0 {list|prepare-demo|train|assemble|bench|all|prepare-core-demo|train-core|fetch-ipn|prepare-ipn-core|prepare-ipn-direction} [specialist ...]" >&2; exit 2 ;;
esac

# Keep the selected interpreter isolated. Inheriting a Homebrew Python 3.11
# site-packages path into the repo's Python 3.12 venv can load an ABI-incompatible
# OpenCV/PyQt wheel before the correct local package.
export PYTHONPATH="train/insect-poc"
ARGS=(-m insect_poc.cli "$COMMAND" \
  --backend "$BACKEND" --config "$CONFIG" --data-dir "$DATA_DIR" --runs-dir "$RUNS_DIR" \
  --samples "$SAMPLES" --iterations "$ITERATIONS" --seed "$SEED" \
  --minutes "${MINUTES:-15}" --fps "${FPS:-15}" --core-units "${CORE_UNITS:-256}" \
  --core-epochs "${CORE_EPOCHS:-120}" --raw-dir "${RAW_DIR:-train/insect-poc/raw/ipn}" \
  --ipn-shard "${IPN_SHARD:-1}")
for name in "$@"; do
  ARGS+=(--specialist "$name")
done
exec "$PY" "${ARGS[@]}"
