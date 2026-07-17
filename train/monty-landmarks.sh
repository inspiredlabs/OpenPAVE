#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# A user-level PYTHONPATH in this workspace can expose an incompatible Qt
# wheel during headless live replay. Every adapter establishes the repository
# root itself, so isolate all training/evaluation actions from it.
unset PYTHONPATH
ACTION="${1:-all}"
if [[ $# -gt 0 ]]; then shift; fi
OPENPAVE_PY="${PYTHON:-.venv/bin/python}"
MONTY_PY="${OPENPAVE_MONTY_PYTHON:-/opt/anaconda3/envs/tbp.monty/bin/python}"
RUN="train/runs/monty_landmark_alignment/tbp_run"
FRAMEWORK_RUN="train/runs/monty_landmark_alignment/framework_verified"

ORACLE_STUDENT="train/runs/monty_landmark_alignment/oracle_student"

case "$ACTION" in
  hanco-prepare)
    exec "$OPENPAVE_PY" train/monty_lab/tbp_adapter/prepare_hanco.py "$@"
    ;;
  hanco-target)
    exec "$OPENPAVE_PY" train/hanco_target_poc.py "$@"
    ;;
  hanco-gestures-prepare)
    exec "$OPENPAVE_PY" train/prepare_hanco_gestures.py "$@"
    ;;
  hanco-gestures)
    exec "$OPENPAVE_PY" train/hanco_crop_gesture.py "$@"
    ;;
  hanco-curriculum)
    exec "$OPENPAVE_PY" train/hanco_crop_curriculum.py "$@"
    ;;
  student)
    exec "$OPENPAVE_PY" train/monty_lab/tbp_adapter/train_oracle_student.py "$@"
    ;;
  prepare)
    exec "$OPENPAVE_PY" train/monty_lab/tbp_adapter/prepare_landmark_pairs.py "$@"
    ;;
  proposed)
    exec "$OPENPAVE_PY" train/monty_lab/tbp_adapter/benchmark_proposed_roi.py \
      --update-meta "$@"
    ;;
  live)
    exec "$OPENPAVE_PY" train/monty_lab/tbp_adapter/replay_crude_videos.py \
      --calibrate --update-meta "$@"
    ;;
  hard)
    exec "$OPENPAVE_PY" train/monty_lab/tbp_adapter/mine_acquisition_hard_examples.py "$@"
    ;;
  alternate)
    exec "$OPENPAVE_PY" train/monty_lab/tbp_adapter/run_alternating_rounds.py "$@"
    ;;
  saccade-prepare)
    exec "$OPENPAVE_PY" train/monty_lab/tbp_adapter/prepare_saccade_simulation.py "$@"
    ;;
  run)
    exec "$MONTY_PY" train/monty_lab/tbp_adapter/run_landmark_alignment.py "$@"
    ;;
  analyze)
    exec "$MONTY_PY" train/monty_lab/tbp_adapter/analyze_landmark_alignment.py "$RUN" "$@"
    ;;
  pretrain)
    # The inherited PYTHONPATH points at Homebrew Python 3.11 packages and
    # breaks cv2 inside Monty's Python 3.8 environment, so isolate this run.
    exec env -u PYTHONPATH MPLCONFIGDIR=/private/tmp/openpave-mpl \
      XDG_CACHE_HOME=/private/tmp/openpave-cache "$MONTY_PY" \
      train/monty_lab/tbp_adapter/run_framework_pretraining.py \
      --out "$FRAMEWORK_RUN" "$@"
    ;;
  replay)
    exec "$OPENPAVE_PY" train/monty_lab/tbp_adapter/render_framework_replay.py \
      --run "$FRAMEWORK_RUN/pretrained" "$@"
    ;;
  all)
    # Retrain the oracle-ROI student only when absent; FORCE=1 is deliberate.
    if [[ "${FORCE:-0}" == "1" || ! -f "$ORACLE_STUDENT/landmarker.onnx" ]]; then
      "$OPENPAVE_PY" train/monty_lab/tbp_adapter/train_oracle_student.py
    fi
    "$OPENPAVE_PY" train/monty_lab/tbp_adapter/prepare_landmark_pairs.py
    "$MONTY_PY" train/monty_lab/tbp_adapter/run_landmark_alignment.py
    "$MONTY_PY" train/monty_lab/tbp_adapter/analyze_landmark_alignment.py "$RUN"
    env -u PYTHONPATH MPLCONFIGDIR=/private/tmp/openpave-mpl \
      XDG_CACHE_HOME=/private/tmp/openpave-cache "$MONTY_PY" \
      train/monty_lab/tbp_adapter/run_framework_pretraining.py \
      --out "$FRAMEWORK_RUN"
    exec "$OPENPAVE_PY" train/monty_lab/tbp_adapter/render_framework_replay.py \
      --run "$FRAMEWORK_RUN/pretrained"
    ;;
  *)
    echo "usage: train/monty-landmarks.sh {hanco-prepare|hanco-target|hanco-gestures-prepare|hanco-gestures|hanco-curriculum|student|proposed|live|hard|alternate|saccade-prepare|prepare|run|analyze|pretrain|replay|all}" >&2
    exit 2
    ;;
esac
