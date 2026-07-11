#!/usr/bin/env bash
# OpenPAVE gesture-detector training procedure (HaGRID -> YOLO nano -> Mali).
#
# Everything is an env var with a sane default, so a plain
#
#   ./train/HaGRID.sh all
#
# sources the data (once), trains yolo26n @ 320px on the 5 intent proxy
# classes + no_gesture, and exports ONNX + NCNN. Stages, if you want them
# separately: prepare | train | export | bench | all
#
# ── If YOLO26n is not fast enough (or its NCNN export misbehaves) ──────────
# Walk DOWN this ladder — each rung is a one-line change, same data, same
# script. Re-run `bench` after each and compare the median ms:
#
#   MODEL=yolo26n.pt  ./train/HaGRID.sh all     # newest; NMS-free, best on paper
#   MODEL=yolo11n.pt  ./train/HaGRID.sh all     # 1 year of NCNN deployment hardening
#   MODEL=yolov8n.pt  ./train/HaGRID.sh all     # ncnn repo ships a ready yolov8.cpp example
#
# Still too slow on the Orion? Shrink the input before reaching for a bigger
# knife — accuracy on close-range hands degrades slowly, latency drops fast:
#
#   IMGSZ=256 ./train/HaGRID.sh train
#   IMGSZ=224 ./train/HaGRID.sh train
#
# ── Slashing the dataset / pruning classes ──────────────────────────────────
# The detector only knows the classes you list here; fewer classes = smaller
# problem. Add real point-left/point-right captures later by extending
# CLASSES and re-running prepare (delete train/HaGRID/images+labels first if
# you change the class list, so indices stay consistent):
#
#   CLASSES=like,stop,fist PER_CLASS=150 ./train/HaGRID.sh all
#
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
STAGE="${1:-all}"

MODEL="${MODEL:-yolo26n.pt}"
IMGSZ="${IMGSZ:-320}"
EPOCHS="${EPOCHS:-60}"
BATCH="${BATCH:-32}"
# 7 intent classes; point_* are synthesized by rotating HaGRID's "one"
# (index-finger-up) frames — see hagrid_yolo.py spec syntax alias=source@deg.
CLASSES="${CLASSES:-stop,fist,like,point_up=one@0,point_right=one@90,point_down=one@180,point_left=one@270}"
PER_CLASS="${PER_CLASS:-300}"
DEVICE="${DEVICE:-mps}"     # ultralytics auto picks CPU on this Mac; mps trains ~3x faster.
                            # bench always times on CPU regardless (honest Mali proxy).
FORMATS="${FORMATS:-onnx,ncnn}"

$PY -c "import ultralytics" 2>/dev/null || $PY -m pip install -q ultralytics

exec $PY train/hagrid_yolo.py "$STAGE" \
  --model "$MODEL" \
  --imgsz "$IMGSZ" \
  --epochs "$EPOCHS" \
  --batch "$BATCH" \
  --classes "$CLASSES" \
  --per-class "$PER_CLASS" \
  --device "$DEVICE" \
  --formats "$FORMATS"
