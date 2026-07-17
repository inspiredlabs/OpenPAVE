"""Swipe-phase distillation: dynamic_gestures' crops_classifier as a MOTION GATE.

Their classifier judges swipe PHASE-states from 128² hand crops — but their
detector finds nothing on the blurred crude captures, so the hybrid is:

  our MediaPipe landmarker (works on crude) -> crop -> THEIR phase classifier

The labels do NOT come from their model (direction names for its 45 outputs
are unpublished and our filenames already carry ground truth); its confidence
is used as a GATE: a confident phase-state means the hand is genuinely
mid-swipe, so only those frames enter the shard. Result: a v2-format shard
(datasets/swipe_phases/prepared.npz) of clean, in-motion, direction-labelled
frames that train-v3's sequence head can consume as an ordered source.

Run:  .venv/bin/python train/phase_distill.py            # builds the shard
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

TRAIN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TRAIN_DIR))
from mediapipe_svm import LANDMARKER, _read_frames, _resolve_source  # noqa: E402
from gesture_lab import IMG, _lm_crop_box, _take_crop  # noqa: E402

MODELS = TRAIN_DIR / "datasets" / "dynamic_gestures" / "models"
OUT = TRAIN_DIR / "datasets" / "swipe_phases" / "prepared.npz"
PHASE_CONF = 0.60      # softmax floor: confident phase == genuinely mid-swipe
# motion videos only; label = the file's ground-truth direction
MOTION_SOURCES = {
    "right-to-left": "point_left",
    "left-to-right": "point_right",
    "up": "point_vertical",
    "down": "point_vertical",
}


def main() -> None:
    import cv2
    import onnxruntime as ort
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    lm = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(LANDMARKER)),
        num_hands=1, min_hand_detection_confidence=0.5))
    phase = ort.InferenceSession(str(MODELS / "crops_classifier.onnx"),
                                 providers=["CPUExecutionProvider"])

    def phase_conf(rgb_crop: np.ndarray) -> float:
        x = cv2.resize(rgb_crop, (128, 128)).astype(np.float32)
        x = ((x - 127.0) / 128.0).transpose(2, 0, 1)[None]
        logits = phase.run(None, {"input": x})[0][0]
        e = np.exp(logits - logits.max())
        return float((e / e.sum()).max())

    imgs, labels, is_val, lms, has_lm = [], [], [], [], []
    t0 = time.perf_counter()
    for stem, label in MOTION_SOURCES.items():
        src = _resolve_source(stem)
        if src is None:
            continue
        frames = _read_frames(src)
        val_from = int(len(frames) * 0.8)
        kept = gated = 0
        for i, f in enumerate(frames):
            res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=f))
            if not res.hand_landmarks:
                continue
            hand = res.hand_landmarks[0]
            lm42 = np.array([[q.x, q.y] for q in hand], dtype=np.float32).reshape(-1)
            crop = _take_crop(f, *_lm_crop_box(lm42))
            if phase_conf(crop) < PHASE_CONF:
                gated += 1
                continue                      # idle/transition frame — not swipe motion
            imgs.append(cv2.resize(f, (IMG, IMG)))
            labels.append(label)
            is_val.append(i >= val_from)
            lms.append(lm42)
            has_lm.append(True)
            kept += 1
        print(f"[phase] {src.name:26s} -> {label:14s} kept {kept}, gated out {gated} idle frames")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, imgs=np.array(imgs, dtype=np.uint8),
                        labels=np.array(labels), is_val=np.array(is_val),
                        landmarks=np.array(lms, dtype=np.float32),
                        has_lm=np.array(has_lm))
    u, c = np.unique(labels, return_counts=True)
    print(f"[phase] {len(imgs)} motion-gated frames in {time.perf_counter() - t0:.0f}s -> {OUT}"
          f"\n        " + ", ".join(f"{l}={n}" for l, n in zip(u, c)))


if __name__ == "__main__":
    main()
