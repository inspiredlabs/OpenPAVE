"""Distil the MediaPipe+SVM gesture pipeline into one tiny CNN (the A/B rig).

TEACHER: the shipping pipeline — MediaPipe hand landmarker -> shape SVM ->
index-finger direction geometry (train/mediapipe_svm.py). It labels every
frame of the train/crude capture videos with the OUTCOME vocabulary:

  stop -> STOP    fist -> HOME    like -> TROT
  point + horizontal   -> point_left / point_right (subject-centric)
  point + vertical     -> point_vertical (no-op)
  no hand / unsure     -> no_hand

STUDENT: a ~100k-param CNN, 128x128 RGB frame -> outcome distribution, in ONE
pass. No landmark stage, no palm detector — the landmarks exist only at
training time, inside the teacher. Target: well under 2ms/frame on CPU.

What this buys and what it costs (the A/B against the baseline):
  + latency: ~0.5ms vs ~5ms (the landmark CNN is gone)
  + one artifact, same ONNX runtime as the YOLO path
  - brittleness: the student sees PIXELS, so unlike the landmark pipeline it
    is NOT invariant to scene/lighting/clothing by construction — it knows
    only the world in train/crude. Judge it with `eval` + live A/B, and
    retrain by re-running this file when captures grow.

Ambiguous teacher frames (hand present but SVM conf below --teacher-conf)
are DROPPED, not taught: distillation should copy decisions, not confusion.
Horizontal-flip augmentation swaps point_left/point_right labels.

Output: train/runs/tiny_gesture/{model.onnx (fp32), meta.json}
Run:  .venv/bin/python train/tiny_gesture.py train
      .venv/bin/python train/tiny_gesture.py eval    # outcome-level, same
        held-out tails + scoring as mediapipe_svm.py eval
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

TRAIN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TRAIN_DIR))
from mediapipe_svm import (  # noqa: E402 — teacher components, single source of truth
    EXPECTED_OUTCOME, LANDMARKER, SOURCES, VAL_FRAC, _read_frames, _resolve_source,
    landmarks_to_features, point_direction,
)

OUT_DIR = TRAIN_DIR / "runs" / "tiny_gesture"
SVM_DIR = TRAIN_DIR / "runs" / "mediapipe_svm"
IMG = 128
CLASSES = ["fist", "like", "no_hand", "point_left", "point_right", "point_vertical", "stop"]
FLIP_SWAP = {"point_left": "point_right", "point_right": "point_left"}


def _teacher_label_frames(teacher_conf: float):
    """Run the full teacher pipeline over every crude video -> (imgs, labels,
    is_val). Ambiguous frames are dropped; genuine no-hand frames are kept."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    import onnxruntime as ort

    meta = json.loads((SVM_DIR / "meta.json").read_text())
    svm_classes = meta["classes"]
    sess = ort.InferenceSession(str(SVM_DIR / "model.onnx"), providers=["CPUExecutionProvider"])
    lm = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(LANDMARKER)),
        num_hands=1, min_hand_detection_confidence=0.5))

    imgs, labels, is_val = [], [], []
    counts: dict[str, int] = {}
    for stem in SOURCES:
        src = _resolve_source(stem)
        if src is None:
            continue
        frames = _read_frames(src)
        val_from = int(len(frames) * (1 - VAL_FRAC))
        for i, frame in enumerate(frames):
            res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame))
            if not res.hand_landmarks:
                label = "no_hand"
            else:
                hand = res.hand_landmarks[0]
                feats = landmarks_to_features(hand).reshape(1, -1)
                proba = sess.run(["probabilities"], {"landmarks": feats})[0][0]
                if float(proba.max()) < teacher_conf:
                    continue                       # don't teach confusion
                cls = svm_classes[int(proba.argmax())]
                if cls == "point":
                    d = point_direction(hand)
                    label = {"LEFT": "point_left", "RIGHT": "point_right"}.get(d, "point_vertical")
                else:
                    label = cls
            imgs.append(cv2.resize(frame, (IMG, IMG)))
            labels.append(label)
            is_val.append(i >= val_from)
            counts[label] = counts.get(label, 0) + 1
    print(f"[tiny] teacher labelled {len(imgs)} frames: "
          + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return np.array(imgs), np.array(labels), np.array(is_val)


def _build_net(n_classes: int):
    import torch.nn as nn

    def block(cin, cout):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=2, padding=1),
                             nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    return nn.Sequential(
        block(3, 16), block(16, 32), block(32, 64), block(64, 128),   # 128 -> 8
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, n_classes),
    )


def train(args: argparse.Namespace) -> None:
    import torch
    import torch.nn as nn

    imgs, labels, is_val = _teacher_label_frames(args.teacher_conf)
    y = np.array([CLASSES.index(c) for c in labels], dtype=np.int64)
    x = imgs.astype(np.float32) / 255.0 - 0.5      # HWC RGB -> normalised
    x = x.transpose(0, 3, 1, 2)                    # NCHW

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    net = _build_net(len(CLASSES)).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    lossf = nn.CrossEntropyLoss()
    xtr, ytr = torch.tensor(x[~is_val]), torch.tensor(y[~is_val])
    xva, yva = torch.tensor(x[is_val]).to(device), torch.tensor(y[is_val]).to(device)
    flip_map = torch.tensor([CLASSES.index(FLIP_SWAP.get(c, c)) for c in CLASSES])

    t0 = time.perf_counter()
    best_acc, best_state = 0.0, None
    for epoch in range(args.epochs):
        net.train()
        perm = torch.randperm(len(xtr))
        for b in range(0, len(perm), 64):
            idx = perm[b:b + 64]
            xb, yb = xtr[idx], ytr[idx]
            flip = torch.rand(len(xb)) < 0.5       # mirror aug swaps left/right labels
            xb[flip] = torch.flip(xb[flip], dims=[3])
            yb[flip] = flip_map[yb[flip]]
            xb = (xb + torch.randn_like(xb) * 0.02).contiguous()  # mild photometric noise
            opt.zero_grad()
            loss = lossf(net(xb.to(device)), yb.to(device))
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            acc = float((net(xva).argmax(1) == yva).float().mean())
        if acc > best_acc:
            best_acc, best_state = acc, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    print(f"[tiny] {n_params / 1e3:.0f}k params, {args.epochs} epochs in "
          f"{time.perf_counter() - t0:.0f}s on {device}; best val acc {best_acc:.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    net = nn.Sequential(net.cpu().eval(), nn.Softmax(dim=1))   # probs in-graph
    torch.onnx.export(net, torch.zeros(1, 3, IMG, IMG), str(OUT_DIR / "model.onnx"),
                      input_names=["frames"], output_names=["probabilities"],
                      dynamic_axes={"frames": {0: "n"}, "probabilities": {0: "n"}},
                      dynamo=False)  # legacy exporter: no onnxscript dependency
    onnx_kb = (OUT_DIR / "model.onnx").stat().st_size // 1024
    meta = {
        "classes": CLASSES,
        "input_px": IMG,
        "params": n_params,
        "precision": "fp32 (model.onnx)",
        "teacher": "mediapipe landmarker + shape SVM + direction geometry",
        "teacher_conf": args.teacher_conf,
        "val_acc": best_acc,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[tiny] saved model.onnx ({onnx_kb} KB)")


def evaluate(args: argparse.Namespace) -> None:
    """Same outcome-level scoreboard as mediapipe_svm.py eval, same held-out
    tails — so the two variants are compared on identical data and metrics."""
    import cv2
    import onnxruntime as ort

    meta = json.loads((OUT_DIR / "meta.json").read_text())
    classes = meta["classes"]
    sess = ort.InferenceSession(str(OUT_DIR / "model.onnx"), providers=["CPUExecutionProvider"])
    to_outcome = {"stop": "STOP", "fist": "HOME", "like": "TROT",
                  "point_left": "LEFT", "point_right": "RIGHT",
                  "point_vertical": "NOOP", "no_hand": "no_hand/unsure"}

    outcomes = ["STOP", "HOME", "TROT", "LEFT", "RIGHT", "NOOP", "no_hand/unsure"]
    table: dict[str, dict[str, int]] = {}
    times = []
    for stem in SOURCES:
        src = _resolve_source(stem)
        if src is None:
            continue
        frames = _read_frames(src)
        tail = frames[int(len(frames) * (1 - VAL_FRAC)):]
        row = {o: 0 for o in outcomes}
        for frame in tail:
            t0 = time.perf_counter()
            x = (cv2.resize(frame, (IMG, IMG)).astype(np.float32) / 255.0 - 0.5)
            proba = sess.run(["probabilities"],
                             {"frames": x.transpose(2, 0, 1)[None]})[0][0]
            times.append((time.perf_counter() - t0) * 1000)
            p, cls = float(proba.max()), classes[int(proba.argmax())]
            out = to_outcome[cls] if p >= args.conf else "no_hand/unsure"
            row[out] += 1
        table[stem] = row

    w = max(len(s) for s in SOURCES) + 2
    print(f"\n[eval] held-out tails, conf>={args.conf} (runtime path). rows=video, cols=OUTCOME")
    print(" " * w + "".join(f"{o[:12]:>13}" for o in outcomes) + "     intended")
    for stem, row in table.items():
        want = EXPECTED_OUTCOME[stem]
        total = sum(row.values()) or 1
        wrong = sum(v for o, v in row.items()
                    if o in ("STOP", "HOME", "TROT", "LEFT", "RIGHT") and o != want)
        print(f"{stem:<{w}}" + "".join(f"{row[o]:>13}" for o in outcomes)
              + f"   {want:>6} ({row[want] / total:.0%} hit, {wrong / total:.0%} wrong-action)")
    times.sort()
    print(f"[eval] latency median {times[len(times) // 2]:.2f}ms, "
          f"p90 {times[int(len(times) * 0.9)]:.2f}ms per frame")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", nargs="?", default="train", choices=["train", "eval"])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--teacher-conf", type=float, default=0.85,
                   help="teacher frames below this SVM confidence are dropped, not taught")
    p.add_argument("--conf", type=float, default=0.85, help="eval confidence threshold")
    args = p.parse_args()
    if args.stage == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
