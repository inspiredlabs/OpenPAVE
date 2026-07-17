"""MediaPipe-landmark -> SVM (RBF) gesture classifier for OpenPAVE.

Architecture (matches the UX intention exactly):
  The SVM classifies hand SHAPE only — 4 classes: stop (palm), fist, like,
  point. Pointing DIRECTION is not a class: it is computed geometrically from
  the index-finger vector (landmark 5 MCP -> 8 tip), because "is it pointing
  left/right?" is a question about an angle, not a cluster.

Intent mapping (pave_ui/perception.py applies it at runtime):
  stop.mp4   PALM               -> STOP  (stop trotting)
  fist.mp4   FIST               -> HOME  (return to origin pose/tween)
  like.mp4   THUMBS-UP          -> TROT  (keep trotting)
  up.mp4     POINT, vertical    -> no-op (shape trains the point class;
  down.mp4   POINT, vertical    -> no-op  vertical points command nothing)
  right-to-left.mp4  POINT toward the SUBJECT's left  -> TURN LEFT
  left-to-right.mp4  POINT toward the SUBJECT's right -> TURN RIGHT
  (subject-centric: the unmirrored camera sees "their left" as image-right)

Direction is deliberately HARD to trigger: the index vector must lie within
±POINT_CONE_DEG of horizontal; everything else is a vertical/no-op point.

Sources (train/crude/, .mp4 preferred over .gif, "-muted" suffix tolerated).
Every extracted feature vector is also added horizontally mirrored — with a
single `point` shape class, a mirror is always a valid example of the same
class, so no mirrored source videos are needed (or written) any more.

Feature vector (must match perception.MediaPipeSvmWorker exactly):
  21 landmarks (x,y,z) -> subtract wrist (landmark 0) -> divide by max |coord|
  -> flatten to 63 float32s. Frames are downscaled to the runtime's max dim.

Model size stays small (offload target): frames are stride-sampled to
--per-class rows; support-vector count follows training rows, not video length.

Output: train/runs/mediapipe_svm/{model.onnx (fp32, primary), model.joblib,
meta.json}

Run:  .venv/bin/python train/mediapipe_svm.py train
      .venv/bin/python train/mediapipe_svm.py eval   # OUTCOME-level empirical
        check: held-out tails of every source video through the exact runtime
        path (shape SVM + direction geometry + confidence gate)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

TRAIN_DIR = Path(__file__).resolve().parent
CRUDE_DIR = TRAIN_DIR / "crude"
OUT_DIR = TRAIN_DIR / "runs" / "mediapipe_svm"
LANDMARKER = TRAIN_DIR / "weights" / "hand_landmarker.task"

SOURCES = {  # logical stem -> SHAPE class (directions collapse into `point`)
    "stop": "stop",
    "fist": "fist",
    "like": "like",
    "up": "point",
    "down": "point",
    "right-to-left": "point",
    "left-to-right": "point",   # optional extra capture; mirrors already cover it
}
# Expected runtime OUTCOME per source video, for the eval stage
EXPECTED_OUTCOME = {
    "stop": "STOP", "fist": "HOME", "like": "TROT",
    "up": "NOOP", "down": "NOOP",
    "right-to-left": "LEFT", "left-to-right": "RIGHT",
}
VAL_FRAC = 0.2
MAX_DIM = 384            # match perception.SVM_MAX_DIM so training == runtime domain
POINT_CONE_DEG = 35.0    # index vector within ±cone of horizontal -> LEFT/RIGHT


def landmarks_to_features(hand) -> np.ndarray:
    """21 landmarks -> 63-float32 position/scale-invariant vector (module doc)."""
    pts = np.array([[p.x, p.y, p.z] for p in hand], dtype=np.float32)
    pts -= pts[0]                                # wrist -> origin
    scale = np.abs(pts).max() or 1.0
    return (pts / scale).reshape(-1)


def point_direction(hand) -> str:
    """'LEFT' | 'RIGHT' | 'VERTICAL' from the index-finger vector (landmark 5
    MCP -> 8 tip).

    Direction is SUBJECT-centric, calibrated against the user's own captures:
    right-to-left.mp4 (defined as TURN LEFT) shows the index toward
    image-RIGHT — an unmirrored camera watching someone point to THEIR left.
    So image-right == subject's LEFT. Set PAVE_POINT_MIRROR=1 if a mirrored
    camera chain ever inverts this."""
    dx = hand[8].x - hand[5].x
    dy = hand[8].y - hand[5].y
    ang = abs(math.degrees(math.atan2(-dy, dx)))   # 0 = image-right, 180 = image-left
    flip = os.environ.get("PAVE_POINT_MIRROR", "") in ("1", "true", "yes")
    if ang <= POINT_CONE_DEG:
        return "RIGHT" if flip else "LEFT"
    if ang >= 180.0 - POINT_CONE_DEG:
        return "LEFT" if flip else "RIGHT"
    return "VERTICAL"


def _resolve_source(stem: str) -> Path | None:
    for cand in (f"{stem}.mp4", f"{stem}-muted.mp4", f"{stem}.gif"):
        p = CRUDE_DIR / cand
        if p.exists():
            return p
    return None


def _read_frames(path: Path) -> list[np.ndarray]:
    """Video/GIF -> list of RGB frames downscaled to the runtime's max dim."""
    import cv2
    frames: list[np.ndarray] = []
    if path.suffix.lower() == ".gif":
        from PIL import Image, ImageSequence
        for f in ImageSequence.Iterator(Image.open(path)):
            frames.append(np.array(f.convert("RGB")))
    else:
        cap = cv2.VideoCapture(str(path))
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        cap.release()
    out = []
    for f in frames:
        h, w = f.shape[:2]
        if max(h, w) > MAX_DIM:
            s = MAX_DIM / max(h, w)
            f = cv2.resize(f, (int(w * s), int(h * s)))
        out.append(f)
    return out


def extract(per_class: int) -> tuple[np.ndarray, np.ndarray, list[bool], dict]:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    lm = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(LANDMARKER)),
        num_hands=1, min_hand_detection_confidence=0.5))

    X, y, split = [], [], []          # split: False train / True val (temporal)
    counts: dict[str, list[int]] = {}
    for stem, cls in SOURCES.items():
        src = _resolve_source(stem)
        if src is None:
            continue
        frames = _read_frames(src)
        if len(frames) > per_class:   # stride-sample -> bounded rows -> small model
            frames = [frames[i] for i in np.linspace(0, len(frames) - 1, per_class, dtype=int)]
        val_from = int(len(frames) * (1 - VAL_FRAC))
        kept = skipped = 0
        for i, frame in enumerate(frames):
            res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame))
            if not res.hand_landmarks:
                skipped += 1
                continue
            feats = landmarks_to_features(res.hand_landmarks[0])
            is_val = i >= val_from
            # every row + its horizontal mirror: with a single `point` shape
            # class a mirror is always a valid example of the SAME class, and
            # it doubles other-hand coverage for the rest
            m = feats.reshape(21, 3).copy(); m[:, 0] *= -1
            for row in (feats, m.reshape(-1)):
                X.append(row); y.append(cls); split.append(is_val)
            kept += 1
        counts[stem] = [kept, skipped]
        print(f"[svm] {src.name:26s} -> {cls:8s} {kept} frames ({skipped} no-hand)")
    return np.array(X), np.array(y), split, counts


def train(args: argparse.Namespace) -> None:
    from sklearn.metrics import classification_report
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    import joblib

    X, y, split, counts = extract(args.per_class)
    split = np.array(split)
    Xtr, ytr, Xva, yva = X[~split], y[~split], X[split], y[split]

    t0 = time.perf_counter()
    clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale",
                                              probability=True, class_weight="balanced"))
    clf.fit(Xtr, ytr)
    fit_s = time.perf_counter() - t0
    report = classification_report(yva, clf.predict(Xva), digits=3)
    print(f"[svm] trained on {len(ytr)} rows in {fit_s:.1f}s; "
          f"temporal-holdout val on {len(yva)}:\n{report}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, OUT_DIR / "model.joblib")   # fp64 reference copy (libsvm is double-only)

    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    onx = convert_sklearn(
        clf, initial_types=[("landmarks", FloatTensorType([None, X.shape[1]]))],
        options={"zipmap": False})
    (OUT_DIR / "model.onnx").write_bytes(onx.SerializeToString())

    import onnxruntime as ort
    sess = ort.InferenceSession(str(OUT_DIR / "model.onnx"), providers=["CPUExecutionProvider"])
    onnx_proba = sess.run(["probabilities"], {"landmarks": Xva.astype(np.float32)})[0]
    skl_proba = clf.predict_proba(Xva)
    agree = float((onnx_proba.argmax(1) == skl_proba.argmax(1)).mean())
    drift = float(np.abs(onnx_proba - skl_proba).max())
    print(f"[svm] fp32 ONNX parity vs fp64 joblib: {agree:.1%} label agreement, "
          f"max |Δproba| {drift:.2e}")

    n_sv = int(clf.named_steps["svc"].support_vectors_.shape[0])
    onnx_kb = (OUT_DIR / "model.onnx").stat().st_size // 1024
    meta = {
        "classes": list(clf.classes_),
        "precision": "fp32 (model.onnx, primary) / fp64 (model.joblib, reference)",
        "n_support_vectors": n_sv,
        "point_cone_deg": POINT_CONE_DEG,
        "train_rows": int(len(ytr)),
        "val_rows": int(len(yva)),
        "per_source_counts": counts,
        "feature_spec": "21 landmarks, wrist-origin, max-abs scale, 63 float32",
        "landmarker": str(LANDMARKER),
        "val_report": report,
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[svm] saved model.onnx ({onnx_kb} KB, {n_sv} support vectors)")


def evaluate(args: argparse.Namespace) -> None:
    """OUTCOME-level empirical check on held-out data: the temporal tail of
    every source video goes through the EXACT runtime decision path — shape
    SVM + index-vector direction + confidence gate — and is scored against
    the intended UX outcome (STOP/TROT/LEFT/RIGHT/NOOP)."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    import onnxruntime as ort

    meta = json.loads((OUT_DIR / "meta.json").read_text())
    classes = meta["classes"]
    sess = ort.InferenceSession(str(OUT_DIR / "model.onnx"), providers=["CPUExecutionProvider"])
    lm = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(LANDMARKER)),
        num_hands=1, min_hand_detection_confidence=0.5))

    outcomes = ["STOP", "HOME", "TROT", "LEFT", "RIGHT", "NOOP", "no_hand/unsure"]
    table: dict[str, dict[str, int]] = {}
    times = []
    for stem in SOURCES:
        src = _resolve_source(stem)
        if src is None:
            continue
        frames = _read_frames(src)
        tail = frames[int(len(frames) * (1 - VAL_FRAC)):]   # held-out temporal tail
        row = {o: 0 for o in outcomes}
        for frame in tail:
            t0 = time.perf_counter()
            res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame))
            out = "no_hand/unsure"
            if res.hand_landmarks:
                hand = res.hand_landmarks[0]
                feats = landmarks_to_features(hand).reshape(1, -1)
                proba = sess.run(["probabilities"], {"landmarks": feats})[0][0]
                if float(proba.max()) >= args.conf:
                    cls = classes[int(proba.argmax())]
                    if cls == "point":
                        d = point_direction(hand)
                        out = d if d in ("LEFT", "RIGHT") else "NOOP"
                    else:
                        out = {"stop": "STOP", "fist": "HOME", "like": "TROT"}.get(cls, "NOOP")
            times.append((time.perf_counter() - t0) * 1000)
            row[out] += 1
        table[stem] = row

    w = max(len(s) for s in SOURCES) + 2
    print(f"\n[eval] held-out tails, conf>={args.conf} (runtime path). rows=video, cols=OUTCOME")
    print(" " * w + "".join(f"{o[:12]:>13}" for o in outcomes) + "     intended")
    for stem, row in table.items():
        want = EXPECTED_OUTCOME[stem]
        total = sum(row.values()) or 1
        # wrong-action = frames that would send a DIFFERENT command; NOOP and
        # no_hand/unsure are inaction, not error (teardown frames land there)
        wrong = sum(v for o, v in row.items()
                    if o in ("STOP", "HOME", "TROT", "LEFT", "RIGHT") and o != want)
        print(f"{stem:<{w}}" + "".join(f"{row[o]:>13}" for o in outcomes)
              + f"   {want:>6} ({row[want] / total:.0%} hit, {wrong / total:.0%} wrong-action)")
    times.sort()
    print(f"[eval] latency median {times[len(times) // 2]:.1f}ms, "
          f"p90 {times[int(len(times) * 0.9)]:.1f}ms per frame")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", nargs="?", default="train", choices=["train", "eval"])
    p.add_argument("--per-class", type=int, default=250,
                   help="max frames per source before training (bounds SV count/model size)")
    p.add_argument("--conf", type=float, default=0.85, help="eval confidence threshold")
    args = p.parse_args()
    if args.stage == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
