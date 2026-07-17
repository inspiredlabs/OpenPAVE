"""Monty-principle 3D gesture recognizer (Thousand Brains direction, stage 1).

Follows the Monty recipe from docs.thousandbrains.org/docs/using-monty-in-a-
custom-application without the framework (no PyPI package exists; the full
tbp.monty integration needs their pinned conda env + Hydra configs — see the
datasets README note). What is kept is the METHOD:

  * an "object" is a gesture's 3D landmark constellation — 21 MediaPipe
    points (x, y, z) in a hand-centric reference frame (wrist origin,
    scale-normalised, orientation KEPT);
  * learning is FEW-SHOT and gradient-free: a handful of exemplar graphs per
    object, learned in seconds from the crude captures — this is the "rapid
    training" property;
  * recognition accumulates EVIDENCE across pose hypotheses: each exemplar is
    aligned to the observation with a Kabsch rotation solve, and the residual
    converts to evidence; the winning object must beat an evidence floor or
    the answer is `noop` (unknown gestures abstain by construction);
  * all pointing is ONE object — `point` — because in a rotation-solving
    recogniser direction is pose, not identity. Direction still maps to
    LEFT/RIGHT/vertical with the shipping cone geometry.

Objects: palm, fist, like, point (+ implicit noop). Recognition itself runs
in tens of microseconds; the pixels->landmarks stage in front of it is
unchanged (MediaPipe at ~5ms, or the v3 trunk at ~0.5ms when it matures) —
Monty does not detect hands in pixels, and nothing on this page claims to.

Run:  .venv/bin/python train/monty_gesture.py learn    # seconds, few-shot
      .venv/bin/python train/monty_gesture.py eval     # yolo26 referee
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
from mediapipe_svm import LANDMARKER, _read_frames, _resolve_source, point_direction  # noqa: E402

OUT_DIR = TRAIN_DIR / "runs" / "monty_gesture"
OBJECT_SOURCES = {  # gesture object -> capture videos that teach it
    "palm": ["stop"],
    "fist": ["fist"],
    "like": ["like"],
    "point": ["up", "down", "right-to-left", "left-to-right"],
}
EXEMPLARS_PER_SOURCE = 10
SIGMA = 0.16            # evidence kernel width (normalised-RMS units)
EVIDENCE_FLOOR = 0.50   # below this the answer is noop (unknown -> abstain)
INTENT = {"palm": "STOP", "fist": "HOME", "like": "TROT"}


def _landmarker():
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    lm = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(LANDMARKER)),
        num_hands=1, min_hand_detection_confidence=0.5))
    return mp, lm


def _constellation(hand) -> np.ndarray:
    """21x3 points in the hand reference frame: wrist origin, scale-norm,
    ORIENTATION KEPT (pose is solved at recognition time, not erased)."""
    pts = np.array([[q.x, q.y, q.z] for q in hand], dtype=np.float32)
    pts -= pts[0]
    return pts / (np.abs(pts).max() or 1.0)


def _kabsch_rms(obs: np.ndarray, ex: np.ndarray) -> float:
    """Residual RMS after the optimal rotation aligning obs -> exemplar —
    one pose hypothesis test, Monty's evidence primitive."""
    h = obs.T @ ex
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1.0, 1.0, d]) @ vt
    return float(np.sqrt(np.mean(np.sum((obs @ r - ex) ** 2, axis=1))))


class MontyGestures:
    """Few-shot 3D constellation recogniser with pose-hypothesis evidence."""

    def __init__(self, model_path: Path | None = None) -> None:
        p = model_path or OUT_DIR / "objects.npz"
        d = np.load(p, allow_pickle=True)
        self.objects = {k: d[k] for k in d.files}   # name -> (E, 21, 3)

    def recognize(self, hand) -> tuple[str, float]:
        """(object name | 'noop', evidence). tens of µs for 4 objects."""
        obs = _constellation(hand)
        best, best_e = "noop", 0.0
        for name, exemplars in self.objects.items():
            for ex in exemplars:
                e = float(np.exp(-(_kabsch_rms(obs, ex) / SIGMA) ** 2))
                if e > best_e:
                    best, best_e = name, e
        if best_e < EVIDENCE_FLOOR:
            return "noop", best_e
        return best, best_e

    def outcome(self, hand) -> tuple[str, str, float]:
        """(object, OUTCOME token, evidence) — direction resolved from pose
        geometry for the `point` object, exactly like the shipping pipeline."""
        obj, e = self.recognize(hand)
        if obj == "point":
            d = point_direction(hand)
            return obj, (d if d in ("LEFT", "RIGHT") else "NOOP"), e
        return obj, INTENT.get(obj, "NOOP" if obj == "noop" else ""), e


def learn(_args) -> None:
    """Few-shot, gradient-free: stride-sample exemplar constellations from the
    capture videos. Seconds, not minutes — re-run any time a video changes."""
    mp, lm = _landmarker()
    store: dict[str, list[np.ndarray]] = {}
    t0 = time.perf_counter()
    for obj, stems in OBJECT_SOURCES.items():
        for stem in stems:
            src = _resolve_source(stem)
            if src is None:
                continue
            frames = _read_frames(src)
            mid = frames[int(len(frames) * 0.3):int(len(frames) * 0.7)]  # held gesture
            found = []
            for f in mid[:: max(1, len(mid) // (EXEMPLARS_PER_SOURCE * 3))]:
                res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=f))
                if res.hand_landmarks:
                    found.append(_constellation(res.hand_landmarks[0]))
                if len(found) >= EXEMPLARS_PER_SOURCE:
                    break
            store.setdefault(obj, []).extend(found)
            print(f"[monty] {obj:6s} += {len(found)} exemplars from {src.name}")
    # cross-subject exemplars from the yolo26 GROUND-TRUTH train split —
    # few-shot means adding a new person costs seconds, not a retrain
    from gesture_lab import _yolo26_items
    y26_to_obj = {"Stop": "palm", "Thumbs up": "like",
                  "Left": "point", "Right": "point", "Up": "point", "Down": "point"}
    added: dict[str, int] = {}
    for rgb, name in _yolo26_items("train"):
        obj = y26_to_obj.get(name)
        if obj is None or added.get(obj, 0) >= 12:
            continue
        res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if res.hand_landmarks:
            store.setdefault(obj, []).append(_constellation(res.hand_landmarks[0]))
            added[obj] = added.get(obj, 0) + 1
    print(f"[monty] cross-subject exemplars from yolo26: "
          + ", ".join(f"{k}+{v}" for k, v in sorted(added.items())))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_DIR / "objects.npz",
                        **{k: np.stack(v) for k, v in store.items()})
    n = sum(len(v) for v in store.values())
    meta = {"objects": {k: len(v) for k, v in store.items()},
            "sigma": SIGMA, "evidence_floor": EVIDENCE_FLOOR,
            "learned_in_s": round(time.perf_counter() - t0, 1),
            "method": "monty-principle: 3D reference-frame exemplars + Kabsch pose evidence"}
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[monty] learned {n} exemplar graphs across {len(store)} objects "
          f"in {meta['learned_in_s']}s -> {OUT_DIR / 'objects.npz'}")


def evaluate(_args) -> None:
    """The same yolo26 referee, recognition layer swapped to Monty evidence."""
    sys.path.insert(0, str(TRAIN_DIR))
    from gesture_lab import _yolo26_items  # noqa: E402

    mp, lm = _landmarker()
    monty = MontyGestures()
    expected = {"Stop": "STOP", "Thumbs up": "TROT", "Left": "LEFT", "Right": "RIGHT",
                "Up": "ABSTAIN", "Down": "ABSTAIN", "Thumbs Down": "ABSTAIN"}
    per: dict[str, list[int]] = {}
    t_rec = []
    for split in ("test", "valid"):
        for rgb, name in _yolo26_items(split):
            res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if not res.hand_landmarks:
                out = "ABSTAIN"
            else:
                t0 = time.perf_counter()
                _, token, _ = monty.outcome(res.hand_landmarks[0])
                t_rec.append((time.perf_counter() - t0) * 1e6)
                out = token if token not in ("", "NOOP") else "ABSTAIN"
            want = expected[name]
            hit = out == want
            wrong = out in ("STOP", "HOME", "TROT", "LEFT", "RIGHT") and out != want
            h, w_, t = per.get(name, [0, 0, 0])
            per[name] = [h + hit, w_ + wrong, t + 1]
    hits = sum(v[0] for v in per.values()); wrongs = sum(v[1] for v in per.values())
    total = sum(v[2] for v in per.values())
    t_rec.sort()
    print(f"\n[eval-monty] yolo26 referee: {hits}/{total} correct ({hits / total:.0%}), "
          f"{wrongs / total:.1%} wrong-action; recognition layer "
          f"{t_rec[len(t_rec) // 2]:.0f}µs median (landmarker in front unchanged)")
    for name, (h, w_, t) in per.items():
        print(f"    {name:12s} want {expected[name]:8s} {h:>3}/{t:<3} hit  {w_:>2} wrong-action")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", nargs="?", default="learn", choices=["learn", "eval"])
    args = p.parse_args()
    if args.stage == "learn":
        learn(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
