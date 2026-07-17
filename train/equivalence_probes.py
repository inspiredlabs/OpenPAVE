"""Equivalence probes: per-capability error tracking, towers vs the VLM teacher.

The Phase-4 harness of the critical path. For every CAPABILITY in the frozen
contract, 1-3 held-out probe images are materialised ONCE as stable artifacts
(train/probes/<capability>/<OUTCOME>_<n>.png + manifest.json). The VLM
(Fourier-Qwen2-VL, MLX) judges each probe once and is CACHED — the teacher is
slow and its answers must not drift between runs. Every tower then answers
the same probes, and the matrix shows EXACTLY which capability × tower cells
have low error — both against ground truth and against the teacher
("feature equality").

Capabilities wired now (extend PROBE_SPECS as towers grow — audio etc.):
  gesture.intent   STOP HOME TROT LEFT RIGHT NOOP   (probes from yolo26 TEST
                   split — foreign hands — plus crude tails where yolo26 has
                   no matching class, e.g. fist/HOME)
  hand.presence    YES NO                            (crude frames, landmarker-
                   verified at build time)

Towers scored: monty (3D evidence), svm (landmark SVM), tiny (v3 crop+seq),
yolo (yolo26n detector). Teacher column = VLM vs ground truth, so a weak
teacher on a capability is visible instead of silently polluting the target.

Run:  .venv/bin/python train/equivalence_probes.py build     # once
      .venv/bin/python train/equivalence_probes.py teacher   # VLM, cached
      .venv/bin/python train/equivalence_probes.py run       # the matrix
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
sys.path.insert(0, str(TRAIN_DIR.parent))
from mediapipe_svm import LANDMARKER, _read_frames, _resolve_source, point_direction  # noqa: E402

PROBE_DIR = TRAIN_DIR / "probes"
TEACHER_CACHE = PROBE_DIR / "teacher_cache.json"
VLM_MODEL = TRAIN_DIR.parent / "models" / "fourier-qwen2vl-2b-4bit"
PER_OUTCOME = 3

# Prompting follows the app's MEASURED lore (pave_ui/perception.py): option
# lists get parroted by this model ("first word wins"), so ask for the gesture
# NAME and clamp synonyms; pointing needs a direction FOLLOW-UP question.
GESTURE_PROMPT = ("What hand gesture is the person making? Answer with a short "
                  "phrase only. If there is no hand, answer NONE.")
DIRECTION_PROMPT = ("The person is pointing with a finger. Which direction is the "
                    "finger pointing? Answer with one word: LEFT, RIGHT, UP or DOWN.")
PRESENCE_PROMPT = ("Answer with a short phrase only: what is the person's hand "
                   "doing? If no hand is visible, answer NONE.")

# yolo26 ground-truth class -> probe outcome (fist/HOME comes from crude)
Y26_OUTCOME = {"Stop": "STOP", "Thumbs up": "TROT", "Left": "LEFT",
               "Right": "RIGHT", "Up": "NOOP", "Down": "NOOP"}


def _ask_gesture(backend, bgr) -> tuple[str, str]:
    """The app's two-step: gesture NAME -> synonyms; pointing -> direction
    follow-up (mirrors perception.py's gesture-name + pointing flow)."""
    name = backend.generate(bgr, GESTURE_PROMPT, max_tokens=16)
    up = (name or "").upper()
    if "THUMB" in up or "LIKE" in up:
        return name, "TROT"
    if "PALM" in up or "STOP" in up or "WAV" in up:
        return name, "STOP"
    if "FIST" in up or "PUNCH" in up:
        return name, "HOME"
    if "POINT" in up or "FINGER" in up:
        d = backend.generate(bgr, DIRECTION_PROMPT, max_tokens=8)
        du = (d or "").upper()
        for tok in ("LEFT", "RIGHT"):
            if tok in du:
                return f"{name} / {d}", tok
        return f"{name} / {d}", "NOOP"
    return name, "NOOP"


def _ask_presence(backend, bgr) -> tuple[str, str]:
    text = backend.generate(bgr, PRESENCE_PROMPT, max_tokens=16)
    return text, ("NO" if "NONE" in (text or "").upper() else "YES")


PROBE_SPECS = {
    "gesture.intent": {"ask": _ask_gesture,
                       "outcomes": ["STOP", "HOME", "TROT", "LEFT", "RIGHT", "NOOP"]},
    "hand.presence": {"ask": _ask_presence,
                      "outcomes": ["YES", "NO"]},
}


def _landmarker():
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    return mp, vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(LANDMARKER)),
        num_hands=1, min_hand_detection_confidence=0.5))


# ── build: materialise stable probe artifacts ────────────────────────────────

def build(args) -> None:
    import cv2
    manifest_p = PROBE_DIR / "manifest.json"
    if manifest_p.exists() and not args.force:
        print(f"[probes] {manifest_p} exists — probes are stable artifacts; --force to rebuild")
        return
    from gesture_lab import _yolo26_items
    manifest: list[dict] = []

    def save(cap: str, outcome: str, rgb: np.ndarray, n: int) -> None:
        d = PROBE_DIR / cap
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{outcome}_{n}.png"
        cv2.imwrite(str(p), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        manifest.append({"capability": cap, "outcome": outcome,
                         "file": str(p.relative_to(PROBE_DIR))})

    # gesture.intent — foreign hands from the yolo26 TEST split (never trained)
    counts: dict[str, int] = {}
    for rgb, name in _yolo26_items("test"):
        out = Y26_OUTCOME.get(name)
        if out is None or counts.get(out, 0) >= PER_OUTCOME:
            continue
        counts[out] = counts.get(out, 0) + 1
        save("gesture.intent", out, rgb, counts[out])
    # HOME (fist) has no yolo26 class: crude tail (held-out end of the video)
    mp, lm = _landmarker()
    fist = _read_frames(_resolve_source("fist"))
    n = 0
    for f in fist[int(len(fist) * 0.82)::5]:
        if n >= PER_OUTCOME:
            break
        if lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=f)).hand_landmarks:
            n += 1
            save("gesture.intent", "HOME", f, n)

    # hand.presence — verified by the landmarker at build time
    yes = no = 0
    for stem in ("like", "stop", "up"):
        for f in _read_frames(_resolve_source(stem)):
            if yes >= PER_OUTCOME and no >= PER_OUTCOME:
                break
            found = bool(lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=f)).hand_landmarks)
            if found and yes < PER_OUTCOME:
                yes += 1
                save("hand.presence", "YES", f, yes)
            elif not found and no < PER_OUTCOME:
                no += 1
                save("hand.presence", "NO", f, no)

    manifest_p.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[probes] built {len(manifest)} probe artifacts -> {PROBE_DIR}")


# ── teacher: VLM judgments, cached ───────────────────────────────────────────

def teacher(args) -> None:
    manifest = json.loads((PROBE_DIR / "manifest.json").read_text())
    cache = json.loads(TEACHER_CACHE.read_text()) if TEACHER_CACHE.exists() else {}
    todo = [m for m in manifest if m["file"] not in cache or args.force]
    if not todo:
        print(f"[probes] teacher cache complete ({len(cache)} judgments) — --force to redo")
        return
    print(f"[probes] loading VLM teacher ({VLM_MODEL.name}) via the repo's own backend …")
    # Use the EXACT backend the app deploys (pave_mlx Fourier4BitBackend), so
    # "teacher" here is byte-for-byte the VLM being compared against — raw
    # mlx_vlm calls with path-string images produced degenerate output on
    # this fork; the backend's PIL-positional path is the proven one.
    import cv2
    from pave_mlx.backends import Fourier4BitBackend
    backend = Fourier4BitBackend()
    for m in todo:
        bgr = cv2.imread(str(PROBE_DIR / m["file"]))
        t0 = time.perf_counter()
        raw, clamped = PROBE_SPECS[m["capability"]]["ask"](backend, bgr)
        cache[m["file"]] = {"raw": (raw or "").strip(), "clamped": clamped,
                            "s": round(time.perf_counter() - t0, 2)}
        print(f"[teacher] {m['file']:34s} -> {cache[m['file']]['clamped']:5s} "
              f"({cache[m['file']]['raw'][:40]!r}, {cache[m['file']]['s']}s)")
    TEACHER_CACHE.write_text(json.dumps(cache, indent=2) + "\n")
    print(f"[probes] teacher cache -> {TEACHER_CACHE}")


# ── run: the capability × tower matrix ───────────────────────────────────────

def _towers():
    """name -> fn(rgb) -> outcome token, built lazily; a missing artifact just
    drops that column instead of failing the harness."""
    import cv2
    towers = {}
    mp, lm = _landmarker()

    def hand_of(rgb):
        res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        return res.hand_landmarks[0] if res.hand_landmarks else None

    # monty — 3D evidence
    try:
        from monty_lab import EvidenceLM
        from monty_lab.tasks.gestures import INTENT as MONTY_INTENT
        from monty_lab.tasks.gestures import hand_to_episode
        mlm = EvidenceLM.load(TRAIN_DIR / "runs" / "monty_gestures" / "objects.npz")

        def monty_fn(rgb, hand):
            if hand is None:
                return "NOOP"
            obj, _e, _ = mlm.infer(hand_to_episode(hand))
            if obj == "point":
                d = point_direction(hand)
                return d if d in ("LEFT", "RIGHT") else "NOOP"
            return MONTY_INTENT.get(obj, "NOOP")
        towers["monty"] = monty_fn
    except Exception as exc:
        print(f"[probes] monty tower unavailable: {exc}")

    # svm — landmark shape SVM
    try:
        import onnxruntime as ort
        from mediapipe_svm import landmarks_to_features
        svm_dir = TRAIN_DIR / "runs" / "mediapipe_svm"
        sess = ort.InferenceSession(str(svm_dir / "model.onnx"), providers=["CPUExecutionProvider"])
        svm_classes = json.loads((svm_dir / "meta.json").read_text())["classes"]
        svm_map = {"stop": "STOP", "fist": "HOME", "like": "TROT"}

        def svm_fn(rgb, hand):
            if hand is None:
                return "NOOP"
            proba = sess.run(["probabilities"],
                             {"landmarks": landmarks_to_features(hand).reshape(1, -1)})[0][0]
            if float(proba.max()) < 0.85:
                return "NOOP"
            cls = svm_classes[int(proba.argmax())]
            if cls == "point":
                d = point_direction(hand)
                return d if d in ("LEFT", "RIGHT") else "NOOP"
            return svm_map.get(cls, "NOOP")
        towers["svm"] = svm_fn
    except Exception as exc:
        print(f"[probes] svm tower unavailable: {exc}")

    # tiny — v3 detect->crop->classify
    try:
        from gesture_lab import TinyV3Runtime, V3_INTENT
        rt = TinyV3Runtime(conf=0.6)

        def tiny_fn(rgb, hand):
            rt.ring.clear()
            tower, _p, _ = rt.step(rgb)
            return V3_INTENT.get(tower, "") or "NOOP"
        towers["tiny"] = tiny_fn
    except Exception as exc:
        print(f"[probes] tiny tower unavailable: {exc}")

    # yolo — pixel detector
    try:
        from ultralytics import YOLO
        best = sorted((TRAIN_DIR / "runs").glob("yolo*/weights/best.pt"))
        ymodel = YOLO(str(best[-1]))
        ymap = {"stop": "STOP", "fist": "HOME", "like": "TROT",
                "point_left": "LEFT", "point_right": "RIGHT"}

        def yolo_fn(rgb, hand):
            r = ymodel.predict(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                               imgsz=320, conf=0.5, verbose=False)[0]
            if not len(r.boxes):
                return "NOOP"
            cls = r.names[int(r.boxes.cls[int(r.boxes.conf.argmax())])]
            return ymap.get(cls, "NOOP")
        towers["yolo"] = yolo_fn
    except Exception as exc:
        print(f"[probes] yolo tower unavailable: {exc}")

    def presence_wrap(fn):
        def inner(rgb, hand):
            return "YES" if hand is not None else "NO"
        return inner

    return towers, hand_of


def run(_args) -> None:
    import cv2
    manifest = json.loads((PROBE_DIR / "manifest.json").read_text())
    cache = json.loads(TEACHER_CACHE.read_text()) if TEACHER_CACHE.exists() else {}
    towers, hand_of = _towers()
    names = list(towers)

    # score[capability.outcome][column] = [hits, total]; plus ≡teacher tallies
    score: dict[str, dict[str, list[int]]] = {}
    equiv: dict[str, list[int]] = {n: [0, 0] for n in names}
    for m in manifest:
        cap, want = m["capability"], m["outcome"]
        rgb = cv2.cvtColor(cv2.imread(str(PROBE_DIR / m["file"])), cv2.COLOR_BGR2RGB)
        hand = hand_of(rgb)
        row = score.setdefault(f"{cap}.{want}", {})
        t_ans = cache.get(m["file"], {}).get("clamped")
        if t_ans is not None:
            h, t = row.get("teacher", [0, 0])
            row["teacher"] = [h + (t_ans == want), t + 1]
        for n in names:
            if cap == "hand.presence":
                ans = "YES" if hand is not None else "NO"
            else:
                ans = towers[n](rgb, hand)
            h, t = row.get(n, [0, 0])
            row[n] = [h + (ans == want), t + 1]
            if t_ans is not None:
                equiv[n][0] += (ans == t_ans); equiv[n][1] += 1

    cols = ["teacher"] + names
    w = max(len(k) for k in score) + 2
    print(f"\n[probes] capability × tower — hits/probes vs GROUND TRUTH "
          f"(teacher column shows where the VLM itself is weak)")
    print(" " * w + "".join(f"{c:>9}" for c in cols))
    for k in sorted(score):
        row = score[k]
        cells = "".join(f"{row[c][0]}/{row[c][1]:<7}" if c in row else f"{'-':>9}" for c in cols)
        print(f"{k:<{w}}" + cells)
    print(f"\n[probes] feature equality (tower ≡ teacher, all probes): "
          + ", ".join(f"{n} {a}/{b}" for n, (a, b) in equiv.items()))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["build", "teacher", "run", "all"])
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    if args.stage in ("build", "all"):
        build(args)
    if args.stage in ("teacher", "all"):
        teacher(args)
    if args.stage in ("run", "all"):
        run(args)


if __name__ == "__main__":
    main()
