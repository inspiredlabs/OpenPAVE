#!/usr/bin/env python3
"""Auditable, offline hand-landmark curriculum builder.

This tool does not download anything and does not train by default.  Expensive
stages require the SHA-256 approval token printed by ``plan`` so the exact
curriculum can be reviewed before wall-clock work starts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TRAIN = HERE.parent
ROOT = TRAIN.parent
DATASETS = TRAIN / "datasets"
TEACHER_TASK = TRAIN / "weights" / "hand_landmarker.task"
DEFAULT_PLAN = HERE / "plan.json"
CURRICULUM = HERE / "curriculum"

# Roles are intentionally strict.  Gesture labels do not become joint labels.
SOURCES = {
    "20bn-jester-raw": {
        "path": DATASETS / "20bn-jester",
        "role": "video diversity + temporal intent; teacher distillation input",
        "joint_truth": False,
    },
    "jester": {
        "path": DATASETS / "jester" / "prepared.npz",
        "role": "filtered teacher landmarks + explicit No-gesture negatives",
        "joint_truth": False,
    },
    "hagrid": {
        "path": DATASETS / "hagrid" / "prepared.npz",
        "role": "camera/person/background diversity; teacher pseudo-landmarks",
        "joint_truth": False,
    },
    "hagrid_shapes": {
        "path": DATASETS / "hagrid_shapes" / "prepared.npz",
        "role": "gesture-shape diversity; teacher pseudo-landmarks",
        "joint_truth": False,
    },
    "crude": {
        "path": DATASETS / "crude" / "prepared.npz",
        "role": "target-camera domain probes; never the only validation source",
        "joint_truth": False,
    },
    "ipn": {
        "path": DATASETS / "ipn" / "prepared.npz",
        "role": "ordered video/background diversity; teacher pseudo-landmarks",
        "joint_truth": False,
    },
    "yolo26": {
        "path": DATASETS / "yolo26" / "prepared.npz",
        "role": "held-out cross-domain gesture probe; teacher pseudo-landmarks",
        "joint_truth": False,
    },
    "swipe_phases": {
        "path": DATASETS / "swipe_phases" / "prepared.npz",
        "role": "motion-phase/trajectory curriculum; not joint ground truth",
        "joint_truth": False,
    },
    "nvgesture": {
        "path": DATASETS / "nvgesture" / "raw",
        "role": "optional RGB/depth video diversity; currently empty",
        "joint_truth": False,
    },
    "dynamic-crops-classifier": {
        "path": DATASETS / "dynamic_gestures" / "models" / "crops_classifier.onnx",
        "role": "optional motion-phase gate only; forbidden as joint supervision",
        "joint_truth": False,
    },
    "onehand10k": {
        "path": DATASETS / "onehand10k",
        "role": "required/recommended manually annotated 21-joint RGB geometry",
        "joint_truth": True,
    },
    "freihand": {
        "path": DATASETS / "freihand",
        "role": "recommended real multi-view 3D hand geometry",
        "joint_truth": True,
    },
    "interhand26m": {
        "path": DATASETS / "interhand26m",
        "role": "optional large-scale 3D hand geometry; use a stratified subset",
        "joint_truth": True,
    },
    "coco-keypoints": {
        "path": DATASETS / "coco-keypoints",
        "role": "separate 17-joint BODY tower; never map body joints to hand joints",
        "joint_truth": False,
    },
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _npz_info(path: Path) -> dict:
    try:
        d = np.load(path, allow_pickle=True)
        keys = list(d.files)
        # Reading `imgs` from a compressed NPZ inflates hundreds of MB merely
        # to learn N. Prefer a small one-dimensional field for a cheap audit.
        count_key = next((k for k in ("labels", "is_val", "has_lm") if k in keys), None)
        n = int(len(d[count_key])) if count_key else 0
        out = {"frames": n, "fields": keys}
        if "has_lm" in keys:
            out["teacher_landmarks"] = int(np.asarray(d["has_lm"]).sum())
        if "is_val" in keys:
            out["validation_frames"] = int(np.asarray(d["is_val"]).sum())
        if "labels" in keys:
            names, counts = np.unique(d["labels"], return_counts=True)
            out["labels"] = {str(k): int(v) for k, v in zip(names, counts)}
        return out
    except Exception as exc:
        return {"error": str(exc)}


def source_audit() -> dict:
    out = {}
    for name, spec in SOURCES.items():
        path = Path(spec["path"])
        item = {
            "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "exists": path.exists(),
            "role": spec["role"],
            "joint_truth": bool(spec["joint_truth"]),
        }
        if spec["joint_truth"]:
            # Raw downloads are not silently considered curriculum-ready. An
            # adapter must emit the normalized, reviewable OpenPAVE manifest.
            manifest = path / "openpave-21.jsonl"
            item["normalized_manifest"] = str(manifest.relative_to(ROOT))
            item["curriculum_ready"] = manifest.is_file() and manifest.stat().st_size > 0
        if path.is_file():
            item["bytes"] = path.stat().st_size
            if path.suffix == ".npz":
                item.update(_npz_info(path))
        elif path.is_dir():
            # Avoid recursive traversal of Jester's ~57k directories.
            item["top_level_entries"] = sum(1 for _ in path.iterdir())
        out[name] = item
    return out


def cmd_audit(args) -> None:
    audit = source_audit()
    if args.json:
        print(json.dumps(audit, indent=2))
        return
    print(f"{'source':27s} {'state':9s} {'joint GT':8s} {'ready':7s} path")
    for name, item in audit.items():
        state = "present" if item["exists"] else "MISSING"
        ready = str(item.get("curriculum_ready", "-"))
        print(f"{name:27s} {state:9s} {str(item['joint_truth']):8s} {ready:7s} {item['path']}")
        if "frames" in item:
            print(f"  frames={item['frames']:,} landmarks={item.get('teacher_landmarks', 0):,} "
                  f"val={item.get('validation_frames', 0):,}")


def teacher_manifest() -> dict:
    if not TEACHER_TASK.exists():
        return {"exists": False, "path": str(TEACHER_TASK)}
    result = {"exists": True, "path": str(TEACHER_TASK.relative_to(ROOT)),
              "bytes": TEACHER_TASK.stat().st_size, "sha256": _sha256(TEACHER_TASK),
              "members": []}
    with zipfile.ZipFile(TEACHER_TASK) as zf:
        for info in zf.infolist():
            with zf.open(info) as f:
                digest = hashlib.sha256(f.read()).hexdigest()
            result["members"].append({"name": info.filename, "bytes": info.file_size,
                                      "sha256": digest})
    return result


def cmd_teacher(args) -> None:
    manifest = teacher_manifest()
    print(json.dumps(manifest, indent=2))
    if args.extract:
        if not manifest.get("exists"):
            raise SystemExit("teacher task is missing")
        dest = Path(args.extract).resolve()
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(TEACHER_TASK) as zf:
            for member in ("hand_detector.tflite", "hand_landmarks_detector.tflite"):
                with zf.open(member) as src, (dest / member).open("wb") as dst:
                    shutil.copyfileobj(src, dst)
        (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"extracted inference models to {dest}")


def build_plan(args) -> dict:
    audit = source_audit()
    exploration = {"crude": {"limit": 600, "stride": 3},
                   "hagrid_shapes": {"limit": 800, "stride": 5},
                   "jester": {"limit": 800, "stride": 10}}
    referee = {"yolo26": {"limit": 500, "stride": 2}}
    return {
        "schema": "openpave.hand-landmark-sensorimotor-plan.v2",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "teacher": teacher_manifest(),
        "objective": "Gradient-free sensorimotor hand reference frames learned from ordered "
                     "MediaPipe 21-point 2.5D constellations",
        "sources": audit,
        "curriculum": [
            {"stage": 0, "name": "freeze teacher", "action": "hash the exact MediaPipe task; "
             "extract detector and landmarker only for inspection"},
            {"stage": 1, "name": "bounded constellation harvest",
             "action": "distill x/y/z and world coordinates; require horizontal-flip consistency",
             "sources": exploration, "referee": referee,
             "accept": {"teacher_score_min": .75, "flip_error_px_max": 6.0}},
            {"stage": 2, "name": "sensorimotor exploration",
             "action": "walk wrist/palm anchors then finger chains; each action is a 3D joint "
                       "displacement and each observation updates pose hypotheses",
             "policies": ["palm_anchor_scan", "finger_chain_scan", "deterministic_random_walk"]},
            {"stage": 3, "name": "additive reference-frame learning",
             "action": "append a constellation only when existing hypotheses cannot explain it; "
                       "no gradients and no destructive retraining",
             "accept": {"max_prototypes": 256, "novelty_rms": .10}},
            {"stage": 4, "name": "next-sensation proof",
             "action": "predict the next landmark location from the current partial walk before "
                       "revealing the MediaPipe observation",
             "accept": {"pck_010_min": .50, "pck_020_min": .80, "median_step_us_max": 1000}},
            {"stage": 5, "name": "pixel sensor adapter (later)",
             "action": "retain palm detector as the initial sensor; search RGB patches around each "
                       "Monty-predicted next location; re-detect only when evidence collapses"},
        ],
        "sensorimotor_training_allowed": bool(teacher_manifest().get("exists")),
        "pixel_landmarker_replacement_allowed": False,
        "blocker": "The learned constellation substrate still needs a pixel Sensor Module. "
                   "Until patch-search validation passes, MediaPipe remains the teacher/reference.",
        "notes": [
            "crops_classifier.onnx is permitted only as a motion-phase gate.",
            "Teacher misses on gesture-labelled frames are unknown, not negative.",
            "Split by subject/clip before sampling frames; never random-split adjacent frames.",
            "This plan learns hand geometry from compact ordered episodes, not a broad RGB corpus.",
            "No pixel runtime artifact is promoted from constellation-only results.",
        ],
    }


def cmd_plan(args) -> None:
    plan = build_plan(args)
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=2) + "\n")
    digest = _sha256(out)
    print(json.dumps(plan, indent=2))
    print(f"\nPLAN_SHA256={digest}\nReview {out}; expensive commands require --approve {digest}")


def require_approval(plan_path: Path, token: str) -> dict:
    if not plan_path.exists():
        raise SystemExit(f"missing plan: {plan_path}; run `offline.py plan` first")
    actual = _sha256(plan_path)
    if token != actual:
        raise SystemExit(f"approval mismatch; reviewed plan hash must be {actual}")
    return json.loads(plan_path.read_text())


def _detect(landmarker, mp, rgb):
    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.hand_landmarks:
        return None
    hand = result.hand_landmarks[0]
    xy = np.asarray([[p.x, p.y] for p in hand], np.float32)
    xyz = np.asarray([[p.x, p.y, p.z] for p in hand], np.float32)
    world = np.asarray([[p.x, p.y, p.z] for p in result.hand_world_landmarks[0]], np.float32)
    category = result.handedness[0][0]
    return xy, xyz, world, str(category.category_name), float(category.score)


def cmd_distill(args) -> None:
    plan_path = Path(args.plan).resolve()
    require_approval(plan_path, args.approve)
    source = args.source
    if source not in SOURCES:
        raise SystemExit(f"unknown source {source!r}")
    shard = Path(SOURCES[source]["path"])
    if not shard.is_file() or shard.suffix != ".npz":
        raise SystemExit(f"distill expects a prepared NPZ source, got {shard}")

    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(TEACHER_TASK),
                                           delegate=mp_python.BaseOptions.Delegate.CPU),
        num_hands=1,
        min_hand_detection_confidence=args.min_score,
        min_hand_presence_confidence=args.min_score))

    d = np.load(shard, allow_pickle=True); images = d["imgs"]
    rows = np.arange(0, len(images), max(1, args.stride))
    if args.limit:
        is_val = np.asarray(d["is_val"], bool)
        tr, va = rows[~is_val[rows]], rows[is_val[rows]]
        nva = min(len(va), max(1, args.limit // 5)); ntr = min(len(tr), args.limit - nva)
        take = lambda a, n: a[np.linspace(0, len(a) - 1, n, dtype=int)] if n else a[:0]
        rows = np.concatenate((take(tr, ntr), take(va, nva)))
    accepted = []
    for n, i in enumerate(rows, 1):
        rgb = images[i]
        a = _detect(landmarker, mp, rgb)
        b = _detect(landmarker, mp, np.ascontiguousarray(rgb[:, ::-1]))
        if a is None or b is None:
            continue
        xy, xyz, world, handed, score = a
        flip_xy = b[0].copy(); flip_xy[:, 0] = 1 - flip_xy[:, 0]
        consistency = float(np.linalg.norm(xy - flip_xy, axis=1).mean() * max(rgb.shape[:2]))
        if score < args.min_score or consistency > args.consistency_px:
            continue
        accepted.append((int(i), xy, xyz, world, handed, score, consistency))
        if args.progress and n % args.progress == 0:
            print(f"[distill] {n}/{len(rows)} checked; {len(accepted)} accepted", flush=True)
    if not accepted:
        raise SystemExit("no tracks passed the confidence/consistency gates")

    dest = Path(args.out or CURRICULUM / f"{source}.teacher.npz").resolve()
    if dest.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {dest}; pass --overwrite")
    dest.parent.mkdir(parents=True, exist_ok=True)
    idx, xy, xyz, world, handed, score, consistency = zip(*accepted)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, source_shard=str(shard), frame_index=np.asarray(idx),
                            xy=np.asarray(xy), xyz=np.asarray(xyz), world_xyz=np.asarray(world),
                            handedness=np.asarray(handed), teacher_score=np.asarray(score),
                            flip_error_px=np.asarray(consistency),
                            is_val=np.asarray(d["is_val"])[np.asarray(idx)])
    os.replace(tmp, dest)
    print(f"[distill] accepted {len(accepted)}/{len(rows)} -> {dest}")


PALM_ANCHORS = np.asarray([0, 5, 9, 13, 17], np.int64)
SCAN_ORDER = np.asarray([1, 2, 3, 4, 6, 7, 8, 10, 11, 12,
                         14, 15, 16, 18, 19, 20], np.int64)


def _normalise_constellation(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, np.float32) - np.asarray(points, np.float32)[0]
    return points / max(float(np.abs(points).max()), 1e-6)


def _align(obs: np.ndarray, ref: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, float]:
    """Rotation taking observed points into one stored object reference frame."""
    h = obs[indices].T @ ref[indices]
    u, _, vt = np.linalg.svd(h)
    fix = np.eye(3, dtype=np.float32)
    fix[2, 2] = np.sign(np.linalg.det(u @ vt)) or 1.0
    r = u @ fix @ vt
    rms = float(np.sqrt(((obs[indices] @ r - ref[indices]) ** 2).sum(-1).mean()))
    return r.astype(np.float32), rms


def _learn_prototypes(train: np.ndarray, novelty: float, cap: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed); order = rng.permutation(len(train)); prototypes = []
    for i in order:
        episode = _normalise_constellation(train[i])
        if not prototypes:
            prototypes.append(episode); continue
        best = min(_align(episode, ref, np.arange(21))[1] for ref in prototypes)
        if best >= novelty:
            prototypes.append(episode)
        if len(prototypes) >= cap:
            break
    return np.asarray(prototypes, np.float32)


def _predictive_walk(target: np.ndarray, prototypes: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Predict each next sensation, then reveal it and update the hypothesis."""
    target = _normalise_constellation(target)
    known = list(PALM_ANCHORS); errors = []; timings = []
    for joint in SCAN_ORDER:
        t0 = time.perf_counter(); idx = np.asarray(known, np.int64)
        observed = target[idx]
        references = prototypes[:, idx]
        # Evaluate every object hypothesis in one batched Kabsch solve. This is
        # mathematically identical to the scalar loop but removes Python from
        # the 256-unit recurrent core's hot path.
        h = np.einsum("na,knb->kab", observed, references, optimize=True)
        u, _, vt = np.linalg.svd(h)
        fix = np.broadcast_to(np.eye(3, dtype=np.float32), h.shape).copy()
        fix[:, 2, 2] = np.where(np.linalg.det(u @ vt) < 0, -1.0, 1.0)
        rotations = u @ fix @ vt
        aligned = np.einsum("na,kab->knb", observed, rotations, optimize=True)
        rms = np.sqrt(((aligned - references) ** 2).sum(-1).mean(-1))
        best = int(np.argmin(rms))
        # obs @ R ~= ref, therefore ref @ R.T predicts observation space.
        prediction = prototypes[best, joint] @ rotations[best].T
        timings.append((time.perf_counter() - t0) * 1e6)
        errors.append(float(np.linalg.norm(prediction - target[joint])))
        known.append(int(joint))                 # movement completes; sensor reveals joint
    return np.asarray(errors, np.float32), timings


def cmd_train(args) -> None:
    plan = require_approval(Path(args.plan).resolve(), args.approve)
    if not plan.get("sensorimotor_training_allowed"):
        raise SystemExit("sensorimotor training blocked: frozen MediaPipe teacher is missing")
    paths = [Path(p).resolve() for p in args.input] if args.input else [
        p for p in sorted(CURRICULUM.glob("*.teacher.npz")) if p.name != "yolo26.teacher.npz"
    ]
    if not paths:
        raise SystemExit("no distilled constellation shards; run the bounded Stage 1 commands first")
    train, val, referee, provenance = [], [], [], {}
    for path in paths:
        d = np.load(path, allow_pickle=True); xyz = np.asarray(d["xyz"], np.float32)
        mask = np.asarray(d["is_val"], bool)
        train.append(xyz[~mask]); val.append(xyz[mask])
        provenance[path.name] = {"train": int((~mask).sum()), "val": int(mask.sum()),
                                 "role": "exploration", "sha256": _sha256(path)}
    referee_paths = [Path(p).resolve() for p in args.referee]
    for path in referee_paths:
        d = np.load(path, allow_pickle=True); xyz = np.asarray(d["xyz"], np.float32)
        referee.append(xyz)
        provenance[path.name] = {"train": 0, "val": int(len(xyz)),
                                 "role": "untouched_referee", "sha256": _sha256(path)}
    train = np.concatenate(train)
    val = np.concatenate(val) if any(len(x) for x in val) else train[-min(100, len(train)):]
    referee = np.concatenate(referee) if referee else np.empty((0, 21, 3), np.float32)
    prototypes = _learn_prototypes(train, args.novelty_rms, args.max_prototypes, args.seed)
    def evaluate(episodes: np.ndarray) -> dict:
        all_errors, all_times = [], []
        for episode in episodes:
            errors, timings = _predictive_walk(episode, prototypes)
            all_errors.extend(errors); all_times.extend(timings)
        errors = np.asarray(all_errors); times = np.asarray(all_times)
        return {"episodes": int(len(episodes)), "next_location_nme": float(errors.mean()),
                "pck_010": float((errors <= .10).mean()), "pck_020": float((errors <= .20).mean()),
                "median_step_us": float(np.median(times)), "p95_step_us": float(np.percentile(times, 95))}

    evaluations = {"exploration_holdout": evaluate(val)}
    if len(referee):
        evaluations["untouched_referee"] = evaluate(referee)
    combined = evaluate(np.concatenate((val, referee)) if len(referee) else val)
    metrics = {"episodes_train": int(len(train)), "episodes_val": combined.pop("episodes"),
               "prototypes": int(len(prototypes)), **combined}
    accepted = (metrics["pck_010"] >= .50 and metrics["pck_020"] >= .80
                and metrics["median_step_us"] <= 1000)
    accepted = accepted and all(m["pck_010"] >= .50 and m["pck_020"] >= .80
                                for m in evaluations.values())
    deltas = prototypes[:, SCAN_ORDER] - prototypes[:, np.asarray(
        [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19])]
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "objects.npz", hand=prototypes, palm_anchors=PALM_ANCHORS,
                        scan_order=SCAN_ORDER, edge_delta_mean=deltas.mean(0), edge_delta_std=deltas.std(0))
    meta = {"contract": "openpave.sensorimotor-hand.v1", "learning": "gradient-free additive",
            "teacher_sha256": plan["teacher"]["sha256"], "novelty_rms": args.novelty_rms,
            "max_prototypes": args.max_prototypes, "provenance": provenance,
            "policies": ["palm_anchor_scan", "finger_chain_scan"],
            "accepted": accepted, "metrics": metrics, "evaluations": evaluations,
            "limitation": "Geometry substrate only: pixel patch Sensor Module is not yet implemented."}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    print(f"[sensorimotor] {'ACCEPTED' if accepted else 'REJECTED'} -> {out}")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=False)
    a = sub.add_parser("audit"); a.add_argument("--json", action="store_true"); a.set_defaults(fn=cmd_audit)
    t = sub.add_parser("teacher"); t.add_argument("--extract", metavar="DIR"); t.set_defaults(fn=cmd_teacher)
    q = sub.add_parser("plan"); q.add_argument("--out", default=str(DEFAULT_PLAN)); q.set_defaults(fn=cmd_plan)
    d = sub.add_parser("distill"); d.add_argument("--plan", default=str(DEFAULT_PLAN)); d.add_argument("--approve", required=True)
    d.add_argument("--source", required=True); d.add_argument("--out"); d.add_argument("--limit", type=int, default=0)
    d.add_argument("--stride", type=int, default=1); d.add_argument("--min-score", type=float, default=.75)
    d.add_argument("--consistency-px", type=float, default=6.0); d.add_argument("--progress", type=int, default=500)
    d.add_argument("--overwrite", action="store_true"); d.set_defaults(fn=cmd_distill)
    r = sub.add_parser("train"); r.add_argument("--plan", default=str(DEFAULT_PLAN)); r.add_argument("--approve", required=True)
    r.add_argument("--input", action="append", default=[]); r.add_argument("--out", default=str(TRAIN / "runs" / "sensorimotor_hand"))
    r.add_argument("--referee", action="append", default=[],
                   help="coordinate shard used only for evaluation; never admitted as prototypes")
    r.add_argument("--novelty-rms", type=float, default=.10); r.add_argument("--max-prototypes", type=int, default=256)
    r.add_argument("--seed", type=int, default=22); r.set_defaults(fn=cmd_train)
    return p


def main() -> None:
    p = parser(); args = p.parse_args()
    if not getattr(args, "command", None):
        args = p.parse_args(["audit"])
    args.fn(args)


if __name__ == "__main__":
    main()
