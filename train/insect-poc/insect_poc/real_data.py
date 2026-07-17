"""Bounded download and feature extraction for genuine IPN Hand sequences."""

from __future__ import annotations

import csv
from collections import deque
import json
import math
import shutil
import tarfile
from pathlib import Path

import numpy as np

IPN_VIDEO_SHARDS = {
    1: "1HylyDnApIRNMloREqvKYWWhB88q7Z1Cg",
    2: "1tqR2FF8OlXGmYACw3TSxyS6QGDit_s3a",
    3: "1Dk7l-jAAvNLlb0faypAco88XYLWPa9g_",
    4: "1x0mDr-QHQtDkfcQAm9bKdWBR7Lj3fLcV",
    5: "1PCjldH6hmVYV7EObPf-jtNWCtDK60o6n",
}
IPN_ANNOTATION_FOLDER = "1-mihJEIFoNDpfo1puF8xAMJz6PGVKsBD"
MOTION_CLASSES = ["still", "left", "right", "up", "down"]


def fetch_ipn(raw_dir: Path, shard: int = 1, remove_archive: bool = True) -> None:
    """Download annotations and one official ~1GB video shard, then extract it."""
    import gdown

    if shard not in IPN_VIDEO_SHARDS:
        raise ValueError(f"IPN shard must be one of {sorted(IPN_VIDEO_SHARDS)}")
    raw_dir.mkdir(parents=True, exist_ok=True)
    annotations = raw_dir / "annotations"
    if not (annotations / "Annot_List.txt").exists():
        gdown.download_folder(id=IPN_ANNOTATION_FOLDER, output=str(annotations), quiet=False)
    videos = raw_dir / "videos"
    videos.mkdir(exist_ok=True)
    marker = videos / f".shard-{shard}-extracted"
    if marker.exists():
        print(f"[fetch-ipn] shard {shard} already extracted in {videos}")
        return
    free = shutil.disk_usage(raw_dir).free
    if free < 2_000_000_000:
        raise RuntimeError(f"need at least 2GB free for bounded IPN extraction; have {free/1e9:.1f}GB")
    archive = raw_dir / f"videos{shard:02d}.tgz"
    if archive.exists():
        try:
            with tarfile.open(archive, "r:gz") as probe:
                # Force a complete checksum/decompression pass; listing only
                # the first member does not detect a truncated Google response.
                for member in probe:
                    if member.isfile():
                        stream = probe.extractfile(member)
                        while stream and stream.read(1024 * 1024):
                            pass
        except (tarfile.TarError, EOFError, OSError):
            archive.unlink()
    if not archive.exists():
        gdown.download(id=IPN_VIDEO_SHARDS[shard], output=str(archive), quiet=False, resume=False)
    with tarfile.open(archive, "r:gz") as tf:
        root = videos.resolve()
        for member in tf.getmembers():
            target = (videos / member.name).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"unsafe archive member: {member.name}")
        tf.extractall(videos, filter="data")
    marker.write_text("ok\n")
    if remove_archive:
        archive.unlink(missing_ok=True)
    print(f"[fetch-ipn] extracted shard {shard} to {videos}")


def _annotations(path: Path) -> dict[str, list[tuple[int, int, int, str]]]:
    result: dict[str, list[tuple[int, int, int, str]]] = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            label = row["label"]
            # Horizontal bounded state: non-gesture, present/non-horizontal,
            # throw-left, throw-right. Labels remain independent of CV features.
            state = 0 if label == "D0X" else 2 if label == "G05" else 3 if label == "G06" else 1
            result.setdefault(row["video"], []).append(
                (int(row["t_start"]), int(row["t_end"]), state, label))
    return result


def _state_for_frame(intervals, frame_number: int) -> tuple[int, str]:
    for start, end, state, source in intervals:
        if start <= frame_number <= end:
            return state, source
    return 0, "unannotated"


def _features(gray: np.ndarray, previous: np.ndarray | None,
              previous_cx: float) -> tuple[np.ndarray, float]:
    import cv2

    gray = cv2.resize(gray, (160, 120), interpolation=cv2.INTER_AREA)
    mean, std = float(gray.mean()) / 255, float(gray.std()) / 128
    edges = cv2.Canny(gray, 60, 120)
    if previous is None:
        return np.asarray([mean, std, 0, 0, 0, 0, 0, 0, 0,
                           float((edges > 0).mean()), 0, 1], np.float32), 0.5
    diff = cv2.absdiff(gray, previous)
    mask = (diff > 18).astype(np.uint8)
    area = float(mask.mean())
    moments = cv2.moments(mask)
    if moments["m00"]:
        cx = float(moments["m10"] / moments["m00"] / mask.shape[1])
        cy = float(moments["m01"] / moments["m00"] / mask.shape[0])
    else:
        cx, cy = previous_cx, 0.5
    left = float(diff[:, :80].mean()) / 255
    right = float(diff[:, 80:].mean()) / 255
    flow = cv2.calcOpticalFlowFarneback(previous, gray, None, 0.5, 2, 11, 2, 5, 1.1, 0)
    moving = mask.astype(bool)
    if moving.any():
        flow_x = float(np.median(flow[..., 0][moving])) / 8.0
        flow_y = float(np.median(flow[..., 1][moving])) / 8.0
        flow_mag = float(np.median(np.linalg.norm(flow[moving], axis=1))) / 8.0
    else:
        flow_x = flow_y = flow_mag = 0.0
    vector = np.asarray([
        mean, std, float(diff.mean()) / 255, area, cx, cy,
        flow_x, flow_y, left - right, abs(mean - float(previous.mean()) / 255),
        flow_mag, 1.0,
    ], dtype=np.float32)
    return vector, cx


def prepare_ipn_core(raw_dir: Path, output: Path, minutes: float = 15.0,
                     target_fps: int = 15) -> None:
    """Extract ordered CV evidence and independent IPN labels into core NPZ."""
    import cv2

    annotations = _annotations(raw_dir / "annotations" / "Annot_List.txt")
    candidates = sorted(p for p in (raw_dir / "videos").rglob("*")
                        if p.suffix.lower() in {".mp4", ".avi"} and p.stem in annotations)
    if len(candidates) < 3:
        raise FileNotFoundError("need at least three extracted IPN videos; run fetch-ipn")
    # Spread recordings across the shard instead of taking adjacent captures
    # from the same subject/session until the duration cap is reached.
    take = min(len(candidates), max(8, int(math.ceil(minutes / 1.8))))
    chosen_indices = np.linspace(0, len(candidates) - 1, take, dtype=int)
    candidates = [candidates[i] for i in sorted(set(chosen_indices.tolist()))]
    max_frames = int(minutes * 60 * target_fps)
    X, y, recording_ids, timestamps, source_labels, provenance = [], [], [], [], [], []
    for rid, video in enumerate(candidates):
        if len(X) >= max_frames:
            break
        cap = cv2.VideoCapture(str(video))
        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        stride = max(1, int(round(source_fps / target_fps)))
        previous = None; previous_cx = 0.5; source_frame = 0; kept = 0
        while len(X) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            source_frame += 1
            if (source_frame - 1) % stride:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            feature, previous_cx = _features(gray, previous, previous_cx)
            previous = cv2.resize(gray, (160, 120), interpolation=cv2.INTER_AREA)
            state, source_label = _state_for_frame(annotations[video.stem], source_frame)
            X.append(feature); y.append(state); recording_ids.append(rid)
            timestamps.append((source_frame - 1) / source_fps); source_labels.append(source_label)
            kept += 1
        cap.release()
        provenance.append({"recording_id": rid, "video": video.name, "source_fps": source_fps,
                           "sample_fps": target_fps, "frames": kept})
    if len(set(recording_ids)) < 3:
        raise RuntimeError("15-minute selection contains fewer than three recordings")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, X=np.asarray(X, np.float32), y=np.asarray(y, np.int32),
                        recording_id=np.asarray(recording_ids, np.int32),
                        timestamp_s=np.asarray(timestamps, np.float32), fps=np.int32(target_fps),
                        classes=np.asarray(["no_gesture", "present_other", "throw_left", "throw_right"]),
                        source_label=np.asarray(source_labels))
    manifest = {"dataset": "IPN Hand", "license": "CC BY 4.0", "minutes": len(X) / target_fps / 60,
                "frames": len(X), "feature_spec": "ipn_cv_flow_v2_12f", "recordings": provenance}
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    counts = np.bincount(np.asarray(y), minlength=4)
    print(f"[core-real] {output}: {len(X)} frames, {len(set(recording_ids))} recordings, "
          f"classes={counts.tolist()}")


def _direction_label(source_label: str) -> str:
    return {"G03": "up", "G04": "down", "G05": "left", "G06": "right"}.get(
        source_label, "still")


def _trajectory_features(points: np.ndarray | None, history: deque) -> np.ndarray:
    """Twelve scale-normalized trajectory values from MediaPipe landmarks."""
    if points is None:
        history.clear()
        return np.zeros(12, dtype=np.float32)
    wrist, tip, centroid = points[0, :2], points[8, :2], points[:, :2].mean(0)
    scale = float(np.linalg.norm(points[9, :2] - wrist)) or 1e-4
    current = np.concatenate([wrist, tip, centroid, [scale]]).astype(np.float32)
    history.append(current)
    while len(history) > 16:
        history.popleft()
    previous = history[-2] if len(history) >= 2 else current
    lag5 = history[-6] if len(history) >= 6 else history[0]
    lag15 = history[-16] if len(history) >= 16 else history[0]
    d1 = (current - previous) / scale
    d5 = (current - lag5) / scale
    d15 = (current - lag15) / scale
    return np.asarray([
        d1[0], d1[1], d5[0], d5[1], d15[0], d15[1],
        d1[2], d1[3], d5[2], d5[3], d15[2], d15[3],
    ], dtype=np.float32)


def prepare_ipn_direction(raw_dir: Path, output: Path, minutes: float = 15.0,
                          target_fps: int = 15) -> None:
    """Create the real MediaPipe-trajectory dataset for the motion specialist."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    annotations = _annotations(raw_dir / "annotations" / "Annot_List.txt")
    candidates = sorted(p for p in (raw_dir / "videos").rglob("*")
                        if p.suffix.lower() in {".mp4", ".avi"} and p.stem in annotations)
    if len(candidates) < 3:
        raise FileNotFoundError("need at least three extracted IPN videos; run fetch-ipn")
    take = min(len(candidates), max(8, int(math.ceil(minutes / 1.8))))
    candidates = [candidates[i] for i in sorted(set(
        np.linspace(0, len(candidates) - 1, take, dtype=int).tolist()))]
    landmarker_path = Path(__file__).resolve().parents[2] / "weights" / "hand_landmarker.task"
    if not landmarker_path.exists():
        raise FileNotFoundError(f"missing MediaPipe model: {landmarker_path}")
    landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(landmarker_path)),
        num_hands=1, min_hand_detection_confidence=0.45,
        min_hand_presence_confidence=0.45, running_mode=vision.RunningMode.IMAGE))

    max_frames = int(minutes * 60 * target_fps)
    X, y, groups, source_labels, provenance = [], [], [], [], []
    detected = 0
    for rid, video in enumerate(candidates):
        if len(X) >= max_frames:
            break
        cap = cv2.VideoCapture(str(video)); source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30)
        stride = max(1, int(round(source_fps / target_fps)))
        source_frame = kept = 0; history: deque = deque()
        while len(X) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            source_frame += 1
            if (source_frame - 1) % stride:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            points = None
            if result.hand_landmarks:
                points = np.asarray([[p.x, p.y, p.z] for p in result.hand_landmarks[0]], np.float32)
                detected += 1
            feature = _trajectory_features(points, history)
            _, source = _state_for_frame(annotations[video.stem], source_frame)
            X.append(feature); y.append(_direction_label(source)); groups.append(rid)
            source_labels.append(source); kept += 1
        cap.release()
        provenance.append({"recording_id": rid, "video": video.name, "frames": kept})
    landmarker.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, X=np.asarray(X, np.float32), y=np.asarray(y),
                        groups=np.asarray(groups, np.int32), source_label=np.asarray(source_labels))
    labels = np.asarray(y)
    counts = {name: int(np.count_nonzero(labels == name)) for name in MOTION_CLASSES}
    manifest = {"dataset": "IPN Hand", "license": "CC BY 4.0",
                "feature_spec": "mediapipe_trajectory_v2_1_5_15f_12d", "frames": len(X),
                "minutes": len(X) / target_fps / 60, "landmark_detection_rate": detected / len(X),
                "classes": counts, "recordings": provenance}
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[direction-real] {output}: {len(X)} frames, detection={detected/len(X):.1%}, "
          f"classes={counts}")
