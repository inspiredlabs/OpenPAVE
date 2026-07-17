#!/usr/bin/env python3
"""Expand the reviewed HanCo gesture manifest into a calibrated sample index."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "train/datasets/hanco/gesture_manifest.json"
DEFAULT_OUT = ROOT / "train/datasets/hanco_gestures/index.npz"


def split_for_sequence(sequence: str) -> str:
    remainder = int(sequence) % 5
    return "evaluation" if remainder == 0 else "calibration" if remainder == 1 else "train"


def expand_frames(root: Path, item: dict) -> list[str]:
    available = sorted((root / "xyz" / item["sequence"]).glob("*.json"))
    if item.get("all"):
        return [path.stem for path in available]
    if "range" in item:
        first, last = item["range"]
        requested = range(int(first), int(last) + 1)
    else:
        requested = map(int, item.get("frames", []))
    frames = [f"{frame:08d}" for frame in requested]
    missing = [frame for frame in frames
               if not (root / "xyz" / item["sequence"] / f"{frame}.json").is_file()]
    if missing:
        raise FileNotFoundError(f"{item['sequence']} missing requested xyz frames: {missing}")
    return frames


def project(xyz: np.ndarray, calibration: dict, camera: int) -> np.ndarray | None:
    homogeneous = np.column_stack((xyz, np.ones(21)))
    camera_xyz = (np.asarray(calibration["M"][camera]) @ homogeneous.T).T[:, :3]
    if np.any(camera_xyz[:, 2] <= 0):
        return None
    pixels_h = (np.asarray(calibration["K"][camera]) @ camera_xyz.T).T
    points = pixels_h[:, :2] / pixels_h[:, 2:3] / 224.0
    if not np.isfinite(points).all() or np.any(points < 0) or np.any(points > 1):
        return None
    return points.astype(np.float32)


def resolve_pose_partitions(root: Path, manifest: dict) -> tuple[dict[str, list[dict]], dict]:
    """Assign transition-sequence frames to the nearest reviewed MANO pose anchor."""
    selections = {label: list(rows) for label, rows in manifest["labels"].items()}
    report = {}
    for item in manifest.get("pose_partitioned_sequences", []):
        sequence = item["sequence"]
        shape_dir = root / "shape" / sequence
        files = sorted(shape_dir.glob("*.json"))
        anchors = {}
        for label, anchor in item["anchors"].items():
            vectors = []
            for frame in anchor["frames"]:
                path = shape_dir / f"{int(frame):08d}.json"
                vectors.append(np.asarray(json.loads(path.read_text())["poses"][0], np.float32))
            anchors[label] = np.median(np.stack(vectors), axis=0)
        assigned = {label: [] for label in anchors}
        for path in files:
            pose = np.asarray(json.loads(path.read_text())["poses"][0], np.float32)
            label = min(anchors, key=lambda name: float(np.linalg.norm(pose - anchors[name])))
            assigned[label].append(int(path.stem))
        report[sequence] = {label: len(frames) for label, frames in assigned.items()}
        for label, frames in assigned.items():
            anchor = item["anchors"][label]
            selections.setdefault(label, []).append({
                "sequence": sequence,
                "frames": frames,
                "orientation": anchor.get("orientation", ""),
                "resolved_by": "nearest MANO pose anchor",
            })
    return selections, report


def prepare(manifest_path: Path, out: Path) -> dict:
    import cv2

    manifest = json.loads(manifest_path.read_text())
    root = Path(manifest["root"]).expanduser().resolve()
    selections_by_label, partition_report = resolve_pose_partitions(root, manifest)
    records: dict[str, list] = {
        "labels": [], "orientations": [], "splits": [], "sequence_ids": [],
        "frame_ids": [], "camera_ids": [], "rgb_paths": [], "mask_paths": [],
        "landmarks": [], "world_landmarks": [], "mano_pose": [], "mask_fraction": [],
    }
    seen: set[tuple[str, str, int]] = set()
    skipped = 0
    for label, selections in selections_by_label.items():
        for item in selections:
            sequence = item["sequence"]
            for frame in expand_frames(root, item):
                xyz_path = root / "xyz" / sequence / f"{frame}.json"
                shape_path = root / "shape" / sequence / f"{frame}.json"
                calib_path = root / "HanCo_calib_meta/calib" / sequence / f"{frame}.json"
                if not shape_path.is_file() or not calib_path.is_file():
                    raise FileNotFoundError(f"incomplete geometry for {sequence}/{frame}")
                xyz = np.asarray(json.loads(xyz_path.read_text()), np.float32)
                shape = json.loads(shape_path.read_text())
                pose = np.asarray(shape["poses"][0], np.float32)
                calibration = json.loads(calib_path.read_text())
                for camera in range(8):
                    key = (sequence, frame, camera)
                    if key in seen:
                        raise ValueError(f"duplicate labeled observation: {key}")
                    rgb_path = root / "rgb" / sequence / f"cam{camera}" / f"{frame}.jpg"
                    mask_path = root / "mask_hand" / sequence / f"cam{camera}" / f"{frame}.jpg"
                    points = project(xyz, calibration, camera)
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    if not rgb_path.is_file() or mask is None or points is None:
                        skipped += 1
                        continue
                    fraction = float((mask >= 128).mean())
                    if fraction < 0.002:
                        skipped += 1
                        continue
                    seen.add(key)
                    records["labels"].append(label)
                    records["orientations"].append(item.get("orientation", ""))
                    records["splits"].append(split_for_sequence(sequence))
                    records["sequence_ids"].append(sequence)
                    records["frame_ids"].append(frame)
                    records["camera_ids"].append(camera)
                    records["rgb_paths"].append(str(rgb_path.relative_to(root)))
                    records["mask_paths"].append(str(mask_path.relative_to(root)))
                    records["landmarks"].append(points.reshape(-1))
                    records["world_landmarks"].append(xyz)
                    records["mano_pose"].append(pose)
                    records["mask_fraction"].append(fraction)

    out.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "labels": np.asarray(records["labels"]),
        "orientations": np.asarray(records["orientations"]),
        "splits": np.asarray(records["splits"]),
        "sequence_ids": np.asarray(records["sequence_ids"]),
        "frame_ids": np.asarray(records["frame_ids"]),
        "camera_ids": np.asarray(records["camera_ids"], np.int8),
        "rgb_paths": np.asarray(records["rgb_paths"]),
        "mask_paths": np.asarray(records["mask_paths"]),
        "landmarks": np.asarray(records["landmarks"], np.float32),
        "world_landmarks": np.asarray(records["world_landmarks"], np.float32),
        "mano_pose": np.asarray(records["mano_pose"], np.float32),
        "mask_fraction": np.asarray(records["mask_fraction"], np.float32),
    }
    np.savez_compressed(out, **arrays)
    counts = {}
    for split in ("train", "calibration", "evaluation"):
        counts[split] = {
            label: int(((arrays["splits"] == split) & (arrays["labels"] == label)).sum())
            for label in manifest["labels"]
        }
    report = {
        "contract": "openpave.hanco-gesture-index.v1",
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "root": str(root),
        "output": str(out),
        "observations": len(arrays["labels"]),
        "skipped_incomplete": skipped,
        "counts": counts,
        "unconfirmed_excluded": manifest.get("unconfirmed", []),
        "pose_partitions": partition_report,
        "leakage_contract": "all cameras for a frame share its sequence-level split",
    }
    out.with_name("meta.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    result.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    print(json.dumps(prepare(args.manifest, args.out), indent=2))
