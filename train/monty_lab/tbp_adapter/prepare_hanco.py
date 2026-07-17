#!/usr/bin/env python3
"""Prepare a bounded HanCo shard for acquisition-matched landmarker training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

DEFAULT_ROOT = Path("~/.cache/HanCo_tester").expanduser()
DEFAULT_OUT = Path("train/datasets/hanco/prepared.npz")
IMAGE_SIZE = 128


def project_landmarks(
    xyz_world: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project HanCo world joints into one cropped camera image."""
    xyz_world = np.asarray(xyz_world, dtype=np.float64)
    homogeneous = np.column_stack((xyz_world, np.ones(len(xyz_world))))
    camera = (np.asarray(extrinsics, dtype=np.float64) @ homogeneous.T).T[:, :3]
    pixels_h = (np.asarray(intrinsics, dtype=np.float64) @ camera.T).T
    pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
    return pixels, camera


def prepare(root: Path, out: Path, validation_fraction: float = 0.2) -> dict:
    """Build the OpenPAVE prepared.npz contract without frame/camera leakage."""
    import cv2

    root = root.expanduser().resolve()
    sequences = sorted(path.name for path in (root / "xyz").iterdir() if path.is_dir())
    if not sequences:
        raise FileNotFoundError(f"No HanCo xyz sequences below {root}")

    images: list[np.ndarray] = []
    landmarks: list[np.ndarray] = []
    world_landmarks: list[np.ndarray] = []
    camera_landmarks: list[np.ndarray] = []
    labels: list[str] = []
    validation: list[bool] = []
    sequence_ids: list[str] = []
    frame_ids: list[str] = []
    camera_ids: list[int] = []
    source_paths: list[str] = []
    split_frames: dict[str, list[str]] = {}

    for sequence in sequences:
        xyz_files = sorted((root / "xyz" / sequence).glob("*.json"))
        if not xyz_files:
            continue
        split_at = max(1, int(round(len(xyz_files) * (1.0 - validation_fraction))))
        validation_frames = {path.stem for path in xyz_files[split_at:]}
        split_frames[sequence] = sorted(validation_frames)
        for xyz_path in xyz_files:
            frame = xyz_path.stem
            calib_path = root / "calib" / sequence / f"{frame}.json"
            if not calib_path.is_file():
                continue
            xyz = np.asarray(json.loads(xyz_path.read_text()), dtype=np.float64)
            calibration = json.loads(calib_path.read_text())
            for camera_id, (intrinsics, extrinsics) in enumerate(
                zip(calibration["K"], calibration["M"], strict=True)
            ):
                image_path = (
                    root / "rgb" / sequence / f"cam{camera_id}" / f"{frame}.jpg"
                )
                if not image_path.is_file():
                    continue
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                height, width = image.shape[:2]
                pixels, camera_xyz = project_landmarks(xyz, intrinsics, extrinsics)
                normalized = pixels / np.asarray([width, height], dtype=np.float64)
                visible = (
                    np.isfinite(normalized).all(axis=1)
                    & (camera_xyz[:, 2] > 0)
                    & (normalized >= 0).all(axis=1)
                    & (normalized <= 1).all(axis=1)
                )
                if not visible.all():
                    continue
                images.append(
                    cv2.resize(
                        image,
                        (IMAGE_SIZE, IMAGE_SIZE),
                        interpolation=cv2.INTER_AREA,
                    )
                )
                landmarks.append(normalized.astype(np.float32).reshape(-1))
                world_landmarks.append(xyz.astype(np.float32))
                camera_landmarks.append(camera_xyz.astype(np.float32))
                labels.append("hand")
                validation.append(frame in validation_frames)
                sequence_ids.append(sequence)
                frame_ids.append(frame)
                camera_ids.append(camera_id)
                source_paths.append(str(image_path.relative_to(root)))

    if not images:
        raise RuntimeError(
            "No complete HanCo RGB/calibration/xyz observations were found"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        imgs=np.asarray(images, dtype=np.uint8),
        labels=np.asarray(labels),
        is_val=np.asarray(validation, dtype=bool),
        landmarks=np.asarray(landmarks, dtype=np.float32),
        has_lm=np.ones(len(images), dtype=bool),
        presence_known=np.ones(len(images), dtype=bool),
        world_landmarks=np.asarray(world_landmarks, dtype=np.float32),
        camera_landmarks=np.asarray(camera_landmarks, dtype=np.float32),
        sequence_ids=np.asarray(sequence_ids),
        frame_ids=np.asarray(frame_ids),
        camera_ids=np.asarray(camera_ids, dtype=np.int8),
        source_paths=np.asarray(source_paths),
    )
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    report = {
        "contract": "openpave.hanco-prepared.v1",
        "root": str(root),
        "output": str(out),
        "sha256": digest,
        "sequences": sequences,
        "observations": len(images),
        "train": int(len(images) - sum(validation)),
        "validation": int(sum(validation)),
        "split_policy": (
            "temporally contiguous frames; all eight cameras share frame split"
        ),
        "validation_frames": split_frames,
        "image_size": IMAGE_SIZE,
        "coordinate_contract": (
            "xy normalized to HanCo per-frame 224px crop; K read per frame"
        ),
        "auxiliary_only": True,
    }
    report_path = out.with_name("prepared.meta.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    result.add_argument("--out", type=Path, default=DEFAULT_OUT)
    result.add_argument("--validation-fraction", type=float, default=0.2)
    return result


if __name__ == "__main__":
    config = parser().parse_args()
    if not 0.0 < config.validation_fraction < 0.5:
        raise SystemExit("--validation-fraction must be in (0, 0.5)")
    print(
        json.dumps(
            prepare(config.root, config.out, config.validation_fraction), indent=2
        )
    )
