#!/usr/bin/env python3
"""Build a versioned, model-neutral HanCo corpus for live-inference parity.

The corpus is an index over the immutable HanCo cache, not another image copy.
It binds each manually labelled observation to its RGB, hand mask, foreground
cutout, calibration, metric landmarks and MANO parameters.  All cameras and
frames belonging to one sequence receive the same deterministic split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import cache
from pathlib import Path

import numpy as np

CONTRACT = "openpave.hanco-inference-parity-corpus.v1"
LABELS = ("fist", "palm", "point", "like")
REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path("~/.cache/hanco").expanduser()
DEFAULT_ASSIGNMENTS = (
    REPO.parent / "monty/tutorials/results/hand_pose_lab/classifier_assignments.json"
)
DEFAULT_PARENT = REPO / "train/datasets/hanco_inference_parity"


def split_for_sequence(sequence: str, seed: str = "openpave-hanco-v1") -> str:
    """Stable 80/10/10 split with sequence, frame and camera isolation."""
    bucket = int(hashlib.sha256(f"{seed}:{sequence}".encode()).hexdigest()[:8], 16) % 10
    return "evaluation" if bucket == 0 else "calibration" if bucket == 1 else "train"


def assign_sequence_splits(
    selected: dict[str, dict[str, tuple[str | None, str]]],
    seed: str = "openpave-hanco-v1",
) -> dict[str, str]:
    """Deterministically stratify multilabel sequence groups into 80/10/10."""
    labels_by_sequence = {
        sequence: {label for label, _scope in frames.values() if label is not None}
        for sequence, frames in selected.items()
    }
    for label in LABELS:
        supporting = [sequence for sequence, labels in labels_by_sequence.items() if label in labels]
        if len(supporting) < 3:
            raise ValueError(
                f"label {label!r} needs at least three independent sequences for train/calibration/evaluation; "
                f"found {len(supporting)}"
            )
    unassigned = set(selected)
    result: dict[str, str] = {}
    holdout_size = max(1, round(len(selected) * 0.1))
    for split in ("evaluation", "calibration"):
        chosen: list[str] = []
        missing = set(LABELS)
        while missing:
            candidates = [sequence for sequence in unassigned if labels_by_sequence[sequence] & missing]
            if not candidates:
                raise ValueError(f"cannot give {split} coverage for labels {sorted(missing)}")
            sequence = min(
                candidates,
                key=lambda value: (
                    -len(labels_by_sequence[value] & missing),
                    hashlib.sha256(f"{seed}:{split}:{value}".encode()).hexdigest(),
                ),
            )
            chosen.append(sequence)
            unassigned.remove(sequence)
            missing -= labels_by_sequence[sequence]
        fill = sorted(
            unassigned,
            key=lambda value: hashlib.sha256(f"{seed}:{split}:fill:{value}".encode()).hexdigest(),
        )
        for sequence in fill[:max(0, holdout_size - len(chosen))]:
            chosen.append(sequence)
            unassigned.remove(sequence)
        result.update({sequence: split for sequence in chosen})
    result.update({sequence: "train" for sequence in unassigned})
    for split in ("train", "calibration", "evaluation"):
        present = set().union(*(labels_by_sequence[sequence] for sequence, value in result.items() if value == split))
        if present != set(LABELS):
            raise ValueError(f"{split} has labels {sorted(present)}, expected {list(LABELS)}")
    return result


def project_landmarks(
    xyz_world: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return pixel XY, camera XYZ, and per-joint geometric validity."""
    xyz = np.asarray(xyz_world, np.float64)
    homogeneous = np.column_stack((xyz, np.ones(len(xyz))))
    camera = (np.asarray(extrinsics, np.float64) @ homogeneous.T).T[:, :3]
    pixels_h = (np.asarray(intrinsics, np.float64) @ camera.T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
    valid = np.isfinite(pixels).all(axis=1) & np.isfinite(camera).all(axis=1) & (camera[:, 2] > 0)
    return pixels.astype(np.float32), camera.astype(np.float32), valid


def _bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _image_stats(path: Path, kind: str) -> tuple[int, int, float, list[int] | None]:
    import cv2

    mode = cv2.IMREAD_GRAYSCALE if kind == "hand_mask" else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise FileNotFoundError(path)
    height, width = image.shape[:2]
    # mask_fg is not a mask: it is an RGB cutout on black.  JPEG ringing makes
    # exact nonzero tests unstable, hence the small intensity floor.
    foreground = image >= 128 if kind == "hand_mask" else image.max(axis=2) > 8
    return width, height, float(foreground.mean()), _bbox(foreground)


@cache
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_assignments(path: Path) -> tuple[dict[str, str | None], dict[str, int], dict]:
    payload = json.loads(path.read_text())
    if not str(payload.get("contract", "")).startswith("monty.hanco-classifier-assignments"):
        raise ValueError(f"unsupported assignments contract in {path}")
    assignments = {str(key): value for key, value in payload["assignments"].items()}
    invalid = {key: value for key, value in assignments.items() if value is not None and value not in LABELS}
    if invalid:
        raise ValueError(f"unsupported classifier labels: {invalid}")
    cameras = {str(key): int(value) for key, value in payload.get("sequence_cameras", {}).items()}
    return assignments, cameras, payload


def selected_frames(
    root: Path,
    assignments: dict[str, str | None],
    context_radius: int,
) -> dict[str, dict[str, tuple[str | None, str]]]:
    """Resolve sequence overrides and retain nearby unlabelled temporal context."""
    direct: dict[str, dict[str, tuple[str | None, str]]] = defaultdict(dict)
    sequence_labels = {
        key.split("/", 1)[0]: value
        for key, value in assignments.items()
        if key.endswith("/*") and value is not None
    }
    for sequence, label in sequence_labels.items():
        frames = sorted(path.stem for path in (root / "rgb" / sequence / "cam0").glob("*.jpg"))
        if not frames:
            raise FileNotFoundError(f"no RGB frames for sequence assignment {sequence}")
        direct[sequence].update({frame: (label, "sequence") for frame in frames})
    for key, label in assignments.items():
        if label is None or key.endswith("/*"):
            continue
        sequence, frame = key.split("/", 1)
        if sequence in sequence_labels:
            continue
        if not (root / "rgb" / sequence / "cam0" / f"{frame}.jpg").is_file():
            raise FileNotFoundError(f"no RGB for frame assignment {key}")
        direct[sequence][frame] = (label, "frame")

    if context_radius:
        for sequence, chosen in list(direct.items()):
            available = sorted(path.stem for path in (root / "rgb" / sequence / "cam0").glob("*.jpg"))
            positions = {frame: index for index, frame in enumerate(available)}
            for frame in list(chosen):
                index = positions[frame]
                for neighbour in available[max(0, index - context_radius): index + context_radius + 1]:
                    chosen.setdefault(neighbour, (None, "context"))
    return direct


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def build(
    root: Path,
    assignments_path: Path,
    out: Path,
    *,
    context_radius: int = 2,
    cameras: tuple[int, ...] = tuple(range(8)),
    max_sequences: int | None = None,
    max_frames_per_sequence: int | None = None,
    split_seed: str = "openpave-hanco-v1",
    hash_assets: bool = False,
) -> dict:
    root = root.expanduser().resolve()
    assignments_path = assignments_path.expanduser().resolve()
    assignments, preferred_cameras, assignment_payload = _load_assignments(assignments_path)
    selected = selected_frames(root, assignments, context_radius)
    sequences = sorted(selected)
    if max_sequences is not None:
        sequences = sequences[:max_sequences]

    temporary = out.with_name(f".{out.name}.{os.getpid()}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    records_path = temporary / "records.jsonl"

    arrays: dict[str, list] = defaultdict(list)
    counts: Counter = Counter()
    skipped: Counter = Counter()
    sequence_splits = assign_sequence_splits(
        {sequence: selected[sequence] for sequence in sequences}, split_seed
    )
    materialized_sequences: set[str] = set()
    incomplete_frames: list[dict] = []
    row = 0
    with records_path.open("w") as records_file:
        for sequence in sequences:
            all_frames = sorted(path.stem for path in (root / "rgb" / sequence / "cam0").glob("*.jpg"))
            neighbour = {
                frame: (
                    all_frames[index - 1] if index else None,
                    all_frames[index + 1] if index + 1 < len(all_frames) else None,
                )
                for index, frame in enumerate(all_frames)
            }
            frames = sorted(selected[sequence])
            if max_frames_per_sequence is not None:
                frames = frames[:max_frames_per_sequence]
            split = sequence_splits[sequence]
            for frame in frames:
                label, scope = selected[sequence][frame]
                xyz_path = root / "xyz" / sequence / f"{frame}.json"
                shape_path = root / "shape" / sequence / f"{frame}.json"
                calibration_path = root / "HanCo_calib_meta/calib" / sequence / f"{frame}.json"
                missing_geometry = [
                    name for name, path in (
                        ("xyz", xyz_path), ("shape", shape_path),
                        ("calibration", calibration_path),
                    ) if not path.is_file()
                ]
                if missing_geometry:
                    skipped["missing_geometry"] += len(cameras)
                    for name in missing_geometry:
                        skipped[f"missing_{name}"] += len(cameras)
                    incomplete_frames.append({
                        "sequence_id": sequence,
                        "frame_id": frame,
                        "label": label,
                        "label_scope": scope,
                        "missing": missing_geometry,
                    })
                    continue
                xyz = np.asarray(json.loads(xyz_path.read_text()), np.float32)
                shape = json.loads(shape_path.read_text())
                calibration = json.loads(calibration_path.read_text())
                if xyz.shape != (21, 3):
                    raise ValueError(f"expected 21x3 xyz at {xyz_path}, got {xyz.shape}")
                for camera in cameras:
                    rgb = root / "rgb" / sequence / f"cam{camera}" / f"{frame}.jpg"
                    hand_mask = root / "mask_hand" / sequence / f"cam{camera}" / f"{frame}.jpg"
                    foreground = root / "mask_fg" / sequence / f"cam{camera}" / f"{frame}.jpg"
                    if not all(path.is_file() for path in (rgb, hand_mask, foreground)):
                        skipped["missing_assets"] += 1
                        continue
                    rgb_width, rgb_height, _, _ = _image_stats(rgb, "rgb")
                    hand_width, hand_height, hand_fraction, hand_bbox = _image_stats(hand_mask, "hand_mask")
                    fg_width, fg_height, fg_fraction, fg_bbox = _image_stats(foreground, "foreground_cutout")
                    if {(rgb_width, rgb_height), (hand_width, hand_height), (fg_width, fg_height)} != {(rgb_width, rgb_height)}:
                        raise ValueError(f"asset dimensions disagree for {sequence}/{frame}/cam{camera}")
                    intrinsics = np.asarray(calibration["K"][camera], np.float32)
                    extrinsics = np.asarray(calibration["M"][camera], np.float32)
                    pixels, camera_xyz, geometrically_valid = project_landmarks(xyz, intrinsics, extrinsics)
                    normalized = pixels / np.asarray([rgb_width, rgb_height], np.float32)
                    visible = geometrically_valid & (normalized >= 0).all(axis=1) & (normalized <= 1).all(axis=1)
                    asset_paths = {
                        "rgb": str(rgb.relative_to(root)),
                        "hand_mask": str(hand_mask.relative_to(root)),
                        "foreground_cutout": str(foreground.relative_to(root)),
                        "xyz": str(xyz_path.relative_to(root)),
                        "shape": str(shape_path.relative_to(root)),
                        "calibration": str(calibration_path.relative_to(root)),
                    }
                    record = {
                        "row": row,
                        "observation_id": f"{sequence}/{frame}/cam{camera}",
                        "sequence_id": sequence,
                        "frame_id": frame,
                        "camera_id": camera,
                        "preferred_camera": preferred_cameras.get(sequence, 0),
                        "synchronized_group": f"{sequence}/{frame}",
                        "previous_frame_id": neighbour[frame][0],
                        "next_frame_id": neighbour[frame][1],
                        "split": split,
                        "label": label,
                        "label_scope": scope,
                        "assets": asset_paths,
                        "asset_semantics": {
                            "hand_mask": "grayscale segmentation; >=128 is hand",
                            "foreground_cutout": "RGB foreground on black; not a binary mask",
                        },
                        "image_size": [rgb_width, rgb_height],
                        "visible_landmarks": int(visible.sum()),
                        "hand_mask_fraction": hand_fraction,
                        "foreground_fraction": fg_fraction,
                        "hand_bbox_xyxy": hand_bbox,
                        "foreground_bbox_xyxy": fg_bbox,
                        "baseline_channels": {},
                    }
                    if hash_assets:
                        record["asset_sha256"] = {name: _sha256(root / path) for name, path in asset_paths.items()}
                    records_file.write(json.dumps(record, sort_keys=True) + "\n")
                    arrays["world_xyz"].append(xyz)
                    arrays["camera_xyz"].append(camera_xyz)
                    arrays["pixel_xy"].append(pixels)
                    arrays["normalized_xy"].append(normalized)
                    arrays["visible"].append(visible)
                    arrays["intrinsics"].append(intrinsics)
                    arrays["extrinsics"].append(extrinsics)
                    arrays["mano_pose"].append(np.asarray(shape["poses"][0], np.float32))
                    arrays["mano_shape"].append(np.asarray(shape["shapes"][0], np.float32))
                    arrays["global_translation"].append(np.asarray(shape["global_t"], np.float32).reshape(-1))
                    counts[(split, label or "unlabelled_context")] += 1
                    materialized_sequences.add(sequence)
                    row += 1

    if not row:
        raise RuntimeError("no complete HanCo observations were produced")
    np.savez_compressed(temporary / "geometry.npz", **{name: np.asarray(values) for name, values in arrays.items()})
    shutil.copy2(assignments_path, temporary / "classifier_assignments.snapshot.json")
    assignments_hash = _sha256(assignments_path)
    record_hash = _sha256(records_path)
    geometry_hash = _sha256(temporary / "geometry.npz")
    report = {
        "contract": CONTRACT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hanco_root": str(root),
        "assignments_source": str(assignments_path),
        "assignments_contract": assignment_payload.get("contract"),
        "assignments_sha256": assignments_hash,
        "records": "records.jsonl",
        "records_sha256": record_hash,
        "geometry": "geometry.npz",
        "geometry_sha256": geometry_hash,
        "observations": row,
        "sequences": sequences,
        "materialized_sequences": sorted(materialized_sequences),
        "sequences_without_observations": sorted(set(sequences) - materialized_sequences),
        "sequence_splits": sequence_splits,
        "split_seed": split_seed,
        "split_policy": (
            "deterministic multilabel sequence-group stratification, approximately 80/10/10; "
            "every split contains every label and no sequence crosses splits"
        ),
        "context_radius": context_radius,
        "cameras": list(cameras),
        "counts": {f"{split}:{label}": count for (split, label), count in sorted(counts.items())},
        "skipped": dict(skipped),
        "incomplete_frames": incomplete_frames,
        "coordinate_contract": {
            "world_xyz": "HanCo metric world coordinates",
            "camera_xyz": "M @ homogeneous(world_xyz), per frame and camera",
            "pixel_xy": "K @ camera_xyz, divided by depth; difficult/out-of-frame views retained",
            "normalized_xy": "pixel_xy / decoded RGB [width,height]",
            "visible": "positive camera depth, finite projection, normalized XY in [0,1]",
        },
        "model_neutrality": "baseline_channels is empty; legacy 71K output is optional evidence, never a target",
        "asset_storage": "relative references into hanco_root; source images are not duplicated",
    }
    _atomic_json(temporary / "manifest.json", report)
    if out.exists():
        raise FileExistsError(f"refusing to replace existing corpus {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(out)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%SZ")
    result.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    result.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    result.add_argument("--out", type=Path, default=DEFAULT_PARENT / timestamp)
    result.add_argument("--context-radius", type=int, default=2)
    result.add_argument("--cameras", default="0,1,2,3,4,5,6,7")
    result.add_argument("--max-sequences", type=int)
    result.add_argument("--max-frames-per-sequence", type=int)
    result.add_argument("--split-seed", default="openpave-hanco-v1")
    result.add_argument("--hash-assets", action="store_true")
    return result


if __name__ == "__main__":
    config = parser().parse_args()
    camera_ids = tuple(int(value) for value in config.cameras.split(",") if value != "")
    if config.context_radius < 0 or not camera_ids or any(value not in range(8) for value in camera_ids):
        raise SystemExit("context radius must be >=0 and cameras must be IDs 0..7")
    print(json.dumps(build(
        config.root, config.assignments, config.out,
        context_radius=config.context_radius,
        cameras=camera_ids,
        max_sequences=config.max_sequences,
        max_frames_per_sequence=config.max_frames_per_sequence,
        split_seed=config.split_seed,
        hash_assets=config.hash_assets,
    ), indent=2))
