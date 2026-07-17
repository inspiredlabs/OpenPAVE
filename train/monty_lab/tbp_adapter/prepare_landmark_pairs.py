#!/usr/bin/env python3
"""Prepare paired MediaPipe-teacher and OpenPAVE-student landmark episodes.

Run in the OpenPAVE arm64 environment.  This does not invoke MediaPipe: it
uses the frozen teacher coordinates already stored in prepared.npz shards.
The resulting file is the only bridge into the separate tbp.monty env.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.monty_lab.tbp_adapter.oracle_roi import (
    crop as roi_crop,
    oracle_roi,
    project_to_source,
)
from train.pixel_sensor.runtime import PatchSensorModule
from train.pixel_sensor.train import DATASETS, OUT, TRUNK, evenly_spaced, trunk_predict

DEFAULT_OUT = ROOT / "train" / "runs" / "monty_landmark_alignment" / "episodes.npz"
ORACLE_STUDENT = ROOT / "train" / "runs" / "monty_landmark_alignment" / "oracle_student"
ORACLE_CROP = 96


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def object_label(label: str) -> str:
    if label == "stop":
        return "palm"
    if label.startswith("point_"):
        return "point"
    return label


def selected_rows(data, split: str, cap: int):
    has = np.asarray(data["has_lm"], bool)
    is_val = np.asarray(data["is_val"], bool)
    # MediaPipe occasionally emitted a constellation on frames explicitly
    # labelled no_hand. Those are negative diagnostics, not hand objects to
    # place in Monty's supervised graph memory.
    is_hand = np.asarray(data["labels"]).astype(str) != "no_hand"
    if split == "train":
        rows = np.where(has & is_hand & ~is_val)[0]
    elif split == "val":
        rows = np.where(has & is_hand & is_val)[0]
    elif split == "all":
        rows = np.where(has & is_hand)[0]
    else:
        raise ValueError(split)
    return evenly_spaced(rows, cap)


def oracle_student_predict(images, teacher, session):
    """Run the oracle-ROI landmarker on teacher-defined crops (doc §4).

    The ROI comes from the frozen teacher constellation — the acquisition-free
    benchmark — so measured error isolates the landmarker from crop error.
    Frames whose teacher cannot orient a ROI yield NaN points at confidence 0.
    """
    student = np.full((len(images), 21, 2), np.nan, dtype=np.float32)
    confidence = np.zeros((len(images), 21), dtype=np.float32)
    for row, (image, points) in enumerate(zip(images, teacher)):
        try:
            roi = oracle_roi(points)
        except ValueError:
            continue
        patch = roi_crop(image, roi, ORACLE_CROP).astype(np.float32) / 255.0 - 0.5
        uv, conf = session.run(
            ["landmarks_uv", "confidence"],
            {"crops": np.transpose(patch[None], (0, 3, 1, 2))})
        student[row] = project_to_source(uv[0], roi).astype(np.float32)
        confidence[row] = conf[0]
    return student, confidence


def main(args=None):
    import onnxruntime as ort

    cfg = parser().parse_args(args)
    if cfg.student == "oracle-roi":
        student_meta = json.loads((ORACLE_STUDENT / "meta.json").read_text())
        oracle_session = ort.InferenceSession(
            str(ORACLE_STUDENT / "landmarker.onnx"), providers=["CPUExecutionProvider"])
        trunk = refiner = None
    else:
        trunk = ort.InferenceSession(str(TRUNK), providers=["CPUExecutionProvider"])
        patch_meta = json.loads((OUT / "meta.json").read_text())
        refiner = PatchSensorModule(
            OUT / "patch_refiner.onnx",
            confidence_threshold=float(patch_meta["confidence_threshold"]),
        )

    fields = {name: [] for name in (
        "teacher_uv", "coarse_uv", "student_uv", "student_confidence",
        "labels", "sources", "source_indices", "is_val", "roles",
    )}
    exploration = [x for x in cfg.sources.split(",") if x]
    jobs = [(name, "train", cfg.train_cap, "exploration") for name in exploration]
    jobs += [(name, "val", cfg.val_cap, "exploration_holdout") for name in exploration]
    jobs += [(cfg.referee, "all", cfg.referee_cap, "untouched_referee")]

    for source, split, cap, role in jobs:
        path = DATASETS / source / "prepared.npz"
        with np.load(path, allow_pickle=True) as data:
            rows = selected_rows(data, split, cap)
            images = np.asarray(data["imgs"])[rows]
            teacher = np.asarray(data["landmarks"], np.float32)[rows].reshape(-1, 21, 2)
            labels = np.asarray(data["labels"])[rows]
        if cfg.student == "oracle-roi":
            refined, confidence = oracle_student_predict(images, teacher, oracle_session)
            coarse = refined.copy()  # the crop model has no separate coarse stage
        else:
            coarse = trunk_predict(images, trunk)
            refined, confidence = [], []
            for image, initial in zip(images, coarse):
                points, scores, _elapsed = refiner.refine(image, initial)
                refined.append(points); confidence.append(scores)
        n = len(rows)
        fields["teacher_uv"].append(teacher)
        fields["coarse_uv"].append(np.asarray(coarse, np.float32))
        fields["student_uv"].append(np.asarray(refined, np.float32))
        fields["student_confidence"].append(np.asarray(confidence, np.float32))
        fields["labels"].append(np.asarray([object_label(str(x)) for x in labels]))
        fields["sources"].append(np.asarray([source] * n))
        fields["source_indices"].append(rows.astype(np.int64))
        fields["is_val"].append(np.asarray([split != "train"] * n, bool))
        fields["roles"].append(np.asarray([role] * n))
        print(f"[pairs] {role:21s} {source:15s} {n:5d}")

    output = {key: np.concatenate(value) for key, value in fields.items()}
    # Preserve the motor/sensor contract used by Monty's 2-D saccade
    # environment. These are teacher positions for supervised pretraining;
    # evaluation must derive the trajectory from the candidate stream.
    teacher_xyz = np.concatenate(
        (output["teacher_uv"],
         np.zeros((*output["teacher_uv"].shape[:-1], 1), dtype=np.float32)),
        axis=-1)
    output["joint_order"] = np.tile(
        np.arange(21, dtype=np.int16), (len(teacher_xyz), 1))
    output["sensor_locations"] = teacher_xyz
    output["motor_deltas"] = np.zeros_like(teacher_xyz)
    output["motor_deltas"][:, 1:] = teacher_xyz[:, 1:] - teacher_xyz[:, :-1]
    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cfg.out, **output)
    manifest = {
        "contract": "openpave.monty-landmark-pairs.v1",
        "episodes": int(len(output["labels"])),
        "teacher": "stored MediaPipe 21-landmark coordinates",
        "split_unit": "prepared shard recording-preserving is_val",
        "fields": {key: list(value.shape) for key, value in output.items()},
    }
    if cfg.student == "oracle-roi":
        manifest.update({
            "student": "oracle-ROI soft-argmax heatmap landmarker "
                       "(acquisition-free teacher-defined crop)",
            "student_sha256": digest(ORACLE_STUDENT / "landmarker.onnx"),
            "student_confidence_threshold": float(
                student_meta["confidence_threshold"]),
            "student_confidence_semantics": student_meta["confidence_semantics"],
        })
    else:
        manifest.update({
            "student": "v3 trunk plus 35,175-parameter patch refiner",
            "trunk_sha256": digest(TRUNK),
            "patch_sha256": digest(OUT / "patch_refiner.onnx"),
        })
    cfg.out.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[pairs] wrote {len(output['labels'])} episodes -> {cfg.out}")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources", default="crude,hagrid_shapes,jester,ipn")
    p.add_argument("--referee", default="yolo26")
    p.add_argument("--train-cap", type=int, default=1000)
    p.add_argument("--val-cap", type=int, default=300)
    p.add_argument("--referee-cap", type=int, default=0)
    p.add_argument("--student", choices=("oracle-roi", "pixel-sensor"),
                   default="oracle-roi",
                   help="oracle-roi: crop-based heatmap landmarker on the "
                        "teacher-defined ROI; pixel-sensor: legacy full-frame "
                        "trunk plus patch refiner")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p


if __name__ == "__main__":
    main()
