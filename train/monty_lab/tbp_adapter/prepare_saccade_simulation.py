#!/usr/bin/env python3
"""Prepare compact Monty-style hand saccade episodes for offline simulation.

Each positive hand frame becomes a 21-step sensor trajectory.  A step stores
the joint-local visual patch, its 2-D sensor position, and the motor delta from
the previous step.  A matched background patch is stored for every positive
step.  Early episodes render the skeleton alone, middle episodes add bounded
background noise, and late episodes use the real RGB frame.

This is a sensor-training bridge, not a claim of 3-D geometry: z is explicitly
zero until RGB-D or a rig supplies depth and surface normals.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.pixel_sensor.train import DATASETS, evenly_spaced

DEFAULT_OUT = (ROOT / "train" / "runs" / "monty_landmark_alignment"
               / "saccade_simulation.npz")
PARENT = np.asarray([0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11,
                     0, 13, 14, 15, 0, 17, 18, 19], dtype=np.int64)


def render_structure(shape, points, phase, rng):
    height, width = shape[:2]
    if phase == "structure_only":
        canvas = np.zeros((height, width, 3), np.uint8)
    else:
        noise = rng.integers(0, 48, (height, width, 1), dtype=np.uint8)
        canvas = np.repeat(noise, 3, axis=2)
    pixels = np.column_stack((points[:, 0] * width, points[:, 1] * height))
    pixels = np.rint(pixels).astype(np.int32)
    thickness = max(1, round(min(height, width) / 128))
    for child in range(1, 21):
        cv2.line(canvas, tuple(pixels[PARENT[child]]), tuple(pixels[child]),
                 (235, 235, 235), thickness, cv2.LINE_AA)
    for point in pixels:
        cv2.circle(canvas, tuple(point), thickness + 1, (255, 255, 255), -1,
                   cv2.LINE_AA)
    return canvas


def patch_at(image, uv, size):
    height, width = image.shape[:2]
    centre = (float(uv[0] * width), float(uv[1] * height))
    return cv2.getRectSubPix(image, (size, size), centre)


def background_uv(points, rng, minimum_distance=0.12):
    best, best_distance = np.asarray([0.5, 0.5]), -1.0
    for _ in range(48):
        candidate = rng.uniform(0.05, 0.95, 2)
        distance = float(np.linalg.norm(points - candidate, axis=1).min())
        if distance > best_distance:
            best, best_distance = candidate, distance
        if distance >= minimum_distance:
            break
    return best.astype(np.float32)


def main(argv=None):
    cfg = parser().parse_args(argv)
    rng = np.random.default_rng(cfg.seed)
    records = {name: [] for name in (
        "patches", "negative_patches", "locations", "motor_deltas",
        "joint_ids", "labels", "sources", "source_indices", "phases")}
    sources = [value for value in cfg.sources.split(",") if value]
    for source in sources:
        with np.load(DATASETS / source / "prepared.npz", allow_pickle=True) as data:
            labels = np.asarray(data["labels"]).astype(str)
            mask = (np.asarray(data["has_lm"], bool)
                    & ~np.asarray(data["is_val"], bool)
                    & (labels != "no_hand"))
            rows = evenly_spaced(np.where(mask)[0], cfg.cap)
            images = np.asarray(data["imgs"])[rows]
            points_all = np.asarray(data["landmarks"], np.float32)[rows].reshape(-1, 21, 2)
        for sequence, (row, image, points) in enumerate(zip(rows, images, points_all)):
            ratio = sequence / max(len(rows) - 1, 1)
            if ratio < cfg.structure_fraction:
                phase = "structure_only"
                sensor_image = render_structure(image.shape, points, phase, rng)
            elif ratio < cfg.structure_fraction + cfg.noise_fraction:
                phase = "contrast_noise"
                sensor_image = render_structure(image.shape, points, phase, rng)
            else:
                phase = "real_rgb"
                sensor_image = image
            locations = np.column_stack(
                (points, np.zeros(21, dtype=np.float32))).astype(np.float32)
            deltas = np.zeros_like(locations)
            deltas[1:] = locations[1:] - locations[:-1]
            positives = np.stack([patch_at(sensor_image, uv, cfg.patch)
                                  for uv in points])
            negatives = np.stack([
                patch_at(sensor_image, background_uv(points, rng), cfg.patch)
                for _ in range(21)])
            records["patches"].append(positives)
            records["negative_patches"].append(negatives)
            records["locations"].append(locations)
            records["motor_deltas"].append(deltas)
            records["joint_ids"].append(np.arange(21, dtype=np.int16))
            records["labels"].append(labels[row])
            records["sources"].append(source)
            records["source_indices"].append(int(row))
            records["phases"].append(phase)
        print(f"[saccade] {source:15s} episodes={len(rows)}")
    if not records["patches"]:
        raise SystemExit("no hand episodes found")
    output = {key: np.asarray(value) for key, value in records.items()}
    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cfg.out, **output)
    manifest = {
        "contract": "openpave.monty-hand-saccades.v1",
        "episodes": int(len(output["labels"])),
        "steps_per_episode": 21,
        "patch_px": cfg.patch,
        "curriculum": ["structure_only", "contrast_noise", "real_rgb"],
        "negative_sampling": "one background patch per positive sensor step",
        "geometry": "normalized image x/y with synthetic z=0",
        "surface_normals": "unavailable; require RGB-D or synthetic mesh",
        "fields": {key: list(value.shape) for key, value in output.items()},
    }
    cfg.out.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[saccade] wrote {cfg.out}")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources", default="crude,hagrid_shapes,jester,ipn")
    p.add_argument("--cap", type=int, default=100)
    p.add_argument("--patch", type=int, default=24)
    p.add_argument("--structure-fraction", type=float, default=0.40)
    p.add_argument("--noise-fraction", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=37)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p


if __name__ == "__main__":
    main()
