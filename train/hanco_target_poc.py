#!/usr/bin/env python3
"""Train a one-gesture HanCo geometry gate behind the legacy 71k acquirer.

The pixel front end is deliberately frozen: this proof-of-concept asks whether
one HanCo tester gesture can be added without sacrificing the incumbent's hand
presence and acquisition behaviour.  HanCo xyz/calibration provides projected
positive and hard-negative constellations; MANO pose parameters select the
closest non-target poses instead of wasting capacity on easy negatives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
HANCO = Path("~/.cache/hanco").expanduser()
PREPARED = ROOT / "train/datasets/hanco/prepared.npz"
CRUDE = ROOT / "train/datasets/crude/prepared.npz"
OUT = ROOT / "train/runs/hanco_target_poc"
TARGET_SEQUENCE = "0110"
PALM = np.asarray([5, 9, 13, 17])
BONES = np.asarray([
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
])


def feature(points: np.ndarray) -> np.ndarray:
    """Translation/scale/in-plane-rotation normalized 2D hand geometry."""
    points = np.asarray(points, np.float64).reshape(21, 2)
    points = points - points[0]
    axis = points[PALM].mean(axis=0)
    scale = max(float(np.linalg.norm(axis)), 1e-6)
    points /= scale
    angle = np.arctan2(axis[1], axis[0])
    c, s = np.cos(angle), np.sin(angle)
    points = points @ np.asarray([[c, -s], [s, c]])
    lengths = np.linalg.norm(points[BONES[:, 1]] - points[BONES[:, 0]], axis=1)
    return np.concatenate((points.reshape(-1), lengths)).astype(np.float32)


def project(xyz: np.ndarray, calibration: dict, camera: int) -> np.ndarray | None:
    homogeneous = np.column_stack((xyz, np.ones(21)))
    camera_xyz = (np.asarray(calibration["M"][camera]) @ homogeneous.T).T[:, :3]
    if np.any(camera_xyz[:, 2] <= 0):
        return None
    pixels_h = (np.asarray(calibration["K"][camera]) @ camera_xyz.T).T
    points = pixels_h[:, :2] / pixels_h[:, 2:3] / 224.0
    if not np.isfinite(points).all() or np.any(points < 0) or np.any(points > 1):
        return None
    return points


def pose_vector(path: Path) -> np.ndarray:
    value = json.loads(path.read_text())["poses"][0]
    return np.asarray(value, np.float32)


def target_pose(root: Path, sequence: str, train_frames: set[str]) -> np.ndarray:
    vectors = [pose_vector(root / "shape" / sequence / f"{frame}.json")
               for frame in sorted(train_frames)
               if (root / "shape" / sequence / f"{frame}.json").is_file()]
    if not vectors:
        raise FileNotFoundError(f"no MANO pose parameters for target sequence {sequence}")
    return np.median(np.stack(vectors), axis=0)


def hard_negative_frames(root: Path, sequence: str, reference: np.ndarray,
                         limit: int, per_sequence: int) -> list[tuple[float, str, str]]:
    candidates: list[tuple[float, str, str]] = []
    for shape_dir in sorted((root / "shape").iterdir()):
        seq = shape_dir.name
        if seq == sequence or not (root / "xyz" / seq).is_dir():
            continue
        files = sorted(shape_dir.glob("*.json"))
        if not files:
            continue
        indices = np.linspace(0, len(files) - 1, min(per_sequence, len(files)), dtype=int)
        for index in indices:
            frame = files[int(index)].stem
            if not (root / "xyz" / seq / f"{frame}.json").is_file():
                continue
            if not (root / "HanCo_calib_meta/calib" / seq / f"{frame}.json").is_file():
                continue
            distance = float(np.linalg.norm(pose_vector(files[int(index)]) - reference))
            candidates.append((distance, seq, frame))
    candidates.sort()
    return candidates[:limit]


def projected_features(root: Path, rows: list[tuple[float, str, str]]) -> tuple[np.ndarray, np.ndarray]:
    values, groups = [], []
    for _, sequence, frame in rows:
        xyz = np.asarray(json.loads((root / "xyz" / sequence / f"{frame}.json").read_text()))
        calibration = json.loads(
            (root / "HanCo_calib_meta/calib" / sequence / f"{frame}.json").read_text())
        for camera in range(8):
            points = project(xyz, calibration, camera)
            if points is not None:
                values.append(feature(points))
                groups.append(sequence)
    if not values:
        raise RuntimeError("no valid projected HanCo hard negatives")
    return np.stack(values), np.asarray(groups)


def target_projection_features(root: Path, sequence: str,
                               frames: set[str]) -> np.ndarray:
    rows = [(0.0, sequence, frame) for frame in sorted(frames)]
    return projected_features(root, rows)[0]


def run_legacy(images: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from train.landmark_tower import LandmarkerRuntime

    runtime = LandmarkerRuntime()
    landmarks, presence, quality = [], [], []
    for image in images:
        points, p, q = runtime.step(image, apply_gate=False)
        landmarks.append(feature(np.asarray(points).reshape(21, 2)))
        presence.append(p)
        quality.append(q)
    return np.stack(landmarks), np.asarray(presence), np.asarray(quality)


def split_group(sequence: str) -> int:
    """Stable sequence split: 0=train, 1=calibration, 2=evaluation."""
    bucket = int(hashlib.sha256(sequence.encode()).hexdigest()[:8], 16) % 5
    return 2 if bucket == 0 else 1 if bucket == 1 else 0


def scores(model, values: np.ndarray) -> np.ndarray:
    return model.predict_proba(values)[:, 1]


def best_threshold(positive: np.ndarray, negative: np.ndarray) -> tuple[float, float]:
    from sklearn.metrics import f1_score

    if len(negative) > len(positive):
        order = np.argsort(negative)[-len(positive):]
        negative = negative[order]
    truth = np.r_[np.ones(len(positive), bool), np.zeros(len(negative), bool)]
    value = np.r_[positive, negative]
    best = (0.5, -1.0)
    for threshold in np.linspace(0.05, 0.95, 181):
        score = float(f1_score(truth, value >= threshold, zero_division=0))
        if score > best[1]:
            best = (float(threshold), score)
    return best


def best_presence_gates(positive_p: np.ndarray, positive_q: np.ndarray,
                        negative_p: np.ndarray, negative_q: np.ndarray
                        ) -> tuple[float, float, float]:
    from sklearn.metrics import f1_score

    truth = np.r_[np.ones(len(positive_p), bool), np.zeros(len(negative_p), bool)]
    best = (0.5, 0.15, -1.0)
    for presence in np.linspace(0.3, 0.95, 27):
        for quality in np.linspace(0.05, 0.8, 31):
            prediction = np.r_[
                (positive_p >= presence) & (positive_q >= quality),
                (negative_p >= presence) & (negative_q >= quality),
            ]
            value = float(f1_score(truth, prediction, zero_division=0))
            if value > best[2]:
                best = (float(presence), float(quality), value)
    return best


def train(config: argparse.Namespace) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    started = time.perf_counter()
    prepared = np.load(config.prepared, allow_pickle=True)
    target_frames = np.asarray(prepared["frame_ids"]).astype(str)
    is_eval = np.asarray(prepared["is_val"], bool)
    available_train_frames = sorted(set(target_frames[~is_eval]))
    calibration_count = max(1, round(len(available_train_frames) * 0.2))
    calibration_frames = set(available_train_frames[-calibration_count:])
    fit_frames = set(available_train_frames[:-calibration_count])
    is_calibration = np.asarray([frame in calibration_frames for frame in target_frames])
    is_fit = ~(is_eval | is_calibration)

    predicted_x, predicted_presence, predicted_quality = run_legacy(prepared["imgs"])
    crude = np.load(config.crude, allow_pickle=True)
    labels = np.asarray(crude["labels"]).astype(str)
    no_hand_rows = np.flatnonzero(labels == "no_hand")
    no_hand_x, no_hand_presence, no_hand_quality = run_legacy(crude["imgs"][no_hand_rows])
    crude_eval = np.asarray(crude["is_val"], bool)[no_hand_rows]
    crude_calibration = (~crude_eval) & (np.arange(len(no_hand_rows)) % 5 == 1)
    crude_fit = ~(crude_eval | crude_calibration)

    if config.presence_gate is None or config.quality_gate is None:
        presence_gate, quality_gate, gate_calibration_f1 = best_presence_gates(
            predicted_presence[is_calibration], predicted_quality[is_calibration],
            no_hand_presence[crude_calibration], no_hand_quality[crude_calibration])
    else:
        presence_gate, quality_gate = config.presence_gate, config.quality_gate
        gate_calibration_f1 = None
    upstream_accept = (predicted_presence >= presence_gate) & (
        predicted_quality >= quality_gate)
    no_hand_upstream = (no_hand_presence >= presence_gate) & (
        no_hand_quality >= quality_gate)

    reference = target_pose(config.hanco, config.target_sequence, fit_frames)
    hard_rows = hard_negative_frames(
        config.hanco, config.target_sequence, reference,
        config.hard_negative_frames, config.frames_per_sequence)
    negative_x, negative_groups = projected_features(config.hanco, hard_rows)
    negative_split = np.asarray([split_group(group) for group in negative_groups])

    projected_train = target_projection_features(config.hanco, config.target_sequence, fit_frames)
    positive_train = predicted_x[is_fit & upstream_accept]
    if len(positive_train):
        positive_train = np.concatenate((projected_train, positive_train))
    else:
        positive_train = projected_train
    fit_negative = negative_x[negative_split == 0]
    if np.any(crude_fit & no_hand_upstream):
        fit_negative = np.concatenate((fit_negative, no_hand_x[crude_fit & no_hand_upstream]))

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=config.c, class_weight="balanced", max_iter=1000,
                           random_state=config.seed),
    )
    model.fit(np.concatenate((positive_train, fit_negative)),
              np.r_[np.ones(len(positive_train)), np.zeros(len(fit_negative))])

    calibration_target_score = scores(model, predicted_x[is_calibration])
    calibration_target_score[~upstream_accept[is_calibration]] = 0.0
    calibration_other_score = scores(model, negative_x[negative_split == 1])
    calibration_no_hand_score = scores(model, no_hand_x[crude_calibration])
    calibration_no_hand_score[~no_hand_upstream[crude_calibration]] = 0.0
    threshold, calibration_f1 = best_threshold(calibration_target_score, np.concatenate(
        (calibration_other_score, calibration_no_hand_score)))

    target_score = scores(model, predicted_x[is_eval])
    target_score[~upstream_accept[is_eval]] = 0.0
    other_score = scores(model, negative_x[negative_split == 2])
    no_hand_score = scores(model, no_hand_x[crude_eval])
    no_hand_score[~no_hand_upstream[crude_eval]] = 0.0

    target_accept = target_score >= threshold
    no_hand_accept = no_hand_score >= threshold
    presence_truth = np.r_[np.ones(is_eval.sum(), bool), np.zeros(crude_eval.sum(), bool)]
    presence_pred = np.r_[upstream_accept[is_eval], no_hand_upstream[crude_eval]]
    presence_f1 = float(f1_score(presence_truth, presence_pred, zero_division=0))

    camera_ids = np.asarray(prepared["camera_ids"])[is_eval]
    val_frames = target_frames[is_eval]
    lock_times = []
    for camera in sorted(set(map(int, camera_ids))):
        rows = np.flatnonzero(camera_ids == camera)
        rows = rows[np.argsort(val_frames[rows])]
        locks = np.flatnonzero(target_accept[rows])
        if len(locks):
            lock_times.append(float(locks[0]) / 5.0)

    scaler = model.named_steps["standardscaler"]
    classifier = model.named_steps["logisticregression"]
    config.out.mkdir(parents=True, exist_ok=True)
    model_path = config.out / "model.npz"
    np.savez_compressed(
        model_path,
        mean=scaler.mean_.astype(np.float32),
        scale=scaler.scale_.astype(np.float32),
        coefficients=classifier.coef_[0].astype(np.float32),
        intercept=np.float32(classifier.intercept_[0]),
        threshold=np.float32(threshold),
    )
    report = {
        "contract": "openpave.hanco-target-poc.v1",
        "target": "HanCo_tester",
        "target_sequence": config.target_sequence,
        "front_end": "legacy 71k landmark_tower (frozen)",
        "outcomes": ["no_hand", "HanCo_tester"],
        "feature": "wrist-centred, palm-axis-normalized 2D joints + bone lengths",
        "sources": {
            "target_rgb": str(config.prepared),
            "calibration_meta": str(config.hanco / "HanCo_calib_meta"),
            "shape": str(config.hanco / "shape"),
            "xyz": str(config.hanco / "xyz"),
            "no_hand": str(config.crude),
        },
        "data": {
            "target_projected_train": int(len(projected_train)),
            "target_runtime_train_accepted": int((is_fit & upstream_accept).sum()),
            "target_runtime_calibration": int(is_calibration.sum()),
            "target_runtime_evaluation": int(is_eval.sum()),
            "hard_negative_frames": int(len(hard_rows)),
            "hard_negative_projections_train": int((negative_split == 0).sum()),
            "hard_negative_projections_calibration": int((negative_split == 1).sum()),
            "hard_negative_projections_evaluation": int((negative_split == 2).sum()),
            "no_hand_train": int(crude_fit.sum()),
            "no_hand_calibration": int(crude_calibration.sum()),
            "no_hand_evaluation": int(crude_eval.sum()),
            "split": "target temporal frame groups; negatives held out by sequence; calibration separate from evaluation",
        },
        "gates": {
            "presence": presence_gate,
            "quality": quality_gate,
            "target_probability": threshold,
            "presence_calibration_f1": gate_calibration_f1,
        },
        "metrics": {
            "presence_f1": presence_f1,
            "target_acquisition_rate": float(target_accept.mean()),
            "no_hand_false_acquisition_rate": float(no_hand_accept.mean()),
            "other_pose_false_acquisition_rate": float((other_score >= threshold).mean()),
            "calibration_f1": calibration_f1,
            "median_time_to_first_lock_s": (
                float(np.median(lock_times)) if lock_times else None),
            "cameras_locked": len(lock_times),
            "cameras_total": int(len(set(map(int, camera_ids)))),
            "legacy_target_presence_rate": float(upstream_accept[is_eval].mean()),
            "legacy_no_hand_false_presence_rate": float(no_hand_upstream[crude_eval].mean()),
        },
        "model": str(model_path),
        "seconds": time.perf_counter() - started,
        "seed": config.seed,
    }
    (config.out / "meta.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--hanco", type=Path, default=HANCO)
    result.add_argument("--prepared", type=Path, default=PREPARED)
    result.add_argument("--crude", type=Path, default=CRUDE)
    result.add_argument("--out", type=Path, default=OUT)
    result.add_argument("--target-sequence", default=TARGET_SEQUENCE)
    result.add_argument("--hard-negative-frames", type=int, default=1200)
    result.add_argument("--frames-per-sequence", type=int, default=3)
    result.add_argument("--presence-gate", type=float)
    result.add_argument("--quality-gate", type=float)
    result.add_argument("--c", type=float, default=1.0)
    result.add_argument("--seed", type=int, default=37)
    return result


if __name__ == "__main__":
    print(json.dumps(train(parser().parse_args()), indent=2))
