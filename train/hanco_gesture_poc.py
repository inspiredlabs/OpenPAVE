#!/usr/bin/env python3
"""Train reviewed HanCo gestures behind the frozen legacy 71k acquirer."""

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
INDEX = ROOT / "train/datasets/hanco_gestures/index.npz"
INDEX_META = INDEX.with_name("meta.json")
OUT = ROOT / "train/runs/hanco_gesture_poc"


def run_legacy_paths(root: Path, paths: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import cv2
    from train.hanco_target_poc import feature
    from train.landmark_tower import LandmarkerRuntime

    runtime = LandmarkerRuntime()
    features, presence, quality = [], [], []
    for relative in paths:
        image = cv2.imread(str(root / str(relative)), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(root / str(relative))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        landmarks, p, q = runtime.step(image, apply_gate=False)
        features.append(feature(np.asarray(landmarks).reshape(21, 2)))
        presence.append(p)
        quality.append(q)
    return np.stack(features), np.asarray(presence), np.asarray(quality)


def presence_gates(positive_p: np.ndarray, positive_q: np.ndarray,
                   negative_p: np.ndarray, negative_q: np.ndarray) -> tuple[float, float, float]:
    from train.hanco_target_poc import best_presence_gates

    return best_presence_gates(positive_p, positive_q, negative_p, negative_q)


def index_digest(index: Path) -> str:
    return hashlib.sha256(index.read_bytes()).hexdigest()


def load_or_extract(index: np.lib.npyio.NpzFile, root: Path, out: Path,
                    digest: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache = out / "legacy_features.npz"
    if cache.is_file():
        stored = np.load(cache)
        if str(stored["index_sha256"]) == digest:
            return stored["features"], stored["presence"], stored["quality"]
    values = run_legacy_paths(root, index["rgb_paths"])
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, features=values[0], presence=values[1], quality=values[2],
                        index_sha256=np.asarray(digest))
    return values


def load_or_extract_masked(index: np.lib.npyio.NpzFile, root: Path, out: Path,
                           digest: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the frozen acquirer on same-domain RGB with hand pixels inpainted."""
    import cv2
    from train.hanco_target_poc import feature
    from train.landmark_tower import LandmarkerRuntime

    cache = out / "masked_no_hand_features.npz"
    if cache.is_file():
        stored = np.load(cache)
        if str(stored["index_sha256"]) == digest:
            return stored["features"], stored["presence"], stored["quality"]
    runtime = LandmarkerRuntime()
    features, presence, quality = [], [], []
    kernel = np.ones((7, 7), np.uint8)
    for rgb_relative, mask_relative in zip(index["rgb_paths"], index["mask_paths"], strict=True):
        image = cv2.imread(str(root / str(rgb_relative)), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(root / str(mask_relative)), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(root / str(rgb_relative))
        binary = cv2.dilate((mask >= 128).astype(np.uint8) * 255, kernel)
        negative = cv2.inpaint(image, binary, 5, cv2.INPAINT_TELEA)
        negative = cv2.cvtColor(negative, cv2.COLOR_BGR2RGB)
        landmarks, p, q = runtime.step(negative, apply_gate=False)
        features.append(feature(np.asarray(landmarks).reshape(21, 2)))
        presence.append(p)
        quality.append(q)
    values = np.stack(features), np.asarray(presence), np.asarray(quality)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, features=values[0], presence=values[1], quality=values[2],
                        index_sha256=np.asarray(digest))
    return values


def ground_truth_features(landmarks: np.ndarray) -> np.ndarray:
    from train.hanco_target_poc import feature

    return np.stack([feature(points.reshape(21, 2)) for points in landmarks])


def best_confidence_gate(probabilities: np.ndarray, truth: np.ndarray,
                         classes: np.ndarray) -> tuple[float, float]:
    from sklearn.metrics import f1_score

    best = (0.0, -1.0)
    top = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    no_hand = int(np.flatnonzero(classes == "no_hand")[0])
    for threshold in np.linspace(0.0, 0.95, 191):
        predicted = top.copy()
        predicted[confidence < threshold] = no_hand
        value = float(f1_score(truth, predicted, average="macro", zero_division=0))
        if value > best[1]:
            best = (float(threshold), value)
    return best


def train(config: argparse.Namespace) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix, f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    started = time.perf_counter()
    index = np.load(config.index, allow_pickle=True)
    index_meta = json.loads(config.index_meta.read_text())
    hanco_root = Path(index_meta["root"])
    digest = index_digest(config.index)
    runtime_x, hand_presence, hand_quality = load_or_extract(
        index, hanco_root, config.out, digest)
    projected_x = ground_truth_features(index["landmarks"])
    labels = np.asarray(index["labels"]).astype(str)
    splits = np.asarray(index["splits"]).astype(str)

    no_hand_x, no_hand_presence, no_hand_quality = load_or_extract_masked(
        index, hanco_root, config.out, digest)

    calibration = splits == "calibration"
    evaluation = splits == "evaluation"
    fitting = splits == "train"
    no_hand_calibration = calibration
    no_hand_eval = evaluation
    no_hand_train = fitting
    presence_gate, quality_gate, presence_calibration_f1 = presence_gates(
        hand_presence[calibration], hand_quality[calibration],
        no_hand_presence[no_hand_calibration], no_hand_quality[no_hand_calibration])
    hand_upstream = (hand_presence >= presence_gate) & (hand_quality >= quality_gate)
    no_hand_upstream = (no_hand_presence >= presence_gate) & (no_hand_quality >= quality_gate)

    fit_x = np.concatenate((projected_x[fitting], runtime_x[fitting & hand_upstream]))
    fit_labels = np.concatenate((labels[fitting], labels[fitting & hand_upstream]))
    if np.any(no_hand_train & no_hand_upstream):
        fit_x = np.concatenate((fit_x, no_hand_x[no_hand_train & no_hand_upstream]))
        fit_labels = np.concatenate((fit_labels, np.full(
            int((no_hand_train & no_hand_upstream).sum()), "no_hand")))

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=config.c, class_weight="balanced", max_iter=1500,
                           random_state=config.seed),
    )
    model.fit(fit_x, fit_labels)
    classes = model.named_steps["logisticregression"].classes_
    class_to_index = {name: index for index, name in enumerate(classes)}

    calibration_x = np.concatenate((runtime_x[calibration], no_hand_x[no_hand_calibration]))
    calibration_labels = np.concatenate((labels[calibration], np.full(no_hand_calibration.sum(), "no_hand")))
    calibration_probabilities = model.predict_proba(calibration_x)
    calibration_hand_count = calibration.sum()
    calibration_probabilities[:calibration_hand_count][~hand_upstream[calibration]] = 0.0
    calibration_probabilities[:calibration_hand_count][~hand_upstream[calibration], class_to_index["no_hand"]] = 1.0
    calibration_probabilities[calibration_hand_count:][~no_hand_upstream[no_hand_calibration]] = 0.0
    calibration_probabilities[calibration_hand_count:][~no_hand_upstream[no_hand_calibration], class_to_index["no_hand"]] = 1.0
    calibration_truth = np.asarray([class_to_index[label] for label in calibration_labels])
    confidence_gate, confidence_calibration_f1 = best_confidence_gate(
        calibration_probabilities, calibration_truth, classes)

    evaluation_x = np.concatenate((runtime_x[evaluation], no_hand_x[no_hand_eval]))
    evaluation_labels = np.concatenate((labels[evaluation], np.full(no_hand_eval.sum(), "no_hand")))
    probabilities = model.predict_proba(evaluation_x)
    hand_count = evaluation.sum()
    probabilities[:hand_count][~hand_upstream[evaluation]] = 0.0
    probabilities[:hand_count][~hand_upstream[evaluation], class_to_index["no_hand"]] = 1.0
    probabilities[hand_count:][~no_hand_upstream[no_hand_eval]] = 0.0
    probabilities[hand_count:][~no_hand_upstream[no_hand_eval], class_to_index["no_hand"]] = 1.0
    predicted = probabilities.argmax(axis=1)
    predicted[probabilities.max(axis=1) < confidence_gate] = class_to_index["no_hand"]
    truth = np.asarray([class_to_index[label] for label in evaluation_labels])

    presence_truth = np.r_[np.ones(hand_count, bool), np.zeros(no_hand_eval.sum(), bool)]
    presence_prediction = np.r_[hand_upstream[evaluation], no_hand_upstream[no_hand_eval]]
    positive_truth = truth[:hand_count]
    positive_prediction = predicted[:hand_count]
    per_class = {}
    for label in ("palm", "like", "fist", "point"):
        class_index = class_to_index[label]
        rows = positive_truth == class_index
        per_class[label] = {
            "frames": int(rows.sum()),
            "correct_acquisition_rate": float((positive_prediction[rows] == class_index).mean()),
            "abstention_rate": float((positive_prediction[rows] == class_to_index["no_hand"]).mean()),
        }

    scaler = model.named_steps["standardscaler"]
    classifier = model.named_steps["logisticregression"]
    model_path = config.out / "model.npz"
    np.savez_compressed(
        model_path,
        mean=scaler.mean_.astype(np.float32), scale=scaler.scale_.astype(np.float32),
        coefficients=classifier.coef_.astype(np.float32),
        intercept=classifier.intercept_.astype(np.float32), classes=classes,
        confidence_gate=np.float32(confidence_gate),
        presence_gate=np.float32(presence_gate), quality_gate=np.float32(quality_gate),
    )
    matrix = confusion_matrix(truth, predicted, labels=np.arange(len(classes)))
    report = {
        "contract": "openpave.hanco-gesture-poc.v1",
        "front_end": "legacy 71k landmark_tower (frozen)",
        "classes": classes.tolist(),
        "manifest": index_meta["manifest"],
        "manifest_sha256": index_meta["manifest_sha256"],
        "index_sha256": digest,
        "data": {
            "indexed_hanco_observations": int(len(labels)),
            "no_hand_source": "mask-inpainted HanCo RGB only",
            "fit_features": int(len(fit_x)),
            "calibration_observations": int(len(calibration_x)),
            "evaluation_observations": int(len(evaluation_x)),
            "index_counts": index_meta["counts"],
            "unconfirmed_excluded": index_meta["unconfirmed_excluded"],
        },
        "gates": {
            "presence": presence_gate, "quality": quality_gate,
            "gesture_confidence": confidence_gate,
            "presence_calibration_f1": presence_calibration_f1,
            "gesture_calibration_macro_f1": confidence_calibration_f1,
        },
        "metrics": {
            "presence_f1": float(f1_score(presence_truth, presence_prediction, zero_division=0)),
            "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
            "overall_accuracy": float((truth == predicted).mean()),
            "correct_gesture_acquisition_rate": float((positive_truth == positive_prediction).mean()),
            "wrong_gesture_rate": float(((positive_prediction != positive_truth)
                                          & (positive_prediction != class_to_index["no_hand"])).mean()),
            "no_hand_false_acquisition_rate": float(
                (predicted[hand_count:] != class_to_index["no_hand"]).mean()),
            "per_class": per_class,
            "confusion_matrix_order": classes.tolist(),
            "confusion_matrix": matrix.tolist(),
        },
        "model": str(model_path), "seconds": time.perf_counter() - started,
        "seed": config.seed,
    }
    (config.out / "meta.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--index", type=Path, default=INDEX)
    result.add_argument("--index-meta", type=Path, default=INDEX_META)
    result.add_argument("--out", type=Path, default=OUT)
    result.add_argument("--c", type=float, default=1.0)
    result.add_argument("--seed", type=int, default=37)
    return result


if __name__ == "__main__":
    print(json.dumps(train(parser().parse_args()), indent=2))
