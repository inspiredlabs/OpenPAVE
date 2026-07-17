#!/usr/bin/env python3
"""Train a HanCo-only crop classifier behind the frozen legacy 71k acquirer."""

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
OUT = ROOT / "train/runs/hanco_crop_gesture"
CROP = 64


def crop_box(points: np.ndarray) -> tuple[float, float, float]:
    points = np.asarray(points, np.float32).reshape(21, 2)
    low, high = points.min(0), points.max(0)
    centre = (low + high) * 0.5
    side = max(float((high - low).max()) * 1.70, 0.08)
    return float(centre[0]), float(centre[1]), side


def take_crop(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    import cv2

    height, width = image.shape[:2]
    cx, cy, side = crop_box(points)
    side_px = max(8, int(round(side * max(height, width))))
    padding = side_px
    padded = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_REFLECT_101)
    patch = cv2.getRectSubPix(
        padded, (side_px, side_px),
        (float(cx * width + padding), float(cy * height + padding)))
    return cv2.resize(patch, (CROP, CROP), interpolation=cv2.INTER_AREA)


def build_crops(index, root: Path, out: Path, digest: str) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    cache = out / "crops.npz"
    if cache.is_file():
        stored = np.load(cache)
        if str(stored["index_sha256"]) == digest:
            return stored["crops"], stored["is_no_hand"]
    crops, negatives = [], []
    kernel = np.ones((7, 7), np.uint8)
    for rgb_relative, mask_relative, landmarks in zip(
            index["rgb_paths"], index["mask_paths"], index["landmarks"], strict=True):
        image = cv2.imread(str(root / str(rgb_relative)), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(root / str(mask_relative)), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(root / str(rgb_relative))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        crops.append(take_crop(image, landmarks))
        negatives.append(False)
        binary = cv2.dilate((mask >= 128).astype(np.uint8) * 255, kernel)
        removed = cv2.inpaint(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), binary, 5, cv2.INPAINT_TELEA)
        removed = cv2.cvtColor(removed, cv2.COLOR_BGR2RGB)
        crops.append(take_crop(removed, landmarks))
        negatives.append(True)
    values = np.asarray(crops, np.uint8), np.asarray(negatives, bool)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, crops=values[0], is_no_hand=values[1],
                        index_sha256=np.asarray(digest))
    return values


def model_definition():
    import torch.nn as nn

    class DS(nn.Module):
        def __init__(self, channels_in, channels_out, stride=1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(channels_in, channels_in, 3, stride, 1, groups=channels_in, bias=False),
                nn.BatchNorm2d(channels_in), nn.SiLU(),
                nn.Conv2d(channels_in, channels_out, 1, bias=False),
                nn.BatchNorm2d(channels_out), nn.SiLU())

        def forward(self, values):
            return self.net(values)

    return nn.Sequential(
        nn.Conv2d(3, 24, 3, 2, 1, bias=False), nn.BatchNorm2d(24), nn.SiLU(),
        DS(24, 40, 2), DS(40, 64, 2), DS(64, 96, 2), DS(96, 128),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 5))


def train(config: argparse.Namespace) -> dict:
    import torch
    import torch.nn.functional as functional
    from sklearn.metrics import confusion_matrix, f1_score

    started = time.perf_counter()
    index = np.load(config.index, allow_pickle=True)
    index_meta = json.loads(config.index_meta.read_text())
    digest = hashlib.sha256(config.index.read_bytes()).hexdigest()
    crops, is_no_hand = build_crops(index, Path(index_meta["root"]), config.out, digest)
    labels = np.repeat(index["labels"].astype(str), 2).astype("<U16")
    labels[is_no_hand] = "no_hand"
    splits = np.repeat(index["splits"].astype(str), 2)
    classes = np.asarray(["fist", "like", "no_hand", "palm", "point"])
    class_to_index = {name: position for position, name in enumerate(classes)}
    targets = np.asarray([class_to_index[label] for label in labels], np.int64)
    train_rows = np.flatnonzero(splits == "train")
    calibration_rows = np.flatnonzero(splits == "calibration")
    evaluation_rows = np.flatnonzero(splits == "evaluation")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model_definition().to(device)
    counts = np.bincount(targets[train_rows], minlength=len(classes)).clip(1)
    weights = torch.tensor(len(train_rows) / (len(classes) * counts), dtype=torch.float32, device=device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=config.epochs)
    rng = np.random.default_rng(config.seed)
    best_state, best_f1 = None, -1.0

    def infer(rows: np.ndarray) -> np.ndarray:
        model.eval()
        output = []
        with torch.no_grad():
            for batch in np.array_split(rows, max(1, int(np.ceil(len(rows) / 512)))):
                values = torch.tensor(crops[batch], device=device).permute(0, 3, 1, 2).float() / 255.0 - 0.5
                output.append(model(values).softmax(1).cpu().numpy())
        return np.concatenate(output)

    for epoch in range(config.epochs):
        model.train()
        order = rng.permutation(train_rows)
        running = 0.0
        for batch in np.array_split(order, max(1, int(np.ceil(len(order) / config.batch)))):
            values = torch.tensor(crops[batch], device=device).permute(0, 3, 1, 2).float() / 255.0 - 0.5
            brightness = 0.85 + 0.3 * torch.rand(len(batch), 1, 1, 1, device=device)
            values = values * brightness + torch.randn_like(values) * 0.015
            flip = torch.rand(len(batch), device=device) < 0.5
            values = torch.where(flip[:, None, None, None], values.flip(-1), values).contiguous()
            truth = torch.tensor(targets[batch], device=device)
            loss = functional.cross_entropy(model(values), truth, weight=weights)
            optimiser.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0, foreach=False)
            optimiser.step(); running += float(loss.detach()) * len(batch)
        schedule.step()
        calibration_probabilities = infer(calibration_rows)
        calibration_prediction = calibration_probabilities.argmax(1)
        value = float(f1_score(targets[calibration_rows], calibration_prediction,
                               average="macro", zero_division=0))
        print(f"[hanco-crop] epoch={epoch + 1}/{config.epochs} "
              f"loss={running / len(train_rows):.4f} calibration_macro_f1={value:.3f}", flush=True)
        if value > best_f1:
            best_f1 = value
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

    model.load_state_dict(best_state)
    probabilities = infer(evaluation_rows)
    predicted = probabilities.argmax(1)
    truth = targets[evaluation_rows]
    matrix = confusion_matrix(truth, predicted, labels=np.arange(len(classes)))
    positive = labels[evaluation_rows] != "no_hand"
    no_hand = ~positive
    metrics = {
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "overall_accuracy": float((truth == predicted).mean()),
        "correct_gesture_acquisition_rate": float((truth[positive] == predicted[positive]).mean()),
        "wrong_gesture_rate": float(((truth[positive] != predicted[positive])
                                      & (predicted[positive] != class_to_index["no_hand"])).mean()),
        "no_hand_false_acquisition_rate": float((predicted[no_hand] != class_to_index["no_hand"]).mean()),
        "confusion_matrix_order": classes.tolist(), "confusion_matrix": matrix.tolist(),
    }
    for label in ("palm", "like", "fist", "point"):
        rows = truth == class_to_index[label]
        metrics[f"{label}_correct_rate"] = float((predicted[rows] == truth[rows]).mean())

    model = model.cpu().eval()
    class Export(torch.nn.Module):
        def __init__(self, inner):
            super().__init__(); self.inner = inner

        def forward(self, values):
            return self.inner(values).softmax(1)

    model_path = config.out / "crop.onnx"
    torch.onnx.export(Export(model), torch.zeros(1, 3, CROP, CROP), model_path,
                      input_names=["crops"], output_names=["probabilities"],
                      dynamic_axes={"crops": {0: "n"}, "probabilities": {0: "n"}}, dynamo=False)
    report = {
        "contract": "openpave.hanco-crop-gesture.v1",
        "front_end": "legacy 71k landmark_tower (frozen)",
        "training_sources": ["HanCo RGB", "HanCo mask_hand", "HanCo calibration/xyz/shape"],
        "external_training_sources": [], "classes": classes.tolist(),
        "index_sha256": digest, "manifest_sha256": index_meta["manifest_sha256"],
        "crop_px": CROP, "params": sum(parameter.numel() for parameter in model.parameters()),
        "epochs": config.epochs, "best_calibration_macro_f1": best_f1,
        "data": {"train": len(train_rows), "calibration": len(calibration_rows),
                 "evaluation": len(evaluation_rows)},
        "metrics": metrics, "model": str(model_path), "seconds": time.perf_counter() - started,
        "seed": config.seed,
    }
    (config.out / "meta.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--index", type=Path, default=INDEX)
    result.add_argument("--index-meta", type=Path, default=INDEX_META)
    result.add_argument("--out", type=Path, default=OUT)
    result.add_argument("--epochs", type=int, default=14)
    result.add_argument("--batch", type=int, default=192)
    result.add_argument("--seed", type=int, default=37)
    return result


if __name__ == "__main__":
    print(json.dumps(train(parser().parse_args()), indent=2))
