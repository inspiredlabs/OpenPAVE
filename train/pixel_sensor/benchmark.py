#!/usr/bin/env python3
"""Single-frame CPU benchmark for the frozen trunk + patch refiner."""
from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

from .runtime import PatchSensorModule
from .train import DATASETS, OUT, TRUNK, trunk_predict


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    import onnxruntime as ort
    trunk = ort.InferenceSession(str(TRUNK), providers=["CPUExecutionProvider"])
    with np.load(DATASETS / "yolo26" / "prepared.npz", allow_pickle=True) as d:
        rows = np.where(np.asarray(d["has_lm"], bool))[0][:200]
        images = np.asarray(d["imgs"])[rows]
    # Predictions are prepared once for functional input; timings below use
    # genuine single-frame ORT calls rather than an amortised evaluation batch.
    initial = trunk_predict(images, trunk)
    runtime = PatchSensorModule(confidence_threshold=json.loads(
        (OUT / "meta.json").read_text())["confidence_threshold"])
    for _ in range(10):
        trunk_predict(images[:1], trunk); runtime.refine(images[0], initial[0])
    trunk_ms, patch_ms = [], []
    for image, landmarks in zip(images, initial):
        t0 = time.perf_counter(); trunk_predict(image[None], trunk)
        trunk_ms.append((time.perf_counter() - t0) * 1000)
        _, _, elapsed = runtime.refine(image, landmarks)
        patch_ms.append(elapsed)
    trunk_ms, patch_ms = np.asarray(trunk_ms), np.asarray(patch_ms)
    total = trunk_ms + patch_ms
    report = {
        "contract": "openpave.pixel-sensor-benchmark.v1",
        "machine": platform.platform(), "processor": platform.processor(),
        "frames": len(images), "execution": "CPU fp32, batch=1 frame / batch=21 patches",
        "trunk_sha256": digest(TRUNK), "patch_sha256": digest(OUT / "patch_refiner.onnx"),
        "trunk_ms": {"median": float(np.median(trunk_ms)), "p95": float(np.percentile(trunk_ms, 95))},
        "patch_ms": {"median": float(np.median(patch_ms)), "p95": float(np.percentile(patch_ms, 95))},
        "combined_ms": {"median": float(np.median(total)), "p95": float(np.percentile(total, 95))},
        "note": "Does not include ROI detector, Monty matching, GUI, or camera capture.",
    }
    (OUT / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
