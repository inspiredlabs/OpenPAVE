"""CPU runtime pieces for the Stage-1, 2D-only pixel sensor."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np

from .train import PATCH, RADIUS, OUT, crop_patch


@dataclass(frozen=True)
class Stage1Sensation:
    joint_id: int
    uv: np.ndarray
    confidence: float
    use_state: bool
    z_source: str = "prior"


class CoordinateAdapter:
    """Attach an explicitly non-observed depth prior to confirmed 2D points."""

    def __init__(self, z_prior: np.ndarray):
        prior = np.asarray(z_prior, np.float32)
        if prior.shape != (21,):
            raise ValueError("z_prior must have shape (21,)")
        self.z_prior = prior

    def compose(self, confirmed_uv: np.ndarray) -> tuple[np.ndarray, dict]:
        xy = np.asarray(confirmed_uv, np.float32)
        if xy.shape != (21, 2):
            raise ValueError("confirmed_uv must have shape (21, 2)")
        centred = xy - xy[:1]
        scale = max(float(np.abs(centred).max()), 1e-6)
        locations = np.column_stack((centred / scale, self.z_prior)).astype(np.float32)
        return locations, {"xy_source": "pixel_patch", "z_source": "prior",
                           "metric_3d": False}


class PatchSensorModule:
    """ONNX patch confirmation/refinement with no MediaPipe dependency."""

    def __init__(self, model_path: Path | str = OUT / "patch_refiner.onnx",
                 confidence_threshold: float = 0.25,
                 collapse_fraction: float = 0.45):
        import onnxruntime as ort
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.confidence_threshold = float(confidence_threshold)
        self.collapse_fraction = float(collapse_fraction)
        self.frame_rgb: np.ndarray | None = None
        self._accepted: list[bool] = []

    def reset(self, frame_bgr: np.ndarray, roi=None) -> None:
        import cv2
        frame = np.asarray(frame_bgr)
        if roi is not None:
            x1, y1, x2, y2 = map(float, roi)
            h, w = frame.shape[:2]
            if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
                x1, x2 = x1 * w, x2 * w; y1, y2 = y1 * h, y2 * h
            ix1, iy1 = max(0, int(x1)), max(0, int(y1))
            ix2, iy2 = min(w, int(x2)), min(h, int(y2))
            if ix2 - ix1 < 4 or iy2 - iy1 < 4:
                raise ValueError("ROI is empty")
            frame = frame[iy1:iy2, ix1:ix2]
        self.frame_rgb = cv2.cvtColor(cv2.resize(frame, (128, 128)), cv2.COLOR_BGR2RGB)
        self._accepted.clear()

    def _run(self, patches: np.ndarray, joint_ids: np.ndarray):
        x = patches.astype(np.float32) / 255.0 - 0.5
        x = np.transpose(x, (0, 3, 1, 2))
        return self.session.run(["delta", "match_probability"],
                                {"patches": x, "joint_id": joint_ids.astype(np.int64)})

    def sense(self, predicted_uv: np.ndarray, joint_id: int) -> Stage1Sensation:
        if self.frame_rgb is None:
            raise RuntimeError("reset must be called before sense")
        uv = np.asarray(predicted_uv, np.float32)
        patch = crop_patch(self.frame_rgb, uv * 128.0)[None]
        delta, confidence = self._run(patch, np.asarray([joint_id]))
        score = float(confidence[0]); accepted = score >= self.confidence_threshold
        self._accepted.append(accepted)
        confirmed = uv + delta[0] * (RADIUS / 128.0) if accepted else uv
        return Stage1Sensation(joint_id, confirmed.astype(np.float32), score, accepted)

    def refine(self, frame_rgb: np.ndarray, initial_uv: np.ndarray,
               iterations: int = 1) -> tuple[np.ndarray, np.ndarray, float]:
        """Batch 21 joints per recurrent pass; returns uv, confidence, ms."""
        image = np.asarray(frame_rgb)
        current = np.asarray(initial_uv, np.float32).copy()
        confidence = np.zeros(21, np.float32)
        started = time.perf_counter()
        for _ in range(iterations):
            patches = np.asarray([crop_patch(image, uv * image.shape[0]) for uv in current])
            delta, confidence = self._run(patches, np.arange(21, dtype=np.int64))
            use = confidence >= self.confidence_threshold
            current[use] += delta[use] * (RADIUS / image.shape[0])
        return current, confidence, (time.perf_counter() - started) * 1000.0

    def fallback_needed(self) -> bool:
        return bool(self._accepted and np.mean(self._accepted) < self.collapse_fraction)
