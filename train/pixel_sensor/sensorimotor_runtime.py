"""Inspectable, command-suppressed sensorimotor hand-landmark runtime.

This is deliberately a diagnostic pipeline, not a promoted controller.  It
uses our v3 trunk only to cold-start a 2D hand pose, then performs the Monty
loop explicitly: predict a graph location, sense a local pixel patch, accept
or reject it, and update the object-to-image reference frame.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np

from .projector import HandFrameProjector
from .runtime import CoordinateAdapter, PatchSensorModule


ROOT = Path(__file__).resolve().parents[2]
TRUNK = ROOT / "train" / "runs" / "tiny_gesture" / "trunk.onnx"
PATCH_MODEL = ROOT / "train" / "runs" / "pixel_sensor" / "patch_refiner.onnx"
PATCH_META = PATCH_MODEL.with_name("meta.json")
GEOMETRY = ROOT / "train" / "runs" / "sensorimotor_hand" / "objects.npz"
GESTURES = ROOT / "train" / "runs" / "monty_gestures" / "objects.npz"


@dataclass(frozen=True)
class SensorimotorResult:
    presence: float
    coarse: np.ndarray
    predicted: np.ndarray
    points: np.ndarray
    accepted: np.ndarray
    rejected: np.ndarray
    anchors: np.ndarray
    confidence: np.ndarray
    prototype: int
    reprojection_rms: float
    hypothesis: str
    evidence: float
    timing: dict

    def overlay(self) -> dict:
        """JSON-like shape consumed directly by the OpenCV GUI overlay."""
        return {
            "kind": "sensorimotor_debug",
            "coarse": self.coarse,
            "predicted": self.predicted,
            "points": self.points,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "anchors": self.anchors,
            "confidence": self.confidence,
            "prototype": self.prototype,
            "reprojection_rms": self.reprojection_rms,
            "hypothesis": self.hypothesis,
            "evidence": self.evidence,
            "total_ms": self.timing.get("total_ms", 0.0),
            "commands_enabled": False,
        }


class SensorimotorHandRuntime:
    """Our pixels -> 2D constellation -> Monty hypothesis diagnostic.

    No MediaPipe or BlazePalm code/model is loaded.  The gesture hypothesis is
    display evidence only; callers must not promote it to a robot intent.
    """

    def __init__(self, presence_gate: float = 0.50) -> None:
        import onnxruntime as ort

        for path in (TRUNK, PATCH_MODEL, GEOMETRY, GESTURES):
            if not path.exists():
                raise FileNotFoundError(path)
        self.presence_gate = float(presence_gate)
        self.trunk = ort.InferenceSession(str(TRUNK), providers=["CPUExecutionProvider"])
        try:
            threshold = float(json.loads(PATCH_META.read_text())["confidence_threshold"])
        except (FileNotFoundError, KeyError, ValueError):
            threshold = 0.25
        self.sensor = PatchSensorModule(PATCH_MODEL, confidence_threshold=threshold)

        geometry = np.load(GEOMETRY, allow_pickle=True)
        self.prototypes = np.asarray(geometry["hand"], np.float32)
        self.anchors = np.asarray(geometry["palm_anchors"], np.int64)
        self.scan_order = np.asarray(geometry["scan_order"], np.int64)

        # The z channel remains an explicit teacher-derived prior.  It never
        # participates in pixel confirmation or the 2D projector fit.
        stored = np.load(GESTURES, allow_pickle=True)
        examples = np.concatenate([stored[k] for k in stored.files]).astype(np.float32)
        centred = examples - examples[:, :1]
        scale = np.maximum(np.abs(centred).max(axis=(1, 2), keepdims=True), 1e-6)
        self.adapter = CoordinateAdapter(np.median(centred / scale, axis=0)[:, 2])
        from train.monty_lab import EvidenceLM
        self.gesture_lm = EvidenceLM.load(GESTURES)

    @staticmethod
    def _sigmoid(value: float) -> float:
        return float(1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0))))

    def _cold_start(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float, float]:
        import cv2

        rgb = cv2.cvtColor(cv2.resize(frame_bgr, (128, 128)), cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) / 255.0 - 0.5).transpose(2, 0, 1)[None]
        started = time.perf_counter()
        lm, logit = self.trunk.run(["landmarks", "presence"], {"frames": x})
        elapsed = (time.perf_counter() - started) * 1000.0
        return lm[0].reshape(21, 2).astype(np.float32), self._sigmoid(float(logit[0, 0])), elapsed

    def _best_projector(self, confirmed: dict[int, np.ndarray]) -> tuple[int, HandFrameProjector, float]:
        ids = np.asarray(sorted(confirmed), np.int64)
        observed = np.stack([confirmed[int(i)] for i in ids])
        best: tuple[float, int, HandFrameProjector] | None = None
        for index, prototype in enumerate(self.prototypes):
            try:
                projector = HandFrameProjector.fit(prototype[ids, :2], observed)
            except ValueError:
                continue
            residual = projector.to_uv(prototype[ids, :2]) - observed
            rms = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
            if best is None or rms < best[0]:
                best = (rms, index, projector)
        if best is None:
            raise ValueError("no non-degenerate hand prototype fits the confirmed anchors")
        return best[1], best[2], best[0]

    def step(self, frame_bgr: np.ndarray) -> SensorimotorResult | None:
        from train.monty_lab.protocol import Episode, Observation

        total_started = time.perf_counter()
        coarse, presence, trunk_ms = self._cold_start(frame_bgr)
        if presence < self.presence_gate:
            return None

        self.sensor.reset(frame_bgr)
        accepted = np.zeros(21, bool)
        rejected = np.zeros(21, bool)
        confidence = np.zeros(21, np.float32)
        confirmed: dict[int, np.ndarray] = {}

        sense_started = time.perf_counter()
        # Anchor observations establish the first object-to-camera pose.
        for joint in self.anchors:
            sensation = self.sensor.sense(coarse[joint], int(joint))
            confidence[joint] = sensation.confidence
            if sensation.use_state:
                accepted[joint] = True
                confirmed[int(joint)] = sensation.uv
            else:
                rejected[joint] = True

        if len(confirmed) < 2:
            # Keep failed cold starts visible, but do not manufacture a Monty
            # pose from rejected pixel evidence.
            timing = {"trunk_ms": trunk_ms,
                      "sense_ms": (time.perf_counter() - sense_started) * 1000.0,
                      "monty_us": 0.0,
                      "total_ms": (time.perf_counter() - total_started) * 1000.0}
            return SensorimotorResult(presence, coarse, coarse.copy(), coarse.copy(),
                                     accepted, rejected, self.anchors.copy(), confidence,
                                     -1, float("inf"), "cold-start-failed", 0.0, timing)

        prototype_index, projector, rms = self._best_projector(confirmed)
        prototype = self.prototypes[prototype_index]
        predicted = projector.to_uv(prototype[:, :2])

        # Sensorimotor exploration: each accepted local sensation changes the
        # reference-frame transform used to predict the next joint.
        for joint in self.scan_order:
            joint = int(joint)
            proposal = projector.to_uv(prototype[joint:joint + 1, :2])[0]
            predicted[joint] = proposal
            sensation = self.sensor.sense(proposal, joint)
            confidence[joint] = sensation.confidence
            if sensation.use_state:
                accepted[joint] = True
                confirmed[joint] = sensation.uv
                try:
                    ids = np.asarray(sorted(confirmed), np.int64)
                    projector = HandFrameProjector.fit(
                        prototype[ids, :2], np.stack([confirmed[int(i)] for i in ids]))
                except ValueError:
                    pass
            else:
                rejected[joint] = True

        points = projector.to_uv(prototype[:, :2])
        for joint, uv in confirmed.items():
            points[joint] = uv
        sense_ms = (time.perf_counter() - sense_started) * 1000.0

        locations, _provenance = self.adapter.compose(points)
        episode = Episode([Observation(location=p) for p in locations])
        monty_started = time.perf_counter()
        hypothesis, evidence, _pose = self.gesture_lm.infer(episode)
        monty_us = (time.perf_counter() - monty_started) * 1e6
        total_ms = (time.perf_counter() - total_started) * 1000.0
        return SensorimotorResult(
            presence, coarse, predicted, points, accepted, rejected,
            self.anchors.copy(), confidence, prototype_index, rms,
            hypothesis, evidence,
            {"trunk_ms": trunk_ms, "sense_ms": sense_ms,
             "monty_us": monty_us, "total_ms": total_ms},
        )
