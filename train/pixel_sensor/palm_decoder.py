"""Exact CPU-side decoder for MediaPipe's bundled 192px BlazePalm outputs.

This module does not execute TFLite.  It converts the model's raw
``[1, 2016, 18]`` regressors and ``[1, 2016, 1]`` logits into boxes, seven
palm keypoints, and the rotation-aligned hand ROI used for cold start.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import zipfile

import numpy as np

INPUT_SIZE = 192
NUM_BOXES = 2016
NUM_COORDS = 18
NUM_KEYPOINTS = 7
STRIDES = (8, 16, 16, 16)
MIN_SCORE = 0.5
NMS_IOU = 0.3


@dataclass(frozen=True)
class PalmDetection:
    box: np.ndarray                 # [xmin, ymin, xmax, ymax], normalised
    keypoints: np.ndarray           # [7, 2], x/y normalised
    score: float
    anchor_index: int = -1


@dataclass(frozen=True)
class HandRoi:
    center: np.ndarray              # x/y normalised in the source image
    size: np.ndarray                # width/height normalised in source image
    rotation: float                 # radians; MediaPipe image convention


def generate_anchors() -> np.ndarray:
    """Reproduce SsdAnchorsCalculator's exact ordering.

    The graph omits ``interpolated_scale_aspect_ratio``, whose proto default is
    1.0.  Thus stride 8 has two anchors at every 24x24 cell and the three
    merged stride-16 layers have six anchors at every 12x12 cell.  Fixed anchor
    size makes every anchor width/height one; learned regressors supply size.
    """
    anchors = []
    layer = 0
    while layer < len(STRIDES):
        last = layer
        anchors_per_cell = 0
        while last < len(STRIDES) and STRIDES[last] == STRIDES[layer]:
            anchors_per_cell += 2  # aspect_ratio=1 plus interpolated ratio=1
            last += 1
        stride = STRIDES[layer]
        feature = math.ceil(INPUT_SIZE / stride)
        for y in range(feature):
            for x in range(feature):
                for _ in range(anchors_per_cell):
                    anchors.append(((x + 0.5) / feature, (y + 0.5) / feature,
                                    1.0, 1.0))
        layer = last
    result = np.asarray(anchors, np.float32)
    if result.shape != (NUM_BOXES, 4):
        raise AssertionError(f"anchor contract changed: {result.shape}")
    return result


ANCHORS = generate_anchors()


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(logits, np.float64), -100.0, 100.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)


def decode(raw_regressors: np.ndarray, raw_scores: np.ndarray,
           min_score: float = MIN_SCORE) -> list[PalmDetection]:
    """Decode raw model tensors before NMS and letterbox removal."""
    raw = np.asarray(raw_regressors, np.float32).reshape(-1, NUM_COORDS)
    logits = np.asarray(raw_scores, np.float32).reshape(-1)
    if raw.shape != (NUM_BOXES, NUM_COORDS) or logits.shape != (NUM_BOXES,):
        raise ValueError("expected regressors [1,2016,18] and scores [1,2016,1]")
    score = _sigmoid(logits)
    keep = np.where(score >= min_score)[0]
    detections = []
    for i in keep:
        anchor_x, anchor_y, anchor_w, anchor_h = ANCHORS[i]
        # reverse_output_order=true selects XYWH in MediaPipe's calculator.
        cx = raw[i, 0] / INPUT_SIZE * anchor_w + anchor_x
        cy = raw[i, 1] / INPUT_SIZE * anchor_h + anchor_y
        width = raw[i, 2] / INPUT_SIZE * anchor_w
        height = raw[i, 3] / INPUT_SIZE * anchor_h
        if width < 0 or height < 0 or not np.isfinite((cx, cy, width, height)).all():
            continue
        keypoints = raw[i, 4:].reshape(NUM_KEYPOINTS, 2).copy()
        keypoints[:, 0] = keypoints[:, 0] / INPUT_SIZE * anchor_w + anchor_x
        keypoints[:, 1] = keypoints[:, 1] / INPUT_SIZE * anchor_h + anchor_y
        box = np.asarray([cx - width / 2, cy - height / 2,
                          cx + width / 2, cy + height / 2], np.float32)
        detections.append(PalmDetection(box, keypoints.astype(np.float32),
                                        float(score[i]), int(i)))
    return detections


def iou(a: np.ndarray, b: np.ndarray) -> float:
    left = max(float(a[0]), float(b[0])); top = max(float(a[1]), float(b[1]))
    right = min(float(a[2]), float(b[2])); bottom = min(float(a[3]), float(b[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return intersection / max(area_a + area_b - intersection, 1e-12)


def weighted_nms(detections: list[PalmDetection], threshold: float = NMS_IOU,
                 max_detections: int | None = None) -> list[PalmDetection]:
    """MediaPipe-style score-weighted coordinate blending by IoU cluster."""
    remaining = sorted(detections, key=lambda d: d.score, reverse=True)
    output = []
    while remaining and (max_detections is None or len(output) < max_detections):
        leader = remaining[0]
        cluster = [d for d in remaining if iou(leader.box, d.box) > threshold]
        cluster_ids = {id(d) for d in cluster}
        remaining = [d for d in remaining if id(d) not in cluster_ids]
        weights = np.asarray([d.score for d in cluster], np.float32)
        total = float(weights.sum())
        box = np.average(np.stack([d.box for d in cluster]), axis=0, weights=weights)
        keypoints = np.average(np.stack([d.keypoints for d in cluster]), axis=0,
                               weights=weights)
        output.append(PalmDetection(box.astype(np.float32), keypoints.astype(np.float32),
                                    leader.score, leader.anchor_index))
    return output


def remove_letterbox(detection: PalmDetection, image_width: int,
                     image_height: int) -> PalmDetection:
    """Map a square 192px FIT tensor detection back to the source image."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if image_width >= image_height:
        content = image_height / image_width
        left, top, usable_x, usable_y = 0.0, (1.0 - content) / 2, 1.0, content
    else:
        content = image_width / image_height
        left, top, usable_x, usable_y = (1.0 - content) / 2, 0.0, content, 1.0
    box = detection.box.copy()
    box[[0, 2]] = (box[[0, 2]] - left) / usable_x
    box[[1, 3]] = (box[[1, 3]] - top) / usable_y
    keypoints = detection.keypoints.copy()
    keypoints[:, 0] = (keypoints[:, 0] - left) / usable_x
    keypoints[:, 1] = (keypoints[:, 1] - top) / usable_y
    return PalmDetection(box, keypoints, detection.score, detection.anchor_index)


def normalise_radians(angle: float) -> float:
    return angle - 2 * math.pi * math.floor((angle + math.pi) / (2 * math.pi))


def detection_to_roi(detection: PalmDetection, image_width: int,
                     image_height: int) -> HandRoi:
    """Apply PalmDetectionDetectionToRoi's 0->2 rotation and 2.6 transform."""
    box = detection.box
    center = np.asarray([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], np.float64)
    width, height = float(box[2] - box[0]), float(box[3] - box[1])
    start, end = detection.keypoints[0], detection.keypoints[2]
    vector_x = (float(end[0]) - float(start[0])) * image_width
    vector_y = (float(end[1]) - float(start[1])) * image_height
    rotation = normalise_radians(math.pi / 2 - math.atan2(-vector_y, vector_x))

    # RectTransformationCalculator: shift_y=-0.5 in the rotated rectangle,
    # square the longer pixel side, then scale both axes by 2.6.
    shift_x_px = 0.0
    shift_y_px = -0.5 * height * image_height
    cos_r, sin_r = math.cos(rotation), math.sin(rotation)
    center[0] += (shift_x_px * cos_r - shift_y_px * sin_r) / image_width
    center[1] += (shift_x_px * sin_r + shift_y_px * cos_r) / image_height
    long_px = max(width * image_width, height * image_height)
    size = np.asarray([long_px * 2.6 / image_width,
                       long_px * 2.6 / image_height], np.float32)
    return HandRoi(center.astype(np.float32), size, float(rotation))


def decode_palms(raw_regressors: np.ndarray, raw_scores: np.ndarray,
                 image_width: int, image_height: int,
                 max_detections: int = 2) -> list[tuple[PalmDetection, HandRoi]]:
    decoded = decode(raw_regressors, raw_scores)
    suppressed = weighted_nms(decoded, max_detections=max_detections)
    source = [remove_letterbox(d, image_width, image_height) for d in suppressed]
    return [(d, detection_to_roi(d, image_width, image_height)) for d in source]


def inspect_task(task_path: Path | str) -> dict:
    """Bind this decoder to the exact TFLite tensor signature on disk."""
    from mediapipe.tasks.python.metadata.metadata_writers import writer_utils
    task = Path(task_path)
    with zipfile.ZipFile(task) as archive:
        model = archive.read("hand_detector.tflite")
    graph = writer_utils.get_subgraph(bytearray(model))

    def tensors(indices) -> list[dict]:
        result = []
        for index in indices:
            tensor = graph.Tensors(index)
            result.append({"name": tensor.Name().decode(),
                           "shape": [int(tensor.Shape(i)) for i in range(tensor.ShapeLength())]})
        return result

    manifest = {
        "task": str(task), "task_sha256": hashlib.sha256(task.read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(model).hexdigest(),
        "inputs": tensors([graph.Inputs(i) for i in range(graph.InputsLength())]),
        "outputs": tensors([graph.Outputs(i) for i in range(graph.OutputsLength())]),
        "decoder": {"anchors": NUM_BOXES, "coords": NUM_COORDS,
                    "keypoints": NUM_KEYPOINTS, "strides": STRIDES,
                    "score_threshold": MIN_SCORE, "nms_iou": NMS_IOU},
    }
    shapes = {tuple(item["shape"]) for item in manifest["outputs"]}
    if shapes != {(1, NUM_BOXES, NUM_COORDS), (1, NUM_BOXES, 1)}:
        raise RuntimeError(f"bundled detector signature changed: {manifest['outputs']}")
    return manifest


if __name__ == "__main__":
    default = Path(__file__).resolve().parents[1] / "weights" / "hand_landmarker.task"
    print(json.dumps(inspect_task(default), indent=2))
