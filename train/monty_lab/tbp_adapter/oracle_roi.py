"""Oracle-ROI geometry for the acquisition-free landmark benchmark.

Implements the coordinate contract from docs/training-with-monty.md §3: the
oriented ROI is determined only from teacher normalized landmarks, its +v axis
points from the MCPs toward the wrist (fingers face the top of the crop), and
the square covering the maximum oriented extent is expanded by 1.25.  Both
directions of the normalized-image <-> unit-ROI mapping are provided.

Numpy-only except for crop(), which imports cv2 lazily so the module stays
importable in either environment.
"""
from __future__ import annotations

import numpy as np

N_LANDMARKS = 21
MCP_IDS = np.asarray([5, 9, 13, 17], dtype=np.int64)
EXPAND = 1.25


def oracle_roi(landmarks):
    """Oriented square ROI from one normalized (21, 2) constellation.

    Returns a dict with ``center``, ``x_axis``, ``y_axis`` and ``size`` in
    normalized source coordinates.  Raises ValueError when the wrist or the
    MCP row is too degenerate to orient a crop.
    """
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (N_LANDMARKS, 2):
        raise ValueError("landmarks must have shape (21, 2)")
    valid = np.isfinite(points).all(axis=1)
    mcp_valid = MCP_IDS[valid[MCP_IDS]]
    if not valid[0] or len(mcp_valid) < 2:
        raise ValueError("wrist and at least two MCPs are required for the ROI")
    centroid = points[mcp_valid].mean(axis=0)
    longitudinal = points[0] - centroid  # +v runs from MCPs toward the wrist
    norm = float(np.linalg.norm(longitudinal))
    if norm < 1e-6:
        raise ValueError("degenerate MCP-to-wrist axis")
    y_axis = longitudinal / norm
    x_axis = np.asarray([y_axis[1], -y_axis[0]], dtype=np.float64)

    offsets = points[valid]
    u = offsets.dot(x_axis)
    v = offsets.dot(y_axis)
    center = (0.5 * (u.min() + u.max()) * x_axis + 0.5 * (v.min() + v.max()) * y_axis)
    size = float(max(u.max() - u.min(), v.max() - v.min())) * EXPAND
    if size < 1e-6:
        raise ValueError("degenerate ROI extent")
    return {"center": center, "x_axis": x_axis, "y_axis": y_axis, "size": size}


def perturb_roi(roi, rng, max_rotation=0.30, scale_range=(0.7, 1.45),
                max_translation=0.12):
    """Controlled acquisition error (curriculum step 3): rotate, scale, shift."""
    angle = rng.uniform(-max_rotation, max_rotation)
    cos, sin = np.cos(angle), np.sin(angle)
    rotation = np.asarray([[cos, -sin], [sin, cos]])
    x_axis = rotation.dot(roi["x_axis"])
    y_axis = rotation.dot(roi["y_axis"])
    size = roi["size"] * rng.uniform(*scale_range)
    shift = rng.uniform(-max_translation, max_translation, size=2) * roi["size"]
    center = roi["center"] + shift[0] * x_axis + shift[1] * y_axis
    return {"center": center, "x_axis": x_axis, "y_axis": y_axis, "size": size}


def project_to_roi(points, roi):
    """Normalized source coordinates -> unit-ROI (u, v) coordinates."""
    offsets = np.asarray(points, dtype=np.float64) - roi["center"]
    u = offsets.dot(roi["x_axis"]) / roi["size"] + 0.5
    v = offsets.dot(roi["y_axis"]) / roi["size"] + 0.5
    return np.stack((u, v), axis=-1)


def project_to_source(uv, roi):
    """Unit-ROI coordinates -> normalized source coordinates."""
    uv = np.asarray(uv, dtype=np.float64)
    return (roi["center"]
            + (uv[..., :1] - 0.5) * roi["size"] * roi["x_axis"]
            + (uv[..., 1:2] - 0.5) * roi["size"] * roi["y_axis"])


def crop(image, roi, crop_px):
    """Extract the oriented ROI as a (crop_px, crop_px) image via one affine."""
    import cv2

    source_px = float(image.shape[0])
    scale = crop_px / (roi["size"] * source_px)
    linear = np.stack((roi["x_axis"], roi["y_axis"])) * scale
    offset = crop_px * 0.5 - linear.dot(roi["center"] * source_px)
    matrix = np.column_stack((linear, offset))
    return cv2.warpAffine(image, matrix, (crop_px, crop_px),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
