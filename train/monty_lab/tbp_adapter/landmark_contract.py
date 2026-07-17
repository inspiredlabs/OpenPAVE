"""Shared hand-landmark reference-frame contract for OpenPAVE and tbp.monty.

This module deliberately has no dependency on either runtime.  The arm64
OpenPAVE environment uses it while preparing paired teacher/student episodes;
the osx-64 tbp.monty environment uses the same code inside its SensorModule.
"""
from __future__ import annotations

import numpy as np

N_LANDMARKS = 21
PALM_IDS = np.asarray([0, 5, 9, 13, 17], dtype=np.int64)
PARENT = np.asarray([
    9, 0, 1, 2, 3,
    0, 5, 6, 7,
    0, 9, 10, 11,
    0, 13, 14, 15,
    0, 17, 18, 19,
], dtype=np.int64)


def _points21(values):
    points = np.asarray(values, dtype=np.float64)
    if points.shape not in ((N_LANDMARKS, 2), (N_LANDMARKS, 3)):
        raise ValueError("landmarks must have shape (21,2) or (21,3)")
    if points.shape[1] == 2:
        points = np.column_stack((points, np.zeros(N_LANDMARKS)))
    return points


MCP_IDS = np.asarray([5, 9, 13, 17], dtype=np.int64)


def canonicalize(values, weights=None):
    """Map one constellation into its own wrist/palm reference frame.

    The transform uses only the stream being transformed.  Student evaluation
    therefore cannot leak the MediaPipe teacher coordinates.  Raw coordinates
    remain in the episode for pixel-error measurement.

    The longitudinal axis runs from the wrist to the (optionally
    confidence-weighted) centroid of the valid MCP anchors [5, 9, 13, 17]
    rather than trusting the single 0->9 pair: one mispredicted anchor then
    shifts the frame instead of rotating and rescaling every joint.
    """
    points = _points21(values)
    valid = np.isfinite(points).all(axis=1)
    if not valid[0]:
        raise ValueError("the wrist is required to define the hand frame")

    if weights is None:
        anchor_weights = valid.astype(np.float64)
    else:
        anchor_weights = np.clip(np.asarray(weights, dtype=np.float64), 0.0, None)
        if anchor_weights.shape != (N_LANDMARKS,):
            raise ValueError("weights must have shape (21,)")
        anchor_weights = anchor_weights * valid

    origin = points[0].copy()
    mcp_weights = anchor_weights[MCP_IDS]
    if float(mcp_weights.sum()) > 1e-8:
        centroid = (points[MCP_IDS, :2] * mcp_weights[:, None]).sum(axis=0) / mcp_weights.sum()
    elif valid[9]:
        centroid = points[9, :2]
    else:
        raise ValueError("no usable MCP anchor to define the hand frame")
    palm = centroid - origin[:2]
    palm_norm = float(np.linalg.norm(palm))
    if palm_norm < 1e-8:
        raise ValueError("degenerate wrist-to-MCP-centroid axis")
    y_axis = palm / palm_norm
    x_axis = np.asarray([y_axis[1], -y_axis[0]], dtype=np.float64)
    basis = np.stack((x_axis, y_axis), axis=1)

    palm_valid = PALM_IDS[anchor_weights[PALM_IDS] > 1e-8]
    distances = np.linalg.norm(points[palm_valid, :2] - origin[:2], axis=1)
    distances = distances[distances > 1e-8]
    if not len(distances):
        raise ValueError("no non-degenerate palm scale")
    scale = float(np.median(distances))

    local_xy = (points[:, :2] - origin[:2]).dot(basis) / scale
    local_z = (points[:, 2] - origin[2]) / scale
    local = np.column_stack((local_xy, local_z))
    local[~valid] = np.nan
    transform = {
        "origin": origin,
        "basis": basis,
        "scale": scale,
        "image_y_down": True,
    }
    return local, transform


def local_pose_vectors(canonical_points, joint_id):
    """Return a fully defined 2D joint frame embedded in 3D."""
    points = _points21(canonical_points)
    joint = int(joint_id)
    parent = int(PARENT[joint])
    tangent = points[joint] - points[parent]
    tangent[2] = 0.0
    norm = float(np.linalg.norm(tangent))
    if norm < 1e-8:
        tangent = np.asarray([0.0, 1.0, 0.0])
    else:
        tangent /= norm
    normal = np.asarray([0.0, 0.0, 1.0])
    across = np.cross(normal, tangent)
    return np.stack((normal, tangent, across))


def joint_feature(joint_id):
    """Categorical landmark identity; unlike i/20 this is not ordinal."""
    feature = np.zeros(N_LANDMARKS, dtype=np.float64)
    feature[int(joint_id)] = 1.0
    return feature


def paired_errors(teacher_uv, student_uv, image_size=384.0):
    teacher = np.asarray(teacher_uv, dtype=np.float64)
    student = np.asarray(student_uv, dtype=np.float64)
    if teacher.shape != student.shape or teacher.shape[-2:] != (N_LANDMARKS, 2):
        raise ValueError("paired landmarks must have matching (...,21,2) shapes")
    valid = np.isfinite(teacher).all(axis=-1) & np.isfinite(student).all(axis=-1)
    error = np.full(valid.shape, np.nan, dtype=np.float64)
    error[valid] = np.linalg.norm(student[valid] - teacher[valid], axis=-1) * image_size
    return error
