"""Geometry intent head for LingBot-Map (responsibility C in §3.2) — STUB.

LingBot returns a point cloud, not a semantic embedding, so its head is a
different modality from the EmbeddingProbe: occupancy / PointNet-style rather than
a linear probe over a CLS vector. Not wired yet — this stub fixes the interface so
the shim and trainer can dispatch to it once the LingBot backend is real.
"""

from __future__ import annotations

import numpy as np


class GeometryHead:
    def __init__(self, feature_dim: int = 0):
        self.feature_dim = int(feature_dim)

    def logits(self, features: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "LingBot geometry head is not wired yet (see docs/dgx-spark-mlx-port.md §3.2). "
            "Planned: occupancy/PointNet over PointCloudResult -> obstacle-aware STOP."
        )
