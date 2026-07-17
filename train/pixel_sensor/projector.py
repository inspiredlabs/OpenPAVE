"""Similarity projection between Monty's 2D hand frame and crop coordinates."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HandFrameProjector:
    """A fitted 2D Umeyama similarity transform.

    Stage 1 deliberately operates on x/y only.  Relative z is attached later
    by the coordinate adapter and is never used to pretend that a 2D camera
    observation is metric 3D evidence.
    """

    rotation: np.ndarray
    scale: float
    translation: np.ndarray

    @classmethod
    def fit(cls, hand_xy: np.ndarray, crop_uv: np.ndarray) -> "HandFrameProjector":
        src = np.asarray(hand_xy, dtype=np.float64)
        dst = np.asarray(crop_uv, dtype=np.float64)
        if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
            raise ValueError("hand_xy and crop_uv must both have shape (N, 2)")
        if len(src) < 2:
            raise ValueError("at least two confirmed joints are required")
        src_mean, dst_mean = src.mean(0), dst.mean(0)
        src0, dst0 = src - src_mean, dst - dst_mean
        variance = float((src0 * src0).sum() / len(src))
        if variance <= 1e-12:
            raise ValueError("confirmed hand-frame joints are degenerate")
        covariance = dst0.T @ src0 / len(src)
        u, singular, vt = np.linalg.svd(covariance)
        correction = np.eye(2)
        if np.linalg.det(u @ vt) < 0:
            correction[-1, -1] = -1.0
        rotation = u @ correction @ vt
        scale = float((singular * np.diag(correction)).sum() / variance)
        if not np.isfinite(scale) or scale <= 1e-12:
            raise ValueError("similarity scale is invalid")
        translation = dst_mean - scale * (rotation @ src_mean)
        return cls(rotation.astype(np.float32), scale,
                   translation.astype(np.float32))

    def to_uv(self, hand_xy: np.ndarray) -> np.ndarray:
        points = np.asarray(hand_xy, dtype=np.float32)
        return (self.scale * (points @ self.rotation.T) + self.translation).astype(np.float32)

    def to_hand(self, crop_uv: np.ndarray) -> np.ndarray:
        points = np.asarray(crop_uv, dtype=np.float32)
        return (((points - self.translation) @ self.rotation) / self.scale).astype(np.float32)


def _self_test() -> None:
    rng = np.random.default_rng(17)
    for count in (2, 5, 21):
        for _ in range(50):
            hand = rng.normal(size=(count, 2)).astype(np.float32)
            angle = rng.uniform(-np.pi, np.pi)
            rotation = np.array([[np.cos(angle), -np.sin(angle)],
                                 [np.sin(angle), np.cos(angle)]], np.float32)
            scale = rng.uniform(0.05, 0.8)
            translation = rng.uniform(-0.3, 1.3, size=2).astype(np.float32)
            uv = scale * (hand @ rotation.T) + translation
            projector = HandFrameProjector.fit(hand, uv)
            projected = projector.to_uv(hand)
            restored = projector.to_hand(projected)
            if float(np.max(np.abs(projected - uv))) > 2e-6:
                raise AssertionError("forward Umeyama projection exceeded tolerance")
            if float(np.max(np.abs(restored - hand))) > 2e-5:
                raise AssertionError("inverse Umeyama projection exceeded tolerance")
    print("[projector] 150 similarity/round-trip cases passed")


if __name__ == "__main__":
    _self_test()
