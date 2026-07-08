"""Shared embedding probe for DINOv3 AND V-JEPA 2.1 (responsibility C in §3.2).

Both produce dense embeddings, so the head is one parameterized softmax-regression
probe (`standardize -> linear -> softmax`); only `feature_dim` differs. This is the
standard frozen-feature linear-probe recipe.

Kept in NumPy: a probe is tiny, and the heavy MLX compute is the *featurizer*
(the real DINOv3 engine already runs on Metal). Swap in an MlxProbe later the same
way the template swaps NumpyPolicy -> MlxPolicy if probe training ever gets large.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pave_mlx.heads.base import INTENT_LABELS, softmax


class EmbeddingProbe:
    def __init__(self, feature_dim: int, n_labels: int = len(INTENT_LABELS)):
        self.feature_dim = int(feature_dim)
        self.n_labels = int(n_labels)
        rng = np.random.default_rng(0)
        self.W = (rng.standard_normal((self.feature_dim, self.n_labels)) * 0.01).astype(np.float32)
        self.b = np.zeros((self.n_labels,), dtype=np.float32)
        # Feature standardization stats, fit during training.
        self.mu = np.zeros((self.feature_dim,), dtype=np.float32)
        self.sigma = np.ones((self.feature_dim,), dtype=np.float32)

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        return (np.atleast_2d(x).astype(np.float32) - self.mu) / self.sigma

    def logits(self, features: np.ndarray) -> np.ndarray:
        return self._standardize(features) @ self.W + self.b

    def predict(self, features: np.ndarray) -> tuple[int, float]:
        p = softmax(self.logits(features))[0]
        idx = int(np.argmax(p))
        return idx, float(p[idx])

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        steps: int = 600,
        lr: float = 0.2,
        l2: float = 1e-4,
    ) -> float:
        """Multinomial logistic regression via full-batch gradient descent."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        self.mu = X.mean(axis=0).astype(np.float32)
        self.sigma = (X.std(axis=0) + 1e-6).astype(np.float32)
        Xs = (X - self.mu) / self.sigma
        n = len(Xs)
        Y = np.zeros((n, self.n_labels), dtype=np.float32)
        Y[np.arange(n), y] = 1.0

        loss = 0.0
        for _ in range(max(1, int(steps))):
            p = softmax(Xs @ self.W + self.b)
            loss = float(-np.mean(np.sum(Y * np.log(p + 1e-9), axis=1)))
            grad = (p - Y) / n
            gW = Xs.T @ grad + l2 * self.W
            gb = grad.sum(axis=0)
            self.W -= lr * gW
            self.b -= lr * gb
        return loss

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, W=self.W, b=self.b, mu=self.mu, sigma=self.sigma)

    @classmethod
    def load(cls, path: str | Path) -> "EmbeddingProbe":
        data = np.load(Path(path))
        probe = cls(feature_dim=data["W"].shape[0], n_labels=data["W"].shape[1])
        probe.W = data["W"].astype(np.float32)
        probe.b = data["b"].astype(np.float32)
        probe.mu = data["mu"].astype(np.float32)
        probe.sigma = data["sigma"].astype(np.float32)
        return probe
