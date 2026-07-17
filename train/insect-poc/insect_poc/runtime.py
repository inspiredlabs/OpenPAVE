"""Dependency-light CPU runtime for exported one-vs-rest RBF specialists."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class PortableRbfSpecialist:
    """An immutable specialist stored as numeric arrays plus JSON metadata."""

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self.meta = json.loads((self.model_dir / "meta.json").read_text())
        data = np.load(self.model_dir / "model.npz", allow_pickle=False)
        self.mean = data["mean"].astype(np.float32)
        self.scale = data["scale"].astype(np.float32)
        self.support = data["support"].astype(np.float32)
        self.coef = data["coef"].astype(np.float32)
        self.intercept = data["intercept"].astype(np.float32)
        self.gamma = data["gamma"].astype(np.float32)
        self.offsets = data["offsets"].astype(np.int32)
        self.classes = list(self.meta["classes"])
        self.accept_margin = float(self.meta.get("accept_margin", 0.0))

    def decision_function(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]
        if X.shape[1] != self.mean.size:
            raise ValueError(f"expected {self.mean.size} features, received {X.shape[1]}")
        X = (X - self.mean) / self.scale
        scores = np.empty((len(X), len(self.classes)), dtype=np.float32)
        for index in range(len(self.classes)):
            lo, hi = self.offsets[index:index + 2]
            sv = self.support[lo:hi]
            # Direct deltas are intentionally used here. Embedded inference is
            # batch=1, so this stays small and avoids cancellation/overflow in
            # the expanded ||x||^2 + ||s||^2 - 2*x.s identity at low precision.
            delta = X[:, None, :] - sv[None, :, :]
            dist2 = np.einsum("nsd,nsd->ns", delta, delta, optimize=True)
            kernel = np.exp(-self.gamma[index] * dist2)
            scores[:, index] = kernel @ self.coef[lo:hi] + self.intercept[index]
        return scores

    def predict(self, X) -> list[str]:
        scores = self.decision_function(X)
        order = np.argsort(scores, axis=1)
        gap = scores[np.arange(len(scores)), order[:, -1]] - scores[np.arange(len(scores)), order[:, -2]]
        return [self.classes[i] if g >= self.accept_margin else "unknown"
                for i, g in zip(order[:, -1], gap)]


class InsectEnsemble:
    """Loads the assembled manifest; missing specialists do not break the ensemble."""

    def __init__(self, manifest: str | Path):
        manifest = Path(manifest)
        spec = json.loads(manifest.read_text())
        root = manifest.parent
        self.models = {
            name: PortableRbfSpecialist(root / item["path"])
            for name, item in spec.get("specialists", {}).items()
        }

    def infer(self, features: dict[str, np.ndarray]) -> dict[str, str]:
        return {name: model.predict(features[name])[0]
                for name, model in self.models.items() if name in features}

    @staticmethod
    def compose_observation(result: dict[str, str]) -> str:
        """Turn bounded specialist outputs into a VLM-like deterministic sentence."""
        if result.get("presence") in {None, "absent", "unknown"}:
            return "No person detected."
        parts = ["Person detected"]
        motion = result.get("motion")
        if motion and motion != "unknown":
            parts.append("standing still" if motion == "still" else f"moving {motion}")
        color = result.get("color")
        if color and color != "unknown":
            parts.append(f"wearing {color}")
        gesture = result.get("gesture")
        if gesture and gesture != "unknown":
            parts.append(f"showing {gesture.replace('_', ' ')}")
        return ", ".join(parts) + "."
