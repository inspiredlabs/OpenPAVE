"""IntentHead protocol + serialised manifest (responsibility B in §3.2).

A trained head is two things: weights (saved by the head itself) and a manifest
(this module). One manifest *schema*, one serialised *instance* per backend.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

# Discrete intent vocabulary. These are exactly the text aliases that
# pave_runtime/intent_schema.py already accepts, so the shim can return them as
# plain text and intent_ingress will normalize them (LEFT/RIGHT -> MOVE+yaw).
INTENT_LABELS = ["STOP", "TROT", "HOME", "LEFT", "RIGHT"]
SAFE_DEFAULT = "STOP"
MANIFEST_SCHEMA = "pave-intent-head/0.1"

PKG_DIR = Path(__file__).resolve().parents[1]  # .../pave_mlx


def softmax(z: np.ndarray) -> np.ndarray:
    z = np.atleast_2d(np.asarray(z, dtype=np.float32))
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


@dataclass
class HeadManifest:
    """Serialised per-backend head description (responsibility B)."""

    backend: str
    model_id: str
    feature_dim: int
    pooling: str
    weights: str                       # path relative to the pave_mlx package
    labels: list = field(default_factory=lambda: list(INTENT_LABELS))
    intent_schema_version: str = "0.1"
    safe_default: str = SAFE_DEFAULT
    trained: bool = False
    schema: str = MANIFEST_SCHEMA

    def weights_path(self) -> Path:
        p = Path(self.weights)
        return p if p.is_absolute() else (PKG_DIR / p)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "HeadManifest":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


class IntentHead(Protocol):
    """Maps a feature vector to per-label logits over INTENT_LABELS."""

    def logits(self, features: np.ndarray) -> np.ndarray: ...
