"""Map head logits to an OpenPAVE intent token (pinned to the intent_schema vocab).

The returned string is a text alias intent_ingress already understands
(STOP/TROT/HOME/LEFT/RIGHT), so the shim can drop it straight into the OpenAI
response `content`. Low-confidence predictions fall back to the safe default so the
control path is always well-defined.
"""

from __future__ import annotations

import numpy as np

from pave_mlx.heads.base import INTENT_LABELS, SAFE_DEFAULT, softmax


def decode(
    logits: np.ndarray,
    labels: list[str] | None = None,
    min_confidence: float = 0.0,
) -> tuple[str, float]:
    labels = labels or INTENT_LABELS
    p = softmax(logits)[0]
    idx = int(np.argmax(p))
    conf = float(p[idx])
    if conf < min_confidence:
        return SAFE_DEFAULT, conf
    return labels[idx], conf
