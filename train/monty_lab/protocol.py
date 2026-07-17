"""The Cortical-Messaging-Protocol-shaped contract every task speaks.

Monty's core idea: learning consumes a stream of (feature, location)
observations gathered by a moving sensor; objects are learned as reference
frames of those observations; recognition returns object AND pose. The
classes here are deliberately tiny — they are the seam where the real
tbp.monty SensorModule/LearningModule can be adapted in later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Protocol

import numpy as np


@dataclass
class Observation:
    """One sensor reading: a feature vector AT a 3D location (euclidean)."""
    location: np.ndarray                  # (3,) sensor location in body/object space
    feature: np.ndarray | None = None     # optional feature vector at that pose


@dataclass
class Episode:
    """One sensor walk over one object (a Monty 'episode')."""
    observations: list[Observation]
    label: str | None = None              # ground truth during learning; None at inference
    meta: dict = field(default_factory=dict)

    def locations(self) -> np.ndarray:
        return np.stack([o.location for o in self.observations])


class Task(Protocol):
    """A trainable application. Implementations live in monty_lab/tasks/."""

    name: str

    def learning_episodes(self) -> Iterator[Episode]:
        """Labelled sensor walks to learn object models from (few-shot)."""

    def eval_episodes(self) -> Iterator[Episode]:
        """Labelled walks the model has never seen, for scoring."""

    def outcome(self, obj: str, episode: Episode) -> str:
        """Map a recognised object (+ episode geometry) to a task outcome
        token, e.g. gesture object 'point' + index vector -> 'LEFT'."""
