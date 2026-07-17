"""tbp.monty Environment over exported OpenPAVE gesture episodes.

RUNS INSIDE THE tbp.monty CONDA ENV (osx-64/Rosetta, Python 3.8) — never in
the OpenPAVE venv. The two stacks bridge via one file:
train/runs/monty_gestures/episodes.npz, produced by
`.venv/bin/python -m monty_lab.runner export --task gestures` on the
OpenPAVE side.

Implements the SimulatedEnvironment protocol
(src/tbp/monty/frameworks/environments/environment.py): the "world" is one
hand constellation per episode; the agent's sensor VISITS one landmark per
step (the sensor walk), so Monty receives features at 3D poses exactly as its
sensorimotor contract requires. Actions are accepted but ignored — the walk
order is the skeleton topology, deterministic by design (their docs allow
returning current observations for empty action sequences).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import quaternion  # numpy-quaternion, ships with the tbp.monty env

from tbp.monty.frameworks.models.abstract_monty_classes import (
    AgentObservations,
    Observations,
)
from tbp.monty.frameworks.models.motor_system_state import (
    AgentState,
    ProprioceptiveState,
    SensorState,
)

AGENT_ID = "agent_id_0"
SENSOR_ID = "joint_sensor"


class HandEpisodesEnvironment:
    """One exported gesture episode at a time; one landmark per step."""

    def __init__(self, episodes_npz: str, split: str = "train") -> None:
        d = np.load(Path(episodes_npz), allow_pickle=True)
        keep = ~d["is_val"] if split == "train" else d["is_val"]
        self._locations = d["locations"][keep]      # (N, 21, 3), hand ref frame
        self._labels = d["labels"][keep]            # (N,) object names
        self._episode = 0
        self._cursor = 0

    # ── protocol: Environment ────────────────────────────────────────
    def step(self, actions: Sequence[object]) -> Tuple[Observations, ProprioceptiveState]:
        if len(actions) > 0 and self._cursor < 20:
            self._cursor += 1                       # any action -> visit next landmark
        return self._observe()

    def close(self) -> None:
        pass

    # ── protocol: ResettableEnvironment ──────────────────────────────
    def reset(self) -> Tuple[Observations, ProprioceptiveState]:
        self._cursor = 0
        return self._observe()

    # ── episode management (driven by the experiment's pre/post hooks) ─
    def next_episode(self) -> Optional[str]:
        """Advance to the next constellation; returns its object label."""
        self._episode = (self._episode + 1) % len(self._labels)
        self._cursor = 0
        return str(self._labels[self._episode])

    @property
    def current_label(self) -> str:
        return str(self._labels[self._episode])

    @property
    def steps_remaining(self) -> int:
        return 20 - self._cursor

    # ── internals ────────────────────────────────────────────────────
    def _observe(self) -> Tuple[Observations, ProprioceptiveState]:
        loc = self._locations[self._episode][self._cursor].astype(np.float64)
        obs = Observations({
            AGENT_ID: AgentObservations({
                SENSOR_ID: {
                    "landmark_index": int(self._cursor),   # which joint (feature)
                    "location": loc,                       # where (pose), hand frame
                }
            })
        })
        state = ProprioceptiveState({
            AGENT_ID: AgentState(
                sensors={SENSOR_ID: SensorState(
                    position=(0.0, 0.0, 0.0), rotation=quaternion.one)},
                position=tuple(float(x) for x in loc),     # sensor AT the landmark
                rotation=quaternion.one,
            )
        })
        return obs, state
