"""Actual Monty supervised-pretraining components for paired hand landmarks.

This module runs only in the dedicated tbp.monty environment.  OpenPAVE and
Monty remain separated by ``episodes.npz``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import quaternion

from tbp.monty.experiment.environment import Interface
from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.experiments.pretraining_experiments import (
    MontySupervisedObjectPretrainingExperiment,
)
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
SENSOR_ID = "hand_landmark_sensor"


def _target(label):
    return {
        "object": str(label),
        "position": np.zeros(3, dtype=np.float64),
        "euler_rotation": np.zeros(3, dtype=np.float64),
        "quat_rotation": np.array([1.0, 0.0, 0.0, 0.0]),
        # The supervised experiment consumes ``quat_rotation`` while Monty's
        # logging utilities consume ``rotation`` and index the uniform scale.
        "rotation": np.array([1.0, 0.0, 0.0, 0.0]),
        "scale": np.ones(3, dtype=np.float64),
        "consistent_child_objects": [],
    }


class PairedHandEnvironment:
    """A stored hand frame whose 21 numbered joints are sensor visits."""

    def __init__(self, episodes_npz, role="exploration", per_object=1):
        with np.load(Path(episodes_npz), allow_pickle=True) as loaded:
            self.data = {key: loaded[key] for key in loaded.files}
        candidate = np.where((self.data["roles"] == role) & ~self.data["is_val"])[0]
        selected = []
        for label in sorted(set(self.data["labels"][candidate].tolist())):
            rows = candidate[self.data["labels"][candidate] == label]
            count = min(int(per_object), len(rows))
            selected.extend(rows[np.linspace(0, len(rows) - 1, count, dtype=np.int64)])
        if not selected:
            raise ValueError("no paired hand episodes matched the requested role")
        self.episode_indices = np.asarray(selected, dtype=np.int64)
        self.sequence_index = 0
        self.cursor = 0

    @property
    def current_episode(self):
        return int(self.episode_indices[self.sequence_index])

    @property
    def current_label(self):
        return str(self.data["labels"][self.current_episode])

    def select(self, sequence_index):
        self.sequence_index = int(sequence_index) % len(self.episode_indices)
        self.cursor = 0

    def reset(self):
        self.cursor = 0
        return self._observe(0)

    def step(self, actions):  # noqa: ARG002 - visits are a fixed sensor trajectory
        joint = min(self.cursor, 20)
        result = self._observe(joint)
        self.cursor = min(self.cursor + 1, 20)
        return result

    def close(self):
        pass

    def _observe(self, joint):
        episode = self.current_episode
        order = (self.data["joint_order"][episode]
                 if "joint_order" in self.data else np.arange(21))
        step_index = min(joint, len(order) - 1)
        joint = int(order[step_index])
        sensor_location = (self.data["sensor_locations"][episode, step_index]
                           if "sensor_locations" in self.data
                           else np.zeros(3, dtype=np.float64))
        raw = {
            "frame_id": episode,
            "source": str(self.data["sources"][episode]),
            "source_index": int(self.data["source_indices"][episode]),
            "role": str(self.data["roles"][episode]),
            "label": str(self.data["labels"][episode]),
            "joint_id": int(joint),
            "teacher_uv": self.data["teacher_uv"][episode],
            "student_uv": self.data["student_uv"][episode],
            "student_confidence": self.data["student_confidence"][episode],
            "motor_delta": (self.data["motor_deltas"][episode, step_index]
                            if "motor_deltas" in self.data else np.zeros(3)),
        }
        observations = Observations({
            AGENT_ID: AgentObservations({SENSOR_ID: raw}),
        })
        state = ProprioceptiveState({
            AGENT_ID: AgentState(
                sensors={SENSOR_ID: SensorState(
                    position=tuple(sensor_location), rotation=quaternion.one)},
                position=(0.0, 0.0, 0.0),
                rotation=quaternion.one,
            )
        })
        return observations, state


class PairedHandInterface(Interface):
    """Advances one stored constellation after each Monty episode."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sequence_index = 0
        self.env.select(0)
        self.primary_target = _target(self.env.current_label)

    def pre_episode(self, rng):
        self.rng = rng
        self.env.select(self.sequence_index)
        self.primary_target = _target(self.env.current_label)
        self.reset(rng)

    def post_episode(self):
        self.sequence_index = (self.sequence_index + 1) % len(self.env.episode_indices)
        self.env.select(self.sequence_index)
        # The supervised experiment reads the target before pre_episode(), so
        # prepare the next target here as well.
        self.primary_target = _target(self.env.current_label)


@dataclass
class _Telemetry:
    pc_heading: list
    avoidance_heading: list
    z_defined_pc: list


class _NoCommandSelector:
    def __init__(self):
        self._selected_goals = []

    def reset(self):
        self._selected_goals = []

    def state_dict(self):
        return {"commands_enabled": False}


class LandmarkReplayMotorSystem:
    """A no-command motor system; the environment owns the joint traversal."""

    def __init__(self):
        self._policy_selector = _NoCommandSelector()
        self._motor_only_step = False
        self.reset()

    @property
    def motor_only_step(self):
        return self._motor_only_step

    @property
    def action_sequence(self):
        return self._action_sequence

    def reset(self):
        self._action_sequence = []
        self._telemetry_surface_action_details = _Telemetry([], [], [])
        self._policy_selector.reset()

    def state_dict(self):
        return {"commands_enabled": False, "policy": self._policy_selector.state_dict()}

    def __call__(self, ctx, observations, proprioceptive_state, percept, goals):  # noqa: ARG002
        self._action_sequence.append(([], None))
        return []


class HandLandmarkPretrainingExperiment(
        MontySupervisedObjectPretrainingExperiment):
    """Supervised pretraining with target-aware logs and replayable SM data."""

    @property
    def logger_args(self):
        args = super().logger_args
        if self.env_interface is not None:
            args["target"] = self.env_interface.primary_target
        return args

    def train(self):
        """Retain raw SM observations required for bounded visual replay."""
        self.experiment_mode = ExperimentMode.TRAIN
        self.logger_handler.pre_train(self.logger_args)
        self.model.set_experiment_mode(self.experiment_mode)
        for sm in self.model.sensor_modules:
            sm.save_raw_obs = True
        for _ in range(self.n_train_epochs):
            self.run_epoch()
        self.logger_handler.post_train(self.logger_args)
        self.save_state_dir(output_dir=self.output_dir)
