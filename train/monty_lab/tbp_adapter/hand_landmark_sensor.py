"""Real tbp.monty SensorModule for paired hand-landmark observations.

This file is imported only inside the dedicated tbp.monty conda environment.
"""
from __future__ import annotations

import copy

import numpy as np

from tbp.monty.cmp import Message
from tbp.monty.frameworks.models.abstract_monty_classes import SensorModule

try:
    from .landmark_contract import canonicalize, joint_feature, local_pose_vectors
except ImportError:  # direct script execution inside the tbp.monty env
    from landmark_contract import canonicalize, joint_feature, local_pose_vectors


class HandLandmarkSensorModule(SensorModule):
    """Convert one numbered hand landmark into a common-frame CMP Message.

    Each raw observation contains both teacher and student constellations for
    logging, but ``stream`` selects the only constellation visible to the LM.
    During evaluation the MediaPipe teacher therefore cannot affect inference.
    """

    def __init__(self, sensor_module_id="hand_landmark_sensor", stream="teacher",
                 confidence_threshold=0.25):
        self.sensor_module_id = sensor_module_id
        self.stream = stream
        self.confidence_threshold = float(confidence_threshold)
        # DetailedGraphMatchingLogger includes SM observations only when this
        # flag is enabled.  The supervised hand experiment deliberately keeps
        # it enabled so the teacher/student geometry can be replayed.
        self.save_raw_obs = True
        self.state = None
        self.reset()

    def reset(self):
        self.raw_observations = []
        self.processed_observations = []

    def update_state(self, agent):
        self.state = agent

    def state_dict(self):
        return {
            "raw_observations": copy.deepcopy(self.raw_observations),
            "processed_observations": copy.deepcopy(self.processed_observations),
            "stream": self.stream,
        }

    def step(self, ctx, observation, motor_only_step=False):  # noqa: ARG002
        joint_id = int(observation["joint_id"])
        teacher_uv = np.asarray(observation["teacher_uv"], dtype=np.float64)
        student_uv = np.asarray(observation["student_uv"], dtype=np.float64)
        student_confidence = np.asarray(observation["student_confidence"], dtype=np.float64)
        if self.stream == "teacher":
            selected, weights, confidence = teacher_uv, None, 1.0
        else:
            selected = student_uv
            weights = student_confidence
            confidence = float(student_confidence[joint_id])

        try:
            canonical, transform = canonicalize(selected, weights=weights)
            location = canonical[joint_id]
            valid = bool(np.isfinite(location).all())
        except ValueError:
            canonical = np.full((21, 3), np.nan)
            transform = {"origin": np.zeros(3), "basis": np.eye(2), "scale": 0.0}
            location = np.zeros(3)
            valid = False

        use_state = bool(valid and confidence >= self.confidence_threshold
                         and not motor_only_step)
        pose_vectors = (local_pose_vectors(canonical, joint_id) if valid
                        else np.eye(3, dtype=np.float64))
        message = Message(
            location=np.asarray(location, dtype=np.float64),
            morphological_features={
                "pose_vectors": pose_vectors,
                "pose_fully_defined": valid,
                "on_object": int(valid),
            },
            non_morphological_features={"joint_id": joint_feature(joint_id)},
            confidence=confidence,
            use_state=use_state,
            sender_id=self.sensor_module_id,
            sender_type="SM",
        )

        raw = {
            "frame_id": int(observation["frame_id"]),
            "source": str(observation["source"]),
            "source_index": int(observation["source_index"]),
            "role": str(observation["role"]),
            "target": str(observation["label"]),
            "joint_id": joint_id,
            "teacher_uv": teacher_uv[joint_id].copy(),
            "student_uv": student_uv[joint_id].copy(),
            "student_confidence": float(
                np.asarray(observation["student_confidence"])[joint_id]),
            "selected_stream": self.stream,
            "selected_transform": transform,
            "motor_delta": np.asarray(
                observation.get("motor_delta", np.zeros(3)), dtype=np.float64),
        }
        self.raw_observations.append(raw)
        self.processed_observations.append(copy.deepcopy(message.__dict__))
        return message
