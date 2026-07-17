"""Gesture recognition as a monty_lab Task — the first application.

The 'sensor walk' over a gesture object is the hand skeleton itself: 21
MediaPipe landmarks visited in topological order, each an Observation at a 3D
euclidean location. Learning episodes come from the crude captures plus a
cross-subject harvest from the yolo26 ground-truth train split; eval episodes
are the yolo26 test+valid referee (never learned from).

All pointing is ONE object: direction is pose, resolved by outcome() via the
shipping index-vector geometry.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator

import numpy as np

TRAIN_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TRAIN_DIR))
from mediapipe_svm import LANDMARKER, _read_frames, _resolve_source, point_direction  # noqa: E402

from ..protocol import Episode, Observation  # noqa: E402

OBJECT_SOURCES = {
    "palm": ["stop"],
    "fist": ["fist"],
    "like": ["like"],
    "point": ["up", "down", "right-to-left", "left-to-right"],
}
Y26_TO_OBJECT = {"Stop": "palm", "Thumbs up": "like",
                 "Left": "point", "Right": "point", "Up": "point", "Down": "point"}
Y26_EXPECTED = {"Stop": "STOP", "Thumbs up": "TROT", "Left": "LEFT", "Right": "RIGHT",
                "Up": "ABSTAIN", "Down": "ABSTAIN", "Thumbs Down": "ABSTAIN"}
INTENT = {"palm": "STOP", "fist": "HOME", "like": "TROT"}
EXEMPLARS_PER_SOURCE = 10
CROSS_SUBJECT_PER_OBJECT = 12


def hand_to_episode(hand, label: str | None = None) -> Episode:
    """MediaPipe hand -> the sensor walk episode (shared with the live worker)."""
    obs = [Observation(location=np.array([q.x, q.y, q.z], dtype=np.float32))
           for q in hand]
    ep = Episode(observations=obs, label=label)
    ep.meta["hand"] = hand
    return ep


class GestureTask:
    name = "gestures"

    def __init__(self) -> None:
        self._mp = None
        self._lm = None

    def _landmarker(self):
        if self._lm is None:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            self._mp = mp
            self._lm = vision.HandLandmarker.create_from_options(
                vision.HandLandmarkerOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=str(LANDMARKER)),
                    num_hands=1, min_hand_detection_confidence=0.5))
        return self._mp, self._lm

    def _episodes_from_video(self, stem: str, obj: str) -> Iterator[Episode]:
        mp, lm = self._landmarker()
        src = _resolve_source(stem)
        if src is None:
            return
        frames = _read_frames(src)
        mid = frames[int(len(frames) * 0.3):int(len(frames) * 0.7)]   # held gesture
        found = 0
        for f in mid[:: max(1, len(mid) // (EXEMPLARS_PER_SOURCE * 3))]:
            res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=f))
            if res.hand_landmarks:
                yield hand_to_episode(res.hand_landmarks[0], obj)
                found += 1
            if found >= EXEMPLARS_PER_SOURCE:
                return

    def learning_episodes(self) -> Iterator[Episode]:
        for obj, stems in OBJECT_SOURCES.items():
            for stem in stems:
                yield from self._episodes_from_video(stem, obj)
        # cross-subject few-shot from yolo26 ground-truth TRAIN split
        from gesture_lab import _yolo26_items
        mp, lm = self._landmarker()
        added: dict[str, int] = {}
        for rgb, name in _yolo26_items("train"):
            obj = Y26_TO_OBJECT.get(name)
            if obj is None or added.get(obj, 0) >= CROSS_SUBJECT_PER_OBJECT:
                continue
            res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if res.hand_landmarks:
                added[obj] = added.get(obj, 0) + 1
                yield hand_to_episode(res.hand_landmarks[0], obj)

    def eval_episodes(self) -> Iterator[Episode]:
        from gesture_lab import _yolo26_items
        mp, lm = self._landmarker()
        for split in ("test", "valid"):
            for rgb, name in _yolo26_items(split):
                res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                if res.hand_landmarks:
                    ep = hand_to_episode(res.hand_landmarks[0], Y26_EXPECTED[name])
                    ep.meta["gt_class"] = name
                    yield ep

    def outcome(self, obj: str, episode: Episode) -> str:
        if obj == "point":
            d = point_direction(episode.meta["hand"])
            return d if d in ("LEFT", "RIGHT") else "ABSTAIN"
        if obj == "noop":
            return "ABSTAIN"
        return INTENT.get(obj, "ABSTAIN")
