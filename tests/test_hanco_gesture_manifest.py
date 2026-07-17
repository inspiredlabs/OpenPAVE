from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from train.prepare_hanco_gestures import split_for_sequence


ROOT = Path(__file__).resolve().parents[1]


def test_only_unconfirmed_counting_sequences_are_not_supervised() -> None:
    manifest = json.loads(
        (ROOT / "train/datasets/hanco/gesture_manifest.json").read_text())
    supervised = {item["sequence"] for rows in manifest["labels"].values() for item in rows}
    assert {"0012", "0018"}.isdisjoint(supervised)
    assert "0032" in supervised


def test_sequence_split_keeps_all_cameras_together() -> None:
    index = np.load(ROOT / "train/datasets/hanco_gestures/index.npz")
    for sequence in np.unique(index["sequence_ids"]):
        rows = index["sequence_ids"] == sequence
        assert set(index["splits"][rows]) == {split_for_sequence(str(sequence))}


def test_reviewed_index_has_expected_labels() -> None:
    index = np.load(ROOT / "train/datasets/hanco_gestures/index.npz")
    assert set(index["labels"]) == {"palm", "like", "fist", "point"}
    assert len(index["labels"]) == 19_705


def test_conflicting_sequences_are_partitioned_without_duplicate_frames() -> None:
    index = np.load(ROOT / "train/datasets/hanco_gestures/index.npz")
    for sequence in ("0006", "0019"):
        rows = index["sequence_ids"] == sequence
        keys = list(zip(index["frame_ids"][rows], index["camera_ids"][rows], strict=True))
        assert len(keys) == len(set(keys))
        assert len(set(index["labels"][rows])) == 2
