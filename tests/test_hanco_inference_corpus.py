from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from train.build_hanco_inference_corpus import (
    CONTRACT,
    assign_sequence_splits,
    build,
    project_landmarks,
    selected_frames,
)


def _mini_hanco(root: Path) -> Path:
    assignments = {
        "contract": "monty.hanco-classifier-assignments.v1",
        "assignments": {"0001/00000001": "like", "0002/*": "fist"},
        "sequence_cameras": {"0001": 3},
    }
    assignment_path = root / "assignments.json"
    assignment_path.write_text(json.dumps(assignments))
    for sequence in ("0001", "0002"):
        for frame_number in range(3):
            frame = f"{frame_number:08d}"
            for directory in ("xyz", "shape", "HanCo_calib_meta/calib"):
                (root / directory / sequence).mkdir(parents=True, exist_ok=True)
            xyz = [[float(index % 5) / 100, float(index // 5) / 100, 1.0] for index in range(21)]
            (root / "xyz" / sequence / f"{frame}.json").write_text(json.dumps(xyz))
            shape = {"poses": [[0.0] * 48], "shapes": [[0.0] * 10], "global_t": [[[0, 0, 1]]]}
            (root / "shape" / sequence / f"{frame}.json").write_text(json.dumps(shape))
            calibration = {"K": [np.eye(3).tolist()] * 8, "M": [np.eye(4).tolist()] * 8}
            (root / "HanCo_calib_meta/calib" / sequence / f"{frame}.json").write_text(json.dumps(calibration))
            for kind in ("rgb", "mask_hand", "mask_fg"):
                directory = root / kind / sequence / "cam0"
                directory.mkdir(parents=True, exist_ok=True)
                image = np.zeros((16, 16, 3), np.uint8)
                image[4:12, 4:12] = 255
                if kind == "mask_hand":
                    image = image[:, :, 0]
                assert cv2.imwrite(str(directory / f"{frame}.jpg"), image)
    return assignment_path


def test_projection_preserves_difficult_views_with_visibility_mask() -> None:
    xyz = np.asarray([[0, 0, 1], [2, 0, 1], [0, 0, -1]], np.float32)
    pixels, camera, valid = project_landmarks(xyz, np.eye(3), np.eye(4))
    np.testing.assert_allclose(pixels[:2], [[0, 0], [2, 0]])
    np.testing.assert_allclose(camera, xyz)
    assert valid.tolist() == [True, True, False]


def test_sequence_override_and_temporal_context(tmp_path: Path) -> None:
    assignment_path = _mini_hanco(tmp_path)
    assignments = json.loads(assignment_path.read_text())["assignments"]
    selected = selected_frames(tmp_path, assignments, context_radius=1)
    assert selected["0001"]["00000001"] == ("like", "frame")
    assert selected["0001"]["00000000"] == (None, "context")
    assert {value for value in selected["0002"].values()} == {("fist", "sequence")}


def test_missing_xyz_is_reported_instead_of_aborting(tmp_path: Path) -> None:
    from unittest.mock import patch

    assignment_path = _mini_hanco(tmp_path)
    (tmp_path / "xyz/0001/00000001.json").unlink()
    with patch(
        "train.build_hanco_inference_corpus.assign_sequence_splits",
        return_value={"0001": "train", "0002": "evaluation"},
    ):
        report = build(
            tmp_path, assignment_path, tmp_path / "corpus", context_radius=0, cameras=(0,)
        )
    assert report["skipped"]["missing_xyz"] == 1
    assert report["observations"] == 3
    assert report["incomplete_frames"] == [{
        "sequence_id": "0001",
        "frame_id": "00000001",
        "label": "like",
        "label_scope": "frame",
        "missing": ["xyz"],
    }]
    assert report["sequences_without_observations"] == ["0001"]


def test_builds_parallel_records_and_geometry_without_71k_targets(tmp_path: Path) -> None:
    assignment_path = _mini_hanco(tmp_path)
    # The production splitter deliberately refuses a two-sequence corpus, so
    # give this I/O-focused fixture a prevalidated split map.
    from unittest.mock import patch

    out = tmp_path / "corpus"
    with patch(
        "train.build_hanco_inference_corpus.assign_sequence_splits",
        return_value={"0001": "train", "0002": "evaluation"},
    ):
        report = build(tmp_path, assignment_path, out, context_radius=1, cameras=(0,))
    assert report["contract"] == CONTRACT
    records = [json.loads(line) for line in (out / "records.jsonl").read_text().splitlines()]
    geometry = np.load(out / "geometry.npz")
    assert len(records) == report["observations"] == len(geometry["world_xyz"])
    assert geometry["world_xyz"].shape[1:] == (21, 3)
    assert all(record["baseline_channels"] == {} for record in records)
    assert {record["split"] for record in records if record["sequence_id"] == "0001"} == {"train"}
    assert {record["preferred_camera"] for record in records if record["sequence_id"] == "0001"} == {3}


def test_multilabel_split_has_all_labels_without_sequence_leakage() -> None:
    selected = {}
    for label in ("fist", "palm", "point", "like"):
        for index in range(4):
            selected[f"{label}-{index}"] = {"0": (label, "sequence")}
    split = assign_sequence_splits(selected)
    for partition in ("train", "calibration", "evaluation"):
        sequences = {sequence for sequence, value in split.items() if value == partition}
        assert {next(iter(selected[sequence].values()))[0] for sequence in sequences} == {
            "fist", "palm", "point", "like"
        }
