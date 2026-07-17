import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
POC = ROOT / "train" / "insect-poc"
sys.path.insert(0, str(POC))

from insect_poc.cli import assemble, load_config, prepare_demo, train_one
from insect_poc.runtime import InsectEnsemble, PortableRbfSpecialist
from insect_poc.core_train import fixed_adjacency, prepare_core_demo
from insect_poc.real_data import _annotations, _direction_label, _state_for_frame, _trajectory_features


class InsectPocTests(unittest.TestCase):
    def test_one_specialist_can_be_retrained_without_touching_another(self):
        config = load_config(POC / "config.json")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); data = root / "data"; runs = root / "runs"
            runs.mkdir()
            prepare_demo(config, ["presence", "motion"], data, 160, 3)
            train_one("presence", config["specialists"]["presence"], data, runs, "sklearn", 3)
            train_one("motion", config["specialists"]["motion"], data, runs, "sklearn", 3)
            before = (runs / "specialists" / "presence" / "model.npz").read_bytes()
            train_one("motion", config["specialists"]["motion"], data, runs, "sklearn", 4)
            self.assertEqual(before, (runs / "specialists" / "presence" / "model.npz").read_bytes())

    def test_portable_runtime_and_manifest(self):
        config = load_config(POC / "config.json")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); data = root / "data"; runs = root / "runs"
            runs.mkdir()
            prepare_demo(config, ["presence"], data, 160, 7)
            train_one("presence", config["specialists"]["presence"], data, runs, "sklearn", 7)
            manifest = assemble(config, runs)
            ensemble = InsectEnsemble(manifest)
            X = np.load(data / "presence.npz", allow_pickle=False)["X"][0]
            result = ensemble.infer({"presence": X})
            self.assertIn(result["presence"], config["specialists"]["presence"]["classes"] + ["unknown"])
            self.assertEqual(PortableRbfSpecialist(runs / "specialists" / "presence").mean.size, 10)

    def test_deterministic_observation_composer(self):
        self.assertEqual(
            InsectEnsemble.compose_observation({
                "presence": "present", "motion": "left", "color": "red", "gesture": "fist"
            }),
            "Person detected, moving left, wearing red, showing fist.",
        )
        self.assertEqual(InsectEnsemble.compose_observation({"presence": "absent"}),
                         "No person detected.")

    def test_core_adjacency_is_stable_and_demo_is_grouped(self):
        A = fixed_adjacency(256, 9)
        self.assertEqual(A.shape, (256, 256))
        self.assertTrue(np.isfinite(A).all())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "core.npz"
            prepare_core_demo(path, minutes=0.1, fps=10, seed=9)
            ds = np.load(path, allow_pickle=False)
            self.assertEqual(ds["X"].shape, (60, 12))
            self.assertEqual(len(ds["y"]), len(ds["recording_id"]))

    def test_ipn_labels_are_independent_and_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "Annot_List.txt"
            path.write_text("video,label,id,t_start,t_end,frames\n"
                            "v,D0X,1,1,5,5\nv,G05,8,6,9,4\nv,G06,9,10,12,3\n"
                            "v,B0A,2,13,20,8\n")
            labels = _annotations(path)["v"]
            self.assertEqual([_state_for_frame(labels, f)[0] for f in (1, 6, 10, 15)],
                             [0, 2, 3, 1])

    def test_direction_labels_and_trajectory_shape(self):
        self.assertEqual([_direction_label(x) for x in ("G03", "G04", "G05", "G06", "D0X")],
                         ["up", "down", "left", "right", "still"])
        history = __import__("collections").deque()
        points = np.zeros((21, 3), np.float32); points[9, 0] = 0.1
        first = _trajectory_features(points, history)
        points[:, 0] += 0.02
        second = _trajectory_features(points, history)
        self.assertEqual(first.shape, (12,))
        self.assertGreater(second[0], 0)


if __name__ == "__main__":
    unittest.main()
