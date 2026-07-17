from __future__ import annotations

import unittest

import numpy as np

from train.monty_lab.tbp_adapter.landmark_contract import (
    canonicalize,
    joint_feature,
    paired_errors,
)
from train.monty_lab.tbp_adapter.oracle_roi import (
    oracle_roi,
    project_to_roi,
    project_to_source,
)


class LandmarkContractTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.points = rng.normal(size=(21, 2)) * 0.1
        self.points[0] = [0.4, 0.7]
        self.points[9] = [0.4, 0.4]

    def test_similarity_transform_is_removed_without_teacher_leakage(self):
        expected, _ = canonicalize(self.points)
        angle = 0.7
        rotation = np.asarray([[np.cos(angle), -np.sin(angle)],
                               [np.sin(angle), np.cos(angle)]])
        moved = self.points.dot(rotation.T) * 1.8 + np.asarray([0.2, -0.1])
        actual, _ = canonicalize(moved)
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_joint_identity_is_categorical(self):
        feature = joint_feature(8)
        self.assertEqual(feature.shape, (21,))
        self.assertEqual(float(feature.sum()), 1.0)
        self.assertEqual(float(feature[8]), 1.0)

    def test_paired_error_keeps_landmark_number(self):
        student = self.points.copy()
        student[8, 0] += 10.0 / 384.0
        error = paired_errors(self.points[None], student[None])[0]
        self.assertAlmostEqual(float(error[8]), 10.0, places=5)
        self.assertEqual(int(np.count_nonzero(error)), 1)

    def test_single_bad_mcp_only_shifts_the_frame(self):
        """Multi-anchor fit: one corrupted MCP must not rotate every joint."""
        clean, _ = canonicalize(self.points)
        corrupted = self.points.copy()
        corrupted[9] += [0.3, 0.1]
        multi, _ = canonicalize(corrupted)
        # With the 0->9-only frame this corruption rotated/rescaled all
        # joints; the centroid frame keeps unrelated fingertips close.
        untouched = [4, 8, 12, 16, 20]
        drift = np.linalg.norm(multi[untouched] - clean[untouched], axis=1)
        self.assertLess(float(drift.max()), 0.6)

    def test_zero_weight_anchor_is_excluded(self):
        weights = np.ones(21)
        weights[9] = 0.0
        corrupted = self.points.copy()
        corrupted[9] += [5.0, 5.0]
        weighted, _ = canonicalize(corrupted, weights=weights)
        clean_weights_frame, _ = canonicalize(self.points, weights=weights)
        others = [j for j in range(21) if j != 9]
        np.testing.assert_allclose(weighted[others], clean_weights_frame[others],
                                   atol=1e-6)

    def test_oracle_roi_round_trip_and_orientation(self):
        points = np.clip(self.points + 0.5, 0.05, 0.95)
        roi = oracle_roi(points)
        uv = project_to_roi(points, roi)
        recovered = project_to_source(uv, roi)
        np.testing.assert_allclose(recovered, points, atol=1e-9)
        # +v points from the MCPs toward the wrist: the wrist sits below
        # (greater v than) the MCP centroid, so fingers face the crop top.
        mcp_v = uv[[5, 9, 13, 17], 1].mean()
        self.assertGreater(float(uv[0, 1]), float(mcp_v))
        self.assertTrue((uv >= -0.2).all() and (uv <= 1.2).all())


if __name__ == "__main__":
    unittest.main()
