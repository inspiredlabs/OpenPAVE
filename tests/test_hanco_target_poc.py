from __future__ import annotations

import numpy as np

from train.hanco_target_poc import feature, split_group


def test_feature_is_translation_scale_and_rotation_invariant() -> None:
    rng = np.random.default_rng(3)
    points = rng.normal(size=(21, 2))
    points[0] = 0.0
    angle = 0.73
    rotation = np.asarray([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])
    transformed = points @ rotation * 2.7 + np.asarray([4.0, -3.0])

    np.testing.assert_allclose(feature(points), feature(transformed), atol=1e-5)


def test_sequence_split_is_stable() -> None:
    assert split_group("0001") == split_group("0001")
    assert split_group("0001") in (0, 1, 2)
