from __future__ import annotations

import numpy as np

from train.monty_lab.tbp_adapter.prepare_hanco import project_landmarks


def test_project_landmarks_uses_world_to_camera_extrinsics() -> None:
    xyz = np.asarray([[0.0, 0.0, 2.0], [1.0, 2.0, 4.0]])
    intrinsics = np.asarray([[100.0, 0.0, 20.0], [0.0, 80.0, 30.0], [0.0, 0.0, 1.0]])
    extrinsics = np.eye(4)
    extrinsics[0, 3] = 1.0

    pixels, camera = project_landmarks(xyz, intrinsics, extrinsics)

    np.testing.assert_allclose(camera, [[1.0, 0.0, 2.0], [2.0, 2.0, 4.0]])
    np.testing.assert_allclose(pixels, [[70.0, 30.0], [70.0, 70.0]])
