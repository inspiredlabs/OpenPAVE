import unittest

import numpy as np

from train.pixel_sensor.projector import HandFrameProjector


class HandFrameProjectorTest(unittest.TestCase):
    def test_round_trip_and_subpixel_projection(self):
        hand = np.array([[0.0, 0.0], [0.2, -0.7], [0.8, 0.1], [-0.3, 0.4]], np.float32)
        angle = np.deg2rad(31)
        rotation = np.array([[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle), np.cos(angle)]], np.float32)
        uv = 0.34 * (hand @ rotation.T) + np.array([0.52, 0.41], np.float32)
        projector = HandFrameProjector.fit(hand, uv)
        np.testing.assert_allclose(projector.to_uv(hand), uv, atol=1e-6)
        np.testing.assert_allclose(projector.to_hand(uv), hand, atol=1e-5)
        # Normalised tolerance corresponds to far less than one pixel at 384.
        self.assertLess(np.max(np.abs(projector.to_uv(hand) - uv)) * 384, 0.001)

    def test_rejects_degenerate_evidence(self):
        with self.assertRaises(ValueError):
            HandFrameProjector.fit(np.zeros((3, 2)), np.ones((3, 2)))


if __name__ == "__main__":
    unittest.main()
