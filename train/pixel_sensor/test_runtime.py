import unittest

import numpy as np

from train.pixel_sensor.runtime import CoordinateAdapter


class CoordinateAdapterTest(unittest.TestCase):
    def test_depth_is_prior_and_xy_is_normalised(self):
        uv = np.linspace(0.2, 0.8, 42, dtype=np.float32).reshape(21, 2)
        prior = np.linspace(-0.2, 0.2, 21, dtype=np.float32)
        locations, provenance = CoordinateAdapter(prior).compose(uv)
        np.testing.assert_allclose(locations[:, 2], prior)
        np.testing.assert_allclose(locations[0, :2], 0.0)
        self.assertAlmostEqual(float(np.abs(locations[:, :2]).max()), 1.0, places=5)
        self.assertEqual(provenance["z_source"], "prior")
        self.assertFalse(provenance["metric_3d"])


if __name__ == "__main__":
    unittest.main()
