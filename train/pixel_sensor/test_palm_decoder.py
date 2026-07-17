import math
import unittest

import numpy as np

from train.pixel_sensor.palm_decoder import (
    ANCHORS, INPUT_SIZE, NUM_BOXES, PalmDetection, decode, detection_to_roi,
    generate_anchors, remove_letterbox, weighted_nms,
)


class PalmDecoderTest(unittest.TestCase):
    def test_anchor_count_and_order(self):
        anchors = generate_anchors()
        self.assertEqual(anchors.shape, (2016, 4))
        np.testing.assert_allclose(anchors[0], [1 / 48, 1 / 48, 1, 1])
        np.testing.assert_allclose(anchors[1], anchors[0])
        np.testing.assert_allclose(anchors[1151, :2], [47 / 48, 47 / 48])
        np.testing.assert_allclose(anchors[1152, :2], [1 / 24, 1 / 24])
        np.testing.assert_allclose(anchors[1152:1158], np.tile(anchors[1152], (6, 1)))
        np.testing.assert_allclose(anchors[-1, :2], [23 / 24, 23 / 24])

    def test_xywh_and_seven_keypoints_decode(self):
        raw = np.zeros((1, NUM_BOXES, 18), np.float32)
        scores = np.full((1, NUM_BOXES, 1), -100, np.float32)
        index = 1152
        raw[0, index, :4] = [19.2, -9.6, 38.4, 19.2]
        raw[0, index, 4:] = np.arange(14, dtype=np.float32)
        scores[0, index, 0] = 2.0
        result = decode(raw, scores)
        self.assertEqual(len(result), 1)
        detection = result[0]
        center = np.array([(detection.box[0] + detection.box[2]) / 2,
                           (detection.box[1] + detection.box[3]) / 2])
        np.testing.assert_allclose(center, ANCHORS[index, :2] + [0.1, -0.05], atol=1e-6)
        np.testing.assert_allclose(detection.box[2:] - detection.box[:2], [0.2, 0.1], atol=1e-6)
        np.testing.assert_allclose(detection.keypoints[0], ANCHORS[index, :2] + [0, 1 / INPUT_SIZE])
        self.assertAlmostEqual(detection.score, 1 / (1 + math.exp(-2)), places=6)

    def test_weighted_nms_blends_coordinates(self):
        keypoints = np.zeros((7, 2), np.float32)
        a = PalmDetection(np.array([0, 0, 1, 1], np.float32), keypoints, 0.9, 1)
        b = PalmDetection(np.array([0.1, 0, 1.1, 1], np.float32), keypoints + 1, 0.6, 2)
        out = weighted_nms([b, a])
        self.assertEqual(len(out), 1)
        np.testing.assert_allclose(out[0].box, (a.box * .9 + b.box * .6) / 1.5)
        self.assertEqual(out[0].anchor_index, 1)

    def test_letterbox_removal(self):
        d = PalmDetection(np.array([.2, .3, .8, .7], np.float32),
                          np.tile([.5, .5], (7, 1)).astype(np.float32), .9)
        out = remove_letterbox(d, 640, 320)
        np.testing.assert_allclose(out.box, [.2, .1, .8, .9], atol=1e-6)
        np.testing.assert_allclose(out.keypoints[0], [.5, .5], atol=1e-6)

    def test_upright_palm_roi(self):
        kp = np.zeros((7, 2), np.float32)
        kp[0] = [.5, .7]       # wrist centre
        kp[2] = [.5, .4]       # middle-finger MCP
        d = PalmDetection(np.array([.4, .35, .6, .75], np.float32), kp, .9)
        roi = detection_to_roi(d, 640, 480)
        self.assertAlmostEqual(roi.rotation, 0.0, places=6)
        self.assertLess(roi.center[1], .55)  # shift toward fingers
        self.assertAlmostEqual(roi.size[0] * 640, roi.size[1] * 480, places=4)


if __name__ == "__main__":
    unittest.main()
