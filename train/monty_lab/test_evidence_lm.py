import unittest

import numpy as np

from train.monty_lab.evidence_lm import EvidenceLM
from train.monty_lab.protocol import Episode, Observation


def episode(points, label=None):
    return Episode([Observation(np.asarray(p, np.float32)) for p in points], label=label)


class PartialEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.a = np.asarray([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, 2, .2]], np.float32)
        self.b = np.asarray([
            [0, 0, 0], [1, 0, 0], [0, .2, 0], [-1, 0, 0], [0, .3, 2]], np.float32)
        self.model = EvidenceLM(sigma=.25, evidence_floor=.1)
        self.model.learn_episode(episode(self.a, "a"))
        self.model.learn_episode(episode(self.b, "b"))

    def test_partial_observations_vote_only_at_their_joint_ids(self):
        ids = np.asarray([0, 1, 2, 4])
        label, evidence, pose = self.model.infer_partial(episode(self.a[ids]), ids)
        self.assertEqual(label, "a")
        self.assertGreater(evidence, .9)
        self.assertEqual(pose.shape, (3, 3))

    def test_too_few_joints_abstain(self):
        label, evidence, pose = self.model.infer_partial(
            episode(self.a[[0, 1]]), [0, 1])
        self.assertEqual((label, evidence, pose), ("noop", 0.0, None))

    def test_duplicate_joint_ids_abstain(self):
        label, _, _ = self.model.infer_partial(
            episode(self.a[[0, 1, 2]]), [0, 1, 1])
        self.assertEqual(label, "noop")


if __name__ == "__main__":
    unittest.main()
