import unittest

import numpy as np

from pave_ui.perception import (
    POINTING_DIRECTION_PROMPT,
    ROBOT_PROMPT,
    ROBOT_PROMPT_GESTURE_NAME,
    ROBOT_PROMPT_QWEN,
    EngineHandle,
    clamp_to_intent,
    has_intent_token,
    infer,
    observe_signature,
    pointing_needs_direction,
    prompt_for_model,
    scene_delta,
    scene_delta_max,
)


class GestureSynonymTests(unittest.TestCase):
    # Fourier Qwen2-VL 2B answers the gesture NAME instead of the intent word
    # (measured on HaGRID samples); the clamp must map those, not STOP them.
    def test_gesture_names_map_to_intents(self):
        self.assertEqual(clamp_to_intent("Thumbs-up"), "TROT")
        self.assertEqual(clamp_to_intent("thumb up"), "TROT")
        self.assertEqual(clamp_to_intent("Open palm"), "STOP")
        self.assertEqual(clamp_to_intent("Waving"), "STOP")
        self.assertEqual(clamp_to_intent("FIST"), "HOME")
        self.assertEqual(clamp_to_intent("Point-right"), "RIGHT")

    def test_intent_tokens_still_win_over_synonyms(self):
        # an explicit intent word anywhere beats the gesture-name fallback
        self.assertEqual(clamp_to_intent("fist means HOME so TROT"), "HOME")
        self.assertEqual(clamp_to_intent("INTENT: STOP\nFEATURE: hand center"), "STOP")

    def test_gesture_names_count_as_valid_intent(self):
        self.assertTrue(has_intent_token("Thumbs-up"))
        self.assertTrue(has_intent_token("fist"))
        self.assertFalse(has_intent_token("a dog on a sofa"))

    def test_unrecognised_text_still_clamps_to_stop(self):
        self.assertEqual(clamp_to_intent("a dog on a sofa"), "STOP")
        self.assertEqual(clamp_to_intent(""), "STOP")


class PromptRoutingTests(unittest.TestCase):
    def test_fourier_gets_gesture_name_prompt(self):
        # strict templates are parroted by this model; it must get the bare question
        self.assertEqual(prompt_for_model("Fourier Qwen2-VL 2B (mradermacher)"), ROBOT_PROMPT_GESTURE_NAME)

    def test_qwen_models_keep_strict_prompt(self):
        self.assertEqual(prompt_for_model("Qwen3-VL"), ROBOT_PROMPT_QWEN)
        self.assertEqual(prompt_for_model("Qwen3.5 2B (Rishu11277)"), ROBOT_PROMPT_QWEN)

    def test_gemma_keeps_default_prompt(self):
        self.assertEqual(prompt_for_model("Gemma 4 E4B"), ROBOT_PROMPT)


class PointingFollowUpTests(unittest.TestCase):
    def test_directionless_pointing_triggers_follow_up(self):
        self.assertTrue(pointing_needs_direction("Pointing"))
        self.assertTrue(pointing_needs_direction("pointing with index finger"))

    def test_direction_or_no_pointing_skips_follow_up(self):
        self.assertFalse(pointing_needs_direction("Pointing left"))
        self.assertFalse(pointing_needs_direction("point-right"))
        self.assertFalse(pointing_needs_direction("Pointing up"))
        self.assertFalse(pointing_needs_direction("INTENT: LEFT\nFEATURE: hand center"))
        self.assertFalse(pointing_needs_direction("Thumbs up"))

    def test_infer_asks_direction_once_and_maps_it(self):
        class FakeEngine:
            def __init__(self):
                self.calls = []

            def generate(self, img, prompt, max_tokens=12):
                self.calls.append(prompt)
                return "LEFT" if prompt == POINTING_DIRECTION_PROMPT else "Pointing"

        engine = FakeEngine()
        handle = EngineHandle(name="Fourier Qwen2-VL 2B (mradermacher)", engine=engine,
                              kind="vlm", processor=None, img_size=448)
        result = infer(handle, np.zeros((8, 8, 3), dtype=np.uint8))

        self.assertEqual(result.intent, "LEFT")
        self.assertTrue(result.ok)
        self.assertEqual(len(engine.calls), 2)
        self.assertEqual(engine.calls[1], POINTING_DIRECTION_PROMPT)

    def test_infer_skips_follow_up_for_plain_gestures(self):
        class FakeEngine:
            def __init__(self):
                self.calls = []

            def generate(self, img, prompt, max_tokens=12):
                self.calls.append(prompt)
                return "Thumbs up"

        engine = FakeEngine()
        handle = EngineHandle(name="Fourier Qwen2-VL 2B (mradermacher)", engine=engine,
                              kind="vlm", processor=None, img_size=448)
        result = infer(handle, np.zeros((8, 8, 3), dtype=np.uint8))

        self.assertEqual(result.intent, "TROT")
        self.assertEqual(len(engine.calls), 1)


class SceneChangeGateTests(unittest.TestCase):
    # the continuous OBSERVE toggle's cheap gate: identical frames must read as
    # no change; a real scene change must clear the default threshold (12.0)
    def test_identical_frames_have_zero_delta(self):
        frame = np.random.default_rng(7).integers(0, 255, (480, 640, 3), dtype=np.uint8)
        self.assertEqual(scene_delta(observe_signature(frame), observe_signature(frame)), 0.0)

    def test_missing_reference_always_counts_as_changed(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.assertEqual(scene_delta(None, observe_signature(frame)), float("inf"))

    def test_dramatic_change_clears_default_threshold(self):
        dark = np.full((480, 640, 3), 20, dtype=np.uint8)
        bright = np.full((480, 640, 3), 200, dtype=np.uint8)
        self.assertGreater(scene_delta(observe_signature(dark), observe_signature(bright)), 12.0)

    def test_sensor_noise_stays_under_threshold(self):
        rng = np.random.default_rng(7)
        base = rng.integers(60, 190, (480, 640, 3)).astype(np.uint8)
        noisy = np.clip(base.astype(np.int16) + rng.integers(-6, 7, base.shape), 0, 255).astype(np.uint8)
        self.assertLess(scene_delta(observe_signature(base), observe_signature(noisy)), 12.0)

    # camera gesture gate (max-cell delta, default threshold 30): a hand in one
    # corner must register; noise and global exposure drift must not
    def test_localized_hand_clears_camera_threshold(self):
        base = np.full((480, 640, 3), 90, dtype=np.uint8)
        hand = base.copy()
        hand[300:420, 480:600] = 210  # hand-sized bright patch, one corner
        self.assertGreater(scene_delta_max(observe_signature(base), observe_signature(hand)), 30.0)

    def test_noise_and_exposure_drift_stay_under_camera_threshold(self):
        rng = np.random.default_rng(11)
        base = rng.integers(60, 190, (480, 640, 3)).astype(np.uint8)
        noisy = np.clip(base.astype(np.int16) + rng.integers(-6, 7, base.shape), 0, 255).astype(np.uint8)
        brighter = np.clip(base.astype(np.int16) + 10, 0, 255).astype(np.uint8)  # exposure shift
        self.assertLess(scene_delta_max(observe_signature(base), observe_signature(noisy)), 30.0)
        self.assertLess(scene_delta_max(observe_signature(base), observe_signature(brighter)), 30.0)

    def test_missing_reference_opens_camera_gate(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.assertEqual(scene_delta_max(None, observe_signature(frame)), float("inf"))


if __name__ == "__main__":
    unittest.main()
