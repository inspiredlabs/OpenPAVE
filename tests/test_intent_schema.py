import unittest

from pave_runtime.intent_schema import IntentValidationError, normalize_intent_payload


class IntentSchemaTests(unittest.TestCase):
    def test_text_stop_normalizes(self):
        intent = normalize_intent_payload({"text": "STOP"}, default_source="test")

        self.assertEqual(intent["schema_version"], "0.1")
        self.assertEqual(intent["intent"], "STOP")
        self.assertEqual(intent["params"], {})
        self.assertEqual(intent["source"], "test")
        self.assertEqual(intent["raw_text"], "STOP")

    def test_legacy_intent_alias_normalizes(self):
        intent = normalize_intent_payload({"intent": "RIGHT"}, default_source="test")

        self.assertEqual(intent["intent"], "MOVE")
        self.assertEqual(intent["raw_text"], "RIGHT")
        self.assertEqual(intent["params"], {"vx": 0.0, "yaw": 0.29, "duration_ms": 300})

    def test_legacy_flat_move_normalizes(self):
        intent = normalize_intent_payload(
            {"intent": "MOVE", "vx": 0.0, "yaw": -0.4, "duration_ms": 600},
            default_source="test",
        )

        self.assertEqual(intent["intent"], "MOVE")
        self.assertEqual(intent["params"], {"vx": 0.0, "yaw": -0.4, "duration_ms": 600})

    def test_unknown_text_falls_back_to_stop(self):
        intent = normalize_intent_payload({"text": "dance"}, default_source="test")

        self.assertEqual(intent["intent"], "STOP")
        self.assertTrue(intent["safety_fallback"])

    def test_invalid_move_param_raises(self):
        with self.assertRaises(IntentValidationError):
            normalize_intent_payload({"intent": "MOVE", "params": {"yaw": 2.5}})

    def test_invalid_confidence_raises(self):
        with self.assertRaises(IntentValidationError):
            normalize_intent_payload({"text": "STOP", "confidence": 1.5})


if __name__ == "__main__":
    unittest.main()
