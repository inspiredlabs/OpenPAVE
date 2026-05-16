import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INGRESS_PATH = ROOT / "intent-ingress" / "intent_ingress.py"
HAS_FLASK = importlib.util.find_spec("flask") is not None


def load_ingress_module():
    spec = importlib.util.spec_from_file_location("intent_ingress_test_module", INGRESS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(HAS_FLASK, "Flask is not installed; run with intent-ingress requirements")
class IntentIngressTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.intent_path = Path(self.tempdir.name) / "vla_intent.json"
        self.previous_intent_path = os.environ.get("INTENT_PATH")
        os.environ["INTENT_PATH"] = str(self.intent_path)
        self.ingress = load_ingress_module()
        self.client = self.ingress.app.test_client()

    def tearDown(self):
        if self.previous_intent_path is None:
            os.environ.pop("INTENT_PATH", None)
        else:
            os.environ["INTENT_PATH"] = self.previous_intent_path
        self.tempdir.cleanup()

    def read_written_intent(self):
        with self.intent_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def test_post_text_intent_writes_normalized_schema(self):
        response = self.client.post("/intent", json={"text": "RIGHT", "confidence": 0.75})

        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        written = self.read_written_intent()

        self.assertTrue(body["ok"])
        self.assertEqual(body["written"], written)
        self.assertEqual(written["schema_version"], "0.1")
        self.assertEqual(written["intent"], "MOVE")
        self.assertEqual(written["params"], {"vx": 0.0, "yaw": 0.6, "duration_ms": 600})
        self.assertEqual(written["source"], "webui")
        self.assertEqual(written["confidence"], 0.75)
        self.assertEqual(written["raw_text"], "RIGHT")
        self.assertIn("request_id", written)
        self.assertIn("timestamp", written)

    def test_post_invalid_payload_returns_400_and_does_not_write(self):
        response = self.client.post(
            "/intent",
            json={"intent": "MOVE", "params": {"yaw": 2.5}},
        )

        self.assertEqual(response.status_code, 400)
        body = response.get_json()
        self.assertFalse(body["ok"])
        self.assertIn("params.yaw", body["error"])
        self.assertFalse(self.intent_path.exists())

    def test_healthz(self):
        response = self.client.get("/healthz")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text, "ok\n")


if __name__ == "__main__":
    unittest.main()
