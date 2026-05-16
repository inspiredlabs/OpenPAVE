import json
import tempfile
import unittest
from pathlib import Path

from control_daemon.feedback import atomic_write_json, command_result, robot_state


class FeedbackTests(unittest.TestCase):
    def test_command_result_shape(self):
        intent = {
            "request_id": "req-1",
            "intent": "MOVE",
            "params": {"vx": 0.0, "yaw": 0.6, "duration_ms": 600},
            "source": "test",
        }

        result = command_result(
            intent=intent,
            adapter_name="mock",
            status="completed",
            started_at="start",
            completed_at="end",
            steps=[{"name": "mock_move", "return_code": 0}],
        )

        self.assertEqual(result["schema_version"], "0.1")
        self.assertEqual(result["request_id"], "req-1")
        self.assertEqual(result["intent"], "MOVE")
        self.assertEqual(result["adapter"], "mock")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["steps"], [{"name": "mock_move", "return_code": 0}])

    def test_invalid_command_status_raises(self):
        with self.assertRaises(ValueError):
            command_result(intent=None, adapter_name="mock", status="unknown")

    def test_robot_state_shape(self):
        last_command = {"status": "completed", "intent": "STOP"}

        state = robot_state(adapter_name="mock", status="idle", last_command=last_command)

        self.assertEqual(state["schema_version"], "0.1")
        self.assertEqual(state["adapter"], "mock")
        self.assertEqual(state["status"], "idle")
        self.assertEqual(state["last_command"], last_command)

    def test_atomic_write_json(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "state.json"
            atomic_write_json(str(path), {"ok": True})

            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)

        self.assertEqual(payload, {"ok": True})


if __name__ == "__main__":
    unittest.main()
