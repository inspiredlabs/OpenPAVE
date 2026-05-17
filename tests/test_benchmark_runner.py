import importlib.util
import contextlib
import io
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_benchmark.py"


def load_runner_module():
    spec = importlib.util.spec_from_file_location("run_benchmark", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BenchmarkRunnerTests(unittest.TestCase):
    def setUp(self):
        self.runner = load_runner_module()

    def test_summarize_counts_pass_fail_and_latency(self):
        records = [
            {"benchmark": {"pass": True, "latency_ms": 10.0}},
            {"benchmark": {"pass": False, "latency_ms": 20.0}},
        ]

        summary = self.runner.summarize(records)

        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["passed"], 1)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["avg_latency_ms"], 15.0)

    def test_write_jsonl_writes_one_record_per_line(self):
        with tempfile.TemporaryDirectory() as tempdir:
            output = Path(tempdir) / "result.jsonl"
            self.runner.write_jsonl(output, [{"a": 1}, {"b": 2}])

            lines = output.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(lines), 2)
        self.assertIn('"a": 1', lines[0])
        self.assertIn('"b": 2', lines[1])

    def test_physical_scenario_requires_explicit_flag(self):
        with contextlib.redirect_stderr(io.StringIO()):
            result = self.runner.main(["scenarios/puppypi-gesture-stop-trot.json"])

        self.assertEqual(result, 2)

    def test_capture_runtime_env_keeps_known_openpave_keys(self):
        env = {
            "ROBOT_ADAPTER": "mock",
            "UI_MODEL": "model-a",
            "UNRELATED": "ignored",
        }

        captured = self.runner.capture_runtime_env(env)

        self.assertEqual(captured, {"ROBOT_ADAPTER": "mock", "UI_MODEL": "model-a"})

    def test_benchmark_records_include_runtime_env(self):
        def fake_post_intent(url, payload, timeout_sec):
            return 200, {"status": "ok"}

        def fake_wait_for_command_result(**kwargs):
            return {
                "request_id": kwargs["request_id"],
                "intent": "STOP",
                "status": "completed",
            }, self.runner.time.monotonic()

        self.runner.post_intent = fake_post_intent
        self.runner.wait_for_command_result = fake_wait_for_command_result
        self.runner.read_json_file = lambda path: {"state": "idle"}

        records = self.runner.benchmark_intents(
            scenario={"id": "mock", "prompt_ref": "prompts/intent-stop-trot.json"},
            prompt={"id": "prompt"},
            intents=["STOP"],
            intent_url="http://127.0.0.1:7071/intent",
            command_result_path="/tmp/result.json",
            robot_state_path="/tmp/state.json",
            timeout_sec=1.0,
            poll_sec=0.01,
            run_id="run-1",
            runtime_env={"ROBOT_ADAPTER": "mock", "UI_MODEL": "model-a"},
        )

        self.assertEqual(records[0]["runtime_env"]["ROBOT_ADAPTER"], "mock")
        self.assertEqual(records[0]["runtime_env"]["UI_MODEL"], "model-a")


if __name__ == "__main__":
    unittest.main()
