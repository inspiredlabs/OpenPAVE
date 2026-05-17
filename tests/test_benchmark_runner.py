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


if __name__ == "__main__":
    unittest.main()
