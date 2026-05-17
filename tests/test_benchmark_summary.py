import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "scripts" / "summarize_benchmarks.py"


def load_summary_module():
    spec = importlib.util.spec_from_file_location("summarize_benchmarks", SUMMARY_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BenchmarkSummaryTests(unittest.TestCase):
    def setUp(self):
        self.summary = load_summary_module()

    def test_get_dotted_returns_nested_values(self):
        record = {"scenario": {"id": "mock"}, "adapter": {"name": "mock"}}

        self.assertEqual(self.summary.get_dotted(record, "scenario.id"), "mock")
        self.assertEqual(self.summary.get_dotted(record, "adapter.name"), "mock")
        self.assertIsNone(self.summary.get_dotted(record, "missing.value"))

    def test_summarize_records_groups_by_requested_fields(self):
        records = [
            {
                "scenario": {"id": "same"},
                "adapter": {"name": "mock"},
                "inference_node": {"default_model": "model-a"},
                "benchmark": {"pass": True, "latency_ms": 10.0},
            },
            {
                "scenario": {"id": "same"},
                "adapter": {"name": "mock"},
                "inference_node": {"default_model": "model-a"},
                "benchmark": {"pass": False, "latency_ms": 30.0},
            },
            {
                "scenario": {"id": "same"},
                "adapter": {"name": "mock"},
                "inference_node": {"default_model": "model-b"},
                "benchmark": {"pass": True, "latency_ms": 20.0},
            },
        ]

        rows = self.summary.summarize_records(records, ["inference_node.default_model"])

        self.assertEqual(len(rows), 2)
        by_model = {row["group"]["inference_node.default_model"]: row for row in rows}
        self.assertEqual(by_model["model-a"]["summary"]["total"], 2)
        self.assertEqual(by_model["model-a"]["summary"]["passed"], 1)
        self.assertEqual(by_model["model-a"]["summary"]["avg_latency_ms"], 20.0)
        self.assertEqual(by_model["model-b"]["summary"]["pass_rate"], 1.0)

    def test_read_jsonl_loads_records(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "bench.jsonl"
            path.write_text('{"benchmark":{"pass":true,"latency_ms":1.0}}\n', encoding="utf-8")

            records = self.summary.read_jsonl([path])

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["benchmark"]["latency_ms"], 1.0)

    def test_format_text_includes_group_and_summary_columns(self):
        rows = [
            {
                "group": {"scenario.id": "mock", "adapter.name": "mock"},
                "summary": {
                    "total": 2,
                    "passed": 2,
                    "failed": 0,
                    "pass_rate": 1.0,
                    "avg_latency_ms": 12.5,
                    "min_latency_ms": 10.0,
                    "max_latency_ms": 15.0,
                },
            }
        ]

        output = self.summary.format_text(rows, ["scenario.id", "adapter.name"])

        self.assertIn("scenario.id\tadapter.name", output)
        self.assertIn("mock\tmock\t2\t2\t0\t100.00%\t12.5", output)

    def test_evaluate_thresholds_reports_pass_rate_and_latency_violations(self):
        rows = [
            {
                "group": {"scenario.id": "mock"},
                "summary": {
                    "total": 2,
                    "passed": 1,
                    "failed": 1,
                    "pass_rate": 0.5,
                    "avg_latency_ms": 250.0,
                },
            }
        ]

        violations = self.summary.evaluate_thresholds(
            rows,
            min_pass_rate=1.0,
            max_avg_latency_ms=200.0,
        )

        self.assertEqual(len(violations), 2)
        self.assertIn("pass_rate 50.00% < 100.00%", violations[0])
        self.assertIn("avg_latency_ms 250.0 > 200.0", violations[1])

    def test_evaluate_thresholds_passes_when_groups_meet_limits(self):
        rows = [
            {
                "group": {"scenario.id": "mock"},
                "summary": {
                    "total": 2,
                    "passed": 2,
                    "failed": 0,
                    "pass_rate": 1.0,
                    "avg_latency_ms": 100.0,
                },
            }
        ]

        violations = self.summary.evaluate_thresholds(
            rows,
            min_pass_rate=1.0,
            max_avg_latency_ms=200.0,
        )

        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
