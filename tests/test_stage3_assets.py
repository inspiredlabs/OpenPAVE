import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = ROOT / "prompts"
SCENARIOS_DIR = ROOT / "scenarios"


class Stage3AssetTests(unittest.TestCase):
    def load_json_files(self, directory):
        files = sorted(directory.glob("*.json"))
        self.assertGreater(len(files), 0, f"no JSON files found in {directory}")
        return [(path, json.loads(path.read_text(encoding="utf-8"))) for path in files]

    def test_prompt_presets_have_required_fields(self):
        for path, payload in self.load_json_files(PROMPTS_DIR):
            with self.subTest(path=path.name):
                for field in ["id", "version", "title", "task_type", "output_contract", "prompt"]:
                    self.assertIn(field, payload)
                self.assertIsInstance(payload["prompt"], str)
                self.assertTrue(payload["prompt"].strip())
                self.assertIsInstance(payload["output_contract"], dict)

    def test_scenarios_have_required_fields_and_valid_prompt_refs(self):
        for path, payload in self.load_json_files(SCENARIOS_DIR):
            with self.subTest(path=path.name):
                for field in [
                    "id",
                    "version",
                    "title",
                    "prompt_ref",
                    "safety_constraints",
                    "robot_sensor_endpoint",
                    "inference_node",
                    "adapter",
                    "success_criteria",
                ]:
                    self.assertIn(field, payload)

                prompt_ref = ROOT / payload["prompt_ref"]
                self.assertTrue(prompt_ref.exists(), f"missing prompt_ref: {prompt_ref}")
                self.assertEqual(prompt_ref.parent, PROMPTS_DIR)
                self.assertIsInstance(payload["success_criteria"], list)
                self.assertGreater(len(payload["success_criteria"]), 0)


if __name__ == "__main__":
    unittest.main()
