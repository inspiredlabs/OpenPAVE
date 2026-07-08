import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pave_mlx.backends import _summarize_vlm_load_error
from pave_mlx.downloads import MODEL_SIZE_GB, VLM_MODELS, model_download_report, print_model_download_report


class VlmDownloadPreflightTests(unittest.TestCase):
    def test_qwen_2b_is_registered_for_preflight(self):
        self.assertEqual(VLM_MODELS["Qwen3-VL 2B"], "qwen_2b")
        self.assertEqual(MODEL_SIZE_GB["Qwen3-VL 2B"], 1.57)

    def test_ab_candidates_are_registered_for_preflight(self):
        self.assertEqual(VLM_MODELS["Qwen3.5 2B (Rishu11277)"], "rishu_qwen35_2b")
        self.assertEqual(MODEL_SIZE_GB["Qwen3.5 2B (Rishu11277)"], 3.76)
        self.assertEqual(VLM_MODELS["Fourier Qwen2-VL 2B (mradermacher)"], "fourier_qwen2vl_2b")
        self.assertEqual(MODEL_SIZE_GB["Fourier Qwen2-VL 2B (mradermacher)"], 4.42)

    def test_gemma_report_uses_indexed_weight_total_and_config_shape(self):
        with tempfile.TemporaryDirectory() as tempdir, patch.dict(os.environ, {"HF_HOME": tempdir}):
            snapshot = self._gemma_snapshot(Path(tempdir), "abc123")
            (snapshot / "model-00001-of-00002.safetensors").write_bytes(b"a" * 100)
            (snapshot / "model-00002-of-00002.safetensors").write_bytes(b"b" * 200)

            report = model_download_report("Gemma 4 E4B")

        self.assertEqual(report["present_weight_bytes"], 300)
        self.assertEqual(report["expected_weight_bytes"], 300)
        self.assertEqual(report["missing_weight_bytes"], 0)
        self.assertEqual(report["missing"], [])
        self.assertEqual(report["config"]["model_type"], "gemma4")
        self.assertEqual(report["config"]["text_shared_kv_layers"], 18)

    def test_gemma_preflight_prints_shared_kv_compat_note(self):
        with tempfile.TemporaryDirectory() as tempdir, patch.dict(os.environ, {"HF_HOME": tempdir}):
            snapshot = self._gemma_snapshot(Path(tempdir), "def456", include_shared_kv_extras=True)
            (snapshot / "model-00001-of-00002.safetensors").write_bytes(b"a" * 100)
            (snapshot / "model-00002-of-00002.safetensors").write_bytes(b"b" * 200)
            stream = io.StringIO()

            with contextlib.redirect_stdout(stream):
                print_model_download_report("Gemma 4 E4B", stream=stream)

        output = stream.getvalue()
        self.assertIn("model shape  : gemma4 / gemma4_text / Gemma4ForConditionalGeneration", output)
        self.assertIn("42 text layers, 18 shared-KV layers", output)
        self.assertIn("loader gap", output)
        self.assertIn("OpenPAVE filters these extras at load_weights()", output)

    def test_parameter_mismatch_summary_says_runtime_not_cache(self):
        exc = ValueError(
            "Received 126 parameters not in model: "
            "language_model.model.layers.24.self_attn.k_norm.weight"
        )

        summary = _summarize_vlm_load_error("gemma", exc, ["Gemma 4 shared-KV sanitizer active"])

        self.assertIn("cache is complete", summary)
        self.assertIn("shared-KV weight shape", summary)

    def test_missing_snapshot_is_never_reported_complete(self):
        with tempfile.TemporaryDirectory() as tempdir, patch.dict(os.environ, {"HF_HOME": tempdir}):
            report = model_download_report("Gemma 4 E4B")
            stream = io.StringIO()

            print_model_download_report("Gemma 4 E4B", stream=stream)

        self.assertIn("snapshot", report["missing"])
        self.assertGreater(report["missing_weight_bytes"], 0)
        self.assertIn("MISSING snapshot", stream.getvalue())
        self.assertNotIn("cache        : complete", stream.getvalue())

    def test_lmstudio_cache_is_active_target(self):
        with tempfile.TemporaryDirectory() as tempdir, patch.dict(os.environ, {"HF_HOME": tempdir}):
            snapshot = self._gemma_snapshot(Path(tempdir), "ghi789")
            (snapshot / "model-00001-of-00002.safetensors").write_bytes(b"a" * 100)
            (snapshot / "model-00002-of-00002.safetensors").write_bytes(b"b" * 200)
            report = model_download_report("Gemma 4 E4B")

        self.assertEqual(report["repo"], "lmstudio-community/gemma-4-E4B-it-MLX-4bit")
        self.assertEqual(report["present_weight_bytes"], 300)
        self.assertEqual(report["missing"], [])

    def _gemma_snapshot(self, hf_home: Path, revision: str, include_shared_kv_extras: bool = False) -> Path:
        root = hf_home / "hub" / "models--lmstudio-community--gemma-4-E4B-it-MLX-4bit"
        snapshot = root / "snapshots" / revision
        snapshot.mkdir(parents=True)
        (root / "refs").mkdir()
        (root / "refs" / "main").write_text(revision, encoding="utf-8")
        (snapshot / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "gemma4",
                    "architectures": ["Gemma4ForConditionalGeneration"],
                    "text_config": {
                        "model_type": "gemma4_text",
                        "num_hidden_layers": 42,
                        "num_kv_shared_layers": 18,
                    },
                }
            ),
            encoding="utf-8",
        )
        (snapshot / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        weight_map = {
            "a": "model-00001-of-00002.safetensors",
            "b": "model-00002-of-00002.safetensors",
        }
        if include_shared_kv_extras:
            weight_map.update(
                {
                    "language_model.model.layers.24.self_attn.k_norm.weight": "model-00001-of-00002.safetensors",
                    "language_model.model.layers.24.self_attn.k_proj.weight": "model-00001-of-00002.safetensors",
                    "language_model.model.layers.24.self_attn.v_proj.weight": "model-00001-of-00002.safetensors",
                }
            )
        (snapshot / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": 300}, "weight_map": weight_map}),
            encoding="utf-8",
        )
        return snapshot


if __name__ == "__main__":
    unittest.main()
