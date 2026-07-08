import unittest

import numpy as np

from pave_mlx.backends import VllmMlxBackend


class VllmMlxBackendTests(unittest.TestCase):
    def _frame(self):
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def _server_backend(self):
        backend = VllmMlxBackend.__new__(VllmMlxBackend)
        backend.name = "qwen"
        backend.model_id = "example/model"
        backend.mode = "loaded"
        backend.load_error = ""
        backend._runtime = "vllm-mlx"          # server path
        backend._server_url = "http://127.0.0.1:8000"
        backend._served_model = "default"
        backend._api_key = "not-needed"
        backend._request_timeout = 1.0
        backend._image_transport = "data-url"
        backend._allow_remote_image_urls = False
        backend._frame_url_server = None
        backend.compat_notes = []
        backend._image_data_url = lambda _image: "data:image/jpeg;base64,ZmFrZQ=="
        return backend

    def test_runtime_name_sets(self):
        self.assertIn("vllm-mlx", VllmMlxBackend._SERVER_BACKEND_NAMES)
        self.assertIn("mlx-vlm", VllmMlxBackend._DIRECT_BACKEND_NAMES)

    def test_qwen_2b_backend_uses_mlx_community_checkpoint(self):
        from pave_mlx.backends import VLM_NAMES, backend_model_id

        self.assertIn("qwen_2b", VLM_NAMES)
        self.assertEqual(backend_model_id("qwen_2b"), "mlx-community/Qwen3-VL-2B-Instruct-3bit")

    def test_display_names_match_dropdown_labels(self):
        from pave_mlx.backends import _REGISTRY
        from pave_ui.perception import VLM_MODELS

        for label, key in VLM_MODELS.items():
            self.assertEqual(_REGISTRY[key].display_name, label)

    def test_timing_trace_includes_runtime_and_dropdown_model_name(self):
        import contextlib
        import io

        backend = self._server_backend()
        backend._timings_enabled = True
        backend.display_name = "Fourier Qwen2-VL 2B (mradermacher)"
        backend._request_json = lambda *a, **k: {
            "choices": [{"message": {"content": "INTENT: STOP"}}]
        }

        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            backend.generate(self._frame(), "INTENT contract", max_tokens=8)

        line = stream.getvalue()
        self.assertIn("[pave_mlx] [ vllm-mlx ] [ Fourier Qwen2-VL 2B (mradermacher) ] timings", line)
        self.assertIn("request_ms=", line)

    def test_rishu_qwen35_backend_uses_mlx_fp16_checkpoint(self):
        from pave_mlx.backends import VLM_NAMES, backend_model_id

        self.assertIn("rishu_qwen35_2b", VLM_NAMES)
        self.assertEqual(backend_model_id("rishu_qwen35_2b"), "Rishu11277/Qwen3.5-2B-mlx-fp16")

    def test_fourier_backend_serves_safetensors_source_not_gguf(self):
        # mradermacher's repo is GGUF-only, which MLX cannot load; the backend
        # must point at the safetensors source repo the quants were made from.
        from pave_mlx.backends import VLM_NAMES, backend_model_id

        self.assertIn("fourier_qwen2vl_2b", VLM_NAMES)
        self.assertEqual(backend_model_id("fourier_qwen2vl_2b"), "whyisverysmart/Fourier-Qwen2-VL-2B-0.67")
        self.assertNotIn("GGUF", backend_model_id("fourier_qwen2vl_2b"))

    def test_generate_sends_prompt_and_image_in_one_user_message(self):
        backend = self._server_backend()
        calls = []

        def fake_request(method, url, body, timeout=None):
            calls.append((method, url, body))
            return {"choices": [{"message": {"content": "INTENT: STOP\nFEATURE: hand center"}}]}

        backend._request_json = fake_request

        text = backend.generate(self._frame(), "INTENT contract", max_tokens=32)

        self.assertEqual(text, "INTENT: STOP\nFEATURE: hand center")
        method, url, body = calls[0]
        self.assertEqual(method, "POST")
        self.assertEqual(url, "http://127.0.0.1:8000/v1/chat/completions")
        self.assertEqual(body["model"], "default")
        self.assertEqual(body["max_tokens"], 32)
        self.assertEqual(body["temperature"], 0)
        # prompt + image in ONE user message (text first) — matches the working
        # 1414 baseline; see VllmMlxBackend.generate().
        self.assertEqual(len(body["messages"]), 1)
        message = body["messages"][0]
        self.assertEqual(message["role"], "user")
        self.assertEqual([part["type"] for part in message["content"]], ["text", "image_url"])
        self.assertEqual(message["content"][0]["text"], "INTENT contract")
        self.assertTrue(message["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    def test_http_image_transport_serves_frame_url_without_base64_payload(self):
        backend = self._server_backend()
        backend._image_transport = "http-url"
        backend._allow_remote_image_urls = True
        # Use the real image encoder for this transport test, but inject the URL
        # publisher so the unit test does not need to bind a localhost port.
        del backend._image_data_url
        published = []

        class FakeFrameUrlServer:
            def publish(self, jpeg):
                published.append(jpeg)
                return "http://127.0.0.1:8989/frame/1.jpg"

        backend._frame_url_server = FakeFrameUrlServer()
        calls = []

        def fake_request(method, url, body, timeout=None):
            calls.append((method, url, body))
            return {"choices": [{"message": {"content": "INTENT: STOP"}}]}

        backend._request_json = fake_request
        frame = self._frame()
        try:
            self.assertEqual(backend.generate(frame, "INTENT contract", max_tokens=8), "INTENT: STOP")
            image_url = calls[0][2]["messages"][0]["content"][1]["image_url"]["url"]
            self.assertEqual(image_url, "http://127.0.0.1:8989/frame/1.jpg")
            self.assertNotIn("base64", image_url)
            self.assertEqual(len(published), 1)
            self.assertTrue(published[0].startswith(b"\xff\xd8"))
        finally:
            backend.close()

    def test_http_image_transport_falls_back_when_server_rejects_remote_media(self):
        backend = self._server_backend()
        backend._image_transport = "http-url"
        backend._allow_remote_image_urls = True

        class FakeFrameUrlServer:
            def publish(self, _jpeg):
                return "http://127.0.0.1:8989/frame/1.jpg"

            def close(self):
                pass

        backend._frame_url_server = FakeFrameUrlServer()
        calls = []

        def fake_request(method, url, body, timeout=None):
            calls.append(body)
            image_url = body["messages"][0]["content"][1]["image_url"]["url"]
            if image_url.startswith("http://"):
                raise RuntimeError('HTTP 400 from vllm-mlx: {"detail":"Remote media URL is not allowed"}')
            return {"choices": [{"message": {"content": "INTENT: STOP"}}]}

        backend._request_json = fake_request

        self.assertEqual(backend.generate(self._frame(), "p", max_tokens=8), "INTENT: STOP")
        self.assertEqual(len(calls), 2)
        first_url = calls[0]["messages"][0]["content"][1]["image_url"]["url"]
        second_url = calls[1]["messages"][0]["content"][1]["image_url"]["url"]
        self.assertTrue(first_url.startswith("http://127.0.0.1:"))
        self.assertTrue(second_url.startswith("data:image/jpeg;base64,"))
        self.assertEqual(backend._image_transport, "data-url")
        self.assertIn("fell back to data-url", " ".join(backend.compat_notes))

    def test_http_image_transport_is_disabled_without_explicit_remote_url_opt_in(self):
        backend = self._server_backend()
        backend._image_transport = "http-url"
        calls = []

        def fake_request(method, url, body, timeout=None):
            calls.append(body)
            return {"choices": [{"message": {"content": "INTENT: STOP"}}]}

        backend._request_json = fake_request

        self.assertEqual(backend.generate(self._frame(), "p", max_tokens=8), "INTENT: STOP")
        image_url = calls[0]["messages"][0]["content"][1]["image_url"]["url"]
        self.assertTrue(image_url.startswith("data:image/jpeg;base64,"))
        self.assertEqual(backend._image_transport, "data-url")
        self.assertIn("disabled", " ".join(backend.compat_notes))

    def test_clean_generated_text_strips_leaked_reasoning_channel(self):
        cleaned = VllmMlxBackend._clean_generated_text(
            "<channel>thought image looks dark\nINTENT: STOP\nFEATURE: hand center"
        )
        self.assertEqual(cleaned, "INTENT: STOP\nFEATURE: hand center")

    def test_generate_raises_on_null_content(self):
        # A poisoned prefix-cache hit returns content: null + finish 'length'.
        # str(None) must never reach the parser (it would clamp to STOP forever);
        # this is an inference FAILURE, not an answer.
        backend = self._server_backend()
        backend._request_json = lambda *a, **k: {
            "choices": [{"message": {"content": None}, "finish_reason": "length"}]
        }
        with self.assertRaises(RuntimeError) as ctx:
            backend.generate(self._frame(), "INTENT contract", max_tokens=32)
        self.assertIn("length", str(ctx.exception))

    def test_generate_raises_on_empty_content(self):
        backend = self._server_backend()
        backend._request_json = lambda *a, **k: {
            "choices": [{"message": {"content": "  "}, "finish_reason": "stop"}]
        }
        with self.assertRaises(RuntimeError):
            backend.generate(self._frame(), "INTENT contract", max_tokens=32)

    def test_generate_falls_back_to_reasoning_content(self):
        backend = self._server_backend()
        backend._request_json = lambda *a, **k: {
            "choices": [{"message": {"content": None, "reasoning_content": "INTENT: TROT"}}]
        }
        self.assertEqual(backend.generate(self._frame(), "p", max_tokens=8), "INTENT: TROT")


class PrefixCacheGuardTests(unittest.TestCase):
    """_guard_prefix_cache must bypass the KV prefix cache for media token
    sequences (camera frames tokenize to IDENTICAL ids — cache collisions
    answer from a stale frame) while leaving text-only sequences cached."""

    class _FakeCache:
        def __init__(self):
            self.fetch_calls, self.store_calls = [], []

        def fetch(self, tokens):
            self.fetch_calls.append(list(tokens))
            return ["kv"], []

        def store(self, tokens, cache):
            self.store_calls.append(list(tokens))
            return True

    class _FakeGenerator:
        class _Model:
            class config:
                image_token_index = 99

        def __init__(self, cache):
            self.prefix_cache = cache
            self.model = self._Model()

    def _guarded(self):
        from pave_mlx.vllm_server import _guard_prefix_cache

        cache = self._FakeCache()
        generator = self._FakeGenerator(cache)
        self.assertTrue(_guard_prefix_cache(generator))
        return cache

    def test_media_tokens_bypass_fetch_and_store(self):
        cache = self._guarded()
        frame_ids = [1, 2, 3] + [99] * 5  # prompt text + image placeholder run
        self.assertEqual(cache.fetch(frame_ids), (None, frame_ids))
        self.assertFalse(cache.store(frame_ids, ["kv"]))
        self.assertEqual(cache.fetch_calls, [])
        self.assertEqual(cache.store_calls, [])

    def test_text_only_tokens_still_cached(self):
        cache = self._guarded()
        text_ids = [1, 2, 3, 4]
        self.assertEqual(cache.fetch(text_ids), (["kv"], []))
        self.assertTrue(cache.store(text_ids, ["kv"]))
        self.assertEqual(cache.fetch_calls, [text_ids])
        self.assertEqual(cache.store_calls, [text_ids])

    def test_unknown_model_config_leaves_cache_untouched(self):
        from pave_mlx.vllm_server import _guard_prefix_cache

        cache = self._FakeCache()
        generator = self._FakeGenerator(cache)
        generator.model = object()  # no config -> no media ids -> stock behavior
        _guard_prefix_cache(generator)
        ids = [1, 99, 2]
        self.assertEqual(cache.fetch(ids), (["kv"], []))
        self.assertEqual(cache.fetch_calls, [ids])


class VlmInputSizeTests(unittest.TestCase):
    def test_qwen_uses_smaller_input(self):
        from pave_ui.perception import VLM_INPUT_SIZE, VLM_INPUT_SIZE_QWEN, vlm_input_size

        self.assertEqual(vlm_input_size("Qwen3-VL"), VLM_INPUT_SIZE_QWEN)
        self.assertEqual(vlm_input_size("Gemma 4 E2B"), VLM_INPUT_SIZE)
        self.assertEqual(vlm_input_size(None), VLM_INPUT_SIZE)


if __name__ == "__main__":
    unittest.main()
