"""Perception backend adapters — the per-model code (responsibility C in §3.2).

Each adapter exposes one method, `embed(image_bgr) -> feature_vector`, wrapping a
real MLX engine from the sibling jepa app. DINOv3 is wired; V-JEPA and LingBot are
stubs. Every adapter degrades to a deterministic NumPy fallback featurizer when its
heavy stack (mlx, mlx-image, the jepa app) is unavailable, so the whole pipeline
stays runnable and testable anywhere — the same MLX-or-fallback philosophy the
template uses for its policy runtime.
"""

from __future__ import annotations

import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import threading
import time
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import warnings

import numpy as np

# Where the real MLX engines (dino_engine.py, engine.py, lingbot_*.py) live.
JEPA_APP_DIR = os.environ.get("JEPA_APP_DIR", "/Users/scottphillips/Documents/jepa")
_CV2_MODULE = None
_CV2_IMPORT_FAILED = False


def _cv2():
    global _CV2_MODULE, _CV2_IMPORT_FAILED
    if _CV2_IMPORT_FAILED:
        return None
    if _CV2_MODULE is not None:
        return _CV2_MODULE
    try:
        import cv2

        _CV2_MODULE = cv2
        return cv2
    except Exception:
        _CV2_IMPORT_FAILED = True
        return None


def _hf_cache_dir() -> Path:
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return home / "hub"


def _local_hf_snapshot(model_id: str) -> str | None:
    if "/" not in model_id:
        return model_id if Path(model_id).exists() else None
    root = _hf_cache_dir() / ("models--" + model_id.replace("/", "--"))
    snapshots = root / "snapshots"
    refs_main = root / "refs" / "main"
    if refs_main.is_file():
        snapshot = snapshots / refs_main.read_text(encoding="utf-8").strip()
        return str(snapshot) if _snapshot_complete(snapshot) else None
    if snapshots.exists():
        dirs = [path for path in snapshots.glob("*") if path.is_dir()]
        if dirs:
            snapshot = max(dirs, key=lambda path: path.stat().st_mtime)
            return str(snapshot) if _snapshot_complete(snapshot) else None
    return None


def _snapshot_complete(snapshot: Path) -> bool:
    if not snapshot.exists():
        return False
    if not (snapshot / "config.json").is_file():
        return False
    single_file = any(
        path.is_file() and path.suffix in {".safetensors", ".npz", ".bin"} for path in snapshot.iterdir()
    )
    index = snapshot / "model.safetensors.index.json"
    if index.is_file():
        try:
            data = json.loads(index.read_text(encoding="utf-8"))
        except Exception:
            return False
        weights = set(data.get("weight_map", {}).values())
        shards_present = bool(weights) and all((snapshot / name).is_file() for name in weights)
        # Accept a stale index that names shards but ships a single consolidated
        # model.safetensors (e.g. Qwen3-VL-*-3bit) — see missing_snapshot_files.
        return shards_present or single_file
    return single_file


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not installed"
    except Exception:
        return "unknown"


def _patch_gemma4_shared_kv_sanitize() -> bool:
    """Make mlx-vlm's Gemma 4 loader accept lmstudio-community checkpoints.

    Two checkpoint/loader gaps, both fixed at load time:
    1. Shared-KV extras — some Gemma 4 MLX checkpoints carry k/v projection and
       norm tensors for layers the small shared-KV architecture intentionally
       does not instantiate. The language model sanitizer knows how to drop
       them, but some mlx-vlm releases skip it for native-MLX safetensors,
       causing "parameters not in model" on otherwise complete caches.
    2. Audio-tower conv layout — mlx-vlm 0.6.4 expects audio conv weights with
       channels-last kernel axes (e.g. (128, 3, 3, 1)) while these checkpoints
       store the last two axes swapped ((128, 3, 1, 3)), failing the load with
       a shape error even though OpenPAVE never feeds audio. Transpose any
       audio weight whose swap exactly matches the instantiated shape.
    """
    try:
        from mlx_vlm.models.gemma4 import gemma4
    except Exception:
        return False

    model_cls = getattr(gemma4, "Model", None)
    if model_cls is None:
        return False
    if getattr(model_cls, "_openpave_shared_kv_sanitize", False):
        return True

    original_sanitize = getattr(model_cls, "sanitize", None)
    original_load_weights = getattr(model_cls, "load_weights", None)
    if not callable(original_sanitize) or not callable(original_load_weights):
        return False

    def fix_audio_conv_layout(self, weight_dict):
        try:
            import mlx.core as mx
            from mlx.utils import tree_flatten

            expected = {k: v.shape for k, v in tree_flatten(self.parameters())
                        if "audio" in k and getattr(v, "ndim", 0) >= 3}
        except Exception:
            return weight_dict
        for key, want in expected.items():
            have = weight_dict.get(key)
            if have is not None and have.ndim == len(want) and tuple(have.shape) != tuple(want) \
                    and tuple(mx.swapaxes(have, -2, -1).shape) == tuple(want):
                weight_dict[key] = mx.swapaxes(have, -2, -1)
        return weight_dict

    def filter_shared_kv(self, weights):
        as_items = not isinstance(weights, dict)
        weight_dict = dict(weights) if as_items else weights
        language_model = getattr(self, "language_model", None)
        language_sanitize = getattr(language_model, "sanitize", None)
        if callable(language_sanitize):
            weight_dict = language_sanitize(weight_dict)
        weight_dict = fix_audio_conv_layout(self, weight_dict)
        return list(weight_dict.items()) if as_items else weight_dict

    def sanitize_with_shared_kv_filter(self, weights):
        sanitized = original_sanitize(self, weights)
        return filter_shared_kv(self, sanitized)

    def load_weights_with_shared_kv_filter(self, weights, *args, **kwargs):
        return original_load_weights(self, filter_shared_kv(self, weights), *args, **kwargs)

    model_cls.sanitize = sanitize_with_shared_kv_filter
    model_cls.load_weights = load_weights_with_shared_kv_filter
    model_cls._openpave_shared_kv_sanitize = True
    return True

def _patch_bpe_streaming_detokenizer_utf8() -> bool:
    """Make mlx-vlm's BPE streaming detokenizer survive split UTF-8 sequences.

    Qwen3.5's byte-level BPE can leave a partial multibyte character in the
    detokenizer's unflushed buffer when the next token starts with a space.
    mlx-vlm 0.6.4's add_token() flushes that buffer with a STRICT utf-8 decode
    and crashes (UnicodeDecodeError) mid-stream — its own finalize() decodes
    the same buffer with errors="ignore". Mirror finalize()'s recovery here.
    Affects both serving tiers: vllm-mlx delegates to the same stream_generate.
    """
    try:
        from mlx_vlm import tokenizer_utils
    except Exception:
        return False

    cls = getattr(tokenizer_utils, "BPEStreamingDetokenizer", None)
    if cls is None or not callable(getattr(cls, "add_token", None)):
        return False
    if getattr(cls, "_openpave_utf8_safe", False):
        return True

    original_add_token = cls.add_token
    remove_space = getattr(tokenizer_utils, "_remove_space", lambda s: s.lstrip())

    def add_token_utf8_safe(self, token, *args, **kwargs):
        try:
            return original_add_token(self, token, *args, **kwargs)
        except UnicodeDecodeError:
            # same flush add_token was attempting, with finalize()'s tolerance
            current_text = bytearray(
                self._byte_decoder[c] for c in self._unflushed
            ).decode("utf-8", errors="ignore")
            if self.text or not self.trim_space:
                self.text += current_text
            else:
                self.text += remove_space(current_text)
            self._unflushed = self.tokenmap[token]

    cls.add_token = add_token_utf8_safe
    cls._openpave_utf8_safe = True
    return True


# don't guess what the API looks like, it's all in here: https://huggingface.co/lmstudio-community/gemma-4-E4B-it-MLX-4bit
def _summarize_vlm_load_error(model_name: str, exc: Exception, compat_notes: list[str]) -> str:
    message = str(exc)
    if "parameters not in model" in message:
        if model_name == "gemma":
            return (
                "runtime mismatch: Gemma 4 cache is complete, but mlx-vlm rejected "
                "the shared-KV weight shape"
            )
        return "runtime mismatch: cache is complete, but mlx-vlm rejected the model shape"
    if "No Metal device" in message or "Metal device" in message:
        return "runtime unavailable: MLX cannot see an Apple Metal device"
    if "No such file" in message or "not found" in message.lower():
        return "cache incomplete: required model files are missing"
    if compat_notes:
        return "runtime load failed after " + "; ".join(compat_notes)
    return f"runtime load failed: {type(exc).__name__}"


def _format_vlm_load_error(model_name: str, model_id: str, exc: Exception, compat_notes: list[str]) -> str:
    summary = _summarize_vlm_load_error(model_name, exc, compat_notes)
    mlx_vlm_version = _package_version("mlx-vlm")
    notes = f"; {'; '.join(compat_notes)}" if compat_notes else ""
    detail = f"{type(exc).__name__}: {exc}"
    if "parameters not in model" in str(exc):
        return (
            f"{summary} ({model_id}, mlx-vlm {mlx_vlm_version}{notes}). "
            f"Detail: {detail}"
        )
    return f"{summary} ({model_id}, mlx-vlm {mlx_vlm_version}{notes}). Detail: {detail}"


def _resize_bgr(image_bgr: np.ndarray, size: int) -> np.ndarray:
    """Dependency-free nearest-neighbour resize to size×size×3."""
    img = np.asarray(image_bgr)
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    h, w = img.shape[:2]
    ys = np.linspace(0, h - 1, size).astype(np.int64)
    xs = np.linspace(0, w - 1, size).astype(np.int64)
    return img[ys][:, xs]


class PerceptionBackend(Protocol):
    name: str
    feature_dim: int
    mode: str  # "<engine>" when the real model is loaded, else "fallback"

    def embed(self, image_bgr: np.ndarray) -> np.ndarray: ...


class DinoBackend:
    """DINOv3 (vit_small_patch16_224.dinov3) -> 384-d patch-pooled embedding."""

    name = "dino"
    feature_dim = 384
    model_id = "vit_small_patch16_224.dinov3"

    def __init__(self) -> None:
        self.mode = "fallback"
        self.load_error = ""
        self._engine = None
        self._proj: np.ndarray | None = None
        try:
            self._engine = self._load_real_engine()
            self.mode = "dinov3"
        except Exception as exc:  # noqa: BLE001 - any import/Metal failure -> fallback
            self.load_error = f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _load_real_engine():
        if JEPA_APP_DIR and JEPA_APP_DIR not in sys.path:
            sys.path.insert(0, JEPA_APP_DIR)
        from dino_engine import DinoV3InferenceEngine  # heavy: mlx + mlx-image

        return DinoV3InferenceEngine()

    def embed(self, image_bgr: np.ndarray) -> np.ndarray:
        if self._engine is not None:
            # Engine expects a 224×224 frame and pools no resize itself.
            frame = _resize_bgr(image_bgr, self._engine.img_size).astype(np.uint8)
            cube = self._engine.process_sequence(frame[None, ...])  # [1,1,14,14,384]
            feats = np.asarray(cube, dtype=np.float32).reshape(-1, self.feature_dim)
            return feats.mean(axis=0)  # mean over patch grid -> 384-d
        return self._fallback_embed(image_bgr)

    def _fallback_embed(self, image_bgr: np.ndarray) -> np.ndarray:
        """Deterministic 384-d projection of a 12×12 thumbnail (no real model)."""
        g = _resize_bgr(image_bgr, 12).astype(np.float32) / 255.0
        v = g.reshape(-1)  # 12*12*3 = 432
        if self._proj is None:
            rng = np.random.default_rng(0)
            self._proj = rng.standard_normal((v.shape[0], self.feature_dim)).astype(np.float32)
        return (v @ self._proj).astype(np.float32)


class _StubBackend:
    """Common shell for not-yet-wired backends."""

    name = "stub"
    feature_dim = 0
    model_id = "stub"

    def __init__(self) -> None:
        self.mode = "stub"
        self.load_error = "not wired yet"

    def embed(self, image_bgr: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            f"{self.name} backend is not wired yet (see docs/dgx-spark-mlx-port.md §3.2)."
        )


class VJepaBackend(_StubBackend):
    """V-JEPA 2.1 -> 768-d pooled spatiotemporal embedding (STUB).

    When wired: load engine.VJepaInferenceEngine, feed a frame *window*, pool the
    [1,T,24,24,768] cube to 768-d, and reuse the same EmbeddingProbe as DINOv3.
    """

    name = "vjepa"
    feature_dim = 768
    model_id = "vjepa2_1_vit_base_384"


class LingBotBackend(_StubBackend):
    """LingBot-Map -> point cloud (STUB). Pairs with the geometry head, not the probe."""

    name = "lingbot"
    feature_dim = 0
    model_id = "lingbot_pointcloud"


class MlxVlmBackend:
    """Base for MLX vision-language models (Qwen3-VL, Gemma 4 E4B).

    Unlike the encoder backends these generate *text*, so the shim calls
    `generate()` directly and skips the intent head. mlx-vlm downloads the (~5GB)
    weights on first load and prints download progress to stderr — which the GUI
    parses for a percentage. Falls back gracefully when mlx-vlm/weights are absent.
    """

    is_vlm = True
    feature_dim = 0
    name = "vlm"
    model_id = ""

    def __init__(self) -> None:
        self.mode = "fallback"
        self.load_error = ""
        self.load_status = ""
        self.compat_notes: list[str] = []
        self._model = self._processor = self._config = None
        self._generate = self._apply = None
        try:
            from mlx_vlm import generate, load
            from mlx_vlm.prompt_utils import apply_chat_template

            try:
                from mlx_vlm.utils import load_config
            except Exception:
                load_config = None
            if self.name == "gemma" and _patch_gemma4_shared_kv_sanitize():
                self.compat_notes.append("Gemma 4 shared-KV load filter active")
            if _patch_bpe_streaming_detokenizer_utf8():
                self.compat_notes.append("utf-8-safe BPE streaming detokenizer")
            self._generate, self._apply = generate, apply_chat_template
            load_target = _local_hf_snapshot(self.model_id) or self.model_id
            if load_target != self.model_id:
                self.compat_notes.append("using local HF snapshot")
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="At least one mel filter has all zero values.*",
                    category=UserWarning,
                    module="transformers.audio_utils",
                )
                self._model, self._processor = load(load_target)
                self._config = load_config(load_target) if load_config else getattr(self._processor, "config", None)
            self.mode = "loaded"
            self.load_status = "; ".join(self.compat_notes)
        except Exception as exc:  # noqa: BLE001
            self.load_status = _summarize_vlm_load_error(self.name, exc, self.compat_notes)
            self.load_error = _format_vlm_load_error(self.name, self.model_id, exc, self.compat_notes)

    def embed(self, image_bgr: np.ndarray) -> np.ndarray:
        raise NotImplementedError(f"{self.name} is a VLM; the shim calls generate(), not embed()")

    def generate(self, image_bgr: np.ndarray, prompt: str, max_tokens: int = 12) -> str:
        if self._model is None:
            raise RuntimeError(f"{self.name} model not loaded")
        from PIL import Image

        pil = Image.fromarray(np.ascontiguousarray(image_bgr[:, :, ::-1]).astype(np.uint8))
        formatted = self._apply(self._processor, self._config, prompt, num_images=1)
        out = self._generate(self._model, self._processor, formatted, [pil], verbose=False, max_tokens=max_tokens)
        return out if isinstance(out, str) else getattr(out, "text", str(out))


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _find_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


class _FrameUrlServer:
    """Tiny local JPEG server for base64-free OpenAI image_url requests."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0, keep: int = 8) -> None:
        self.host = host
        self.port = int(port)
        self.keep = max(1, int(keep))
        self._frames: dict[int, bytes] = {}
        self._lock = threading.Lock()
        self._seq = 0
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._httpd is not None:
            return
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - stdlib callback name
                parsed = urlparse(self.path)
                prefix = "/frame/"
                if not parsed.path.startswith(prefix) or not parsed.path.endswith(".jpg"):
                    self.send_error(404)
                    return
                raw_id = parsed.path[len(prefix):-4]
                try:
                    frame_id = int(raw_id)
                except ValueError:
                    self.send_error(404)
                    return
                data = owner.get(frame_id)
                if data is None:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, _format: str, *_args) -> None:
                return

        self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self.port = int(self._httpd.server_address[1])
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="openpave-vllm-frame-url",
            daemon=True,
        )
        self._thread.start()

    def publish(self, jpeg: bytes) -> str:
        self.start()
        with self._lock:
            self._seq += 1
            frame_id = self._seq
            self._frames[frame_id] = bytes(jpeg)
            stale = sorted(self._frames)[:-self.keep]
            for key in stale:
                self._frames.pop(key, None)
        return f"http://{self.host}:{self.port}/frame/{frame_id}.jpg"

    def get(self, frame_id: int) -> bytes | None:
        with self._lock:
            return self._frames.get(frame_id)

    def close(self) -> None:
        httpd = self._httpd
        self._httpd = None
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()


class VllmMlxBackend(MlxVlmBackend):
    """vLLM-MLX serving backend for live VLM inference on Apple Silicon.

    The older :class:`MlxVlmBackend` calls ``mlx-vlm`` in-process, one request at a
    time. This adapter starts ``waybarrios/vllm-mlx`` as a local OpenAI-compatible
    server so OpenPAVE gets continuous batching, paged KV cache, prefix caching,
    and optional SSD-tiered KV cache without changing the UI's ``generate()``
    contract.
    """

    _SERVER_BACKEND_NAMES = {"vllm-mlx", "vllm_mlx", "vllm"}
    _DIRECT_BACKEND_NAMES = {"mlx-vlm", "mlx_vlm", "direct", "inprocess", "in-process"}
    # Whether this model's VISION path is known-good under vllm-mlx. False pins the
    # model to the direct mlx-vlm path (see __init__). Gemma 4 sets this False.
    supports_vllm = True

    @property
    def display_name(self) -> str:
        """Checkpoint basename ("Qwen3-VL-4B-Instruct-3bit") — the one title
        used by the dropdown, the preflight, and the timing trace, so A/B runs
        in the logs match the HF cache directory names on disk."""
        return str(getattr(self, "model_id", "default")).rstrip("/").split("/")[-1]

    def __init__(self) -> None:
        self.mode = "fallback"
        self.load_error = ""
        self.load_status = ""
        self.compat_notes: list[str] = []
        self._model = self._processor = self._config = None
        self._generate = self._apply = None
        self._proc: subprocess.Popen | None = None
        self._server_logs: list[str] = []
        self._server_url = ""
        self._served_model = os.environ.get("PAVE_VLLM_MLX_MODEL_NAME", "default")
        self._api_key = os.environ.get("PAVE_VLLM_MLX_API_KEY", "not-needed")
        self._request_timeout = float(os.environ.get("PAVE_VLLM_MLX_TIMEOUT_S", "120"))
        self._image_transport = os.environ.get("PAVE_VLLM_MLX_IMAGE_TRANSPORT", "data-url").strip().lower()
        self._allow_remote_image_urls = _env_flag("PAVE_VLLM_MLX_ALLOW_REMOTE_IMAGE_URLS", False)
        self._timings_enabled = _env_flag("PAVE_VLLM_MLX_TIMINGS", False)
        self._last_timings: dict[str, float] = {}
        self._frame_url_server: _FrameUrlServer | None = None
        # Runtime resolution, most explicit first: the UI's Runtime selector
        # (set_runtime_override), then PAVE_VLM_BACKEND/PAVE_VLM_RUNTIME, then
        # the model's measured default. supports_vllm=False is that DEFAULT for
        # models whose vision path misbehaves under vllm-mlx (Gemma 4: garbage
        # on image input; Qwen3.5: reasoning-mode burns the token budget) — an
        # explicit choice overrides it so both runtimes stay A/B-testable.
        self._runtime = planned_runtime(self.name)

        if self._runtime in self._DIRECT_BACKEND_NAMES:
            MlxVlmBackend.__init__(self)
            if self.mode == "loaded":
                self.load_status = "; ".join(filter(None, ["legacy mlx-vlm direct backend", self.load_status]))
            return
        if self._runtime not in self._SERVER_BACKEND_NAMES:
            self.load_error = f"unknown PAVE_VLM_BACKEND={self._runtime!r}; use vllm-mlx or mlx-vlm"
            self.load_status = "VLM runtime not configured"
            return

        try:
            self._server_url = self._ensure_server()
            self.mode = "loaded"
            self.load_status = "; ".join(filter(None, self.compat_notes))
        except Exception as exc:  # noqa: BLE001 - unavailable server -> graceful fallback
            self.close()
            self.load_status = "vllm-mlx unavailable"
            self.load_error = self._format_server_error(exc)

    def _format_server_error(self, exc: Exception) -> str:
        details = f"{type(exc).__name__}: {exc}"
        if self._server_logs:
            tail = " | ".join(self._server_logs[-8:])
            details = f"{details}; server log: {tail}"
        return (
            f"vllm-mlx server failed for {self.model_id}. "
            f"Install with `pip install vllm-mlx`, or set PAVE_VLM_BACKEND=mlx-vlm "
            f"for the old direct backend. Detail: {details}"
        )

    def _ensure_server(self) -> str:
        external = os.environ.get("PAVE_VLLM_MLX_URL") or os.environ.get("PAVE_VLLM_MLX_BASE_URL")
        if external:
            root = self._normalise_server_root(external)
            self.compat_notes.append(f"using external vllm-mlx server {root}")
            self._wait_until_ready(root, startup=False)
            return root

        import importlib.util

        if importlib.util.find_spec("vllm_mlx") is None:
            raise RuntimeError("Python package vllm_mlx is not installed")

        # Local-dir model ids (env overrides pointing at mlx_vlm convert output)
        # must be absolutized: vllm-mlx treats any string it can't stat as a HF
        # repo id and fails with a misleading "Failed to download ./models/..."
        model_arg = self.model_id
        if model_arg.startswith((".", "/", "~")):
            local = Path(model_arg).expanduser()
            if not local.is_dir():
                raise RuntimeError(
                    f"local model path {model_arg} does not exist — run the "
                    "mlx_vlm convert step first (see the backend's comment)"
                )
            model_arg = str(local.resolve())

        host = os.environ.get("PAVE_VLLM_MLX_HOST", "127.0.0.1")
        port = int(os.environ.get("PAVE_VLLM_MLX_PORT") or _find_free_port(host))
        low_latency = _env_flag("PAVE_VLLM_MLX_LOW_LATENCY", False)
        root = f"http://{host}:{port}"
        cmd = [
            sys.executable, "-m", "pave_mlx.vllm_server",
            "serve", model_arg,
            "--host", host,
            "--port", str(port),
            "--served-model-name", self._served_model,
        ]
        if _env_flag("PAVE_VLLM_MLX_CONTINUOUS_BATCHING", not low_latency):
            cmd.append("--continuous-batching")
        if _env_flag("PAVE_VLLM_MLX_PAGED_CACHE", not low_latency):
            cmd.append("--use-paged-cache")
        if _env_flag("PAVE_VLLM_MLX_PREFIX_CACHE", not low_latency):
            cmd.append("--enable-prefix-cache")
            prefix_cache_size = os.environ.get("PAVE_VLLM_MLX_PREFIX_CACHE_SIZE")
            if prefix_cache_size:
                cmd.extend(["--prefix-cache-size", prefix_cache_size])
        else:
            # the server default is ENABLED; omitting --enable-prefix-cache
            # does not turn it off
            cmd.append("--disable-prefix-cache")
        if _env_flag("PAVE_VLLM_MLX_MLLM_CACHE", False):
            cmd.append("--enable-mllm-cache")
            mllm_cache_mb = os.environ.get("PAVE_VLLM_MLX_MLLM_CACHE_MB")
            if mllm_cache_mb:
                cmd.extend(["--mllm-cache-max-mb", mllm_cache_mb])
        if _env_flag("PAVE_VLLM_MLX_METRICS", False):
            cmd.append("--enable-metrics")

        ssd_cache_dir = os.environ.get(
            "PAVE_VLLM_MLX_SSD_CACHE_DIR",
            str(Path.home() / ".cache" / "openpave" / "vllm-mlx-kv"),
        ).strip()
        if ssd_cache_dir:
            cmd.extend(["--ssd-cache-dir", ssd_cache_dir])
            self.compat_notes.append(f"ssd KV cache {ssd_cache_dir}")

        cache_mb = os.environ.get("PAVE_VLLM_MLX_CACHE_MEMORY_MB")
        if cache_mb:
            cmd.extend(["--cache-memory-mb", cache_mb])
        cache_percent = os.environ.get("PAVE_VLLM_MLX_CACHE_MEMORY_PERCENT")
        if cache_percent:
            cmd.extend(["--cache-memory-percent", cache_percent])
        extra_args = os.environ.get("PAVE_VLLM_MLX_EXTRA_ARGS", "").strip()
        if extra_args:
            cmd.extend(shlex.split(extra_args))

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        threading.Thread(target=self._collect_server_logs, name=f"{self.name}-vllm-mlx-log", daemon=True).start()
        if low_latency:
            self.compat_notes.append("vllm-mlx serve: low-latency single-request mode")
        else:
            self.compat_notes.append("vllm-mlx serve: throughput mode")
        self.compat_notes.append(f"OpenAI endpoint {root}/v1")
        self._wait_until_ready(root, startup=True)
        return root

    @staticmethod
    def _normalise_server_root(url: str) -> str:
        root = url.strip().rstrip("/")
        if root.endswith("/v1"):
            root = root[:-3]
        return root

    def _collect_server_logs(self) -> None:
        stream = self._proc.stdout if self._proc is not None else None
        if stream is None:
            return
        for line in stream:
            text = line.strip()
            if text:
                self._server_logs.append(text)
                del self._server_logs[:-80]

    def _wait_until_ready(self, root: str, startup: bool) -> None:
        timeout = float(os.environ.get("PAVE_VLLM_MLX_STARTUP_TIMEOUT_S", "900" if startup else "20"))
        deadline = time.time() + timeout
        last_error = ""
        while time.time() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(f"vllm-mlx exited with code {self._proc.returncode}")
            try:
                data = self._request_json("GET", f"{root}/v1/models", None, timeout=2.0)
                models = data.get("data") or []
                if models and not os.environ.get("PAVE_VLLM_MLX_MODEL_NAME"):
                    self._served_model = str(models[0].get("id") or self._served_model)
                return
            except Exception as exc:  # noqa: BLE001 - keep polling while the model loads
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(0.5)
        raise TimeoutError(f"vllm-mlx did not become ready at {root}/v1/models ({last_error})")

    def _request_json(self, method: str, url: str, body: dict | None, timeout: float | None = None) -> dict:
        payload = None if body is None else json.dumps(body).encode("utf-8")
        headers = {"Accept": "application/json"}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        req = Request(url, data=payload, headers=headers, method=method)
        try:
            with urlopen(req, timeout=self._request_timeout if timeout is None else timeout) as resp:
                raw = resp.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise RuntimeError(f"HTTP {exc.code} from vllm-mlx: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(str(exc.reason)) from exc
        return json.loads(raw.decode("utf-8") or "{}")

    @staticmethod
    def _image_jpeg_bytes(image_bgr: np.ndarray) -> bytes:
        quality = int(os.environ.get("PAVE_VLLM_MLX_JPEG_QUALITY", "80"))
        encoder = os.environ.get("PAVE_VLLM_MLX_JPEG_ENCODER", "auto").strip().lower()
        if encoder in {"auto", "cv2", "opencv"}:
            cv2 = _cv2()
            if cv2 is not None:
                frame = image_bgr if image_bgr.dtype == np.uint8 else image_bgr.astype(np.uint8)
                frame = np.ascontiguousarray(frame)
                ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                if ok:
                    return encoded.tobytes()
            elif encoder in {"cv2", "opencv"}:
                raise RuntimeError("OpenCV JPEG encoder requested but cv2 is unavailable")

        from PIL import Image

        rgb = np.ascontiguousarray(image_bgr[:, :, ::-1]).astype(np.uint8)
        image = Image.fromarray(rgb)
        buf = io.BytesIO()
        optimize = _env_flag("PAVE_VLLM_MLX_JPEG_OPTIMIZE", False)
        image.save(buf, format="JPEG", quality=quality, optimize=optimize)
        return buf.getvalue()

    @staticmethod
    def _image_data_url(image_bgr: np.ndarray) -> str:
        encoded = base64.b64encode(VllmMlxBackend._image_jpeg_bytes(image_bgr)).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    @staticmethod
    def _image_data_url_from_jpeg(jpeg: bytes) -> str:
        encoded = base64.b64encode(jpeg).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def _image_url(self, image_bgr: np.ndarray, timings: dict[str, float] | None = None) -> str:
        transport = getattr(self, "_image_transport", "data-url")
        t_jpeg = time.perf_counter()
        jpeg = self._image_jpeg_bytes(image_bgr)
        if timings is not None:
            timings["jpeg_ms"] = (time.perf_counter() - t_jpeg) * 1000.0
            timings["jpeg_kb"] = len(jpeg) / 1000.0
        if transport in {"data-url", "data_url", "base64"}:
            t_b64 = time.perf_counter()
            url = self._image_data_url_from_jpeg(jpeg)
            if timings is not None:
                timings["base64_ms"] = (time.perf_counter() - t_b64) * 1000.0
                timings["payload_kb"] = len(url) / 1000.0
            return url
        if transport in {"http-url", "http_url", "url"}:
            if not getattr(self, "_allow_remote_image_urls", False):
                compat_notes = getattr(self, "compat_notes", None)
                note = "http-url image transport disabled; vllm-mlx rejects remote media URLs"
                if compat_notes is not None and note not in compat_notes:
                    compat_notes.append(note)
                self._image_transport = "data-url"
                t_b64 = time.perf_counter()
                url = self._image_data_url_from_jpeg(jpeg)
                if timings is not None:
                    timings["base64_ms"] = (time.perf_counter() - t_b64) * 1000.0
                    timings["payload_kb"] = len(url) / 1000.0
                return url
            server = getattr(self, "_frame_url_server", None)
            if server is None:
                host = os.environ.get("PAVE_VLLM_MLX_FRAME_HOST", "127.0.0.1")
                port = int(os.environ.get("PAVE_VLLM_MLX_FRAME_PORT", "0"))
                keep = int(os.environ.get("PAVE_VLLM_MLX_FRAME_KEEP", "8"))
                server = _FrameUrlServer(host=host, port=port, keep=keep)
                self._frame_url_server = server
                compat_notes = getattr(self, "compat_notes", None)
                if compat_notes is not None:
                    compat_notes.append("experimental remote frame URLs enabled")
            return server.publish(jpeg)
        raise RuntimeError(
            f"unknown PAVE_VLLM_MLX_IMAGE_TRANSPORT={transport!r}; "
            "use data-url, or http-url with PAVE_VLLM_MLX_ALLOW_REMOTE_IMAGE_URLS=1"
        )

    def _close_frame_url_server(self) -> None:
        frame_server = getattr(self, "_frame_url_server", None)
        self._frame_url_server = None
        if frame_server is not None:
            close_frame_server = getattr(frame_server, "close", None)
            if callable(close_frame_server):
                close_frame_server()

    def _fallback_to_data_url(self, reason: Exception) -> None:
        self._image_transport = "data-url"
        self._close_frame_url_server()
        compat_notes = getattr(self, "compat_notes", None)
        if compat_notes is not None:
            note = "http-url image transport rejected by vllm-mlx; fell back to data-url"
            if note not in compat_notes:
                compat_notes.append(note)

    @staticmethod
    def _remote_media_blocked(exc: Exception) -> bool:
        text = str(exc).lower()
        return "remote media url is not allowed" in text or ("remote media" in text and "not allowed" in text)

    def _chat_body(self, image_url: str, prompt: str, max_tokens: int) -> dict:
        return {
            "model": self._served_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
            "max_tokens": int(max_tokens),
            "temperature": 0,
        }

    @staticmethod
    def _message_text(message_content) -> str:
        if isinstance(message_content, str):
            return message_content
        if isinstance(message_content, list):
            parts = []
            for part in message_content:
                if isinstance(part, dict):
                    parts.append(str(part.get("text") or part.get("content") or ""))
                else:
                    parts.append(str(part))
            return " ".join(p for p in parts if p)
        return str(message_content)

    @staticmethod
    def _clean_generated_text(text: str) -> str:
        """Drop leaked reasoning-channel text before UI parsing/display.

        Some vllm-mlx/mlx-vlm model combinations can surface internal channel
        markers such as ``<channel>thought ...`` in ``message.content`` instead of
        a separate reasoning field. The OpenPAVE contract only consumes the final
        structured answer, so remove those leaked thought spans if they appear.
        """
        cleaned = str(text or "")
        cleaned = cleaned.replace("<|channel|>final", "").replace("<channel>final", "")
        cleaned = re.sub(
            r"<\|?channel\|?>\s*thought\b.*?(?=<\|?channel\|?>\s*final\b|INTENT:|FEATURE:|$)",
            "",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )
        cleaned = re.sub(
            r"<\|?channel\|?>\s*(analysis|commentary|thought|final)\b",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        return cleaned.strip()

    def generate(self, image_bgr: np.ndarray, prompt: str, max_tokens: int = 12) -> str:
        if self._runtime in self._DIRECT_BACKEND_NAMES:
            return MlxVlmBackend.generate(self, image_bgr, prompt, max_tokens=max_tokens)
        if not self._server_url:
            raise RuntimeError(self.load_error or "vllm-mlx server is not loaded")
        timings: dict[str, float] = {}
        t_total = time.perf_counter()
        image_url = self._image_url(image_bgr, timings)
        body = self._chat_body(image_url, prompt, max_tokens)
        try:
            t_request = time.perf_counter()
            data = self._request_json("POST", f"{self._server_url}/v1/chat/completions", body)
            timings["request_ms"] = (time.perf_counter() - t_request) * 1000.0
        except RuntimeError as exc:
            if image_url.startswith("http://") and self._remote_media_blocked(exc):
                self._fallback_to_data_url(exc)
                t_b64 = time.perf_counter()
                body = self._chat_body(self._image_data_url(image_bgr), prompt, max_tokens)
                timings["fallback_base64_ms"] = (time.perf_counter() - t_b64) * 1000.0
                t_request = time.perf_counter()
                data = self._request_json("POST", f"{self._server_url}/v1/chat/completions", body)
                timings["request_ms"] = (time.perf_counter() - t_request) * 1000.0
            else:
                raise
        timings["total_ms"] = (time.perf_counter() - t_total) * 1000.0
        self._last_timings = timings
        if getattr(self, "_timings_enabled", False):
            print(
                f"[pave_mlx] [ {self._runtime} ] [ {self.display_name} ] timings "
                + " ".join(f"{key}={value:.1f}" for key, value in timings.items()),
                flush=True,
            )
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"vllm-mlx response had no choices: {data!r}")
        message = choices[0].get("message") or {}
        content = message.get("content")
        # content can be JSON null (observed on poisoned prefix-cache hits);
        # str(None) would parse as a "None" reply and silently command STOP.
        text = self._message_text(content) if content is not None else ""
        if not text and message.get("reasoning_content"):
            text = self._message_text(message.get("reasoning_content", ""))
        text = self._clean_generated_text(text)
        if not text:
            finish = choices[0].get("finish_reason")
            raise RuntimeError(
                f"vllm-mlx returned empty content (finish_reason={finish!r}); "
                f"treating as inference failure, not an answer"
            )
        return text

    def close(self) -> None:
        self._close_frame_url_server()
        proc = getattr(self, "_proc", None)
        self._proc = None
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)

    def __del__(self) -> None:  # pragma: no cover - best-effort process cleanup
        try:
            self.close()
        except Exception:
            pass


# ── VLM backends — one per HF cache dir, measured verdicts inline ──────────
# Every entry names the exact ~/.cache/huggingface/hub directory it loads from
# and what it measured on the 12-image HaGRID gesture set, so a disk audit can
# be read straight off this file.

class QwenVLMBackend(VllmMlxBackend):
    # cache: models--mlx-community--Qwen3-VL-4B-Instruct-3bit (2.4 GB) — KEEP.
    # The reference model: vllm-mlx-verified, ~0.9s/frame warm @448px, coherent
    # INTENT + FEATURE output, best structured-reply quality of the Qwen3-VL
    # quants. (Gemma 4's vision path is broken under vllm-mlx, so this is the
    # serving tier's default.) Native video pipeline.
    name = "qwen"
    model_id = os.environ.get("QWEN_VLM_MODEL", "mlx-community/Qwen3-VL-4B-Instruct-3bit")


class Qwen2BVLMBackend(QwenVLMBackend):
    # cache: models--mlx-community--Qwen3-VL-2B-Instruct-4bit (1.7 GB) — KEEP.
    # Fastest tier measured: ~440ms/frame warm via vllm-mlx with clean
    # INTENT/FEATURE structure, but weak gesture recognition (3/12) — a speed
    # A/B option, not a gesture driver.
    # cache: models--mlx-community--Qwen3-VL-2B-Instruct-3bit (1.5 GB) — DELETE.
    # The old default; 3-bit collapses this 2B into token repetition
    # ("INTENT: 1 1 1 1 ...", measured live) and is superseded by the 4-bit.
    name = "qwen_2b"
    model_id = os.environ.get("QWEN_2B_VLM_MODEL", "mlx-community/Qwen3-VL-2B-Instruct-4bit")


class Qwen8BVLMBackend(QwenVLMBackend):
    # cache: models--lmstudio-community--Qwen3-VL-8B-Instruct-MLX-4bit (5.4 GB).
    # Already on disk, so exposed as the quality-ceiling A/B option; same
    # Qwen3-VL family as the verified 4B. Untested on the gesture set — expect
    # roughly 2x the 4B's latency per frame.
    name = "qwen_8b"
    model_id = os.environ.get("QWEN_8B_VLM_MODEL", "lmstudio-community/Qwen3-VL-8B-Instruct-MLX-4bit")


class Qwen35VLMBackend(QwenVLMBackend):
    # cache: models--Qwen--Qwen3.5-2B (4.3 GB, official bf16) — KEEP; this is
    # what the entry serves: ~670ms warm on direct mlx-vlm, real INTENT/FEATURE.
    # cache: models--Rishu11277--Qwen3.5-2B-mlx-fp16 (3.5 GB) — DELETE. That
    # export is the SAME language model re-serialized by mlx-lm (measured
    # max|Δ| 3e-8 vs official; layernorms differ only by the +1 RMSNorm storage
    # convention) with the entire vision tower stripped (0 vision weights) —
    # nothing to A/B that this entry doesn't already serve, minus the eyes.
    # Serve EXACTLY the official repo, all measured on mlx-vlm 0.6.4:
    #   * NOT a local `mlx_vlm convert -q` — this hybrid (Gated DeltaNet)
    #     architecture does not survive quantization: 4-bit AND 8-bit both emit
    #     pure gibberish even on text-only prompts.
    #   * NOT vllm-mlx — coherent there, but the chat template puts the model in
    #     reasoning mode ("The user wants me to...") and it burns the whole
    #     token budget before INTENT; direct mlx-vlm answers the strict format
    #     in ~670ms warm, which is already the fast band. Hence supports_vllm
    #     False, same treatment as Gemma 4.
    name = "qwen35_2b"
    supports_vllm = False
    model_id = os.environ.get("QWEN35_VLM_MODEL", "Qwen/Qwen3.5-2B")


class FourierQwen2VLBackend(QwenVLMBackend):
    # cache: models--whyisverysmart--Fourier-Qwen2-VL-2B-0.67 (4.1 GB) — KEEP.
    # Best gesture recognition measured: 10/12 with the gesture-name prompt +
    # pointing follow-up, ~650-900ms/frame as the local 4-bit conversion
    # (./models/fourier-qwen2vl-2b-4bit). No FEATURE overlay (parrots strict
    # templates). mradermacher publishes this model only as GGUF (llama.cpp
    # format), which MLX cannot load — this is the safetensors source repo the
    # GGUF quants were made from: identical weights, native qwen2_vl support.
    # Quantized variant: python -m mlx_vlm.convert --hf-path whyisverysmart/... -q
    # and point FOURIER_QWEN2VL_MODEL at the output directory.
    name = "fourier_qwen2vl_2b"
    model_id = os.environ.get("FOURIER_QWEN2VL_MODEL", "whyisverysmart/Fourier-Qwen2-VL-2B-0.67")


class GemmaVLMBackend(VllmMlxBackend):
    # cache: models--lmstudio-community--gemma-4-E4B-it-MLX-4bit (6.4 GB).
    name = "gemma"
    supports_vllm = False  # Gemma 4 vision is broken under vllm-mlx -> pin to direct mlx-vlm
    model_id = os.environ.get("GEMMA_VLM_MODEL", "lmstudio-community/gemma-4-E4B-it-MLX-4bit")


class GemmaE2BVLMBackend(VllmMlxBackend):
    # cache: models--lmstudio-community--gemma-4-E2B-it-MLX-4bit (4.1 GB).
    # Gemma 4 E2B — the lighter sibling of E4B. name is still "gemma" so the shared-KV
    # load filter (_patch_gemma4_shared_kv_sanitize) applies to both variants.
    name = "gemma"
    supports_vllm = False  # Gemma 4 vision is broken under vllm-mlx -> pin to direct mlx-vlm
    model_id = os.environ.get("GEMMA_E2B_VLM_MODEL", "lmstudio-community/gemma-4-E2B-it-MLX-4bit")


# ── Detection / segmentation (Falcon Perception, MLX) ──────────────────

# RGB palette for detection/segmentation overlays (matches the UI's box colours).
_DET_PALETTE_RGB = [
    (99, 102, 241), (16, 185, 129), (245, 158, 11), (239, 68, 68),
    (139, 92, 246), (6, 182, 212), (236, 72, 153), (34, 197, 94),
]


def annotate_detections(image_bgr: np.ndarray, dets: list[dict]) -> np.ndarray:
    """Render Falcon detections (masks + boxes) onto a copy of the BGR frame.

    Each det is `{"bbox":[x1,y1,x2,y2], "mask": bool ndarray | None, ...}` as
    returned by :meth:`FalconPerceptionBackend.detect`. Returns a BGR uint8 array.
    """
    from PIL import Image, ImageDraw

    base = Image.fromarray(np.ascontiguousarray(image_bgr[:, :, ::-1]).astype(np.uint8)).convert("RGBA")
    for i, det in enumerate(dets):  # masks first (under the boxes)
        mask = det.get("mask")
        if mask is not None:
            colour = _DET_PALETTE_RGB[i % len(_DET_PALETTE_RGB)]
            layer = np.zeros((*mask.shape, 4), dtype=np.uint8)
            layer[mask] = (*colour, 90)
            base = Image.alpha_composite(base, Image.fromarray(layer, "RGBA"))
    draw = ImageDraw.Draw(base)
    for i, det in enumerate(dets):
        colour = _DET_PALETTE_RGB[i % len(_DET_PALETTE_RGB)]
        bbox = det.get("bbox")
        if bbox is not None:
            draw.rectangle(list(bbox), outline=(*colour, 255), width=2)
    rgb = np.asarray(base.convert("RGB"), dtype=np.uint8)
    return rgb[:, :, ::-1].copy()  # RGB -> BGR


class FalconPerceptionBackend:
    """Falcon Perception (MLX) — open-vocabulary detection + instance segmentation.

    Unlike the encoder backends (`embed`) and the VLMs (`generate`), this returns
    *structured detections*: normalised boxes plus per-instance masks for a natural
    -language query. mlx-vlm is not involved — Falcon ships its own MLX runtime. The
    weights (~1.2 GB) download on first load; the backend degrades to a `fallback`
    mode (no detections) when falcon-perception or MLX is unavailable, so the rest
    of the pipeline stays runnable — the same MLX-or-fallback philosophy as the VLMs.
    """

    is_detector = True
    feature_dim = 0
    name = "falcon"
    model_id = os.environ.get("FALCON_PERCEPTION_MODEL", "tiiuae/Falcon-Perception")

    def __init__(self, dtype: str = "float16", model_id: str | None = None,
                 min_dim: int = 256, max_dim: int = 1024,
                 max_new_tokens: int = 200, warmup: bool = False) -> None:
        self.mode = "fallback"
        self.load_error = ""
        self.load_status = ""
        self.compat_notes: list[str] = []
        self._model = self._tokenizer = self._args = self._engine = None
        self._build_prompt = self._process_batch = None
        self._min_dim, self._max_dim = min_dim, max_dim
        self._default_max_new_tokens = int(max_new_tokens)
        self.supports_segmentation = False
        # Instance model id (defaults to the class/env value). The live UI passes the
        # lighter Falcon-Perception-300M here for a near-real-time DETECTION path
        # (~0.3s/frame vs ~5-16s for the 1.5B); segment_reason keeps the default 1.5B,
        # which is the only variant carrying a mask head. See docs/detection-guide.md.
        self.model_id = model_id or type(self).model_id
        try:
            from falcon_perception import build_prompt_for_task, load_from_hf_export_mlx
            from falcon_perception.mlx.batch_inference import (
                BatchInferenceEngine,
                process_batch_and_generate,
            )

            self._build_prompt = build_prompt_for_task
            self._process_batch = process_batch_and_generate
            local_dir = _local_hf_snapshot(self.model_id)
            if local_dir:
                self.compat_notes.append("using local HF snapshot")
                self._model, self._tokenizer, self._args = load_from_hf_export_mlx(
                    hf_local_dir=local_dir, dtype=dtype)
            else:
                self._model, self._tokenizer, self._args = load_from_hf_export_mlx(
                    hf_model_id=self.model_id, dtype=dtype)
            self._engine = BatchInferenceEngine(self._model, self._tokenizer)
            # The 300M variant is detection-only (no conv_segm mask head); probe the
            # capability so a segmentation request degrades to boxes, never crashes.
            self.supports_segmentation = hasattr(self._model, "conv_segm")
            self.mode = "loaded"
            self.load_status = "; ".join(self.compat_notes)
            if warmup:
                self._warmup()
        except Exception as exc:  # noqa: BLE001 - any import/Metal/download failure -> fallback
            self.load_error = f"{type(exc).__name__}: {exc}"
            self.load_status = "falcon-perception unavailable"

    def _warmup(self) -> None:
        """Run one tiny detect so mx.compile / Metal kernels build at load time, not
        on the first live camera frame. Best-effort: a warmup failure never blocks
        the backend (it stays 'loaded'; the real call surfaces any genuine error)."""
        try:
            dummy = np.zeros((self._max_dim, self._max_dim, 3), dtype=np.uint8)
            self.detect(dummy, "object", task="detection", max_new_tokens=8)
            self.compat_notes.append("warmed up")
        except Exception as exc:  # noqa: BLE001 - warmup is optional
            self.compat_notes.append(f"warmup skipped ({type(exc).__name__})")
        self.load_status = "; ".join(self.compat_notes)

    def embed(self, image_bgr: np.ndarray) -> np.ndarray:
        raise NotImplementedError(f"{self.name} is a detector; call detect(), not embed()")

    def detect(
        self,
        image_bgr: np.ndarray,
        query: str,
        task: str = "segmentation",
        max_new_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> list[dict]:
        """Detect/segment `query` in the frame. Returns a list of detections, each
        `{"bbox":[x1,y1,x2,y2], "cx","cy","w","h" (normalised), "mask": bool|None,
        "mask_area_px": int}`. `task` is "segmentation" (boxes+masks) or "detection".
        A segmentation request on a detection-only model (300M) degrades to boxes."""
        if self._engine is None:
            raise RuntimeError(f"{self.name} model not loaded ({self.load_status or self.load_error})")
        if task == "segmentation" and not self.supports_segmentation:
            task = "detection"  # this model has no mask head -> boxes only, don't crash
        tokens = int(max_new_tokens) if max_new_tokens else self._default_max_new_tokens
        from PIL import Image

        pil = Image.fromarray(np.ascontiguousarray(image_bgr[:, :, ::-1]).astype(np.uint8))
        width, height = pil.size
        prompt = self._build_prompt(query, task)
        batch = self._process_batch(
            self._tokenizer, [(pil, prompt)],
            max_length=self._args.max_seq_len,
            min_dimension=self._min_dim, max_dimension=self._max_dim,
        )
        _, aux_outputs = self._engine.generate(
            tokens=batch["tokens"], pos_t=batch["pos_t"], pos_hw=batch["pos_hw"],
            pixel_values=batch["pixel_values"], pixel_mask=batch["pixel_mask"],
            max_new_tokens=tokens, temperature=temperature, task=task,
        )
        return self._parse_aux(aux_outputs[0], width, height, task)

    @staticmethod
    def _parse_aux(aux, width: int, height: int, task: str) -> list[dict]:
        # Falcon emits bbox coords as a stream of {x,y} then {h,w} dicts; pair them.
        boxes, current = [], {}
        for entry in getattr(aux, "bboxes_raw", []) or []:
            if not isinstance(entry, dict):
                continue
            current.update(entry)
            if all(k in current for k in ("x", "y", "h", "w")):
                boxes.append(dict(current))
                current = {}
        masks_rle = getattr(aux, "masks_rle", []) or []

        dets: list[dict] = []
        for i, box in enumerate(boxes):
            cx, cy = box["x"] * width, box["y"] * height
            bw, bh = box["w"] * width, box["h"] * height
            det = {
                "bbox": [max(0, int(cx - bw / 2)), max(0, int(cy - bh / 2)),
                         min(width, int(cx + bw / 2)), min(height, int(cy + bh / 2))],
                "cx": box["x"], "cy": box["y"], "w": box["w"], "h": box["h"],
                "mask": None, "mask_area_px": 0,
            }
            if task == "segmentation" and i < len(masks_rle):
                try:
                    from pycocotools import mask as mask_utils

                    mask = mask_utils.decode(masks_rle[i]).astype(bool)
                    if mask.shape != (height, width):
                        from PIL import Image as _Image

                        mask = np.array(_Image.fromarray(mask.astype(np.uint8) * 255)
                                        .resize((width, height), _Image.NEAREST)) > 127
                    det["mask"] = mask
                    det["mask_area_px"] = int(mask.sum())
                except Exception:  # noqa: BLE001 - mask decode is best-effort
                    pass
            dets.append(det)
        return dets


_REGISTRY = {
    "dino": DinoBackend,
    "vjepa": VJepaBackend,
    "lingbot": LingBotBackend,
    "qwen": QwenVLMBackend,
    "qwen_2b": Qwen2BVLMBackend,
    "qwen_8b": Qwen8BVLMBackend,
    "qwen35_2b": Qwen35VLMBackend,
    "fourier_qwen2vl_2b": FourierQwen2VLBackend,
    "gemma": GemmaVLMBackend,
    "gemma_e2b": GemmaE2BVLMBackend,
    "falcon": FalconPerceptionBackend,
}


def make_backend(name: str) -> PerceptionBackend:
    key = (name or "").strip().lower()
    if key not in _REGISTRY:
        raise ValueError(f"unknown backend '{name}'; choose from {sorted(_REGISTRY)}")
    return _REGISTRY[key]()


def backend_model_id(name: str) -> str:
    """Model id from the class, without constructing (and thus without loading)."""
    cls = _REGISTRY.get((name or "").strip().lower())
    return getattr(cls, "model_id", name) if cls else name


_RUNTIME_OVERRIDE = ""  # "", "vllm-mlx" or "mlx-vlm" — set by the UI Runtime selector


def set_runtime_override(value: str | None) -> None:
    """Explicitly choose the serving runtime for subsequent model loads.

    "" / "auto" restores per-model defaults (supports_vllm). An explicit
    "vllm-mlx" or "mlx-vlm" wins over the model's default so the two runtimes
    can be speed-compared on any model — including ones whose default pins
    them elsewhere."""
    global _RUNTIME_OVERRIDE
    value = (value or "").strip().lower()
    _RUNTIME_OVERRIDE = "" if value == "auto" else value


def planned_runtime(name: str) -> str:
    """The runtime a backend will use if constructed now (override > env > default)."""
    explicit = _RUNTIME_OVERRIDE or (
        os.environ.get("PAVE_VLM_BACKEND") or os.environ.get("PAVE_VLM_RUNTIME") or ""
    ).strip().lower()
    if explicit:
        return explicit
    cls = _REGISTRY.get((name or "").strip().lower())
    return "vllm-mlx" if getattr(cls, "supports_vllm", True) else "mlx-vlm"


def checkpoint_label(name: str) -> str:
    """Dropdown/preflight/trace title for a backend: the checkpoint's basename.

    "Qwen3-VL-4B-Instruct-3bit", not a hand-written alias — so the UI title,
    the CLI preflight line, and the HF cache directory always agree. Env
    overrides flow through automatically (a local ./models/foo-4bit dir shows
    as "foo-4bit")."""
    return backend_model_id(name).rstrip("/").split("/")[-1]


VLM_NAMES = {"qwen", "qwen_2b", "qwen_8b", "qwen35_2b", "fourier_qwen2vl_2b", "gemma", "gemma_e2b"}
DETECTOR_NAMES = {"falcon"}  # structured detect/segment backends (not intent producers)
