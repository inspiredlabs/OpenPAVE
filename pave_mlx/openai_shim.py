"""Universal OpenAI-compatible shim (responsibility A in §3.2).

Exposes the slice of the OpenAI API that live-vlm-webui / OpenPAVE Tier A use:

  GET  /v1/models
  POST /v1/chat/completions   (image in -> intent token out)
  GET  /healthz

Internally it runs a perception backend + an intent head and returns the predicted
intent (STOP/TROT/HOME/LEFT/RIGHT) as the assistant message `content`. OpenPAVE
itself is unchanged — point `UI_API_BASE` at this server:

    python -m pave_mlx.openai_shim --backend dino --port 8000

Until a head is trained (manifest.trained == false) it returns the safe default
STOP, so the control path is well-defined out of the box.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from pave_mlx.backends import VLM_NAMES, backend_model_id, make_backend
from pave_mlx.heads.base import INTENT_LABELS, PKG_DIR, SAFE_DEFAULT, HeadManifest
from pave_mlx.heads.embedding_probe import EmbeddingProbe
from pave_mlx.intent_decode import decode

# Gesture-aware, but deliberately NOT restricted to a narrow STOP/TROT-only
# vocabulary — kept in sync with pave_ui/perception.py's ROBOT_PROMPT default.
ROBOT_PROMPT = os.environ.get(
    "PAVE_ROBOT_PROMPT",
    "You control a quadruped robot from its forward camera. If you see a clear "
    "hand gesture, use it: thumbs-up means TROT, an open palm means STOP, a "
    "closed fist means HOME, pointing left means LEFT, pointing right means "
    "RIGHT. Otherwise reason about the scene. Reply with EXACTLY ONE word — "
    "STOP, TROT, HOME, LEFT, or RIGHT.",
)


def clamp_to_intent(text: str) -> str:
    """Pin free-form VLM text to the intent vocab (earliest-mentioned word wins)."""
    up = (text or "").upper()
    best, best_i = None, len(up) + 1
    for tok in INTENT_LABELS:
        i = up.find(tok)
        if 0 <= i < best_i:
            best, best_i = tok, i
    return best or SAFE_DEFAULT


def _decode_image(data_url: str) -> np.ndarray | None:
    """Decode a base64 data URL (or bare base64) into a BGR uint8 array."""
    if not data_url:
        return None
    raw = data_url.split(",", 1)[1] if data_url.startswith("data:") else data_url
    try:
        blob = base64.b64decode(raw)
    except Exception:
        return None
    try:
        from PIL import Image  # optional; the shim's only image dependency

        img = Image.open(io.BytesIO(blob)).convert("RGB")
        rgb = np.asarray(img, dtype=np.uint8)
        return rgb[:, :, ::-1].copy()  # RGB -> BGR
    except Exception:
        try:
            import cv2

            arr = np.frombuffer(blob, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            return None


def _extract_image(body: dict) -> str | None:
    for message in body.get("messages", []):
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    url = part.get("image_url", {})
                    return url.get("url") if isinstance(url, dict) else url
    return None


class IntentModel:
    """Backend + head (A consumes B & C).

    The model loads *off the request path* (like the template's EngineBuilder): the
    HTTP server starts immediately and `infer()` returns the safe default until the
    backend is ready. So a multi-GB VLM download never blocks the port, and the GUI
    gates inference on the `READY` line rather than hammering an unbound server.
    """

    def __init__(self, backend_name: str, config_path: str | None, min_confidence: float):
        self.backend = None  # set last, once fully loaded
        self.backend_name = backend_name
        self.min_confidence = float(min_confidence)
        self.is_vlm = backend_name in VLM_NAMES
        self.model_id = backend_model_id(backend_name)  # known pre-load
        self.probe: EmbeddingProbe | None = None
        self.manifest = None
        self.last_text = ""
        self.status = "loading"
        self._config_path = config_path

        if self.is_vlm:  # heavy download/load -> background thread
            threading.Thread(target=self._load, name="model-load", daemon=True).start()
        else:            # encoders are fast (or fall back fast) -> load now
            self._load()

    def _load(self) -> None:
        try:
            backend = make_backend(self.backend_name)
        except Exception as exc:  # noqa: BLE001
            self.status = f"error: {exc}"
            print(f"[pave_mlx] READY backend={self.backend_name} mode=error status={self.status}", flush=True)
            return

        if self.is_vlm:
            self.model_id = getattr(backend, "model_id", self.backend_name)
            self.status = backend.mode
        else:
            cfg = Path(self._config_path) if self._config_path else (PKG_DIR / "heads" / "configs" / f"{self.backend_name}.json")
            try:
                self.manifest = HeadManifest.load(cfg)
                self.model_id = self.manifest.model_id
                self.status = "untrained"
                wpath = self.manifest.weights_path()
                if self.manifest.trained and wpath.is_file():
                    self.probe = EmbeddingProbe.load(wpath)
                    self.status = "trained"
            except Exception as exc:  # noqa: BLE001
                self.status = f"manifest-error: {exc}"

        self.backend = backend  # ready
        print(f"[pave_mlx] READY backend={self.backend_name} mode={getattr(backend, 'mode', '?')} status={self.status}", flush=True)

    def infer(self, image_bgr: np.ndarray | None) -> tuple[str, float]:
        if image_bgr is None or self.backend is None:
            return SAFE_DEFAULT, 1.0  # safe default while loading / no image

        if self.is_vlm:
            try:
                self.last_text = self.backend.generate(image_bgr, ROBOT_PROMPT)
                return clamp_to_intent(self.last_text), 1.0
            except Exception as exc:  # noqa: BLE001
                self.last_text = f"error: {exc}"
                return SAFE_DEFAULT, 0.0

        if self.probe is None:
            return SAFE_DEFAULT, 1.0
        feats = self.backend.embed(image_bgr)
        return decode(self.probe.logits(feats), self.manifest.labels, self.min_confidence)


def make_handler(model: IntentModel):
    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            return

        def _json(self, obj: dict, code: int = 200) -> None:
            data = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/healthz":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok\n")
            elif path in ("/v1/models", "/models"):
                self._json({
                    "object": "list",
                    "data": [{"id": model.model_id, "object": "model", "owned_by": "pave_mlx"}],
                })
            else:
                self._json({"error": {"message": f"not found: {path}"}}, 404)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path not in ("/v1/chat/completions", "/chat/completions"):
                self._json({"error": {"message": f"not found: {path}"}}, 404)
                return
            length = int(self.headers.get("Content-Length", "0") or "0")
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except Exception:
                body = {}

            image = _decode_image(_extract_image(body))
            intent, conf = model.infer(image)

            self._json({
                "id": f"chatcmpl-pave-{int(time.time()*1000)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model.model_id,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": intent},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 1, "total_tokens": 1},
                "x_pave": {
                    "backend": model.backend_name,
                    "backend_mode": getattr(model.backend, "mode", "loading") if model.backend else "loading",
                    "head_status": model.status,
                    "confidence": round(conf, 4),
                    "vlm_text": model.last_text if model.is_vlm else None,
                },
            })

    return _Handler


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, request, client_address) -> None:
        import sys

        if isinstance(sys.exc_info()[1], (ConnectionResetError, BrokenPipeError, ConnectionAbortedError)):
            return
        super().handle_error(request, client_address)


def main() -> None:
    ap = argparse.ArgumentParser(description="OpenPAVE Tier A OpenAI shim")
    ap.add_argument("--backend", default="dino", choices=["dino", "vjepa", "lingbot", "qwen", "gemma", "gemma_e2b"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--config", default=None, help="path to a head manifest json")
    ap.add_argument("--min-confidence", type=float, default=0.0)
    args = ap.parse_args()

    model = IntentModel(args.backend, args.config, args.min_confidence)
    httpd = _Server((args.host, args.port), make_handler(model))
    url = f"http://{args.host}:{args.port}/v1"
    print(f"[pave_mlx] serving '{args.backend}' at {url} (model loads in background; READY line follows)")
    print(f"[pave_mlx] point OpenPAVE at it:  UI_API_BASE={url}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
