"""Streaming robot-state viewer server for the OpenPAVE MLX runtime.

Why this exists
---------------
Intent Ingress (Flask, port 7071) is POST-only — it has no page to open in a
browser, which is why `http://127.0.0.1:7071` shows nothing streamable. This
module is the *observation* half the template used: a small stdlib HTTP server
(no Flask) that

  * serves the basic Three.js visualiser (`visualiser/index.html` +
    `visualiser/static/*.STL`) used by the console's Internal button, and
  * streams the control daemon's robot-state / command-result feedback as
    Server-Sent Events (SSE) so the browser receives a consistent time series.

It only reads the feedback files the daemon already writes; it never controls the
robot. Open the URL printed by `mlx-runtime/main.py` (default
`http://127.0.0.1:7080/`).
"""

from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
VIS_DIR = REPO_ROOT / "visualiser"
# This server hosts the BASIC single-file viewer (visualiser/index.html) used
# by the console's "Internal (QtWebEngine)" button — no build step, no webcam.
# The full SvelteKit app (visualiser/src) runs on vite instead ("Browser"
# button; needs webcam access) and reaches /api/* here through vite's proxy.
STATIC_DIR = VIS_DIR / "static"

ROBOT_STATE_PATH = os.environ.get("ROBOT_STATE_PATH", "/tmp/vla_robot_state.json")
COMMAND_RESULT_PATH = os.environ.get("COMMAND_RESULT_PATH", "/tmp/vla_command_result.json")
# "10s OBSERVE" scene observations (pave_ui/viewer.py) — rendered by the
# visualisers as a speech bubble under the Robot State banner.
OBSERVATION_PATH = os.environ.get("OBSERVATION_PATH", "/tmp/vla_observation.json")
# SSE poll cadence. 50ms (was 200ms) so a state change reaches the browser within
# ~50ms. _significant() still gates actual sends, so this only lowers change-detection
# latency; it does not push duplicate frames faster.
STREAM_INTERVAL_S = float(os.environ.get("STATE_STREAM_INTERVAL_S", "0.05"))

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".stl": "model/stl",
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".css": "text/css",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".png": "image/png",
}


def _read_json(path: str) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def _frame(seq: int) -> dict:
    """One time-series sample: server clock + the latest feedback files."""
    return {
        "seq": seq,
        "server_time": time.time(),
        "robot_state": _read_json(ROBOT_STATE_PATH),
        "command_result": _read_json(COMMAND_RESULT_PATH),
        "observation": _read_json(OBSERVATION_PATH),
    }


def _significant(frame: dict) -> tuple:
    """Reduce a frame to the parts that represent an ACTUAL change — a real
    command, not the daemon's own low-frequency heartbeat ticking or the wall
    clock advancing. `_stream()` only pushes (and the browser only renders) a
    new row when this changes. `heartbeat_seq`/`updated_at`/`server_time` are
    deliberately excluded: they change every tick even when the robot is
    completely idle, and including them here is exactly what was flooding the
    "ROBOT STATE — live time series" panel with a new row every
    STREAM_INTERVAL_S regardless of whether anything happened.
    """
    rs = frame.get("robot_state") or {}
    cr = frame.get("command_result") or {}
    return (
        json.dumps(rs.get("status")),
        json.dumps(rs.get("pose")),
        json.dumps(rs.get("joint_state")),
        json.dumps(rs.get("last_command")),
        json.dumps(cr),
        json.dumps(frame.get("observation")),  # each OBSERVE tick pushes a frame
    )


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:  # silence default stderr spam
        return

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send_file(VIS_DIR / "index.html")
        elif path == "/api/state":
            self._send_json(_frame(0))
        elif path == "/api/stream":
            self._stream()
        elif path.startswith("/static/"):
            target = (STATIC_DIR / Path(path).name).resolve()
            if target.parent == STATIC_DIR.resolve() and target.is_file():
                self._send_file(target)
            else:
                self.send_error(404)
        else:
            self.send_error(404)

    def _send_file(self, fp: Path) -> None:
        data = fp.read_bytes()
        ctype = CONTENT_TYPES.get(fp.suffix.lower(), "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, obj: dict) -> None:
        data = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        seq = 0
        last_sig = None  # sentinel: guarantees the first real frame always sends
        try:
            while True:
                frame = _frame(seq)
                sig = _significant(frame)
                if sig != last_sig:      # idle (heartbeat-only) ticks are silently
                    last_sig = sig       # dropped here — see _significant()'s docstring
                    payload = json.dumps(frame)
                    self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    seq += 1
                time.sleep(STREAM_INTERVAL_S)
        except (BrokenPipeError, ConnectionResetError):
            return  # client navigated away


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, request, client_address) -> None:
        # A browser closing an SSE stream resets the socket; that is normal, so
        # do not dump a traceback for it.
        import sys

        if isinstance(sys.exc_info()[1], (ConnectionResetError, BrokenPipeError, ConnectionAbortedError)):
            return
        super().handle_error(request, client_address)


class StateServer:
    """Background SSE viewer server, started/stopped like the template service."""

    def __init__(self, host: str = "127.0.0.1", port: int = 7080):
        self.host = host
        self.port = port
        self.httpd: _Server | None = None
        self.thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def start(self) -> str:
        port = self.port
        for _ in range(20):  # skip busy ports like the template does
            try:
                self.httpd = _Server((self.host, port), _Handler)
                self.port = port
                break
            except OSError:
                port += 1
        if self.httpd is None:
            raise RuntimeError("could not bind robot-state viewer server")
        self.thread = threading.Thread(
            target=self.httpd.serve_forever, name="state-server", daemon=True
        )
        self.thread.start()
        return self.url

    def stop(self) -> None:
        if self.httpd is None:
            return
        self.httpd.shutdown()
        self.httpd.server_close()
        if self.thread:
            self.thread.join(timeout=1.0)
        self.httpd = None
        self.thread = None


if __name__ == "__main__":  # standalone: serve until Ctrl+C
    server = StateServer(port=int(os.environ.get("STATE_SERVER_PORT", "7080")))
    print(f"robot-state viewer streaming at {server.start()}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()
