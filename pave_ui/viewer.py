"""OpenPAVE PyQt6 operator console.

Top bar = three control rows (WORKING/17 template):

  Row 1  Experience   : main-view selector (+ live status, render-in-browser button)
  Row 2  Vision model : No model (startup), Qwen3-VL, DINOv3, V-JEPA 2.1, LingBot-Map
  Row 3  Runtime      : compute badge, camera, input source, control-plane status

Below the bar, two columns:

  Column 1 : live camera preview + prompt probes
  Column 2 : the ThreeJS robot visualiser (STL + SSE time series) + console

Live camera frames are pushed through the VLM pipeline verbatim — frame → the
selected model's OpenAI-compatible shim → intent token → intent_ingress → the
ThreeJS robot — so the simulated robot is driven by the real camera input before
hardware arrives.

Run:  python -m pave_ui.viewer   (or ./mlx-runtime.sh)
"""

from __future__ import annotations

import base64
import contextlib
import importlib.util
import json
import os
import re
import shutil
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from collections import deque
from pathlib import Path

os.environ.setdefault("INTENT_PATH", "/tmp/vla_intent.json")
os.environ.setdefault("COMMAND_RESULT_PATH", "/tmp/vla_command_result.json")
os.environ.setdefault("ROBOT_STATE_PATH", "/tmp/vla_robot_state.json")
os.environ.setdefault("ROBOT_ADAPTER", "mock")

# Quiet noisy native logs BEFORE importing the libraries that emit them.
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("OPENCV_VIDEOIO_DEBUG", "0")
_wf = os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS", "").split()
for _f in ("--disable-logging", "--log-level=3", "--use-angle=metal"):
    if _f not in _wf:
        _wf.append(_f)
os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = " ".join(_wf)


@contextlib.contextmanager
def _quiet_fd2():
    """Silence C-level stderr (OpenCV/AVFoundation probe spam) for a block."""
    saved = None
    try:
        sys.stderr.flush()
        devnull = os.open(os.devnull, os.O_WRONLY)
        saved = os.dup(2)
        os.dup2(devnull, 2)
        os.close(devnull)
        yield
    finally:
        if saved is not None:
            os.dup2(saved, 2)
            os.close(saved)

REPO_ROOT = Path(__file__).resolve().parents[1]
MLX_RUNTIME_DIR = REPO_ROOT / "mlx-runtime"
PAVE_MLX_CONFIGS = REPO_ROOT / "pave_mlx" / "heads" / "configs"
for _p in (str(REPO_ROOT), str(MLX_RUNTIME_DIR)):  # robust regardless of launch style
    if _p not in sys.path:
        sys.path.insert(0, _p)
from state_server import StateServer  # noqa: E402

from PyQt6.QtCore import (  # noqa: E402
    Qt, QProcess, QProcessEnvironment, QRect, QTimer, QUrl, pyqtSignal, qInstallMessageHandler,
)
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap  # noqa: E402
from PyQt6.QtWidgets import (  # noqa: E402
    QApplication, QCheckBox, QComboBox, QHBoxLayout, QLabel, QLineEdit,
    QPlainTextEdit, QPushButton, QSplitter, QStackedWidget, QVBoxLayout, QWidget,
)

try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
except Exception:  # pragma: no cover
    QWebEngineView = None

try:
    import cv2
    import numpy as np
except Exception:  # pragma: no cover - camera optional
    cv2 = None
    np = None

from pave_ui import perception  # noqa: E402  (in-process feature-overlay engines)
from pave_mlx import mlx_mem  # noqa: E402  (Metal wired-limit / cache helpers)

EXPERIENCES = ["Robot State Viewer", "Physics Simulator"]
# Gemma 4 E4B is the default vision-language model on startup (falls back to
# whatever fits on disk if it isn't cached yet — see _start_default_model).
# Gemma 4 E2B is the default: ~1.1s/frame vs ~1.8s for E4B and ~2.5GB less peak
# memory (measured), with the same intent + FEATURE self-report contract.
# Qwen3-VL (4B, 3-bit) is the default: it's the vision model verified to work on
# the fast vllm-mlx serving tier (~0.3s/frame, real INTENT + FEATURE output).
# Gemma stays selectable but runs on the direct mlx-vlm path (its vllm vision path
# is broken). See the SUPPORT reduced test case.
DEFAULT_MODEL = os.environ.get("PAVE_DEFAULT_MODEL", "Qwen3-VL")
MODELS = [
    "No model", "Qwen3-VL", "Qwen3-VL 2B",
    "Qwen3.5 2B (Rishu11277)", "Fourier Qwen2-VL 2B (mradermacher)",
    "Gemma 4 E2B", "Gemma 4 E4B", "DINOv3", "V-JEPA 2.1", "LingBot-Map",
]
MODEL_BACKEND = {
    "No model": "none",
    "Qwen3-VL": "qwen", "Qwen3-VL 2B": "qwen_2b",
    "Qwen3.5 2B (Rishu11277)": "rishu_qwen35_2b",
    "Fourier Qwen2-VL 2B (mradermacher)": "fourier_qwen2vl_2b",
    "Gemma 4 E2B": "gemma_e2b", "Gemma 4 E4B": "gemma",
    "DINOv3": "dino", "V-JEPA 2.1": "vjepa", "LingBot-Map": "lingbot",
}
VLM_BACKENDS = {"qwen", "qwen_2b", "rishu_qwen35_2b", "fourier_qwen2vl_2b", "gemma", "gemma_e2b"}
# "Idle" is the default so nothing is posted until a person actually selects a
# real input; switch to "Camera (VLM)" to drive the robot from the live model
# (gesture recognition) instead. Idle intentionally sends NOTHING — the
# control_daemon's own low-frequency heartbeat (ROBOT_HEARTBEAT_SEC, see
# control_daemon/daemon.py) is the only thing that still nudges the ROBOT STATE
# live time-series panel while idle, and it does so occasionally, not per-tick.
SOURCES = ["Idle", "Camera (VLM)"]
# Zero-click startup: once a camera + the loaded VLM are ready, drive live inference
# automatically instead of waiting for the user to switch the input off "Idle".
# Set PAVE_DEFAULT_SOURCE=Idle to restore the old opt-in behaviour.
DEFAULT_SOURCE = os.environ.get("PAVE_DEFAULT_SOURCE", "Camera (VLM)")
_PCT_RE = re.compile(r"(\d{1,3})\s*%")
INTENT_PORT = 7071
SHIM_PORT = 8000
# SvelteKit visualiser dev server ("Browser" button — needs webcam access,
# which the embedded QtWebEngine page does not get). 888 rather than vite's
# usual 5173 to avoid colliding with other dev servers on this machine.
VITE_PORT = int(os.environ.get("PAVE_VITE_PORT", "888"))
# Feature gate: the Browser button starts the SvelteKit dev server (vite).
# Deployments that don't ship vite/node set this to 0 — Browser then opens the
# basic single-file viewer (visualiser/index.html on the state server) instead.
BROWSER_USES_VITE = os.environ.get("PAVE_BROWSER_USES_VITE", "1") == "1"


def _node_binary() -> str | None:
    """Node >=20.19 for vite 8: PAVE_NODE_BIN, newest nvm install, then PATH.

    The system node (v20.11 here) is too old for vite 8, so PATH alone is not
    good enough — prefer an explicit override, then the newest nvm version."""
    env_bin = os.environ.get("PAVE_NODE_BIN")
    if env_bin and Path(env_bin).is_file():
        return env_bin
    candidates = []
    for d in (Path.home() / ".nvm" / "versions" / "node").glob("v*"):
        try:
            version = tuple(int(x) for x in d.name.lstrip("v").split("."))
        except ValueError:
            continue
        if version >= (20, 19) and (d / "bin" / "node").is_file():
            candidates.append((version, d / "bin" / "node"))
    if candidates:
        return str(max(candidates)[1])
    return shutil.which("node")
SHIM_URL = f"http://127.0.0.1:{SHIM_PORT}/v1/chat/completions"

# TROT confirmation gate — ported from the DGX Spark scenario
# (scenarios/puppypi-gesture-stop-trot.json: safety_constraints.trot_requires_confirmation,
# recommended_env.TROT_CONFIRMATIONS / TROT_CONFIRMATION_WINDOW_MS). TROT must be
# seen this many times in a row, within this window, from EITHER the gesture
# tick or a prompt-probe tap, before it is actually forwarded to intent_ingress.
# STOP/HOME/LEFT/RIGHT are never gated — only TROT, matching the DGX behavior.
TROT_CONFIRMATIONS = int(os.environ.get("TROT_CONFIRMATIONS", "1"))
TROT_CONFIRMATION_WINDOW_MS = float(os.environ.get("TROT_CONFIRMATION_WINDOW_MS", "10"))

# Camera→VLM inference cadence. This is a SHORT poll (not the inference time): the
# tick self-gates on `_camera_job_pending`, so when the model is busy it returns
# immediately, and the instant a frame's inference finishes the next tick submits
# the freshest frame — i.e. back-to-back inference, matching the WORKING/17
# reference (80-140ms). The old 1500ms value added up to 1.5s of dead waiting
# before each inference even started, which is the latency the robot felt.
INFER_INTERVAL_MS = int(os.environ.get("PAVE_INFER_INTERVAL_MS", "120"))

# Camera gesture gate — the "early-out" that keeps the fans off. Back-to-back
# camera inference is ~100% GPU duty forever; but a gesture IS a scene change,
# so when the frame hasn't visibly changed since the last inference there is
# nothing new to recognize and the tick returns before touching the VLM. The
# check is the same tiny MLX kernel the OBSERVE gate uses (~0.4ms), with the
# MAX single-cell delta so a hand raised in one corner always registers while
# global exposure drift stays under the bar. A keepalive inference still runs
# every few seconds so the pipeline can never go permanently blind.
CAMERA_SCENE_GATE = os.environ.get("PAVE_CAMERA_SCENE_GATE", "1") == "1"
CAMERA_DIFF_THRESHOLD = float(os.environ.get("PAVE_CAMERA_DIFF_THRESHOLD", "30.0"))
CAMERA_IDLE_KEEPALIVE_S = float(os.environ.get("PAVE_CAMERA_IDLE_KEEPALIVE_S", "10"))

# Falcon detector cadence. With the 300M detection fast-path (~0.3s/frame warm)
# this is a short poll so boxes update near real-time; it is self-throttled to one
# job in flight (a longer decode — e.g. segmentation on the 1.5B — simply skips
# ticks), and the worker coalesces to the newest frame, so it never backs up.
FALCON_INTERVAL_MS = int(os.environ.get("PAVE_FALCON_INTERVAL_MS", "200"))
FALCON_DEFAULT_QUERY = os.environ.get("PAVE_FALCON_QUERY", "person")

# Idempotency: repeated identical camera intents (STOP, TROT, ...) must NOT flood
# intent_ingress / the control daemon / the ROBOT STATE feed. A recognized camera
# intent is only repeated after this cooldown, and never while the prior command
# is still in flight.
# Prompt-probe clicks are explicit commands and keep their own confirmation gate.
CAMERA_INTENT_REPEAT_MS = float(os.environ.get("PAVE_CAMERA_INTENT_REPEAT_MS", "2500"))
COMMAND_RESULT_PATH = os.environ["COMMAND_RESULT_PATH"]
_TERMINAL_COMMAND_STATUSES = {"completed", "failed", "rejected"}
_PENDING_FALLBACK_MS = 5000  # safety valve: never suppress forever if we can't confirm

# Prompt-probe buttons: explicit one-off prompts (task 3A). Gesture recognition
# (task 3B) instead runs continuously off the camera tick using perception's
# gesture-aware ROBOT_PROMPT default — see pave_ui/perception.py.
PROMPT_BUTTONS = [
    ("STOP", "The robot must freeze immediately. Reply with exactly one word: STOP."),
    ("TROT", "The camera shows a clear path forward. Reply with exactly one word: TROT."),
    ("LEFT", "The camera shows blocked space on the right and free space on the left. Reply with exactly one word: LEFT."),
    ("RIGHT", "The camera shows blocked space on the left and free space on the right. Reply with exactly one word: RIGHT."),
    ("HOME", "The robot should return to its home pose. Reply with exactly one word: HOME."),
]

# "10s OBSERVE": one scene observation per second for 10 seconds. Every tick
# feeds the previous answers back and demands something NEW — a repeated
# observation is wasted request_ms; the point is scene richness over time.
OBSERVE_TICKS = int(os.environ.get("PAVE_OBSERVE_TICKS", "10"))
OBSERVE_INTERVAL_MS = int(os.environ.get("PAVE_OBSERVE_INTERVAL_MS", "1000"))
# Each observation is also written here; the state server streams it to the
# visualisers, which show it as a speech bubble under the Robot State banner —
# that is where a human expects the robot to "speak".
OBSERVATION_PATH = os.environ.get("OBSERVATION_PATH", "/tmp/vla_observation.json")

# Continuous OBSERVE toggle — unlike the 10s burst (fixed 1Hz, ~90% GPU duty
# cycle, fans on), this idles near zero. The economics, in order:
#   1. the scene-change gate runs as one tiny MLX kernel on the GPU (see
#      perception.observe_signature) — no cv2/CPU image work competing with
#      the UI while the model runs, and no request_ms spent on a static scene;
#   2. checks are spaced out (2s) — the gate itself stays a rounding error in
#      the GPU budget;
#   3. a cooldown floor caps the duty cycle even in a busy scene (3s gap
#      around a ~0.8s inference ≈ ~25% GPU duty — under the thermal envelope);
#   4. when the model repeats itself anyway, the cooldown doubles (up to the
#      max) and resets on the next genuinely new observation;
#   5. at most ONE observation is ever in flight — a slow reply postpones the
#      next check instead of queueing behind it.
OBSERVE_CHECK_MS = int(os.environ.get("PAVE_OBSERVE_CHECK_MS", "2000"))
OBSERVE_DIFF_THRESHOLD = float(os.environ.get("PAVE_OBSERVE_DIFF_THRESHOLD", "12.0"))
OBSERVE_COOLDOWN_MS = int(os.environ.get("PAVE_OBSERVE_COOLDOWN_MS", "3000"))
OBSERVE_BACKOFF_MAX_MS = int(os.environ.get("PAVE_OBSERVE_BACKOFF_MAX_MS", "20000"))

# macOS camera probe (Continuity Camera shows up as a higher AVFoundation index).
# Only AVFOUNDATION — CAP_ANY triggers the noisy ffmpeg "list devices" failure.
_PROBE_COMBOS = []
if cv2 is not None:
    _PROBE_COMBOS = [
        (0, cv2.CAP_AVFOUNDATION), (1, cv2.CAP_AVFOUNDATION), (2, cv2.CAP_AVFOUNDATION),
    ]


def _quick_open(index, backend, width=1280, height=720, tries=8):
    with _quiet_fd2():  # hide "out device of bound" probe spam
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        for _ in range(tries):
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap
            time.sleep(0.05)
        cap.release()
        return None


def enumerate_cameras():
    if cv2 is None:
        return []
    found, seen = [], set()
    for index, backend in _PROBE_COMBOS:
        cap = _quick_open(index, backend)
        if cap is None:
            continue
        bname = cap.getBackendName()
        cap.release()
        key = (index, bname)
        if key in seen:
            continue
        seen.add(key)
        tag = "laptop (default)" if index == 0 else f"camera {index}"
        found.append((f"{tag} — {bname}", index, backend))
    return found


def detect_compute_backend() -> tuple[str, str]:
    try:
        import mlx.core as mx

        return "MLX", str(mx.default_device())
    except Exception:
        return "NumPy", "cpu fallback"


def head_status(backend_key: str) -> str:
    if backend_key == "none":
        return "idle"
    if backend_key in VLM_BACKENDS:
        return "VLM (generates intent text)"
    if backend_key in ("vjepa", "lingbot"):
        return "stub (not wired)"
    cfg = PAVE_MLX_CONFIGS / f"{backend_key}.json"
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
        return "trained" if data.get("trained") else "untrained → safe STOP"
    except Exception:
        return "no manifest"


def backend_mode_hint(backend_key: str) -> str:
    if backend_key == "none":
        return "not loaded"
    if backend_key in VLM_BACKENDS:
        runtime = (os.environ.get("PAVE_VLM_BACKEND") or os.environ.get("PAVE_VLM_RUNTIME") or "vllm-mlx").strip().lower()
        if runtime in {"mlx-vlm", "mlx_vlm", "direct", "inprocess", "in-process"}:
            return "real MLX (mlx-vlm)" if importlib.util.find_spec("mlx_vlm") else "needs mlx-vlm + ~5GB"
        return "vLLM-MLX server" if importlib.util.find_spec("vllm_mlx") else "needs vllm-mlx + ~5GB"
    if backend_key != "dino":
        return "stub"
    jepa = Path(os.environ.get("JEPA_APP_DIR", "/Users/scottphillips/Documents/jepa"))
    real = (
        importlib.util.find_spec("mlx") is not None
        and importlib.util.find_spec("mlxim") is not None
        and (jepa / "dino_engine.py").is_file()
    )
    return "dinov3 (real MLX)" if real else "fallback (NumPy)"


class PaveConsole(QWidget):
    log_line = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("OpenPAVE — MLX Operator Console")
        self.resize(1380, 880)
        self.experience_name = EXPERIENCES[0]
        self.procs: dict[str, QProcess] = {}
        # Default to Browser (no embedded ThreeJS): the QtWebEngine viewer is not
        # created until the operator picks "Internal", so Chromium isn't spun up at
        # startup — saving ~1-3 GB of Metal working set for the VLM. The Browser
        # button is highlighted by default (see _update_viz_buttons).
        self._rendering_embedded = False
        self.cap = None
        self.current_frame = None
        self._cam_fail = 0
        self._model_ready = False
        self._ui_ready = False
        # perception state — the selected model's features composited on the feed
        self.overlay_on = True
        self.last_intent = "-"
        self.last_infer_ms = 0.0
        self.last_dims = ""
        self.last_raw_text = ""            # Gemma/Qwen's raw answer, shown as a caption
        self._features = []                # (label, box) pairs Gemma/Qwen reported, for the overlay
        self.model_status = "idle"
        self._analyzing = False
        self._mode = "vlm"                 # "vlm" (shim) or "engine" (in-process)
        self.engine_handle = None
        self._overlay_rgba = None
        self.frame_ring = deque(maxlen=16)
        self._engine_worker = None       # one persistent EngineWorker per loaded model
        self._loading = False            # True between _build_engine() and ready/failed
        self._camera_job_pending = False  # self-throttle: at most one camera-tick job in flight
        self._camera_ref = None           # frame signature at the last camera inference
        self._camera_last_infer = 0.0     # time.time() of the last camera inference
        self._camera_gate_idle = False    # True while the gate is skipping (log transitions once)
        self._observations: list[str] = []  # this OBSERVE run's answers (novelty context)
        self._observe_ticks_left = 0        # >0 while a 10s OBSERVE run is active
        # continuous OBSERVE toggle state (scene-change gated — see constants)
        self._observe_ref = None            # frame signature at the last observation
        self._observe_job_pending = False   # one observation in flight, never more
        self._observe_last_request = 0.0    # time.time() of the last VLM request
        self._observe_cooldown_ms = float(OBSERVE_COOLDOWN_MS)
        self._observe_timer = QTimer(self)
        self._observe_timer.setInterval(OBSERVE_CHECK_MS)
        self._observe_timer.timeout.connect(self._observe_scene_tick)
        self._trot_streak = 0            # consecutive TROT results seen (TROT confirmation gate)
        self._trot_streak_started_ms = 0.0
        self._pending_intent = None      # last intent word POSTed, awaiting completion
        self._pending_since = 0.0        # for the fallback timeout in _action_in_flight()
        self._last_camera_posted_intent = None  # idempotency: rate-limit steady-state camera repeats
        self._last_camera_posted_at = 0.0
        self._last_logged_inference_sig = None  # console quieting: don't log unchanged camera results
        # Falcon Perception — annotated bounding boxes + segmentation masks drawn
        # ON TOP of the video feed (docs/detection-guide.md). Loads lazily (only
        # when enabled); its own MLX runtime, independent of the VLM engine.
        self.falcon_worker = None
        self.falcon_enabled = False
        self.falcon_query = FALCON_DEFAULT_QUERY
        self.falcon_task = "detection"      # fast 300M boxes by default; "segmentation" -> 1.5B + masks
        self.falcon_dets: list = []         # list[perception.Detection], normalised corners
        self.falcon_last_query = ""
        self.falcon_last_s = 0.0
        self._falcon_busy = False        # at most one detect job in flight (self-throttle)

        self.state_server = StateServer(port=int(os.environ.get("STATE_SERVER_PORT", "7080")))
        self.viewer_url = self.state_server.start()
        self.browser_url = self.viewer_url  # replaced by the vite URL on "Browser"

        self.log_line.connect(self._append_log)
        self._build_ui()
        self._ui_ready = True

        self._start_control_plane()

        # create timers BEFORE loading a model (the model load uses dl_poll_timer)
        self.ingress_timer = QTimer(self); self.ingress_timer.timeout.connect(self._check_ingress_ready)
        self.ingress_timer.start(400)
        self.capture_timer = QTimer(self); self.capture_timer.timeout.connect(self._tick)
        self.infer_timer = QTimer(self); self.infer_timer.timeout.connect(self._infer_tick)
        self.disk_timer = QTimer(self); self.disk_timer.timeout.connect(self._refresh_model_availability)
        self.disk_timer.start(8000)             # disk frees/fills as downloads run
        self.dl_poll_timer = QTimer(self); self.dl_poll_timer.timeout.connect(self._poll_download)
        self.falcon_timer = QTimer(self); self.falcon_timer.timeout.connect(self._falcon_tick)

        self._refresh_model_availability()      # grey out models that won't fit on disk
        # Kick off the default model load ~100ms after this constructor returns
        # (i.e. after the window has painted and is on-screen), not synchronously
        # here — so the console appears immediately instead of the whole app
        # looking hung while Gemma loads.
        QTimer.singleShot(100, self._start_default_model)

        self._init_cameras()
        self._log(f"state viewer streaming at {self.viewer_url}")
        self._show_experience_view()

    # ── UI ──────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        mono = QFont("Menlo"); mono.setStyleHint(QFont.StyleHint.Monospace); mono.setPointSize(11)
        # compact controls so the three rows stay tidy at the very top
        self.setStyleSheet(
            "QComboBox{padding:1px 6px; min-height:20px;} QPushButton{padding:2px 10px;}"
            " QLabel{font-size:12px;} QCheckBox{font-size:12px;}"
        )

        # Row 1 — Experience + Visualiser render target
        self.experience_box = QComboBox(); self.experience_box.addItems(EXPERIENCES)
        self.experience_box.currentTextChanged.connect(self._select_experience)
        self.info = QLabel("starting…")
        # Two explicit buttons instead of a single toggle: Browser mode frees the
        # embedded QtWebEngine (Chromium) page so its ~1-3 GB returns to Metal for
        # the VLM. Internal re-embeds it. Layout is otherwise unchanged.
        self.viz_internal_btn = QPushButton("Internal (QtWebEngine)")
        self.viz_browser_btn = QPushButton("Browser")
        for b in (self.viz_internal_btn, self.viz_browser_btn):
            b.setCheckable(True)
        self.viz_internal_btn.clicked.connect(lambda: self._set_visualiser("internal"))
        self.viz_browser_btn.clicked.connect(lambda: self._set_visualiser("browser"))
        if QWebEngineView is None:                       # no embedded engine available
            self.viz_internal_btn.setEnabled(False)
        row1 = self._row()
        row1.addWidget(QLabel("Experience:")); row1.addWidget(self.experience_box)
        row1.addStretch(); row1.addWidget(self.info)
        row1.addWidget(QLabel("Visualiser:")); row1.addWidget(self.viz_internal_btn); row1.addWidget(self.viz_browser_btn)

        # Row 2 — Vision model + live status (replaces the on-frame status band)
        # + free disk aligned right. The model name lives in the dropdown, so the
        # status label deliberately omits it (no duplication): "[state] intent: … Nms".
        self.model_box = QComboBox(); self.model_box.addItems(MODELS)
        self.model_box.currentTextChanged.connect(self._select_model)
        self.status_label = QLabel("")   # live: [state] intent + dims + ms
        self.status_label.setStyleSheet("color:#555555; font-family:Menlo,monospace;")
        self.disk_label = QLabel("")     # "n GB free", right-aligned
        self.disk_label.setStyleSheet("font-family:Menlo,monospace;")
        # Retained (set elsewhere) but no longer shown in the row — their content is
        # folded into status_label / disk_label to reduce the duplicated readouts.
        self.head_label = QLabel(""); self.mode_label = QLabel("")
        self.dl_label = QLabel(""); self.endpoint_label = QLabel("endpoint: —")
        row2 = self._row()
        row2.addWidget(QLabel("Model:")); row2.addWidget(self.model_box)
        row2.addWidget(self.status_label)
        row2.addStretch(); row2.addWidget(self.disk_label)

        # Row 3 — runtime
        kind, device = detect_compute_backend()
        self.compute_label = QLabel(f"Compute: {kind} · {device}")
        self.camera_box = QComboBox(); self.camera_box.currentIndexChanged.connect(self._select_camera)
        self.source_box = QComboBox(); self.source_box.addItems(SOURCES)
        self.source_box.currentTextChanged.connect(self._select_source)
        self.overlay_chk = QCheckBox("Feature overlay"); self.overlay_chk.setChecked(True)
        self.overlay_chk.toggled.connect(lambda v: setattr(self, "overlay_on", v))
        # Pin the loaded model's weights as wired (non-evictable) so Metal can't
        # page them out under memory pressure. On by default; uncheck to A/B it.
        self.wired_chk = QCheckBox("Pin weights (wired)")
        self.wired_chk.setChecked(mlx_mem.available())
        self.wired_chk.setEnabled(mlx_mem.available())
        self.wired_chk.setToolTip("Pin the loaded VLM's weights in Metal memory (wired limit) so "
                                  "they aren't evicted under pressure. Uncheck to compare.")
        self.wired_chk.toggled.connect(self._toggle_wired)
        self.plane_label = QLabel("control plane: starting…")
        row3 = self._row()
        row3.addWidget(self.compute_label); row3.addWidget(self._sep())
        row3.addWidget(QLabel("Camera:")); row3.addWidget(self.camera_box)
        row3.addWidget(QLabel("Input:")); row3.addWidget(self.source_box)
        row3.addWidget(self.overlay_chk); row3.addWidget(self.wired_chk)
        row3.addStretch(); row3.addWidget(self.plane_label)

        header = QVBoxLayout(); header.setContentsMargins(0, 0, 0, 0); header.setSpacing(3)
        header.addLayout(row1); header.addLayout(row2); header.addLayout(row3)

        # Falcon Perception controls — enable + open-vocabulary query + task. The
        # ANNOTATION (bounding boxes + segmentation masks) is drawn on the video
        # feed itself (see _draw_overlay / _draw_detections), per detection-guide.md.
        self.falcon_chk = QCheckBox("Falcon boxes")
        self.falcon_chk.setToolTip("Open-vocabulary detector/segmenter (Falcon Perception, MLX). "
                                   "Detection uses the fast 300M model (~real-time); Segmentation "
                                   "uses the 1.5B model for instance masks (slower).")
        self.falcon_chk.toggled.connect(self._toggle_falcon)
        self.falcon_query_edit = QLineEdit(self.falcon_query)
        self.falcon_query_edit.setMaximumWidth(180)
        self.falcon_query_edit.setPlaceholderText("query, e.g. person, hand, cup")
        self.falcon_query_edit.textChanged.connect(self._falcon_set_query)
        self.falcon_task_box = QComboBox()
        self.falcon_task_box.addItems(["Detection", "Segmentation"])  # fast 300M boxes / 1.5B boxes+masks
        self.falcon_task_box.currentTextChanged.connect(self._falcon_set_task)
        self.falcon_status = QLabel("")
        self.falcon_status.setStyleSheet("color:#9ad; font-family:Menlo,monospace;")
        falcon_bar = self._row()
        falcon_bar.addWidget(self.falcon_chk)
        falcon_bar.addWidget(QLabel("query:")); falcon_bar.addWidget(self.falcon_query_edit)
        falcon_bar.addWidget(QLabel("task:")); falcon_bar.addWidget(self.falcon_task_box)
        falcon_bar.addWidget(self.falcon_status); falcon_bar.addStretch()

        # Column 1 — Falcon controls, camera preview (annotated), prompt probes
        self.camera_view = QLabel("camera preview", alignment=Qt.AlignmentFlag.AlignCenter)
        self.camera_view.setMinimumSize(360, 240)
        self.camera_view.setStyleSheet("background:#0c0e10; color:#789; border:1px solid rgba(255,255,255,0.12);")
        self.prompt_status = QLabel("Prompt probes: load a VLM, then click a button to drive intent_ingress -> mock adapter.")
        self.prompt_status.setWordWrap(True)
        self.prompt_status.setStyleSheet("color:#9fb6a0; font-family:Menlo,monospace;")
        prompt_rows = QVBoxLayout(); prompt_rows.setContentsMargins(0, 0, 0, 0); prompt_rows.setSpacing(4)
        prompt_row = self._row()
        self.prompt_buttons: list[QPushButton] = []
        for i, (label, prompt) in enumerate(PROMPT_BUTTONS):
            btn = QPushButton(label)
            btn.setToolTip(prompt)
            btn.setEnabled(False)  # greyed out until a VLM is loaded — see _set_prompt_buttons_enabled
            btn.clicked.connect(lambda _=False, p=prompt, l=label: self._run_prompt_probe(l, p))
            prompt_row.addWidget(btn)
            self.prompt_buttons.append(btn)
            if i == 2:
                prompt_rows.addLayout(prompt_row)
                prompt_row = self._row()
        self.observe_btn = QPushButton("10s OBSERVE")
        self.observe_btn.setToolTip(
            "Say what the camera sees once a second for 10 seconds — each observation "
            "must add something not already mentioned."
        )
        self.observe_btn.setEnabled(False)  # gated with the other probe buttons
        self.observe_btn.clicked.connect(self._start_observe)
        prompt_row.addWidget(self.observe_btn)
        self.prompt_buttons.append(self.observe_btn)
        self.observe_toggle = QPushButton("OBSERVE")
        self.observe_toggle.setCheckable(True)
        self.observe_toggle.setToolTip(
            "Keep observing: speak only when the scene visibly changes. A cheap frame "
            "diff gates each model request, with a cooldown and repeat back-off, so an "
            "unchanged scene costs no GPU time (and no fan noise)."
        )
        self.observe_toggle.setEnabled(False)  # gated with the other probe buttons
        self.observe_toggle.toggled.connect(self._toggle_observe)
        prompt_row.addWidget(self.observe_toggle)
        self.prompt_buttons.append(self.observe_toggle)
        prompt_row.addStretch()
        prompt_rows.addLayout(prompt_row)
        prompt_box = QWidget()
        prompt_layout = QVBoxLayout(prompt_box)
        prompt_layout.setContentsMargins(0, 4, 0, 0)
        prompt_layout.setSpacing(4)
        prompt_layout.addWidget(self.prompt_status)
        prompt_layout.addLayout(prompt_rows)
        camera_col = QWidget()
        camera_layout = QVBoxLayout(camera_col)
        camera_layout.setContentsMargins(0, 0, 0, 0)
        camera_layout.setSpacing(6)
        camera_layout.addLayout(falcon_bar, 0)          # Falcon controls, above the webcam
        camera_layout.addWidget(self.camera_view, 1)    # annotated video feed (boxes + masks)
        camera_layout.addWidget(prompt_box, 0)

        # Right column — ThreeJS visualiser over console
        self.console = QPlainTextEdit(); self.console.setReadOnly(True)
        self.console.setFont(mono); self.console.setMaximumBlockCount(4000)
        self.console.setStyleSheet("background:#0c0e10; color:#cfe8cf; border:none;")

        self.content_stack = QStackedWidget()
        self.placeholder = QLabel("", alignment=Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setWordWrap(True); self.placeholder.setStyleSheet("color:#cfe8cf; font-size:14px;")
        self.content_stack.addWidget(self.placeholder)
        # web_view is created lazily (only when Internal is chosen), so no Chromium
        # process exists at startup — see _ensure_web_view.
        self.web_view = None
        visual_col = QSplitter(Qt.Orientation.Vertical)
        visual_col.addWidget(self.content_stack)
        visual_col.addWidget(self.console)
        visual_col.setStretchFactor(0, 4)
        visual_col.setStretchFactor(1, 1)

        columns = QSplitter(Qt.Orientation.Horizontal)
        columns.addWidget(camera_col); columns.addWidget(visual_col)
        columns.setStretchFactor(0, 4); columns.setStretchFactor(1, 5)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 8); root.setSpacing(6)
        root.addLayout(header)            # tidy, fixed-height top bar
        root.addWidget(columns, 1)        # camera/console/ThreeJS fill the rest
        self._update_viz_buttons()
        self._refresh_status_label()
        self._select_model(self.model_box.currentText())

    @staticmethod
    def _row() -> QHBoxLayout:
        row = QHBoxLayout(); row.setContentsMargins(0, 0, 0, 0); row.setSpacing(6)
        return row

    @staticmethod
    def _sep() -> QLabel:
        s = QLabel("|"); s.setStyleSheet("color:rgba(255,255,255,0.25);"); return s

    # ── logging ─────────────────────────────────────────────────────
    def _log(self, text: str) -> None:
        self.log_line.emit(f"[pave_ui] {text}")

    def _append_log(self, line: str) -> None:
        self.console.appendPlainText(line)

    # ── experience / render ─────────────────────────────────────────
    def _select_experience(self, name: str) -> None:
        self.experience_name = name
        self._show_experience_view()

    def _show_experience_view(self) -> None:
        if self.experience_name == "Physics Simulator":
            self._placeholder(
                "Physics Simulator\n\nThe pave_sim digital twin is not wired yet "
                "(docs §5.1). Use 'Robot State Viewer' for the live STL + stream."
            )
            return
        wv = self._ensure_web_view() if self._rendering_embedded else self.web_view
        if self._rendering_embedded and wv is not None:
            wv.setUrl(QUrl(self.viewer_url))
            self.content_stack.setCurrentWidget(wv)
        else:
            self._placeholder(f"Robot State Viewer open in your browser:\n{self.browser_url}")

    def _ensure_web_view(self):
        """Create the embedded QtWebEngine view on first use (lazy — no Chromium at
        startup). Returns None if QtWebEngine is unavailable."""
        if self.web_view is None and QWebEngineView is not None:
            self.web_view = QWebEngineView()
            self.content_stack.addWidget(self.web_view)
        return self.web_view

    def _placeholder(self, message: str) -> None:
        self.placeholder.setText(message)
        self.content_stack.setCurrentWidget(self.placeholder)

    def _set_visualiser(self, mode: str) -> None:
        """mode 'internal' embeds the QtWebEngine viewer; 'browser' frees it.

        Internal embeds the BASIC single-file viewer (visualiser/index.html,
        served by the state server) — light, no webcam. Browser mode blanks the
        embedded page so QtWebEngine releases the ThreeJS scene and its GPU
        surface (~1-3 GB back to Metal for the VLM), starts the SvelteKit dev
        server, and opens IT in the system browser — that app needs webcam
        access, which only a real browser grants."""
        if mode == "internal" and QWebEngineView is None:
            mode = "browser"  # QtWebEngine not installed -> browser only
        if mode == "browser":
            self._rendering_embedded = False
            if self.web_view is not None:
                self.web_view.setUrl(QUrl("about:blank"))  # release the embedded page
            mlx_mem.clear_cache()
            if not BROWSER_USES_VITE:
                self.browser_url = self.viewer_url
                webbrowser.open(self.viewer_url)
                self._log(f"visualiser -> Browser (basic viewer; vite gated off): {self.viewer_url}")
            else:
                url, starting = self._ensure_vite()
                self.browser_url = url
                if starting:
                    # open localhost only AFTER vite accepts connections —
                    # opening on a fixed delay raced a cold start (blank page).
                    self._open_when_listening(url, VITE_PORT)
                else:
                    webbrowser.open(url)
                self._log(f"visualiser -> Browser (SvelteKit + webcam): {url}")
        else:
            self._rendering_embedded = True
            self._log("visualiser -> Internal (QtWebEngine, basic viewer)")
        self._update_viz_buttons()
        self._show_experience_view()

    def _ensure_vite(self) -> tuple[str, bool]:
        """(url, starting_now) for the SvelteKit visualiser; start vite once.

        Reuses a dev server that is already listening (yours or a previous
        run's). Falls back to the basic state-server viewer when Node/vite are
        unavailable, so the Browser button never opens a dead URL."""
        # localhost (not a raw IP) so the page is a secure context -> webcam allowed
        url = f"http://localhost:{VITE_PORT}"
        try:
            with socket.create_connection(("127.0.0.1", VITE_PORT), timeout=0.2):
                return url, False
        except OSError:
            pass
        node = _node_binary()
        vite_js = REPO_ROOT / "visualiser" / "node_modules" / "vite" / "bin" / "vite.js"
        if node is None or not vite_js.is_file():
            self._log("vite unavailable (need Node >=20.19 and `pnpm install` in visualiser/); "
                      "opening the basic viewer instead")
            return self.viewer_url, False
        args = [str(vite_js), "dev", "--port", str(VITE_PORT)]
        if VITE_PORT < 1024:
            # macOS quirk: unprivileged binds to low ports (like 888) only work
            # on the wildcard interface — 127.0.0.1:888 is EPERM. --host makes
            # vite bind 0.0.0.0; the browser still opens localhost.
            args += ["--host", "0.0.0.0"]
        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.setProcessEnvironment(self._proc_env())
        proc.setWorkingDirectory(str(REPO_ROOT / "visualiser"))
        proc.readyReadStandardOutput.connect(lambda p=proc: self._drain(p, "vite"))
        proc.start(node, args)
        self.procs["vite"] = proc
        self._log(f"vite dev server starting on {url} (SvelteKit visualiser)")
        return url, True

    def _open_when_listening(self, url: str, port: int, timeout_s: float = 20.0) -> None:
        """Open the system browser once the freshly-spawned server accepts
        connections. If vite dies or never binds, fall back to the basic viewer
        so the button never lands on a dead page."""
        deadline = time.time() + timeout_s

        def poll() -> None:
            proc = self.procs.get("vite")
            vite_dead = proc is None or proc.state() == QProcess.ProcessState.NotRunning
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    pass
            except OSError:
                if time.time() < deadline and not vite_dead:
                    QTimer.singleShot(250, poll)
                    return
                self._log(f"vite never came up on {url}; opening the basic viewer instead")
                self.browser_url = self.viewer_url
                webbrowser.open(self.viewer_url)
                self._show_experience_view()
                return
            webbrowser.open(url)
            self._show_experience_view()

        QTimer.singleShot(300, poll)

    def _update_viz_buttons(self) -> None:
        self.viz_internal_btn.setChecked(self._rendering_embedded)
        self.viz_browser_btn.setChecked(not self._rendering_embedded)

    # ── MLX memory: wired-limit pin (keep weights resident) ─────────────
    def _apply_wired_limit(self) -> None:
        """Pin the loaded model's weights (checkbox on) or release the pin (off).
        Sized to the model's peak so it fits under the Metal working set."""
        if not mlx_mem.available():
            return
        if self.wired_chk.isChecked():
            nbytes = mlx_mem.suggested_wired_bytes(mlx_mem.peak_bytes() or None)
            prev = mlx_mem.set_wired_limit(nbytes)
            snap = mlx_mem.snapshot()
            self._log(f"[mem] wired pin ON -> {nbytes / 1e9:.1f}GB "
                      f"(was {(prev or 0) / 1e9:.1f}GB); active {snap.get('active', 0) / 1e9:.1f}GB "
                      f"peak {snap.get('peak', 0) / 1e9:.1f}GB")
        else:
            mlx_mem.set_wired_limit(0)
            self._log("[mem] wired pin OFF (weights may be evicted under pressure)")

    def _toggle_wired(self, on: bool) -> None:
        self._apply_wired_limit()

    # ── model row → manage the inference shim ───────────────────────
    def _refresh_model_availability(self) -> None:
        """Grey out models whose download won't fit, show free disk by the dropdown."""
        free = perception.free_disk_gb()
        self.disk_label.setText(f"{free:.1f} GB free")
        tight = any(perception.download_blocked(m)[0] for m in MODELS)
        self.disk_label.setStyleSheet(
            "font-family:Menlo,monospace; color:%s;" % ("#d47b5d" if tight else "#9fb6a0")
        )
        mdl = self.model_box.model()
        for i in range(self.model_box.count()):
            blocked, reason = perception.download_blocked(self.model_box.itemText(i))
            item = mdl.item(i) if hasattr(mdl, "item") else None
            if item is not None:
                item.setEnabled(True)
                item.setToolTip(reason if blocked else "")

    def _start_default_model(self) -> None:
        name = DEFAULT_MODEL if DEFAULT_MODEL in MODELS else "No model"
        if self.model_box.currentText() != name:
            self.model_box.blockSignals(True)
            self.model_box.setCurrentText(name)
            self.model_box.blockSignals(False)
        if perception.download_blocked(name)[0]:
            # default won't fit — fall back to the first model that does
            for i in range(self.model_box.count()):
                cand = self.model_box.itemText(i)
                if not perception.download_blocked(cand)[0]:
                    self.model_box.blockSignals(True)
                    self.model_box.setCurrentIndex(i)
                    self.model_box.blockSignals(False)
                    name = cand
                    break
            self._log(f"default model needs more disk; loading '{name}' instead")
        key = MODEL_BACKEND.get(name, "qwen")
        self.head_label.setText(f"head: {head_status(key)}")
        self.mode_label.setText(f"backend: {backend_mode_hint(key)}")
        self._set_model_backend(name)

    def _select_model(self, name: str) -> None:
        key = MODEL_BACKEND.get(name, "qwen")
        self.head_label.setText(f"head: {head_status(key)}")
        self.mode_label.setText(f"backend: {backend_mode_hint(key)}")
        if self._ui_ready:  # only user-initiated changes reload the model
            self._set_model_backend(name)

    def _set_model_backend(self, name: str) -> None:
        """Load the selected model IN-PROCESS (no subprocess shim → a multi-GB VLM
        download can't deadlock the UI). Encoders overlay their dense features on
        the camera; VLMs emit an intent word."""
        if self._loading:
            self._log(f"model load already in progress; ignoring selection '{name}'")
            return

        self._model_ready = False
        self.last_intent = "waiting for model"
        self.last_dims = ""
        self.model_status = "loading"
        self._overlay_rgba = None
        self.frame_ring.clear()
        self._last_camera_posted_intent = None
        self._last_camera_posted_at = 0.0
        self._last_logged_inference_sig = None
        self._teardown_engine()
        self._set_prompt_buttons_enabled(False)  # re-enabled only once a VLM is actually ready

        if name == "No model":
            self.dl_poll_timer.stop()
            self.dl_label.setText("idle")
            self.endpoint_label.setText("model: —")
            self.last_intent = "-"
            self.last_dims = ""
            self.model_status = "idle"
            self._log("startup idle: select a model to load weights")
            return

        blocked, reason = perception.download_blocked(name)
        if blocked:  # never start a download that can't fit on disk
            self.dl_label.setText(f"blocked: {reason}")
            self.last_intent = "blocked"
            self.last_dims = reason
            self.model_status = "blocked"
            self._log(f"'{name}' not loaded — {reason}. Free disk space or pick a smaller model.")
            return

        self.dl_label.setText("loading model…")
        if name in perception.VLM_MODELS and not perception.model_cached(name):
            missing = ", ".join(perception.missing_snapshot_files(name)[:2])
            self.last_dims = f"fetching {missing}" if missing else "fetching weights"
            self.model_status = "downloading"
        if name in perception.VLM_MODELS:
            runtime = (os.environ.get("PAVE_VLM_BACKEND") or os.environ.get("PAVE_VLM_RUNTIME") or "vllm-mlx").strip().lower()
            self.endpoint_label.setText("model: vllm-mlx server" if "vllm" in runtime else "model: mlx-vlm direct")
        else:
            self.endpoint_label.setText("model: in-process")
        self._build_engine(name)

    # ── in-process model engine (off the UI thread, like WORKING/17) ─
    def _build_engine(self, name: str) -> None:
        if self._loading:
            self._log(f"model load already in progress; not starting '{name}'")
            return
        tier = "via vllm-mlx server" if name in perception.VLM_MODELS else "in-process"
        self._log(f"loading model {tier}: {name}")
        self.model_box.setEnabled(False)
        self._loading = True
        # ONE persistent worker per engine: it loads the model AND serves every
        # inference job for this engine's lifetime from the same native thread
        # (see perception.EngineWorker for why that matters for MLX/Metal).
        self._engine_worker = perception.EngineWorker(name)
        self._engine_worker.progress.connect(self._on_engine_progress)
        self._engine_worker.ready.connect(self._on_engine_ready)
        self._engine_worker.failed.connect(self._on_engine_failed)
        self._engine_worker.done.connect(self._on_engine_infer)
        self._engine_worker.start()
        # mirror the hf download % into the GUI by polling the cache (safe — no fd tricks)
        if name in perception.VLM_MODELS:
            self.dl_poll_timer.start(1000)
        else:
            self.dl_poll_timer.stop()

    def _poll_download(self) -> None:
        pct = perception.download_progress(self.model_box.currentText())
        if pct is not None and not self._model_ready:
            self.dl_label.setText(f"downloading {pct:.0f}%")
            self.model_status = f"downloading {pct:.0f}%"
            if not self.last_dims:
                self.last_dims = "fetching weights"

    def _on_engine_progress(self, pct: int, status: str) -> None:
        self.dl_label.setText(f"{status} {pct}%")
        self.model_status = status

    def _on_engine_ready(self, handle) -> None:
        self.engine_handle = handle
        self._mode = handle.kind
        self.frame_ring.clear()
        self.dl_poll_timer.stop()
        self.last_intent = "-"
        self._model_ready = True
        self._loading = False
        self.model_box.setEnabled(True)
        # surface whether the REAL model loaded or it fell back, and why
        eng_mode = getattr(handle.engine, "mode", "ready")
        err = getattr(handle.engine, "load_error", "")
        status = getattr(handle.engine, "load_status", "")
        if eng_mode == "loaded":                 # it's now fully on disk -> mark cached
            perception.mark_cached(handle.name)
        self.dl_label.setText(eng_mode)
        self.mode_label.setText(f"backend: {eng_mode}")
        self.model_status = eng_mode
        if err:
            self.last_dims = status or err
        else:
            if handle.kind == "vlm":
                if self.source_box.currentText() == "Camera (VLM)":
                    self.last_intent = "waiting for frame"
                    self.last_dims = "VLM loaded; waiting for first image+prompt inference"
                    self.model_status = "ready"
                else:
                    self.last_intent = "camera idle"
                    self.last_dims = "select Camera (VLM) to run image+prompt inference"
                    self.model_status = "loaded"
            else:
                self.last_dims = status or ""
        self._set_prompt_buttons_enabled(handle.kind == "vlm" and not err)
        self._refresh_model_availability()
        # Memory hygiene without killing the overlay: keep the Falcon detector
        # (its annotated boxes/masks) resident ALONGSIDE the VLM, and only free it
        # if memory is genuinely tight (would risk evicting VLM weights). On a roomy
        # Mac both coexist, so the annotated overlay survives the VLM load.
        if handle.kind == "vlm" and eng_mode == "loaded" and self.falcon_worker is not None \
                and mlx_mem.under_pressure():
            self._log("[mem] memory tight — freeing Falcon detector to protect VLM speed")
            self._free_falcon_for_vlm()
        mlx_mem.clear_cache()
        self._apply_wired_limit()
        self._refresh_status_label()
        self._log(f"model ready: {handle.name} ({handle.kind}, mode={eng_mode})")
        if err:
            self._log(f"  model note: {err}")

    def _set_prompt_buttons_enabled(self, enabled: bool) -> None:
        """Grey out (and make unclickable) the prompt-probe buttons until a VLM
        has actually finished loading — clicking them with no model ready used
        to just show an error message; now it's not possible to click at all."""
        for btn in getattr(self, "prompt_buttons", []):
            btn.setEnabled(enabled)

    def _on_engine_failed(self, msg: str) -> None:
        self.engine_handle = None
        self._model_ready = False
        self._loading = False
        self.model_box.setEnabled(True)
        self.dl_label.setText("engine unavailable")
        self.last_intent = "unavailable"
        self.last_dims = msg
        self.model_status = "unavailable"
        self.dl_poll_timer.stop()
        self._log(f"engine load failed: {msg} (needs the jepa MLX stack at JEPA_APP_DIR)")

    def _teardown_engine(self) -> None:
        w = self._engine_worker
        if w is not None:
            for sig in (w.progress, w.ready, w.failed, w.done):
                try:
                    sig.disconnect()
                except Exception:
                    pass
            w.stop()  # push the sentinel so its run() loop can exit
            if w.isRunning():
                w.wait(3000)
        self._engine_worker = None
        self.engine_handle = None
        self._overlay_rgba = None
        self._analyzing = False
        self._camera_job_pending = False
        self._loading = False
        self.last_raw_text = ""
        self._features = []
        self._last_logged_inference_sig = None
        mlx_mem.clear_cache()  # return the torn-down model's buffers to the OS

    # ── camera ──────────────────────────────────────────────────────
    def _init_cameras(self) -> None:
        if cv2 is None:
            self.camera_view.setText("OpenCV not installed\n(pip install opencv-python)\ncamera preview disabled")
            self.camera_box.addItem("none", None)
            self.source_box.setCurrentText("Idle")
            return
        cams = enumerate_cameras()
        if not cams:
            self.camera_view.setText("no camera found\n(staying idle)")
            self.camera_box.addItem("none", None)
            self.source_box.setCurrentText("Idle")
            return
        for label, index, backend in cams:
            self.camera_box.addItem(label, (index, backend))
        self._select_camera(0)
        # Auto-start live inference (no click needed). _infer_tick self-gates on
        # self._model_ready, so this is safe before Gemma finishes loading — the
        # first inference simply runs the moment the model is ready.
        if self.cap is not None and DEFAULT_SOURCE in SOURCES:
            self.source_box.setCurrentText(DEFAULT_SOURCE)

    def _select_camera(self, idx: int) -> None:
        if cv2 is None:
            return
        data = self.camera_box.itemData(idx)
        if self.cap is not None:
            self.cap.release(); self.cap = None
        if not data:
            return
        index, backend = data
        self.cap = _quick_open(index, backend)
        if self.cap is None:
            self.camera_view.setText(f"failed to open {self.camera_box.itemText(idx)}")
        else:
            self.capture_timer.start(33)  # ~30 FPS preview
            if self.falcon_enabled and self.falcon_worker is not None:
                self.falcon_timer.start(FALCON_INTERVAL_MS)

    def _tick(self) -> None:
        if self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self._cam_fail += 1
            if self._cam_fail == 60:                     # ~2s of dead frames -> recover once
                self._log("camera stalled; reopening…")
                self._select_camera(self.camera_box.currentIndex())
            elif self._cam_fail > 240:                   # give up rather than spin forever
                self.capture_timer.stop()
                self.camera_view.setText("camera lost — reselect a camera to resume")
            return
        self._cam_fail = 0
        self.current_frame = frame                       # clean frame for inference
        engine_handle = self.engine_handle
        if engine_handle is not None and engine_handle.kind != "vlm":  # feed encoders square frames
            self.frame_ring.append(cv2.resize(frame, (engine_handle.img_size, engine_handle.img_size)))
        # PASS 1 (frame resolution): camera + feature heatmap + translucent Falcon
        # segmentation masks + status band + VLM feature boxes, then scale to the
        # display widget. Masks belong here — they read as background tint.
        disp = self._draw_overlay(frame.copy())
        rgb = np.ascontiguousarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))
        h, w, _ = rgb.shape
        img = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(img).scaled(
            self.camera_view.width(), self.camera_view.height(),
            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation,
        )
        # PASS 2 (already-scaled pixmap): annotated bounding boxes + labels in a
        # SEPARATE, LATER QPainter — the last pixels written, crisp (never blurred
        # by the scale above) and never covered (z-index guarantee, detection-guide
        # §1.4). BOTH sources go through the same painter: Falcon's real per-instance
        # boxes and the VLM's own self-reported FEATURE grid cells (Detection structs).
        paint_dets = []
        if self.falcon_enabled and self.falcon_dets:
            paint_dets.extend(self.falcon_dets)
        if self._features:                               # VLM self-report (list[Detection])
            paint_dets.extend(self._features)
        if paint_dets:
            painter = QPainter(pixmap)
            self._draw_detections(painter, pixmap.width(), pixmap.height(), paint_dets)
            painter.end()
        self.camera_view.setPixmap(pixmap)
        self._refresh_status_label()   # live status now lives in Row 2, not on the frame

    def _draw_overlay(self, frame):
        """Pass 1 (frame resolution): the SELECTED model's feature map (swaps per
        model), Falcon segmentation masks, and — for VLM backends — a caption band
        with the model's raw answer.

        The live status (`[state] intent … ms`) is NOT drawn on the frame anymore —
        it lives in Row 2 next to the model dropdown (see `_refresh_status_label`).
        The annotated BOXES (Falcon's and the VLM's self-reported `FEATURE` cells)
        are drawn on top in _tick's second QPainter pass (_draw_detections).
        """
        if cv2 is None:
            return frame
        # Falcon segmentation masks: translucent per-instance fills UNDER every
        # other annotation (the boxes/labels go on top in _tick's second QPainter
        # pass). Independent of the "Feature overlay" toggle — this is the Falcon
        # detector, not the selected model's dense-feature heatmap.
        if self.falcon_enabled and self.falcon_dets:
            perception.draw_falcon_masks(frame, self.falcon_dets)
        if not self.overlay_on:
            return frame
        if self._overlay_rgba is not None:               # the model's dense features
            frame = perception.composite(frame, self._overlay_rgba)
        if self.engine_handle is not None and self.engine_handle.kind == "vlm":
            perception.draw_caption(frame, self.last_raw_text)  # boxes drawn in _tick's pass 2
        return frame

    def _refresh_status_label(self) -> None:
        """Update the Row 2 live status: the info that used to be the on-frame band,
        minus the model name (the dropdown already shows it)."""
        state = f"[{self.model_status}] " if self.model_status else ""
        extra = f"   {self.last_dims}" if self.last_dims else ""
        ms = f"   {self.last_infer_ms:.0f}ms" if self.last_infer_ms else ""
        self.status_label.setText(f"{state}intent: {self.last_intent}{extra}{ms}")

    # ── Falcon Perception: annotated boxes + masks on the video feed ──
    def _falcon_set_query(self, text: str) -> None:
        self.falcon_query = text.strip() or FALCON_DEFAULT_QUERY

    def _falcon_set_task(self, text: str) -> None:
        self.falcon_task = "detection" if text.lower().startswith("detect") else "segmentation"

    def _toggle_falcon(self, on: bool) -> None:
        self.falcon_enabled = on
        if on:
            if cv2 is None:
                self.falcon_status.setText("OpenCV unavailable")
                self.falcon_chk.setChecked(False)   # re-enters here with on=False
                return
            if self.engine_handle is not None and self.engine_handle.kind == "vlm" \
                    and mlx_mem.under_pressure():
                self._log("[mem] note: Falcon + VLM resident and memory is tight — watch for eviction")
            self._ensure_falcon_worker()
            if self.cap is not None:
                self.falcon_timer.start(FALCON_INTERVAL_MS)
            self._log(f"falcon on (query={self.falcon_query!r}, task={self.falcon_task})")
        else:
            self._teardown_falcon()          # free the detector model + its buffers
            self.falcon_status.setText("")
            self._log("falcon off")

    def _teardown_falcon(self) -> None:
        """Stop the Falcon detector and free its model so only one big model stays
        resident. Re-enabling reloads it (300M ~1s)."""
        if getattr(self, "falcon_timer", None) is not None:
            self.falcon_timer.stop()
        self._falcon_busy = False
        self.falcon_dets = []
        w = self.falcon_worker
        if w is not None:
            for sig in (w.progress, w.ready, w.failed, w.detections_ready):
                try:
                    sig.disconnect()
                except Exception:
                    pass
            w.stop()
            if w.isRunning():
                w.wait(3000)
            self.falcon_worker = None
            mlx_mem.clear_cache()

    def _free_falcon_for_vlm(self) -> None:
        """Called when a VLM becomes the active model: drop the Falcon detector so
        the VLM is the only big model resident (uncheck the box to match)."""
        if self.falcon_worker is None and not self.falcon_enabled:
            return
        self._log("[mem] freeing Falcon detector to keep one big model resident")
        self._teardown_falcon()
        if self.falcon_enabled:
            self.falcon_enabled = False
            self.falcon_chk.blockSignals(True)
            self.falcon_chk.setChecked(False)
            self.falcon_chk.blockSignals(False)
            self.falcon_status.setText("freed (VLM active)")

    def _ensure_falcon_worker(self) -> None:
        """Spin up the persistent Falcon detector thread once (lazy load)."""
        if self.falcon_worker is not None:
            return
        self.falcon_status.setText("loading Falcon…")
        w = perception.FalconDetectorWorker()
        w.progress.connect(self.falcon_status.setText)
        w.ready.connect(self._on_falcon_ready)
        w.failed.connect(self._on_falcon_failed)
        w.detections_ready.connect(self._on_falcon_detections)
        w.start()
        self.falcon_worker = w

    def _on_falcon_ready(self, status: str) -> None:
        self.falcon_status.setText(f"Falcon: {status}")
        self._log(f"falcon ready: {status}")
        if "loaded" not in status:   # fallback/unavailable -> stop polling
            self.falcon_timer.stop()

    def _on_falcon_failed(self, msg: str) -> None:
        self.falcon_timer.stop()
        self._falcon_busy = False
        self.falcon_status.setText(f"Falcon unavailable: {msg}")
        self._log(f"falcon load failed: {msg}")

    def _on_falcon_detections(self, dets, query: str, task: str, dt: float) -> None:
        self._falcon_busy = False
        self.falcon_dets = list(dets or [])          # list[perception.Detection]
        self.falcon_last_query = query
        self.falcon_last_s = dt
        self.falcon_status.setText(f"Falcon: {query} ×{len(self.falcon_dets)} ({task}) · {dt:.1f}s")

    def _falcon_tick(self) -> None:
        """Submit the current frame to the Falcon detector, one job at a time."""
        if not self.falcon_enabled or self.current_frame is None or cv2 is None:
            return
        if self.falcon_worker is None or self._falcon_busy:
            return
        self._falcon_busy = True
        self.falcon_worker.submit(self.current_frame.copy(), self.falcon_query, self.falcon_task)

    def _draw_detections(self, painter: QPainter, sw: int, sh: int, dets: list) -> None:
        """Paint annotated bounding boxes + labels onto the already-scaled pixmap
        (docs/detection-guide.md §1.5), for any list of `perception.Detection`
        (Falcon's real boxes AND the VLM's self-reported grid cells). Detections
        are normalised [0,1] corners mapped to the scaled pixmap; OpenPAVE's preview
        is NOT mirrored (unlike WORKING/17's selfie view), so no x-flip. `det.fill`
        (coarse regions, e.g. VLM grid cells) adds a translucent box fill under the
        outline; `det.cls_id` picks the palette colour (stable per instance/label);
        the label chip sits INSIDE the box corner so an edge box is never clipped."""
        font = QFont("Menlo"); font.setPixelSize(max(12, perception.DETECTION_LABEL_PX + 1))
        painter.setFont(font)
        for det in dets:
            r, g, b = perception.FALCON_PALETTE_RGB[det.cls_id % len(perception.FALCON_PALETTE_RGB)]
            color = QColor(r, g, b)
            px1 = int(round(det.x1 * sw)); px2 = int(round(det.x2 * sw))
            py1 = int(round(det.y1 * sh)); py2 = int(round(det.y2 * sh))
            x1, x2 = max(0, min(px1, sw - 1)), max(0, min(px2, sw))
            y1, y2 = max(0, min(py1, sh - 1)), max(0, min(py2, sh))
            bw, bh = max(1, x2 - x1), max(1, y2 - y1)

            if getattr(det, "fill", False):           # coarse region -> translucent fill under outline
                fill = QColor(r, g, b); fill.setAlpha(70)
                painter.fillRect(x1, y1, bw, bh, fill)
            box_pen = QPen(color); box_pen.setWidth(2)
            painter.setPen(box_pen)
            painter.drawRect(x1, y1, bw, bh)

            label = det.label
            th = max(16, perception.DETECTION_LABEL_PX + 6)
            tw = max(44, len(label) * 9 + 8)
            chip = QRect(x1 + 1, y1 + 1, tw, th)
            painter.fillRect(chip, color)             # label chip inside the box corner
            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.drawText(x1 + 4, y1 + 1 + th - 5, label)

    # ── input source ────────────────────────────────────────────────
    def _select_source(self, name: str) -> None:
        if name == "Camera (VLM)":
            if self.cap is not None:
                self.infer_timer.start(INFER_INTERVAL_MS)  # back-to-back (busy-gated), not a 1.5s wait
                self._last_logged_inference_sig = None
                if self.engine_handle is not None and self.engine_handle.kind == "vlm" and self._model_ready:
                    self.model_status = "ready"
                    self.last_intent = "waiting for frame"
                    self.last_dims = "VLM loaded; waiting for first image+prompt inference"
                self._log("input: live camera → model (gesture) → overlay + robot")
            else:
                self._log("camera unavailable; staying idle")
                self.source_box.setCurrentText("Idle")
        else:
            # Idle sends NOTHING — no fake intents, no synthetic traffic, and the
            # ROBOT STATE live time-series panel stays silent too (state_server's
            # SSE stream only pushes a frame when something actually changes —
            # see mlx-runtime/state_server.py's _significant()). Prompt-probe
            # buttons still work on demand while idle.
            self.infer_timer.stop()
            if self.engine_handle is not None and self.engine_handle.kind == "vlm" and self._model_ready:
                self.model_status = "loaded"
                self.last_intent = "camera idle"
                self.last_dims = "select Camera (VLM) to run image+prompt inference"
            self._log("input: idle — no intents posted")

    # ── camera → selected model → overlay + intent → ingress ────────
    def _infer_tick(self) -> None:
        """Periodic gesture/scene inference on the live camera frame.

        Self-throttled (at most one camera job in flight at a time) so a slow
        model can't flood the queue with stale frames — but this never blocks or
        drops a prompt-probe submission; those always go in via _run_prompt_probe
        regardless of what the camera tick is doing.
        """
        if not self._model_ready or self.current_frame is None or cv2 is None:
            return
        if self._camera_job_pending:
            return
        w, h = self._engine_worker, self.engine_handle
        if w is None or h is None:
            return
        if h.kind == "vlm":
            frames = self.current_frame.copy()           # single frame for the VLM
            if CAMERA_SCENE_GATE and not self._camera_gate_pass(frames):
                return                                   # EARLY-OUT: unchanged scene, no VLM
        elif self.frame_ring:
            frames = np.stack(list(self.frame_ring))     # [T,S,S,3] for encoders
        else:
            return
        self._camera_job_pending = True
        self._analyzing = True
        if h.kind == "vlm":
            self.model_status = "inferencing"
            self.last_intent = "thinking"
            self.last_dims = "running image+prompt inference"
        w.submit(frames, prompt=None, source="camera-model")

    def _camera_gate_pass(self, frame) -> bool:
        """True when the camera tick should actually spend a VLM request.

        Cheapest checks run first; the frame signature (one tiny GPU kernel) is
        computed only after those pass. Skipped ticks cost ~0.4ms; a keepalive
        inference runs every CAMERA_IDLE_KEEPALIVE_S so the gate can never make
        the pipeline permanently blind to slow drift."""
        now = time.time()
        if now - self._camera_last_infer > CAMERA_IDLE_KEEPALIVE_S:
            self._camera_gate_open(perception.observe_signature(frame), now, "keepalive")
            return True
        sig = perception.observe_signature(frame)
        delta = perception.scene_delta_max(self._camera_ref, sig)
        if delta < CAMERA_DIFF_THRESHOLD:
            if not self._camera_gate_idle:
                self._camera_gate_idle = True
                self._log(
                    f"camera gate: scene static (Δmax {delta:.0f} < {CAMERA_DIFF_THRESHOLD:.0f}) — "
                    f"VLM paused, keepalive every {CAMERA_IDLE_KEEPALIVE_S:.0f}s"
                )
            self.model_status = "watching"
            return False
        self._camera_gate_open(sig, now, f"Δmax {delta:.0f}")
        return True

    def _camera_gate_open(self, sig, now: float, why: str) -> None:
        if self._camera_gate_idle:
            self._camera_gate_idle = False
            self._log(f"camera gate: resuming VLM inference ({why})")
        self._camera_ref = sig
        self._camera_last_infer = now

    def _run_prompt_probe(self, label: str, prompt: str) -> None:
        """Enqueue a one-off prompt from a probe button.

        Always enqueues immediately — it never waits for, is blocked by, or
        cancels whatever the periodic camera/gesture tick is doing. The worker's
        FIFO queue is the only thing that orders the two.
        """
        h, w = self.engine_handle, self._engine_worker
        if h is None or w is None or not self._model_ready:
            self.prompt_status.setText("Load Gemma 4 E4B or Qwen3-VL before running prompt probes.")
            self._log(f"prompt probe {label} skipped: no loaded VLM")
            return
        if h.kind != "vlm":
            self.prompt_status.setText("Prompt probes require a VLM backend; select Gemma 4 E4B or Qwen3-VL.")
            self._log(f"prompt probe {label} skipped: selected backend is {h.kind}")
            return
        if self.current_frame is not None:
            frame = self.current_frame.copy()
            frame_note = "current camera frame"
        elif np is not None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame_note = "blank fallback frame"
        else:
            self.prompt_status.setText("No camera frame and NumPy is unavailable; cannot build VLM image input.")
            self._log(f"prompt probe {label} skipped: no image input")
            return
        self.prompt_status.setText(f"{label}: queued prompt on {frame_note} -> intent_ingress -> mock")
        self._log(f"prompt probe {label}: {prompt}")
        w.submit(frame, prompt=prompt, source="prompt-probe", label=label)

    # ── 10s OBSERVE: 1 observation/second, novelty required ─────────
    def _observe_prompt(self) -> str:
        if not self._observations:
            return ("You are watching a live camera stream. In ONE short sentence, "
                    "describe the most prominent thing you see.")
        seen = "; ".join(self._observations[-8:])
        return ("You are watching a live camera stream. In ONE short sentence, name "
                "something you see RIGHT NOW that is NOT in the already-seen list — "
                "a different object, person, action, or a new detail of the scene.\n"
                f"Already seen: {seen}.")

    def _start_observe(self) -> None:
        if self._observe_ticks_left > 0:
            return  # a run is already in progress
        self._observations = []
        self._observe_ticks_left = OBSERVE_TICKS
        self.observe_btn.setEnabled(False)
        self._log(f"observe: {OBSERVE_TICKS} observations at {OBSERVE_INTERVAL_MS}ms — "
                  "each must add something new")
        self._observe_tick()

    def _observe_tick(self) -> None:
        h, w = self.engine_handle, self._engine_worker
        if (h is None or w is None or not self._model_ready or h.kind != "vlm"
                or self.current_frame is None):
            self._log("observe stopped: needs a loaded VLM and a live camera frame")
            self._observe_ticks_left = 0
            self.observe_btn.setEnabled(True)
            return
        n = OBSERVE_TICKS - self._observe_ticks_left + 1
        w.submit(self.current_frame.copy(), prompt=self._observe_prompt(),
                 source="observe", label=f"observe {n}/{OBSERVE_TICKS}")
        self._observe_ticks_left -= 1
        if self._observe_ticks_left > 0:
            QTimer.singleShot(OBSERVE_INTERVAL_MS, self._observe_tick)
        else:
            self.observe_btn.setEnabled(True)

    # ── continuous OBSERVE toggle: speak only when the scene changes ─
    def _toggle_observe(self, on: bool) -> None:
        if on:
            self._observations = []
            self._observe_ref = None          # first tick always speaks once
            self._observe_job_pending = False
            self._observe_last_request = 0.0
            self._observe_cooldown_ms = float(OBSERVE_COOLDOWN_MS)
            self._observe_timer.start()
            self._log(
                f"observe toggle ON: checking every {OBSERVE_CHECK_MS}ms, speaking only "
                f"on scene change (Δ>{OBSERVE_DIFF_THRESHOLD:.0f}), cooldown "
                f"{OBSERVE_COOLDOWN_MS}ms with back-off to {OBSERVE_BACKOFF_MAX_MS}ms"
            )
        else:
            self._observe_timer.stop()
            self._publish_observation("", "", hold=False)  # toggle off -> bubble goes away
            self._log(f"observe toggle OFF ({len(self._observations)} observations this run)")

    def _observe_scene_tick(self) -> None:
        """One cheap gate pass — the expensive VLM call only happens when every
        gate opens: model ready, nothing in flight, cooldown expired, AND the
        frame visibly differs from the one last spoken about."""
        h, w = self.engine_handle, self._engine_worker
        frame = self.current_frame
        if h is None or w is None or not self._model_ready or h.kind != "vlm" or frame is None:
            return                                       # not ready — keep idling cheaply
        if self._observe_job_pending:
            return                                       # one in flight, never queue more
        if (time.time() - self._observe_last_request) * 1000.0 < self._observe_cooldown_ms:
            return                                       # thermal floor / back-off window
        sig = perception.observe_signature(frame)
        delta = perception.scene_delta(self._observe_ref, sig)
        if delta < OBSERVE_DIFF_THRESHOLD:
            return                                       # static scene: zero GPU cost
        self._observe_ref = sig                          # this frame becomes the reference
        self._observe_job_pending = True
        self._observe_last_request = time.time()
        label = "observe" if delta == float("inf") else f"observe Δ{delta:.0f}"
        w.submit(frame.copy(), prompt=self._observe_prompt(), source="observe-live", label=label)

    def _publish_observation(self, text: str, label, hold: bool = False) -> None:
        """Write the observation for the visualisers' speech bubble (streamed by
        the state server alongside robot_state/command_result).

        hold=True pins the bubble on screen (continuous OBSERVE: it must stay
        visible until the toggle goes off); hold=False lets it retire on the
        clients' 15s TTL (10s OBSERVE bursts)."""
        try:
            Path(OBSERVATION_PATH).write_text(json.dumps({
                "text": text,
                "label": str(label or ""),
                "at": time.time(),
                "hold": bool(hold),
            }), encoding="utf-8")
        except Exception as exc:  # a broken bubble must never break inference
            self._log(f"observe: could not write {OBSERVATION_PATH}: {exc}")

    def _on_engine_infer(self, result, ms: float, source: str, label) -> None:
        name = self.engine_handle.name if self.engine_handle else "model"
        if source in ("observe", "observe-live"):
            # observations are commentary, never robot commands — record, show,
            # and stop here so nothing reaches the TROT/idempotency gates below
            raw = getattr(result, "raw_text", "") or "(no reply)"
            is_repeat = raw in self._observations
            note = ""
            if source == "observe-live":
                self._observe_job_pending = False
                if is_repeat:
                    # the scene diff fired but the model saw nothing new — back
                    # off exponentially so a busy-but-boring scene (flicker,
                    # shadows, fans in frame) stops burning request_ms
                    self._observe_cooldown_ms = min(self._observe_cooldown_ms * 2.0,
                                                    float(OBSERVE_BACKOFF_MAX_MS))
                    note = f" (repeat — backing off to {self._observe_cooldown_ms / 1000:.0f}s)"
                else:
                    self._observe_cooldown_ms = float(OBSERVE_COOLDOWN_MS)
            elif is_repeat:
                note = " (repeat — no new detail)"
            self._observations.append(raw)
            self._observations = self._observations[-24:]  # bounded novelty context
            self.last_raw_text = raw           # shown as the camera caption
            self.prompt_status.setText(f"{label}: {raw}")
            self.log_line.emit(f"[observe] {label}: {raw}  ({ms:.0f}ms){note}")
            if not is_repeat or source == "observe":
                # repeats don't refresh the bubble; continuous mode pins it on
                # screen (hold) until the toggle goes off
                self._publish_observation(raw, label, hold=source == "observe-live")
            return
        if source == "camera-model":
            self._camera_job_pending = False
        self._overlay_rgba = result.overlay
        self.last_intent = result.intent
        self.last_dims = result.dims
        self.last_infer_ms = ms
        self._analyzing = False
        if self.engine_handle and self.engine_handle.kind == "vlm":
            self.model_status = "prompt ok" if getattr(result, "ok", True) else "prompt error"
            raw = getattr(result, "raw_text", "")
            if raw:
                self.last_raw_text = raw   # drawn as the camera caption — see _draw_overlay
            self._features = getattr(result, "features", []) or []  # drawn as boxes — see _draw_overlay
            suffix = f", raw={raw!r}" if raw else ""
            feature_sig = tuple((getattr(f, "label", ""), getattr(f, "x1", 0), getattr(f, "y1", 0),
                                 getattr(f, "x2", 0), getattr(f, "y2", 0)) for f in self._features)
            infer_sig = (name, result.intent, feature_sig, getattr(result, "ok", True))
            should_log_infer = source != "camera-model" or infer_sig != self._last_logged_inference_sig
            if should_log_infer:
                self.log_line.emit(f"[{name}] -> {result.intent}  ({ms:.0f}ms, {result.dims}{suffix})")
                if source == "camera-model":
                    self._last_logged_inference_sig = infer_sig
            if label:
                if getattr(result, "ok", True):
                    self.prompt_status.setText(f"{label}: model returned {result.intent}")
                else:
                    self.prompt_status.setText(f"{label}: model prompt failed; not posted")
        else:
            self.model_status = "ready"
            infer_sig = (name, result.intent, result.dims)
            should_log_infer = source != "camera-model" or infer_sig != self._last_logged_inference_sig
            if should_log_infer:
                self.log_line.emit(f"[{name}] -> {result.intent}  ({ms:.0f}ms, {result.dims})")
                if source == "camera-model":
                    self._last_logged_inference_sig = infer_sig
        if not getattr(result, "ok", True):
            return
        if not self._gate_trot(result.intent):
            if label:
                self.prompt_status.setText(
                    f"{label}: TROT needs confirmation ({self._trot_streak}/{TROT_CONFIRMATIONS}) — not posted yet"
                )
            return
        # IDEMPOTENCY GATE — critical, do not remove: a repeated identical intent
        # (e.g. an open palm held in frame, or a prompt-probe button tapped twice)
        # must be a no-op while the previous one is still executing, or a
        # continuously-recognized gesture floods intent_ingress/the daemon.
        if source == "camera-model" and result.intent == self._last_camera_posted_intent:
            elapsed_ms = (time.time() - self._last_camera_posted_at) * 1000.0
            if self._action_in_flight() or elapsed_ms < CAMERA_INTENT_REPEAT_MS:
                return
        if source != "camera-model" and result.intent == self._pending_intent and self._action_in_flight():
            self._log(f"suppressed duplicate {result.intent} — previous action still in progress")
            if label:
                self.prompt_status.setText(f"{label}: {result.intent} already in progress; not re-sent")
            return
        self._pending_intent = result.intent
        self._pending_since = time.time()
        if source == "camera-model":
            self._last_camera_posted_intent = result.intent
            self._last_camera_posted_at = self._pending_since
        # post off the UI thread — never block the event loop on the network
        threading.Thread(target=self._post_intent,
                         args=({"text": result.intent, "source": source},), daemon=True).start()

    def _action_in_flight(self) -> bool:
        """True while the last-posted intent's command_result hasn't reached a
        terminal status yet (see COMMAND_RESULT_PATH / _TERMINAL_COMMAND_STATUSES
        above). Backs the idempotency gate in _on_engine_infer."""
        if self._pending_intent is None:
            return False
        if (time.time() - self._pending_since) * 1000.0 > _PENDING_FALLBACK_MS:
            return False  # safety valve — never stay "stuck" if we can't confirm
        try:
            data = json.loads(Path(COMMAND_RESULT_PATH).read_text(encoding="utf-8"))
        except Exception:
            return False  # can't tell -> don't block
        return data.get("status") not in _TERMINAL_COMMAND_STATUSES

    def _gate_trot(self, intent: str) -> bool:
        """TROT confirmation gate, ported from the DGX Spark scenario
        (scenarios/puppypi-gesture-stop-trot.json: trot_requires_confirmation).
        TROT only actually gets forwarded to intent_ingress once it has been seen
        TROT_CONFIRMATIONS times in a row within TROT_CONFIRMATION_WINDOW_MS — from
        EITHER the gesture tick or a prompt-probe tap, since they're the same
        stream of evidence. Every other intent resets the streak immediately and
        is never gated (matches "open palm or uncertainty emits STOP" from the
        scenario's success_criteria — STOP is always immediate)."""
        if intent != "TROT":
            self._trot_streak = 0
            return True
        now_ms = time.time() * 1000.0
        if self._trot_streak == 0 or (now_ms - self._trot_streak_started_ms) > TROT_CONFIRMATION_WINDOW_MS:
            self._trot_streak_started_ms = now_ms
            self._trot_streak = 1
        else:
            self._trot_streak += 1
        if self._trot_streak < TROT_CONFIRMATIONS:
            self._log(f"TROT seen ({self._trot_streak}/{TROT_CONFIRMATIONS}) — waiting for confirmation")
            return False
        self._trot_streak = 0  # consumed — the next TROT starts a fresh streak
        return True

    # ── control plane orchestration ─────────────────────────────────
    def _proc_env(self) -> QProcessEnvironment:
        env = QProcessEnvironment.systemEnvironment(); env.insert("PYTHONUNBUFFERED", "1"); return env

    def _spawn(self, name: str, args: list[str]) -> None:
        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.setProcessEnvironment(self._proc_env())
        proc.setWorkingDirectory(str(REPO_ROOT))
        proc.readyReadStandardOutput.connect(lambda p=proc, n=name: self._drain(p, n))
        proc.start(sys.executable, args)
        self.procs[name] = proc

    def _kill(self, name: str) -> None:
        proc = self.procs.pop(name, None)
        if proc is not None:
            proc.terminate()
            if not proc.waitForFinished(1500):
                proc.kill()

    def _drain(self, proc: QProcess, name: str) -> None:
        tag = f"[{name}]"
        text = bytes(proc.readAllStandardOutput()).decode("utf-8", "replace")
        for line in text.splitlines():
            line = line.rstrip()
            if not line:
                continue
            self.log_line.emit(line if line.startswith(tag) else f"{tag} {line}")

    def _start_control_plane(self) -> None:
        self._spawn("ingress", ["-m", "intent_ingress.server"])
        self._spawn("daemon", ["-m", "control_daemon.daemon"])
        self._log("control plane spawned (intent_ingress + control_daemon, ROBOT_ADAPTER=mock)")

    def _check_ingress_ready(self) -> None:
        try:
            with socket.create_connection(("127.0.0.1", INTENT_PORT), timeout=0.1):
                pass
        except OSError:
            return
        self.ingress_timer.stop()
        self.plane_label.setText("control plane: ready")
        self.info.setText("live")
        self._log("intent ingress healthy")
        self._select_source(self.source_box.currentText())  # start the chosen input

    def _post_intent(self, payload: dict, label: str | None = None) -> None:
        label = label or payload.get("text") or payload.get("intent")
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(f"http://127.0.0.1:{INTENT_PORT}/intent", data=data,
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            urllib.request.urlopen(req, timeout=1.5).read()
            source = payload.get("source", "intent")
            if source in {"prompt-probe", "camera-model"}:
                self.log_line.emit(f"[{source}] sent {label} to intent_ingress")
        except Exception as exc:  # noqa: BLE001
            self.log_line.emit(f"[{payload.get('source', 'intent')}] {label} failed: {exc}")

    # ── shutdown ────────────────────────────────────────────────────
    def _shutdown(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self._teardown_falcon()
        self._teardown_engine()
        for name in list(self.procs):
            self._kill(name)
        try:
            self.state_server.stop()
        except Exception:
            pass

    def closeEvent(self, event) -> None:
        self._shutdown()
        super().closeEvent(event)


_BENIGN_QT = (
    "DIR_APP_DICTIONARIES", "propagateSizeHints", "qt.qpa.fonts",
    "Populating font family", "qtwebengine_dictionaries",
)


def _qt_message_filter(mode, ctx, message) -> None:
    if not any(b in message for b in _BENIGN_QT):
        sys.stderr.write(message + "\n")


def main() -> None:
    import signal

    qInstallMessageHandler(_qt_message_filter)  # drop known-benign WebEngine/Qt noise
    app = QApplication(sys.argv)
    console = PaveConsole()
    console.show()
    app.aboutToQuit.connect(console._shutdown)
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    signal.signal(signal.SIGTERM, lambda *_: app.quit())
    keepalive = QTimer(); keepalive.timeout.connect(lambda: None); keepalive.start(200)
    print("OpenPAVE GUI running. Control-plane CLI is healthy — its logs stream in "
          "the in-app Console panel (not duplicated here).", flush=True)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
