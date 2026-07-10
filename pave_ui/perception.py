"""In-process perception engines + feature overlay (ports WORKING/17).

The selected *encoder* model is loaded IN-PROCESS (DINOv3 / V-JEPA 2.1 / LingBot
from JEPA_APP_DIR), run on the live camera frames, and ITS features are composited
on top of the webcam — and the overlay swaps when the model changes:

  DINOv3      → ViT-S/16 patch tokens  [1,1,14,14,384]  → PCA-RGB feature map
  V-JEPA 2.1  → ViT-B/16 spacetime cube [1,T,24,24,768] → PCA-RGB feature map
  LingBot-Map → point cloud                              → depth-binned map

VLMs (Qwen/Gemma) emit text, not dense features, so they stay in the subprocess
shim (handled in viewer.py); this module is only the feature-overlay path.
"""

from __future__ import annotations

import os
import json
import queue
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from PyQt6.QtCore import QThread, pyqtSignal

JEPA_APP_DIR = os.environ.get("JEPA_APP_DIR", "/Users/scottphillips/Documents/jepa")
if JEPA_APP_DIR and JEPA_APP_DIR not in sys.path:
    sys.path.insert(0, JEPA_APP_DIR)
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ENGINE_MODELS = {"DINOv3", "V-JEPA 2.1", "LingBot-Map"}
# VLM titles are DERIVED from each backend's checkpoint id (basename), so the
# dropdown, the CLI preflight, and the HF cache dirs on disk always agree —
# "Qwen3-VL-4B-Instruct-3bit", never a hand-written alias that can drift.
# Approximate download sizes (GB) are keyed by backend, label maps derived.
_VLM_SIZE_BY_KEY = {
    "qwen": 2.5,
    "qwen_2b": 1.78,
    "qwen_8b": 5.4,
    "qwen35_2b": 4.5,
    "fourier_qwen2vl_2b": 4.42,
    "gemma_e2b": 4.3,
    "gemma": 6.83,
}
from pave_mlx.backends import checkpoint_label as _checkpoint_label  # noqa: E402

VLM_MODELS = {_checkpoint_label(key): key for key in _VLM_SIZE_BY_KEY}
DEFAULT_FEATURE = os.environ.get("PAVE_FEATURE", "pca_rgb")
# Live VLM tuning (see infer()). These do NOT make Gemma sub-second — its cost is
# prefill of ~280 image tokens through an ~8B model (~1.6s), a hard floor — but
# they keep per-frame cost minimal and stable: cap oversized frames to the model's
# own working size (896, no upscale blur, trims CPU preprocessing) and cap a
# runaway reply. Disciplined replies (see ROBOT_PROMPT) hit EOS well under the cap.
VLM_INPUT_SIZE = int(os.environ.get("PAVE_VLM_INPUT_SIZE", "448"))
# Qwen3-VL tokenizes the camera frame at native resolution, so its prefill cost
# scales with input size (measured, 4B-3bit on vllm-mlx, honest no-cache numbers:
# ~2060ms @ 896px, ~1320ms @ 672px, ~900ms @ 448px, ~600ms @ 112px
# — same INTENT/FEATURE quality
# on the coarse 3x3-grid task). Gemma re-encodes to a fixed size internally, so
# this knob only matters for Qwen.
VLM_INPUT_SIZE_QWEN = int(os.environ.get("PAVE_VLM_INPUT_SIZE_QWEN", "448"))
VLM_MAX_TOKENS = int(os.environ.get("PAVE_VLM_MAX_TOKENS", "48"))
VLM_FAST_INTENT_ONLY = os.environ.get("PAVE_VLM_FAST_INTENT_ONLY", "0") == "1"


def vlm_input_size(model_name: str) -> int:
    """Camera input size for a VLM (Qwen: smaller square = fewer image tokens)."""
    return VLM_INPUT_SIZE_QWEN if "qwen" in (model_name or "").lower() else VLM_INPUT_SIZE


_MLX_DATA_RESIZE_FAILED = False


def _resize_with_mlx_data(img: np.ndarray, size: int) -> np.ndarray:
    import mlx.data as dx

    sample = {"image": np.ascontiguousarray(img)}
    resized = dx.buffer_from_vector([sample]).image_resize("image", size, size)[0]["image"]
    return np.ascontiguousarray(np.asarray(resized, dtype=np.uint8))


def resize_vlm_frame(img: np.ndarray, size: int) -> tuple[np.ndarray, str]:
    """Resize a live VLM frame through the selected backend without upscaling."""
    if max(img.shape[:2]) <= size:
        return img, "none"
    backend = os.environ.get("PAVE_VLM_RESIZE_BACKEND", "cv2").strip().lower()
    global _MLX_DATA_RESIZE_FAILED
    if backend in {"mlx-data", "mlx_data", "mlx.data"} and not _MLX_DATA_RESIZE_FAILED:
        try:
            return _resize_with_mlx_data(img, size), "mlx.data"
        except Exception:
            _MLX_DATA_RESIZE_FAILED = True
    if cv2 is not None:
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA), "cv2"
    return img, "none"

# Approximate download sizes (GB) for the multi-GB VLMs — used by the pre-flight
# disk check so the GUI won't start a download that cannot fit.
MODEL_SIZE_GB = {label: _VLM_SIZE_BY_KEY[key] for label, key in VLM_MODELS.items()}
_DISK_MARGIN_GB = 0.5
_WEIGHT_EXTENSIONS = (".safetensors", ".npz", ".bin")
_RUNTIME_EXTENSIONS = (".json", ".jinja", ".txt", ".safetensors", ".npz", ".bin")
ALLOW_MODEL_DOWNLOADS = os.environ.get("PAVE_ALLOW_MODEL_DOWNLOADS", "0") == "1"
_RED = "\033[31m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RESET = "\033[0m"


def _color(text: str, color: str) -> str:
    if os.environ.get("NO_COLOR"):
        return text
    return f"{color}{text}{_RESET}"


def _fmt_bytes(size: int | float | None) -> str:
    if size is None:
        return "unknown"
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1000.0 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.2f} {unit}"
        value /= 1000.0


def hf_cache_dir() -> Path:
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return home / "hub"


def free_disk_gb() -> float:
    import shutil

    probe = hf_cache_dir()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return shutil.disk_usage(probe).free / 1e9
    except Exception:
        return 0.0


_CACHE_OK: set[str] = set()       # VLMs verified fully cached (permanent once True)
_CACHE_CHECK: dict[str, tuple] = {}  # name -> (timestamp, result) — throttle network


def mark_cached(name: str) -> None:
    if name in VLM_MODELS:
        _CACHE_OK.add(name)


def _local_model_dir(name: str) -> bool:
    """True when the backend's model id is a local directory, not a HF repo id."""
    from pave_mlx.backends import backend_model_id

    path = Path(backend_model_id(VLM_MODELS[name])).expanduser()
    return path.is_dir() and (path / "config.json").is_file()


def _repo_files(name: str):
    from pave_mlx.backends import backend_model_id

    repo = backend_model_id(VLM_MODELS[name])
    try:
        from huggingface_hub import HfApi

        return repo, [sib.rfilename for sib in HfApi().model_info(repo).siblings]
    except Exception:
        return repo, []


def model_cached(name: str) -> bool:
    """True only when the runtime snapshot for the VLM is present locally.

    The cache may contain stale ``*.incomplete`` files from interrupted downloads;
    those must never count as progress. True is memoized permanently; the one
    network call for the repo file list is throttled so periodic disk refreshes
    do not spam the Hub."""
    if name not in VLM_MODELS:
        return False
    if name in _CACHE_OK:
        return True
    if _local_model_dir(name):   # env override points at a local folder (e.g. mlx_vlm.convert output)
        _CACHE_OK.add(name)
        return True
    if _local_snapshot_complete(name):
        _CACHE_OK.add(name)
        return True
    if not ALLOW_MODEL_DOWNLOADS:
        return False
    ts, res = _CACHE_CHECK.get(name, (0.0, False))
    if time.time() - ts < 60.0:
        return res

    result = False
    try:
        from huggingface_hub import try_to_load_from_cache

        repo, files = _repo_files(name)
        required = [f for f in files if f.endswith(_RUNTIME_EXTENSIONS)]
        result = bool(required) and all(isinstance(try_to_load_from_cache(repo, f), str) for f in required)
    except Exception:
        result = False
    _CACHE_CHECK[name] = (time.time(), result)
    if result:
        _CACHE_OK.add(name)
    return result


def download_blocked(name: str) -> tuple[bool, str]:
    """(blocked, reason). Gate large VLMs unless fully cached.

    By default the GUI is cache-only for VLM weights: selecting a model must not
    start a surprise multi-GB download. Set PAVE_ALLOW_MODEL_DOWNLOADS=1 to allow
    Hugging Face to fetch missing files deliberately.
    """
    if name not in VLM_MODELS:
        return False, ""
    if model_cached(name):       # already downloaded -> loads from cache, no fetch
        return False, "cached"
    missing = ", ".join(missing_snapshot_files(name)[:3])
    reason = f"missing {missing}" if missing else "not fully cached"
    if not ALLOW_MODEL_DOWNLOADS:
        return True, f"{reason}; set PAVE_ALLOW_MODEL_DOWNLOADS=1 to download"
    size = MODEL_SIZE_GB.get(name, 5.0)
    free = free_disk_gb()
    if free >= size + _DISK_MARGIN_GB:
        return False, ""
    return True, f"needs ~{size:.0f}GB · {free:.1f}GB free"

# Default camera prompt: gesture-aware, but deliberately NOT restricted to a
# narrow STOP/TROT-only vocabulary. The model can answer with any of the five
# OpenPAVE intents from a hand gesture OR general scene reasoning; clamp_to_intent
# below always matches against the full _LABELS set. Narrowing the vocabulary
# per-prompt was considered and intentionally rejected — see
# docs/dgx-spark-mlx-port.md §10.3.
#
# The prompt also asks for up to 3 FEATURE lines — this is what backs the camera
# overlay's annotated boxes (see "visual feedback" below). A generic OpenCV
# Haar-cascade / skin-color detector was tried first and pulled entirely after
# testing showed frequent false positives; the model's OWN reported features are
# what get annotated now, not a classical-CV guess.
# IMPORTANT: this prompt is tuned for SHORT, disciplined replies, not for text
# brevity. Gemma's per-frame cost is ~90% prefill (~280 image tokens + this
# text); trimming the text saves only ~150ms, but a LOOSELY-formatted prompt
# makes the model ramble — and every extra generated token costs ~22ms of decode.
# The "EXACTLY this format, one item per line, nothing else" wording keeps replies
# to ~10-15 tokens, which is worth far more than a shorter prompt. Do not "shorten"
# this without checking that replies stay strictly one-item-per-line.
ROBOT_PROMPT = os.environ.get(
    "PAVE_ROBOT_PROMPT",
    "You control a quadruped robot from its forward camera. Reply in EXACTLY "
    "this format, one item per line, nothing else:\n"
    "INTENT: <one word: STOP, TROT, HOME, LEFT, or RIGHT>\n"
    "FEATURE: <short label> <grid cell>\n"
    "(1 to 3 FEATURE lines for things you actually see — face, hand, body, or a "
    "notable object; grid cell is exactly one of: top-left, top-center, "
    "top-right, middle-left, center, middle-right, bottom-left, bottom-center, "
    "bottom-right — pick whichever cell the feature is mostly in.)\n"
    "Gesture guide for INTENT: thumbs-up means TROT, an open palm means STOP, a "
    "closed fist means HOME, pointing left means LEFT, pointing right means "
    "RIGHT. Otherwise choose INTENT by reasoning about the scene.",
)

# Qwen3-VL does NOT respect Gemma's "one item per line" as reliably — it emits
# MULTIPLE INTENT lines (STOP...TROT...) which makes the intent unstable. This
# variant is blunt about emitting the INTENT line exactly ONCE, first, and never
# repeating it. parse_model_response also takes the first INTENT as a safety net.
ROBOT_PROMPT_QWEN = os.environ.get(
    "PAVE_ROBOT_PROMPT_QWEN",
    "You control a quadruped robot from its forward camera.\n"
    "Output ONLY lines in this format, nothing else, no prose:\n"
    "INTENT: <STOP|TROT|HOME|LEFT|RIGHT>\n"
    "FEATURE: <label> <cell>\n"
    "Each FEATURE line has ONE short label (a noun like face, hand, body) then ONE "
    "cell. cell is one of: top-left top-center top-right middle-left center "
    "middle-right bottom-left bottom-center bottom-right.\n"
    "Example (copy this shape exactly):\n"
    "INTENT: STOP\n"
    "FEATURE: face top-center\n"
    "FEATURE: hand middle-right\n"
    "Rules: exactly ONE INTENT line, first, never repeated. 1-3 FEATURE lines, each "
    "a label THEN a cell. Gestures set INTENT: thumbs-up=TROT, open palm=STOP, "
    "fist=HOME, point-left=LEFT, point-right=RIGHT; no gesture -> STOP.",
)
# Qwen3.5-2B follows the strict template perfectly but never flags pointing on
# its own (0/4 measured) — the extra rule makes it either set a direction or
# emit a 'pointing hand' FEATURE, and EITHER mention triggers the direction
# follow-up in infer(), whose focused answer (4/4 measured) overrides the
# unreliable inline guess (~50%). Non-pointing gestures went 8/8 with this
# prompt vs 6/8 without — the emphasis line sharpens the whole gesture read.
ROBOT_PROMPT_QWEN35 = os.environ.get(
    "PAVE_ROBOT_PROMPT_QWEN35",
    ROBOT_PROMPT_QWEN + (
        "\nIf a finger points toward a side of the picture, INTENT MUST be LEFT or "
        "RIGHT (the side the finger points toward), and one FEATURE label must be "
        "'pointing hand'."
    ),
)
ROBOT_PROMPT_FAST_INTENT = os.environ.get(
    "PAVE_ROBOT_PROMPT_FAST_INTENT",
    "Reply with exactly one word and nothing else: STOP, TROT, HOME, LEFT, or RIGHT. "
    "Use the image: thumbs-up=TROT, open palm=STOP, fist=HOME, point-left=LEFT, "
    "point-right=RIGHT; no clear gesture or unsafe scene=STOP.",
)
# Fourier Qwen2-VL's camera prompt. Every intent-word/template variant measured
# WORSE on the HaGRID gesture set: option lists get parroted (first word wins),
# and demanding a direction makes it answer Left/Right for everything. The bare
# name question is what this model does best — 8/8 on non-pointing gestures and
# a reliable "Pointing" (4/4) that triggers the direction follow-up below.
ROBOT_PROMPT_GESTURE_NAME = os.environ.get(
    "PAVE_ROBOT_PROMPT_GESTURE_NAME",
    "What hand gesture is the person making? Answer with a short phrase only. "
    "If there is no hand, answer NONE.",
)
# Second stage, sent only when the reply says "pointing" without a direction:
# one extra ~0.7s request for pointing frames only, nothing for the rest.
POINTING_DIRECTION_PROMPT = os.environ.get(
    "PAVE_POINTING_DIRECTION_PROMPT",
    "The person is pointing with a finger. Toward which side of the picture "
    "does the finger point? Answer with exactly one word: LEFT or RIGHT.",
)


def pointing_needs_direction(text: str) -> bool:
    """True when the model saw sideways pointing — run the direction follow-up.

    The focused LEFT-or-RIGHT question measures far more reliable than any
    inline direction (Qwen3.5: follow-up 4/4 vs inline ~50%), so an inline
    LEFT/RIGHT does NOT suppress the follow-up — it gets overridden. Grid-cell
    names (top-right, middle-left, ...) are stripped first so a FEATURE line
    like "pointing hand top-right" can't masquerade as a direction. Vertical
    or at-camera pointing is not a turn command and skips the follow-up."""
    up = (text or "").upper()
    if "POINT" not in up:
        return False
    cleaned = re.sub(r"\b(TOP|MIDDLE|BOTTOM)-(LEFT|RIGHT|CENTER)\b", " ", up)
    return not re.search(r"\b(UP|UPWARD|DOWN|DOWNWARD|FORWARD|CAMERA)\b", cleaned)


def prompt_for_model(model_name: str) -> str:
    """Pick the camera prompt tuned for the loaded model (Qwen needs a stricter one)."""
    name = (model_name or "").lower()
    if VLM_FAST_INTENT_ONLY:
        return ROBOT_PROMPT_FAST_INTENT
    if "fourier" in name:
        # Fourier Qwen2-VL 2B parrots strict templates — it answers a template's
        # example intent for every frame (measured 4/12 on a HaGRID gesture set,
        # all STOP) and echoes "<a|b|c>" placeholders verbatim. The bare
        # gesture-name question + the pointing direction follow-up in infer()
        # scores best (10/12). This prompt yields no FEATURE lines; infer()
        # draws the coarse "vision center" marker so the overlay stays visibly
        # alive.
        return ROBOT_PROMPT_GESTURE_NAME
    if "qwen3.5" in name or "qwen35" in name:
        return ROBOT_PROMPT_QWEN35  # strict template + pointing flag (see above)
    return ROBOT_PROMPT_QWEN if "qwen" in name else ROBOT_PROMPT


def vlm_max_tokens(prompt: str | None) -> int:
    if prompt is None and VLM_FAST_INTENT_ONLY:
        return min(VLM_MAX_TOKENS, 4)
    return VLM_MAX_TOKENS


# ── continuous OBSERVE: scene-change gate, computed on the GPU ──────────────
# A VLM observation costs ~700-900ms of GPU (request_ms). The gate that decides
# whether to spend that runs as ONE tiny MLX kernel on the same GPU (grayscale
# + 32x32 mean-pool of the frame) — no cv2/CPU image work, so gating never
# competes with the UI or camera threads for CPU while the model is busy. The
# residual CPU cost is a 32x32 numpy diff (~a microsecond). NumPy fallback
# keeps the gate working when MLX is unavailable.

try:
    import mlx.core as _mx
except Exception:  # pragma: no cover - MLX optional
    _mx = None


def observe_signature(image_bgr: np.ndarray) -> np.ndarray:
    """32x32 grayscale float thumbnail — the fingerprint of a frame.

    Heavy work (grayscale + pooling over the full frame) happens on the GPU via
    MLX; only the final 32x32 result is materialised for the caller."""
    h, w = image_bgr.shape[:2]
    hc, wc = (h // 32) * 32, (w // 32) * 32
    if _mx is not None and hc and wc:
        x = _mx.array(np.ascontiguousarray(image_bgr[:hc, :wc])).astype(_mx.float32)
        sig = x.mean(axis=2).reshape(32, hc // 32, 32, wc // 32).mean(axis=(1, 3))
        return np.array(sig, dtype=np.float32)
    gray = image_bgr.astype(np.float32).mean(axis=2)
    ys = np.linspace(0, gray.shape[0] - 1, 32).astype(int)
    xs = np.linspace(0, gray.shape[1] - 1, 32).astype(int)
    return gray[np.ix_(ys, xs)]


def scene_delta(sig_a: np.ndarray | None, sig_b: np.ndarray | None) -> float:
    """Mean absolute pixel difference (0..255) between two 32x32 signatures.

    Returns +inf when either side is missing, so the first frame of an OBSERVE
    session always counts as "changed" and produces an opening observation."""
    if sig_a is None or sig_b is None:
        return float("inf")
    return float(np.abs(sig_a - sig_b).mean())


def scene_delta_max(sig_a: np.ndarray | None, sig_b: np.ndarray | None) -> float:
    """Largest single-cell |Δ| (0..255) between two 32x32 signatures.

    The camera gesture gate uses this instead of the mean: a hand raised in one
    corner changes a few cells a lot while barely moving the global mean, and a
    gesture must never be missed. Global nuisance shifts (exposure/lighting)
    move every cell a little, so they stay under a max-cell threshold that a
    real gesture clears easily."""
    if sig_a is None or sig_b is None:
        return float("inf")
    return float(np.abs(sig_a - sig_b).max())


_LABELS = ["STOP", "TROT", "HOME", "LEFT", "RIGHT"]

# Small models (Fourier Qwen2-VL 2B, measured) often answer the camera prompt
# with the gesture NAME ("Thumbs-up", "FIST") instead of the mapped intent
# word. The name still identifies the intent unambiguously, so accept it as a
# fallback instead of discarding the reply as "no intent token" (which would
# silently command STOP). Substring matches on purpose: THUMB covers
# thumbs-up/thumb up, WAV covers wave/waving.
_GESTURE_SYNONYMS = [
    ("THUMB", "TROT"),
    ("PALM", "STOP"),
    ("WAV", "STOP"),
    ("FIST", "HOME"),
    ("PUNCH", "HOME"),
    ("NONE", "STOP"),   # the gesture-name prompt answers NONE for "no hand"
]


def has_intent_token(text: str) -> bool:
    up = (text or "").upper()
    if any(tok in up for tok in _LABELS):
        return True
    return any(word in up for word, _ in _GESTURE_SYNONYMS)


def clamp_to_intent(text: str) -> str:
    up = (text or "").upper()
    best, best_i = None, len(up) + 1
    for tok in _LABELS:
        i = up.find(tok)
        if 0 <= i < best_i:
            best, best_i = tok, i
    if best:
        return best
    for word, intent in _GESTURE_SYNONYMS:
        if word in up:
            return intent
    return "STOP"


# ── visual feedback: "what the model sees" ──────────────────────────────────
# An earlier version of this overlay ran a generic OpenCV Haar-cascade
# face/body detector plus an HSV skin-color contour heuristic for "hand", and
# drew small labelled boxes from those. In practice it produced frequent,
# confident-looking false positives on ordinary background — worse than useless,
# since a confidently-wrong box is misleading. It was pulled entirely.
#
# What replaced it: ask Gemma/Qwen ITSELF to report what it sees, via the
# `FEATURE: <label> <grid-cell>` lines in ROBOT_PROMPT above, and annotate boxes
# from THAT — real (if coarse) model output, not a classical-CV fabrication. A
# small on-device VLM cannot reliably regress precise pixel coordinates, so the
# model only has to pick one of 9 named grid cells, which maps deterministically
# to a fixed box below. That's a real tradeoff (coarse, not tight-fitting boxes)
# but every box drawn is something the model actually reported, not a guess
# layered on top of it.
GRID_CELLS = {
    "top-left": (0, 0), "top-center": (1, 0), "top-right": (2, 0),
    "middle-left": (0, 1), "center": (1, 1), "middle-right": (2, 1),
    "bottom-left": (0, 2), "bottom-center": (1, 2), "bottom-right": (2, 2),
}
_FEATURE_LINE_RE = re.compile(r"^\s*FEATURE:\s*([A-Za-z0-9][A-Za-z0-9 _-]{0,19})\s+([A-Za-z-]+)\s*$", re.IGNORECASE)
_INTENT_LINE_RE = re.compile(r"^\s*INTENT:\s*([A-Za-z]+)", re.IGNORECASE)
_MAX_FEATURES = 3
DETECTION_LABEL_PX = 13
DETECTION_LABEL_FOREGROUND = (255, 255, 255)
def _grid_box_norm(cell: str):
    """Named 3x3 grid cell -> NORMALISED [0,1] corners (x1,y1,x2,y2), or None if
    the model misspelled the cell. Normalised so the model's self-reported feature
    goes through the SAME Detection pipeline that draws Falcon's boxes
    (viewer._draw_detections maps normalised corners to any display size)."""
    pos = GRID_CELLS.get((cell or "").strip().lower())
    if pos is None:
        return None
    col, row = pos
    return col / 3.0, row / 3.0, (col + 1) / 3.0, (row + 1) / 3.0


def parse_model_response(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Split the model's structured reply into (intent_text, [(label, grid_cell), ...]).

    Degrades gracefully: prompt-probe buttons ask for a bare one-word answer
    with no INTENT:/FEATURE: lines at all — in that case intent_text falls back
    to the whole response (clamp_to_intent scans it as before) and the feature
    list is simply empty, so no boxes get drawn for those calls. A small
    quantized model not perfectly following the format is expected sometimes;
    this never raises, it just finds fewer/no FEATURE lines that tick.
    """
    intent_text = ""
    features: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for line in (text or "").splitlines():
        m = _INTENT_LINE_RE.match(line)
        if m:
            if not intent_text:      # FIRST INTENT wins — Qwen3-VL emits several
                intent_text = m.group(1)  # (STOP...TROT...); the first is its primary answer
            continue
        m = _FEATURE_LINE_RE.match(line)
        if m:
            pair = (m.group(1).strip(), m.group(2).strip())
            if pair not in seen:     # de-dupe repeated FEATURE lines (Qwen repeats them)
                seen.add(pair)
                features.append(pair)
    if not intent_text:
        intent_text = text
    return intent_text, features[:_MAX_FEATURES]


def features_to_detections(features: list[tuple[str, str]]) -> list["Detection"]:
    """(label, grid_cell) pairs from the VLM's self-report -> `Detection` structs
    in normalised corners, so the model's OWN reported features render through the
    exact same compositing as Falcon: a translucent cell fill (marked `fill=True`)
    with a crisp box + label chip on top in viewer._draw_detections' later QPainter
    pass. Cell names the model misspelled are dropped rather than guessed. Colour
    is stable per label (not per index), so a given label keeps its colour across
    ticks even as the visible feature set changes. `fill=True` distinguishes these
    coarse 3x3-grid regions from Falcon's pixel-accurate instance masks."""
    out: list[Detection] = []
    for label, cell in features:
        box = _grid_box_norm(cell)
        if box is None:
            continue
        cls_id = sum(map(ord, label.lower())) % len(FALCON_PALETTE_RGB)
        out.append(Detection(box[0], box[1], box[2], box[3], 1.0, cls_id, label, mask=None, fill=True))
    return out


# ── Falcon Perception: annotated bounding boxes + masks on the video feed ────
# Implements docs/detection-guide.md Part 2 (Falcon Perception port) against
# OpenPAVE's webcam feed: a REAL open-vocabulary detector/segmenter (unlike the
# VLM's coarse self-reported 3x3 grid cells) returning tight per-instance boxes
# AND pixel-accurate instance masks for any natural-language query.
#
# The bounding boxes are drawn in a SEPARATE, LATER QPainter pass on the
# already-scaled pixmap (viewer._draw_detections / _tick) so they are the last
# pixels written and never blurred by the scale step or covered by anything —
# the z-index rule of §1.4/§1.5. The translucent segmentation masks are the one
# thing drawn UNDERNEATH (frame-resolution fill, draw_falcon_masks below), since
# masks read as background tint, boxes/labels as foreground annotation.
#
# Palette mirrors pave_mlx.backends._DET_PALETTE_RGB (same order) so a box here
# matches the same instance in the segment_reason overlay.
FALCON_PALETTE_RGB = [
    (99, 102, 241), (16, 185, 129), (245, 158, 11), (239, 68, 68),
    (139, 92, 246), (6, 182, 212), (236, 72, 153), (34, 197, 94),
]


def falcon_palette_bgr(i: int) -> tuple[int, int, int]:
    r, g, b = FALCON_PALETTE_RGB[i % len(FALCON_PALETTE_RGB)]
    return (b, g, r)


class Detection:
    """One Falcon detection in NORMALISED frame coords [0,1] (x1<x2, y1<y2).

    Mirrors the reference `Detection` struct in docs/detection-guide.md §1.1 (the
    caller maps normalised corners to whatever display size it wants), extended
    with an optional boolean instance `mask` at full frame resolution and its
    pixel area for the segmentation task. Falcon emits no per-box confidence/class
    id, so `score`/`cls_id` are placeholders and `label` is the query text (§2.4).
    """

    __slots__ = ("x1", "y1", "x2", "y2", "score", "cls_id", "label", "mask", "area_px", "fill")

    def __init__(self, x1, y1, x2, y2, score, cls_id, label, mask=None, area_px=0, fill=False):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.score, self.cls_id, self.label = score, cls_id, label
        self.mask, self.area_px = mask, area_px
        # fill=True -> draw a translucent box fill under the outline (coarse region,
        # e.g. a VLM self-reported grid cell); Falcon boxes rely on their pixel mask.
        self.fill = fill


def backend_dets_to_detections(dets: list, label: str) -> list["Detection"]:
    """FalconPerceptionBackend.detect() dicts -> `Detection` structs.

    The backend returns each box as a normalised centre+size (`cx,cy,w,h`) plus an
    optional mask; this converts centre+size -> normalised corners exactly as
    docs/detection-guide.md §2.4's `falcon_bboxes_to_detections` does, and carries
    the instance mask through for the segmentation task. `label` (the query text)
    becomes the box label since Falcon returns one flat list of boxes it decided
    match the query, with no per-box class/score."""
    out: list[Detection] = []
    for i, d in enumerate(dets):
        cx, cy = d.get("cx", 0.0), d.get("cy", 0.0)
        w, h = d.get("w", 0.0), d.get("h", 0.0)
        out.append(Detection(
            max(0.0, cx - w / 2.0), max(0.0, cy - h / 2.0),
            min(1.0, cx + w / 2.0), min(1.0, cy + h / 2.0),
            1.0, i, f"{label} {i + 1}", d.get("mask"), int(d.get("mask_area_px", 0)), fill=False))
    return out


def draw_falcon_masks(frame_bgr: np.ndarray, dets: list, alpha: float = 0.45) -> None:
    """Blend translucent per-instance segmentation masks onto the frame (BGR,
    full resolution). Colours match each box (FALCON_PALETTE_RGB). Drawn UNDER the
    boxes — the boxes/labels themselves go on top in a later QPainter pass
    (viewer._draw_detections), per the z-index rule of docs/detection-guide.md §1.4.
    Detection-task detections carry no mask and are skipped here (boxes only)."""
    if cv2 is None:
        return
    h, w = frame_bgr.shape[:2]
    for det in dets:
        mask = getattr(det, "mask", None)
        if mask is None or mask.shape != (h, w):
            continue
        color = np.asarray(falcon_palette_bgr(det.cls_id), dtype=np.float32)
        region = frame_bgr[mask].astype(np.float32)
        frame_bgr[mask] = (region * (1.0 - alpha) + color * alpha).astype(np.uint8)


def draw_caption(frame_bgr: np.ndarray, text: str) -> None:
    """Draw the model's own raw answer as a caption band at the bottom of the
    frame, alongside (not instead of) the feature boxes above."""
    if cv2 is None or not text:
        return
    h, w = frame_bgr.shape[:2]
    band_h = 24
    cv2.rectangle(frame_bgr, (0, h - band_h), (w, h), (16, 14, 12), -1)
    shown = " ".join(text.split())
    if len(shown) > 90:
        shown = shown[:87] + "..."
    cv2.putText(frame_bgr, shown, (8, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (207, 232, 207), 1, cv2.LINE_AA)


# 5-anchor viridis LUT (ported from WORKING/17).
_ANCHORS = np.array([(68, 1, 84), (62, 74, 137), (38, 130, 142),
                     (53, 183, 121), (253, 231, 37)], np.float32)
_POS = np.linspace(0.0, 1.0, len(_ANCHORS))


def colorize_scalar(m: np.ndarray) -> np.ndarray:
    """[G,G] in [0,1] -> [G,G,3] uint8 viridis (RGB)."""
    flat = m.ravel()
    rgb = np.stack([np.interp(flat, _POS, _ANCHORS[:, c]) for c in range(3)], -1)
    return rgb.reshape(*m.shape, 3).astype(np.uint8)


def _model_blobs_dir(name: str):
    from pave_mlx.backends import backend_model_id

    repo = backend_model_id(VLM_MODELS[name])
    return hf_cache_dir() / ("models--" + repo.replace("/", "--")) / "blobs"


def _model_snapshot_dir(name: str) -> Path | None:
    from pave_mlx.backends import backend_model_id

    repo = backend_model_id(VLM_MODELS[name])
    root = hf_cache_dir() / ("models--" + repo.replace("/", "--"))
    refs_main = root / "refs" / "main"
    snapshots = root / "snapshots"
    if refs_main.is_file():
        snapshot = snapshots / refs_main.read_text(encoding="utf-8").strip()
    else:
        dirs = [p for p in snapshots.glob("*") if p.is_dir()] if snapshots.exists() else []
        snapshot = max(dirs, key=lambda p: p.stat().st_mtime) if dirs else None
    if not snapshot or not snapshot.exists():
        return None
    return snapshot


def _snapshot_runtime_files(name: str) -> list[str]:
    snapshot = _model_snapshot_dir(name)
    if snapshot is None:
        return []
    try:
        return sorted(path.name for path in snapshot.iterdir() if path.name.endswith(_RUNTIME_EXTENSIONS))
    except OSError:
        return []


def _indexed_weight_files(name: str) -> tuple[list[str], int | None]:
    snapshot = _model_snapshot_dir(name)
    if snapshot is None:
        return [], None
    index = snapshot / "model.safetensors.index.json"
    if not index.exists():
        return [], None
    try:
        data = json.loads(index.read_text(encoding="utf-8"))
    except Exception:
        return [], None
    return sorted(set(data.get("weight_map", {}).values())), data.get("metadata", {}).get("total_size")


def model_download_report(name: str) -> dict[str, object]:
    """Deterministic local cache report for a VLM before mlx-vlm may download."""
    if name not in VLM_MODELS:
        return {}
    from pave_mlx.backends import backend_model_id

    repo = backend_model_id(VLM_MODELS[name])
    snapshot = _model_snapshot_dir(name)
    runtime_files = _snapshot_runtime_files(name)
    weight_files, indexed_total = _indexed_weight_files(name)
    expected_files = sorted(set(runtime_files) | set(weight_files))
    present: list[dict[str, object]] = []
    missing: list[str] = []
    present_weight_bytes = 0
    indexed_weight_set = set(weight_files)

    if snapshot is None:
        missing.append("snapshot")

    for filename in expected_files:
        path = snapshot / filename if snapshot is not None else None
        if path is not None and path.is_file():
            size = path.stat().st_size
            present.append({"name": filename, "size": size})
            if filename in indexed_weight_set or (not indexed_weight_set and filename.endswith(_WEIGHT_EXTENSIONS)):
                present_weight_bytes += size
        else:
            missing.append(filename)

    incomplete = []
    for path in incomplete_cache_files(name):
        try:
            incomplete.append({"name": path.name, "size": path.stat().st_size})
        except OSError:
            pass

    expected_weight_bytes = int(indexed_total or _snapshot_weight_size(name) or MODEL_SIZE_GB.get(name, 0) * 1e9)
    missing_weight_bytes = max(0, expected_weight_bytes - present_weight_bytes)
    if missing_weight_bytes > 0 and not any(item in missing for item in ("snapshot", "weights")):
        missing.append("weights")
    return {
        "name": name,
        "repo": repo,
        "snapshot": str(snapshot) if snapshot else None,
        "expected_files": expected_files,
        "present": present,
        "missing": missing,
        "incomplete": incomplete,
        "expected_weight_bytes": expected_weight_bytes,
        "present_weight_bytes": present_weight_bytes,
        "missing_weight_bytes": missing_weight_bytes,
        "allow_downloads": ALLOW_MODEL_DOWNLOADS,
    }


def print_model_download_report(name: str, stream=None) -> None:
    """Print a human-readable, colorized cache/download expectation report."""
    stream = stream or sys.stderr
    report = model_download_report(name)
    if not report:
        return
    expected_files = report["expected_files"]
    present = report["present"]
    missing = report["missing"]
    incomplete = report["incomplete"]
    print("", file=stream)
    print(f"[openpave] VLM cache preflight: {report['name']} ({report['repo']})", file=stream)
    print(f"[openpave]   runtime files : {len(present)}/{len(expected_files)} present", file=stream)
    print(
        f"[openpave]   weights       : {_fmt_bytes(report['present_weight_bytes'])} present / "
        f"{_fmt_bytes(report['expected_weight_bytes'])} expected / "
        f"{_fmt_bytes(report['missing_weight_bytes'])} missing",
        file=stream,
    )
    if incomplete:
        total = sum(item["size"] for item in incomplete)
        print(
            _color(f"[openpave]   partials      : {len(incomplete)} incomplete files ({_fmt_bytes(total)})", _YELLOW),
            file=stream,
        )
        for item in incomplete[:4]:
            print(_color(f"[openpave]     partial {item['name']} {_fmt_bytes(item['size'])}", _YELLOW), file=stream)
    cache_complete = not missing and report["missing_weight_bytes"] == 0 and bool(expected_files)
    if not cache_complete:
        print(_color(f"[openpave]   missing      : {len(missing)} file(s)", _RED), file=stream)
        for filename in missing[:8]:
            print(_color(f"[openpave]     MISSING {filename}", _RED), file=stream)
        if report["allow_downloads"]:
            print(_color("[openpave]   action       : downloads enabled; missing files may be fetched now", _YELLOW), file=stream)
        else:
            print(_color("[openpave]   action       : downloads disabled; model load will be blocked", _RED), file=stream)
    else:
        print(_color("[openpave]   cache        : complete; no model bytes expected", _GREEN), file=stream)
    print("", file=stream)


def _local_snapshot_complete(name: str) -> bool:
    return not missing_snapshot_files(name)


def missing_snapshot_files(name: str) -> list[str]:
    """Runtime files absent from the local snapshot.

    For sharded safetensors, every shard named by the weight map must exist. This
    prevents mlx-vlm from silently fetching a missing shard when only part of a
    model is cached.
    """
    snapshot = _model_snapshot_dir(name)
    if snapshot is None:
        return ["snapshot"]
    try:
        files = [p for p in snapshot.iterdir() if p.is_file()]
    except OSError:
        return ["snapshot"]
    missing: list[str] = []
    if not any((snapshot / f).is_file() for f in ("config.json", "tokenizer_config.json")):
        missing.append("config/tokenizer")
    missing.extend(path.name for path in files if not path.exists())
    single_file_weights = any(
        p.name.endswith(_WEIGHT_EXTENSIONS) and p.stat().st_size > 0 for p in files
    )
    index = snapshot / "model.safetensors.index.json"
    if index.exists():
        try:
            data = json.loads(index.read_text(encoding="utf-8"))
            required_weights = sorted(set(data.get("weight_map", {}).values()))
        except Exception:
            required_weights = []
        shards_present = bool(required_weights) and all(
            (snapshot / w).is_file() and (snapshot / w).stat().st_size > 0 for w in required_weights
        )
        # Some repos (e.g. mlx-community/Qwen3-VL-*-3bit) ship a STALE index that
        # names shards (model-0000N-of-...) but actually consolidate the weights into
        # a single model.safetensors. Treat the model as complete when the indexed
        # shards are present OR a standalone weight file exists; only report missing
        # shards when neither is true (the real "partial download" case).
        if not shards_present and not single_file_weights:
            if not required_weights:
                missing.append("model.safetensors.index.json")
            for weight in required_weights:
                path = snapshot / weight
                if not path.is_file() or path.stat().st_size <= 0:
                    missing.append(weight)
    elif not single_file_weights:
        missing.append("weights")
    return sorted(set(missing))


def _snapshot_weight_size(name: str) -> int:
    snapshot = _model_snapshot_dir(name)
    if snapshot is None:
        return 0
    total = 0
    for path in snapshot.iterdir():
        if path.name.endswith(_WEIGHT_EXTENSIONS):
            try:
                total += path.stat().st_size
            except OSError:
                pass
    return total


def incomplete_cache_files(name: str | None = None) -> list[Path]:
    """Return stale Hugging Face partial downloads for OpenPAVE VLMs.

    Callers should delete these only when no OpenPAVE/HF download is running.
    """
    names = [name] if name else list(VLM_MODELS)
    files: list[Path] = []
    for model_name in names:
        if model_name not in VLM_MODELS:
            continue
        blobs = _model_blobs_dir(model_name)
        if blobs.exists():
            files.extend(sorted(blobs.glob("*.incomplete")))
    return files


def download_progress(name: str):
    """VLM download % estimated from completed snapshot weight files.

    Stale ``*.incomplete`` blobs are intentionally ignored. They are leftovers
    from aborted downloads, not usable cache, and counting them made the GUI show
    fake progress above the real model size.
    """
    if name not in VLM_MODELS:
        return None
    total = _snapshot_weight_size(name)
    expected = MODEL_SIZE_GB.get(name, 5.0) * 1e9
    return max(0.0, min(99.0, total / expected * 100.0)) if expected else 0.0


def build_engine(name: str, progress=None):
    """Load an encoder engine in-process (returns (engine, kind))."""
    def emit(p, s):
        if progress:
            progress(p, s)

    if name == "DINOv3":
        emit(15, "importing DINOv3"); from dino_engine import DinoV3InferenceEngine
        emit(45, "loading weights"); eng = DinoV3InferenceEngine(dtype="fp32")
        emit(100, "ready"); return eng, "feature_cube"
    if name == "V-JEPA 2.1":
        emit(10, "importing V-JEPA"); from engine import VJepaInferenceEngine
        emit(45, "loading MLX weights")
        eng = VJepaInferenceEngine(fast_mode=True, use_compiled_backend=False)
        emit(100, "ready"); return eng, "feature_cube"
    if name == "LingBot-Map":
        emit(20, "loading LingBot"); from lingbot_pointcloud_engine import LingBotPointCloudEngine
        emit(100, "ready"); return LingBotPointCloudEngine(), "point_cloud"
    if name in VLM_MODELS:
        # VLM backend. By default make_backend starts the vllm-mlx serving tier;
        # PAVE_VLM_BACKEND=mlx-vlm keeps the old direct in-process path.
        emit(5, "starting VLM backend"); from pave_mlx.backends import make_backend
        from pave_mlx.downloads import print_model_download_report

        print_model_download_report(name)
        emit(10, "downloading / loading weights")
        backend = make_backend(VLM_MODELS[name])  # mlx_vlm.load / HF download
        if getattr(backend, "mode", "") == "loaded":
            # Warm up on this same (Metal-bound) thread so the first LIVE frame
            # doesn't eat the one-time mx.compile / kernel-build cost.
            emit(90, "warming up")
            try:
                size = vlm_input_size(name)
                backend.generate(np.zeros((size, size, 3), np.uint8),
                                 prompt_for_model(name), max_tokens=4)
            except Exception:  # noqa: BLE001 - warmup is best-effort
                pass
        emit(100, "ready" if getattr(backend, "mode", "") == "loaded" else "fallback")
        return backend, "vlm"
    raise ValueError(f"no in-process engine for {name}")


@dataclass
class EngineHandle:
    name: str
    engine: object
    kind: str
    processor: object
    img_size: int
    probe: object = None


def make_handle(name: str, engine, kind: str) -> EngineHandle:
    if kind == "vlm":  # text model: no feature processor / intent head
        return EngineHandle(name, engine, kind, None, 0, None)
    img_size = int(getattr(engine, "img_size", 224))
    proc = None
    if kind == "feature_cube":
        from spatio_temporal import SpatioTemporalProcessor

        proc = SpatioTemporalProcessor(
            grid_size=int(getattr(engine, "grid_size", 14)),
            embed_dim=int(getattr(engine, "embed_dim", 384)),
        )
    return EngineHandle(name, engine, kind, proc, img_size, _load_probe(name))


def _load_probe(name: str):
    if name != "DINOv3":  # only DINOv3 has a trained intent head; others -> STOP
        return None
    try:
        from pave_mlx.heads.base import PKG_DIR, HeadManifest
        from pave_mlx.heads.embedding_probe import EmbeddingProbe

        man = HeadManifest.load(PKG_DIR / "heads" / "configs" / "dino.json")
        wp = man.weights_path()
        if man.trained and wp.is_file():
            return EmbeddingProbe.load(wp)
    except Exception:
        pass
    return None


def _probe_intent(handle: EngineHandle, cube: np.ndarray) -> str:
    if handle.probe is None:
        return "STOP"
    try:
        from pave_mlx.intent_decode import decode

        feats = cube[0].reshape(-1, cube.shape[-1]).mean(0)  # mean-pool patches
        intent, _ = decode(handle.probe.logits(feats))
        return intent
    except Exception:
        return "STOP"


def cube_overlay(handle: EngineHandle, cube: np.ndarray) -> np.ndarray:
    from spatio_temporal import FEATURE_KIND

    m = handle.processor.compute(DEFAULT_FEATURE, cube)
    rgb = (m * 255).astype(np.uint8) if FEATURE_KIND[DEFAULT_FEATURE] == "rgb" else colorize_scalar(m)
    alpha = np.full(rgb.shape[:2], 150, np.uint8)
    return np.dstack([rgb, alpha])  # [G,G,4], rgb channels are RGB


def points_overlay(res, grid: int = 64):
    pts = np.asarray(getattr(res, "points_xyz", []), np.float32)
    if pts.size == 0:
        return None
    xy, z = pts[:, :2], pts[:, 2]
    lo, rng = xy.min(0), np.ptp(xy, 0) + 1e-6
    ij = np.clip(((xy - lo) / rng * (grid - 1)).astype(int), 0, grid - 1)
    acc = np.zeros((grid, grid), np.float32)
    cnt = np.zeros((grid, grid), np.float32)
    for (i, j), zz in zip(ij, z):
        acc[j, i] += zz
        cnt[j, i] += 1
    m = np.divide(acc, np.maximum(cnt, 1))
    occupied = cnt > 0
    if occupied.any():
        lo_z, hi_z = m[occupied].min(), m[occupied].max()
        m = (m - lo_z) / (hi_z - lo_z + 1e-6)
    rgb = colorize_scalar(m)
    alpha = np.where(occupied, 190, 0).astype(np.uint8)
    return np.dstack([rgb, alpha])


def composite(frame_bgr, overlay_rgba):
    """Alpha-blend the GxG feature overlay over the full-res camera frame."""
    if overlay_rgba is None or cv2 is None:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    rgb = np.ascontiguousarray(overlay_rgba[..., :3])
    a = overlay_rgba[..., 3].astype(np.float32) / 255.0
    big = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_NEAREST)[..., ::-1]  # RGB->BGR
    biga = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)[..., None]
    out = frame_bgr.astype(np.float32) * (1.0 - biga) + big.astype(np.float32) * biga
    return out.astype(np.uint8)


@dataclass
class InferResult:
    overlay: object
    intent: str
    dims: str
    ok: bool = True
    raw_text: str = ""
    features: list = field(default_factory=list)  # list[Detection] — VLM self-report, see features_to_detections


def infer(handle: EngineHandle, frames_bgr: np.ndarray, prompt: str | None = None) -> InferResult:
    if handle.kind == "vlm":  # text out, no spatial overlay
        img = frames_bgr if frames_bgr.ndim == 3 else frames_bgr[-1]
        # Hand the model a modest square rather than a full 1280x720 frame.
        # For Gemma this only trims CPU preprocessing (it re-encodes to a fixed
        # size internally); for Qwen it directly sets the image-token count and
        # thus the prefill cost (see VLM_INPUT_SIZE_QWEN). Features are
        # normalised grid cells, so the boxes are unaffected by this resize.
        size = vlm_input_size(handle.name)
        t_resize = time.perf_counter()
        img, resize_backend = resize_vlm_frame(img, size)
        resize_ms = (time.perf_counter() - t_resize) * 1000.0
        try:
            # max_tokens only caps a runaway reply — the structured answer (INTENT +
            # up to 3 FEATURE lines) hits EOS in ~10-15 tokens, and a bare one-word
            # prompt-probe reply stops even sooner, so this never slows the hot path.
            # Camera ticks pass prompt=None -> use the prompt tuned for THIS model
            # (Qwen needs a stricter single-INTENT prompt). Prompt-probe buttons pass
            # their own prompt and are used verbatim.
            camera_prompt = prompt_for_model(handle.name)
            t_generate = time.perf_counter()
            text = handle.engine.generate(img, prompt or camera_prompt, max_tokens=vlm_max_tokens(prompt))
            forced_intent = None
            if prompt is None and pointing_needs_direction(text):
                # Two-stage pointing: small models flag pointing far more
                # reliably than they name its direction (Fourier says a bare
                # "Pointing"; Qwen3.5's inline guess is ~50% while the focused
                # question below is 4/4 measured) — so one follow-up request on
                # pointing frames only, and its answer OVERRIDES any inline
                # direction.
                direction = handle.engine.generate(img, POINTING_DIRECTION_PROMPT, max_tokens=4)
                text = f"{text} {direction}"
                d = (direction or "").upper()
                forced_intent = "LEFT" if "LEFT" in d else ("RIGHT" if "RIGHT" in d else None)
            generate_ms = (time.perf_counter() - t_generate) * 1000.0
            intent_text, feature_lines = parse_model_response(text)
            intent_ok = bool(forced_intent) or has_intent_token(intent_text)
            dims = "VLM image+prompt ok"
            if os.environ.get("PAVE_VLM_TIMINGS", "0") == "1":
                pieces = [f"resize[{resize_backend}]={resize_ms:.1f}ms", f"generate={generate_ms:.1f}ms"]
                backend_timings = getattr(handle.engine, "_last_timings", {}) or {}
                for key in ("jpeg_ms", "base64_ms", "request_ms", "total_ms"):
                    value = backend_timings.get(key)
                    if isinstance(value, (int, float)):
                        pieces.append(f"{key}={value:.1f}")
                dims = "; ".join(pieces)
            # The model's self-reported FEATURE lines become Detection structs, so
            # they composite through the exact same box/label pipeline as Falcon.
            if intent_ok and not feature_lines and (prompt is None or prompt in (ROBOT_PROMPT, ROBOT_PROMPT_QWEN)):
                # Live camera VLMs must visibly prove the vision path is active.
                # If a small model follows INTENT but omits FEATURE lines anyway,
                # draw one coarse centre marker instead of leaving the overlay blank.
                feature_lines = [("vision", "center")]
            features = features_to_detections(feature_lines)
            raw = " ".join(str(text).split())
            if len(raw) > 140:
                raw = raw[:137] + "..."
            if not intent_ok:
                return InferResult(
                    None,
                    "STOP",
                    f"VLM invalid response: no intent token ({dims})",
                    False,
                    raw,
                    features,
                )
            return InferResult(None, forced_intent or clamp_to_intent(intent_text), dims, True, raw, features)
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            if len(msg) > 120:
                msg = msg[:117] + "..."
            return InferResult(None, "STOP", f"VLM inference error: {msg}", False, "")
    res = handle.engine.process_sequence(frames_bgr)
    if hasattr(res, "points_xyz"):
        return InferResult(points_overlay(res), "STOP", f"pts {len(res.points_xyz)}")
    cube = res
    t, g, d = cube.shape[1], cube.shape[2], cube.shape[4]
    return InferResult(cube_overlay(handle, cube), _probe_intent(handle, cube), f"{t}x{g}x{g}x{d}")


class EngineWorker(QThread):
    """One persistent native thread per loaded engine — loads the model, then
    serves inference jobs from a FIFO queue for the rest of the engine's life.

    Why one thread for both: MLX/mlx-vlm's Metal stream state is bound to the
    thread that first touches it. The previous design loaded weights on one
    QThread (``EngineBuilder``) and ran every inference call on a *fresh* QThread
    (Qt spins up a new native thread each time ``start()`` is called on a
    finished QThread), so each inference call happened on a native thread MLX
    had never seen. That surfaced as
    ``RuntimeError: There is no Stream(gpu, 1) in current thread.`` Loading and
    every subsequent inference call for this engine now happen on the exact same
    native thread, so MLX only ever touches Metal from one place.

    Jobs are also the mechanism for "camera/gesture ticks and prompt-probe
    button taps never compete": both call ``submit()``, which only ever enqueues
    (never blocks, never drops, never overwrites another job's frame/prompt) —
    the queue is the sole arbiter of order, so a gesture-tick result and a
    prompt-probe result cannot race or clobber each other; they're just two
    events processed strictly in the order they arrived.
    """

    progress = pyqtSignal(int, str)
    ready = pyqtSignal(object)
    failed = pyqtSignal(str)
    done = pyqtSignal(object, float, str, object)  # result, ms, source, label

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self._jobs: "queue.Queue[tuple | None]" = queue.Queue()

    def submit(self, frames: np.ndarray, prompt: str | None = None,
               source: str = "camera-model", label: str | None = None) -> None:
        """Enqueue one inference job. Always succeeds immediately."""
        self._jobs.put((frames, prompt, source, label))

    def stop(self) -> None:
        """Ask the run() loop to exit after its current/queued jobs drain."""
        self._jobs.put(None)  # sentinel

    def run(self) -> None:
        try:
            eng, kind = build_engine(self.name, lambda p, s: self.progress.emit(p, s))
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        handle = make_handle(self.name, eng, kind)
        self.ready.emit(handle)

        while True:
            job = self._jobs.get()
            if job is None:  # stop() sentinel
                break
            frames, prompt, source, label = job
            try:
                t0 = time.perf_counter()
                result = infer(handle, frames, prompt)
                ms = (time.perf_counter() - t0) * 1000.0
            except Exception:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                continue
            self.done.emit(result, ms, source, label)
        closer = getattr(handle.engine, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                pass


# Falcon fast-path (docs/detection-guide.md + the MLX batch-inference engine):
#   - DETECTION runs on the lightweight Falcon-Perception-300M (~0.3s/frame vs
#     ~5-16s for the 1.5B) — near real-time boxes. The 300M is detection-only.
#   - SEGMENTATION (instance masks) runs on the 1.5B, the only variant with a
#     mask head. Switching task in the UI transparently (re)loads the right model.
# All tunable via env so the cadence can be dialled per machine.
FALCON_FAST_MODEL = os.environ.get("PAVE_FALCON_FAST_MODEL", "tiiuae/Falcon-Perception-300M")
FALCON_QUALITY_MODEL = os.environ.get("PAVE_FALCON_QUALITY_MODEL", "tiiuae/Falcon-Perception")
FALCON_MAX_DIM = int(os.environ.get("PAVE_FALCON_MAX_DIM", "640"))
FALCON_MAX_NEW_TOKENS = int(os.environ.get("PAVE_FALCON_MAX_NEW_TOKENS", "64"))


def _falcon_fast_model() -> str:
    """The detection model: Falcon-Perception-300M if its weights are cached, else
    fall back to the (already-cached) 1.5B so detection still works without the
    300M download. Pure local-cache check — never hits the network."""
    from pave_mlx.backends import _local_hf_snapshot

    if _local_hf_snapshot(FALCON_FAST_MODEL):
        return FALCON_FAST_MODEL
    return FALCON_QUALITY_MODEL


class FalconDetectorWorker(QThread):
    """Falcon Perception (MLX) detector on its own persistent native thread.

    Runs the model behind Falcon's MLX **batch-inference engine** — the same engine
    the segment_reason pipeline uses — but tuned for a live single-camera feed:
    the lightweight 300M for detection (near real-time), warmed up at load, on a
    decoupled timer, with the queue coalesced to the newest frame so a decode never
    backs up on stale frames. Segmentation transparently loads the 1.5B (the only
    variant with a mask head); switching task reloads the matching model.

    Same one-thread-owns-Metal rule as EngineWorker: every model load AND detect
    runs on this single native thread, so MLX only ever touches Metal from one
    place. It loads lazily (only when the user enables Falcon), independent of the
    VLM/encoder engine, so app startup and the VLM path are untouched if unused.
    """

    progress = pyqtSignal(str)
    ready = pyqtSignal(str)                             # "<model> <mode> — <note>"
    failed = pyqtSignal(str)
    detections_ready = pyqtSignal(object, str, str, float)  # [Detection], query, task, seconds

    def __init__(self) -> None:
        super().__init__()
        self._jobs: "queue.Queue[tuple | None]" = queue.Queue()
        self._backend = None
        self._backend_model = None

    def submit(self, frame_bgr: np.ndarray, query: str, task: str) -> None:
        """Enqueue one job (BGR uint8 frame + query + task). Always succeeds; the
        run loop coalesces to the newest job so stale frames are dropped. `task`
        is "detection" (fast, boxes) or "segmentation" (1.5B, boxes + masks)."""
        self._jobs.put((frame_bgr, query, task))

    def stop(self) -> None:
        self._jobs.put(None)  # sentinel

    @staticmethod
    def _model_for_task(task: str) -> str:
        return FALCON_QUALITY_MODEL if task == "segmentation" else _falcon_fast_model()

    def _ensure_backend(self, model_id: str):
        """Lazily (re)load the backend for `model_id`, warmed up. Reused across jobs
        until the required model changes (task switch), then swapped in place."""
        if self._backend is not None and self._backend_model == model_id:
            return self._backend
        from pave_mlx.backends import FalconPerceptionBackend

        short = model_id.split("/")[-1]
        self.progress.emit(f"loading {short}…")
        self._backend = None  # drop the previous model first to bound peak memory
        self._backend_model = None
        backend = FalconPerceptionBackend(
            model_id=model_id, min_dim=256, max_dim=FALCON_MAX_DIM,
            max_new_tokens=FALCON_MAX_NEW_TOKENS, warmup=True,
        )
        self._backend = backend
        self._backend_model = model_id
        mode = getattr(backend, "mode", "loaded")
        note = getattr(backend, "load_status", "") or getattr(backend, "load_error", "")
        self.ready.emit(f"{short} {mode}{(' — ' + note) if note else ''}")
        return backend

    def run(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                break
            while True:                       # coalesce: keep only the newest frame
                try:
                    nxt = self._jobs.get_nowait()
                except queue.Empty:
                    break
                if nxt is None:
                    job = None
                    break
                job = nxt
            if job is None:
                break
            frame_bgr, query, task = job
            try:
                backend = self._ensure_backend(self._model_for_task(task))
                if getattr(backend, "mode", "") != "loaded":
                    continue  # fallback mode never detects; wait for the next job
                t0 = time.perf_counter()
                dets = backend.detect(frame_bgr, query, task=task)
                detections = backend_dets_to_detections(dets, query)
                dt = time.perf_counter() - t0
            except Exception:  # noqa: BLE001 - a transient detect error must not kill the detector
                import traceback
                traceback.print_exc()
                continue
            self.detections_ready.emit(detections, query, task, dt)
