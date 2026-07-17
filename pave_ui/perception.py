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
    "moondream3": 5.5,
    "smolvlm_256m": 0.52,
    "fourier_qwen2vl_2b": 4.42,
    "fourier_4bit": 1.1,
    "fourier_3bit": 1.9,
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
# Qwen3.5-2B: prefill (image tokens) dominates request_ms; 336px measured
# median 857ms vs 1058ms @448 AND a better gesture score (11/12 vs 10/12).
# 280px is faster still (699ms) but drops pointing accuracy — don't go lower.
VLM_INPUT_SIZE_QWEN35 = int(os.environ.get("PAVE_VLM_INPUT_SIZE_QWEN35", "336"))
VLM_MAX_TOKENS = int(os.environ.get("PAVE_VLM_MAX_TOKENS", "48"))
VLM_FAST_INTENT_ONLY = os.environ.get("PAVE_VLM_FAST_INTENT_ONLY", "0") == "1"


_INPUT_SIZE_OVERRIDE = 0  # 0 = per-model defaults; set by the UI "Input px" selector


def set_input_size_override(px: int | None) -> None:
    """Explicit camera input size for ALL VLMs (UI selector). 0/None = auto."""
    global _INPUT_SIZE_OVERRIDE
    _INPUT_SIZE_OVERRIDE = int(px or 0)


def vlm_input_size(model_name: str) -> int:
    """Camera input size for a VLM (Qwen: smaller square = fewer image tokens)."""
    if _INPUT_SIZE_OVERRIDE:
        return _INPUT_SIZE_OVERRIDE
    name = (model_name or "").lower()
    if "qwen3.5" in name or "qwen35" in name:
        return VLM_INPUT_SIZE_QWEN35
    return VLM_INPUT_SIZE_QWEN if "qwen" in name else VLM_INPUT_SIZE


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


# ── YOLO gesture detector (train/HaGRID.sh output) ───────────────────────────
# A fine-tuned YOLO-nano (yolo26n by default) replaces the ~857ms VLM gesture
# gate with a single-digit-ms detector. Runs on CPU ON PURPOSE: the nano model
# is fast enough there, it leaves Metal entirely to the MLX engines (same
# one-thread-owns-Metal rule as EngineWorker), and CPU latency on this Mac is
# the honest proxy for the Orion O6 / Mali deployment target.

# detector class -> OpenPAVE intent token. UP/DOWN are new (not in _LABELS yet)
# so the viewer shows them but does not post them to intent_ingress.
YOLO_CLASS_TO_INTENT = {
    "stop": "STOP",
    "fist": "HOME",
    "like": "TROT",
    "point_left": "LEFT",
    "point_right": "RIGHT",
    "point_up": "UP",
    "point_down": "DOWN",
}
YOLO_CONF_MIN = float(os.environ.get("PAVE_YOLO_CONF", "0.50"))
YOLO_IMGSZ = int(os.environ.get("PAVE_YOLO_IMGSZ", "320"))


def default_yolo_weights() -> str | None:
    """Newest trained gesture model under train/runs; PAVE_YOLO_MODEL overrides."""
    env = os.environ.get("PAVE_YOLO_MODEL", "").strip()
    if env:
        return env
    runs = Path(__file__).resolve().parents[1] / "train" / "runs"
    cands = sorted(runs.glob("*/weights/best.pt"), key=lambda p: p.stat().st_mtime)
    return str(cands[-1]) if cands else None


class YoloGestureWorker(QThread):
    """YOLO gesture detector on its own thread, mirroring FalconDetectorWorker:
    lazy load, coalescing queue (stale frames dropped), one job in flight."""

    progress = pyqtSignal(str)
    ready = pyqtSignal(str)
    failed = pyqtSignal(str)
    detections_ready = pyqtSignal(object, str, float)  # [Detection], intent, ms

    def __init__(self, weights: str | None = None) -> None:
        super().__init__()
        self._jobs: "queue.Queue[np.ndarray | None]" = queue.Queue()
        self._weights = weights or default_yolo_weights()
        self._model = None
        self._names: dict[int, str] = {}

    def submit(self, frame_bgr: np.ndarray) -> None:
        self._jobs.put(frame_bgr)

    def stop(self) -> None:
        self._jobs.put(None)

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        if not self._weights or not Path(self._weights).exists():
            raise FileNotFoundError(
                f"no trained gesture model at {self._weights!r} — run ./train/HaGRID.sh all"
            )
        self.progress.emit(f"loading {Path(self._weights).name}…")
        from ultralytics import YOLO  # heavy import, deferred to this thread

        self._model = YOLO(self._weights)
        # warmup so the first live frame isn't a 200ms outlier
        self._model.predict(np.zeros((YOLO_IMGSZ, YOLO_IMGSZ, 3), dtype=np.uint8),
                            imgsz=YOLO_IMGSZ, device="cpu", verbose=False)
        self._names = dict(self._model.names)
        self.ready.emit(f"{Path(self._weights).name} loaded ({len(self._names)} classes, CPU)")
        return self._model

    def _detect(self, frame_bgr: np.ndarray) -> tuple[list, str]:
        res = self._model.predict(frame_bgr, imgsz=YOLO_IMGSZ, conf=YOLO_CONF_MIN,
                                  device="cpu", verbose=False)[0]
        dets, best_conf, intent = [], 0.0, ""
        for xyxyn, conf, cls in zip(res.boxes.xyxyn.tolist(),
                                    res.boxes.conf.tolist(),
                                    res.boxes.cls.tolist()):
            name = self._names.get(int(cls), str(int(cls)))
            dets.append(Detection(*xyxyn, conf, int(cls), f"{name} {conf:.2f}"))
            mapped = YOLO_CLASS_TO_INTENT.get(name, "")
            if mapped and conf > best_conf:
                best_conf, intent = conf, mapped
        return dets, intent

    def run(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                break
            while True:  # coalesce: keep only the newest frame
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
            try:
                self._ensure_model()
                t0 = time.perf_counter()
                dets, intent = self._detect(job)
                ms = (time.perf_counter() - t0) * 1000.0
            except FileNotFoundError as exc:
                self.failed.emit(str(exc))
                break
            except Exception:  # noqa: BLE001 - transient detect error must not kill the thread
                import traceback
                traceback.print_exc()
                continue
            self.detections_ready.emit(dets, intent, ms)


# Known param counts for the nano ladder (fused, from ultralytics summaries);
# anything unknown shows "?" rather than a wrong number.
_YOLO_PARAMS = {"yolo26n": "2.4M", "yolo11n": "2.6M", "yolov8n": "3.2M"}
# runtime key -> (weights artifact inside <run>/weights/, bit depth of that artifact)
YOLO_RUNTIME_ARTIFACTS = {
    "pt": ("best.pt", "fp32"),
    "onnx": ("best.onnx", "fp32"),
    "ncnn": ("best_ncnn_model", "fp16"),
}


def discover_yolo_models(runtime: str = "pt") -> dict[str, str]:
    """Trained gesture detectors under train/runs -> {dropdown title: weights path}.

    Titles follow 'basename · params · detail · bit-depth', e.g.
    'yolo26n · 2.4M · HaGRID-8c-320px · fp16', so delegates/runtimes can be
    compared by eye. Only runs whose artifact for `runtime` (pt|onnx|ncnn)
    exists on disk are listed — export missing ones with ./train/HaGRID.sh export."""
    artifact, bits = YOLO_RUNTIME_ARTIFACTS[runtime]
    out: dict[str, str] = {}
    runs = Path(__file__).resolve().parents[1] / "train" / "runs"
    for run in sorted(runs.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
        w = run / "weights" / artifact
        if not w.exists():
            continue
        base, detail = run.name, ""
        try:
            import yaml
            args = yaml.safe_load((run / "args.yaml").read_text())
            base = Path(str(args.get("model", ""))).stem or run.name
            names = yaml.safe_load(Path(str(args.get("data", ""))).read_text()).get("names", [])
            detail = f"HaGRID-{len(names)}c-{args.get('imgsz', '?')}px"
        except Exception:
            detail = run.name  # still listable when metadata is gone
        out[f"{base} · {_YOLO_PARAMS.get(base, '?')} · {detail} · {bits}"] = str(w)
    return out


# ── MediaPipe-landmark SVM gesture classifier (train/mediapipe_svm.py) ───────
# The "traditional" pipeline: MediaPipe hand landmarker (21 points, CPU/TFLite)
# -> wrist-normalised geometry -> RBF-SVM. Classifies SHAPE+ORIENTATION from
# landmarks, so the user's lighting/background are normalised away — the crude
# GIF captures ARE its training domain. Same CPU-only rationale as YOLO.

# SHAPE classes only. `point` is intentionally absent: a confident point is
# resolved by index-finger geometry (train.mediapipe_svm.point_direction) into
# LEFT / RIGHT / vertical-no-op — direction is an angle, not a cluster.
SVM_CLASS_TO_INTENT = {
    "stop": "STOP",           # open palm -> stop trotting
    "fist": "HOME",           # fist -> return to origin pose (rotation + position tween)
    "like": "TROT",           # thumbs-up -> trot / keep trotting
}
SVM_CONF_MIN = float(os.environ.get("PAVE_SVM_CONF", "0.5"))  # deliberately hard to trigger
# Landmarker gates, raised from MediaPipe's defaults because a face at low
# detection confidence is the classic hand false-positive — it then classifies
# as a random gesture and (worst case) posts a MOVE that kills a running trot.
SVM_DET_CONF = float(os.environ.get("PAVE_SVM_DET_CONF", "0.65"))
SVM_PRESENCE_CONF = float(os.environ.get("PAVE_SVM_PRESENCE_CONF", "0.60"))
SVM_TRACK_CONF = float(os.environ.get("PAVE_SVM_TRACK_CONF", "0.60"))
# Frames wider than this are downscaled before landmarking (aspect kept;
# normalised landmark coords are unaffected). The landmark CNN dominates the
# cost, so this mostly trims the colour-convert/resize preamble.
SVM_MAX_DIM = int(os.environ.get("PAVE_SVM_MAX_DIM", "384"))
# Idle gate: after SVM_IDLE_AFTER consecutive no-hand results, a static scene
# skips the landmark CNN entirely (a 32x32 grey frame-diff, ~0.1ms, stands
# watch). Any real motion — a hand entering frame — trips the delta and wakes
# the full pipeline on the same tick. This is what keeps average CPU load
# near zero while nobody is gesturing; per-frame latency was never the load.
SVM_IDLE_AFTER = int(os.environ.get("PAVE_SVM_IDLE_AFTER", "3"))
SVM_WAKE_DELTA = float(os.environ.get("PAVE_SVM_WAKE_DELTA", "3.0"))
_SVM_RUN_DIR = Path(__file__).resolve().parents[1] / "train" / "runs" / "mediapipe_svm"
_HAND_LANDMARKER = Path(__file__).resolve().parents[1] / "train" / "weights" / "hand_landmarker.task"

# MediaPipe HAND_CONNECTIONS topology (fixed): thumb, index, middle, ring,
# pinky chains + palm ring — for the classic skeleton overlay.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm base
)


def draw_hand_skeleton(frame_bgr: np.ndarray, hands: list) -> None:
    """Classic MediaPipe-style overlay: green bone connections under red
    landmark dots, drawn on the full-res BGR frame (same z-order rule as
    draw_falcon_masks — boxes/labels go on top in the later QPainter pass)."""
    if cv2 is None or not hands:
        return
    h, w = frame_bgr.shape[:2]
    for hand in hands:
        if isinstance(hand, dict) and hand.get("kind") == "sensorimotor_debug":
            _draw_sensorimotor_debug(frame_bgr, hand)
            continue
        finite = [bool(np.isfinite(p).all()) for p in hand]
        pts = [(int(x * w), int(y * h)) if finite[i] else (0, 0)
               for i, (x, y) in enumerate(hand)]
        for a, b in HAND_CONNECTIONS:
            if finite[a] and finite[b]:
                cv2.line(frame_bgr, pts[a], pts[b], (48, 255, 48), 2)
        for index, p in enumerate(pts):
            if finite[index]:
                cv2.circle(frame_bgr, p, 4, (48, 48, 255), -1)


def _draw_sensorimotor_debug(frame_bgr: np.ndarray, state: dict) -> None:
    """Draw every stage of the command-suppressed landmark experiment.

    Blue = trunk cold start; amber = Monty graph prediction; green = accepted
    pixel sensation; red X = rejected sensation; cyan ring = palm anchor.
    """
    h, w = frame_bgr.shape[:2]

    def pixels(values):
        a = np.asarray(values, np.float32).reshape(21, 2)
        a = np.nan_to_num(a, nan=-1.0, posinf=2.0, neginf=-1.0)
        return [(int(round(float(x) * w)), int(round(float(y) * h))) for x, y in a]

    coarse = pixels(state["coarse"])
    predicted = pixels(state["predicted"])
    points = pixels(state["points"])
    accepted = np.asarray(state["accepted"], bool)
    rejected = np.asarray(state["rejected"], bool)
    anchors = set(map(int, np.asarray(state["anchors"]).tolist()))

    for p in coarse:
        cv2.circle(frame_bgr, p, 2, (255, 128, 32), -1)
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame_bgr, predicted[a], predicted[b], (0, 190, 255), 1)
    for joint, p in enumerate(points):
        if accepted[joint]:
            cv2.circle(frame_bgr, p, 4, (60, 255, 60), -1)
        elif rejected[joint]:
            cv2.line(frame_bgr, (p[0] - 4, p[1] - 4), (p[0] + 4, p[1] + 4), (40, 40, 255), 2)
            cv2.line(frame_bgr, (p[0] + 4, p[1] - 4), (p[0] - 4, p[1] + 4), (40, 40, 255), 2)
        else:
            cv2.circle(frame_bgr, p, 2, (0, 190, 255), -1)
        if joint in anchors:
            cv2.circle(frame_bgr, p, 7, (255, 255, 0), 1)

    count = int(accepted.sum())
    hypothesis = str(state.get("hypothesis", "noop"))
    prototype = int(state.get("prototype", -1))
    elapsed = float(state.get("total_ms", 0.0))
    line1 = f"SENSORIMOTOR {count}/21 accepted  proto {prototype}  {elapsed:.1f} ms"
    line2 = f"hypothesis: {hypothesis} (display only)  COMMANDS OFF"
    cv2.rectangle(frame_bgr, (8, 8), (min(w - 8, 590), 56), (18, 18, 18), -1)
    cv2.putText(frame_bgr, line1, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, .48,
                (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(frame_bgr, line2, (16, 49), cv2.FONT_HERSHEY_SIMPLEX, .48,
                (80, 220, 255), 1, cv2.LINE_AA)


def discover_svm_models() -> dict[str, str]:
    """Trained SVM classifiers -> {dropdown title: model path}, titled with the
    same 'basename · params · detail · bit-depth' convention as the YOLOs.
    model.onnx (float32 end-to-end, the deployment artifact) is preferred;
    the fp64 joblib is only listed when no ONNX export exists (old runs)."""
    out: dict[str, str] = {}
    onnx_p = _SVM_RUN_DIR / "model.onnx"
    joblib_p = _SVM_RUN_DIR / "model.joblib"
    model_p, bits = (onnx_p, "fp32") if onnx_p.exists() else (joblib_p, "fp64")
    if not model_p.exists():
        return out
    n_sv, nc = "?", "?"
    try:
        meta = json.loads((_SVM_RUN_DIR / "meta.json").read_text())
        n_sv, nc = meta.get("n_support_vectors", "?"), len(meta.get("classes", []))
    except Exception:
        pass
    out[f"svm-rbf · {n_sv}SV · crude-{nc}c-63f · {bits}"] = str(model_p)
    return out


class MediaPipeSvmWorker(QThread):
    """MediaPipe landmarker + SVM on its own thread (same coalescing-queue,
    one-job pattern as YoloGestureWorker). Emits the landmark skeleton too so
    the viewer can render the traditional MediaPipe overlay."""

    progress = pyqtSignal(str)
    ready = pyqtSignal(str)
    failed = pyqtSignal(str)
    # [Detection], intent, timing {lm_ms, svm_us, gated}, hands
    results_ready = pyqtSignal(object, str, object, object)

    def __init__(self, model_path: str | None = None) -> None:
        super().__init__()
        self._jobs: "queue.Queue[np.ndarray | None]" = queue.Queue()
        default = _SVM_RUN_DIR / "model.onnx"
        if model_path is None and not default.exists():
            default = _SVM_RUN_DIR / "model.joblib"    # pre-fp32 runs
        self._model_path = model_path or str(default)
        self._predict_proba = None                     # (1,63) float32 -> (n_classes,) proba
        self._landmarker = None
        self._classes: list[str] = []
        self._miss_streak = 0                          # consecutive no-hand results
        self._prev_small: np.ndarray | None = None     # 32x32 grey, idle-gate watchdog

    def submit(self, frame_bgr: np.ndarray) -> None:
        self._jobs.put(frame_bgr)

    def stop(self) -> None:
        self._jobs.put(None)

    def _ensure_models(self):
        if self._predict_proba is not None:
            return
        if not Path(self._model_path).exists():
            raise FileNotFoundError(
                f"no SVM at {self._model_path!r} — run .venv/bin/python train/mediapipe_svm.py")
        if not _HAND_LANDMARKER.exists():
            raise FileNotFoundError(f"missing {_HAND_LANDMARKER} (see train/mediapipe_svm.py)")
        self.progress.emit("loading MediaPipe + SVM…")
        root = str(Path(__file__).resolve().parents[1])
        if root not in sys.path:                 # for `from train.mediapipe_svm import …`
            sys.path.insert(0, root)
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        bits = "fp32"
        if self._model_path.endswith("objects.npz"):
            # Monty backend: few-shot 3D evidence over landmark constellations
            from train.monty_lab import EvidenceLM
            from train.monty_lab.tasks.gestures import INTENT as MONTY_INTENT, hand_to_episode
            from train.mediapipe_svm import point_direction as _pd
            self.STATUS_PREFIX = "MNTY"
            lm_model = EvidenceLM.load(Path(self._model_path))

            def _monty_classify(hand):
                ep = hand_to_episode(hand)
                obj, e, _pose = lm_model.infer(ep)
                if obj == "point":
                    d = _pd(hand)
                    return f"point→{d.lower()}", (d if d in ("LEFT", "RIGHT") else ""), e
                return obj, MONTY_INTENT.get(obj, ""), e
            self._monty_classify = _monty_classify
            self._predict_proba = lambda feats: (_ for _ in ()).throw(RuntimeError("monty backend"))
            self._classes = ["palm", "fist", "like", "point", "noop"]
        elif self._model_path.endswith(".onnx"):
            # float32 end-to-end: float32 landmark features into an ONNX
            # SVMClassifier scored by onnxruntime — nothing upcasts to double.
            import onnxruntime as ort
            sess = ort.InferenceSession(self._model_path, providers=["CPUExecutionProvider"])
            self._classes = json.loads((_SVM_RUN_DIR / "meta.json").read_text())["classes"]
            self._predict_proba = lambda feats: sess.run(
                ["probabilities"], {"landmarks": feats})[0][0]
        else:                                    # legacy joblib (sklearn/libsvm = fp64)
            import joblib
            clf = joblib.load(self._model_path)
            self._classes = list(clf.classes_)
            self._predict_proba = lambda feats: clf.predict_proba(feats)[0]
            bits = "fp64"
        # VIDEO mode: the palm-detector CNN runs only when tracking is lost;
        # steady-state frames run just the landmark model on the tracked ROI —
        # ~2.4x faster than IMAGE mode (9.2ms -> 3.9ms on the M4).
        self._landmarker = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(_HAND_LANDMARKER)),
                num_hands=1,
                min_hand_detection_confidence=SVM_DET_CONF,
                min_hand_presence_confidence=SVM_PRESENCE_CONF,
                min_tracking_confidence=SVM_TRACK_CONF,
                running_mode=vision.RunningMode.VIDEO))
        self._ts_ms = 0
        self.ready.emit(f"MediaPipe+SVM loaded ({len(self._classes)} classes, CPU, video mode, {bits})")

    def _detect(self, frame_bgr: np.ndarray) -> tuple[list, str, list, dict]:
        from train.mediapipe_svm import landmarks_to_features, point_direction  # single source of truth

        h, w = frame_bgr.shape[:2]
        if max(h, w) > SVM_MAX_DIM:
            s = SVM_MAX_DIM / max(h, w)
            frame_bgr = cv2.resize(frame_bgr, (int(w * s), int(h * s)))
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self._ts_ms += 33  # monotonic video timestamp (tick cadence)
        t0 = time.perf_counter()
        res = self._landmarker.detect_for_video(
            self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb), self._ts_ms)
        timing = {"lm_ms": (time.perf_counter() - t0) * 1000.0, "svm_us": 0.0, "gated": False}
        if not res.hand_landmarks:
            return [], "", [], timing
        hand = res.hand_landmarks[0]
        if getattr(self, "_monty_classify", None) is not None:
            t1 = time.perf_counter()
            label_txt, intent, e = self._monty_classify(hand)
            timing["svm_us"] = (time.perf_counter() - t1) * 1e6
            if label_txt == "noop":
                return [], "", [[(q.x, q.y) for q in hand]], timing
            xs, ys = [q.x for q in hand], [q.y for q in hand]
            dets = [Detection(max(0.0, min(xs) - 0.02), max(0.0, min(ys) - 0.02),
                              min(1.0, max(xs) + 0.02), min(1.0, max(ys) + 0.02),
                              e, 0, f"{label_txt} {e:.2f}")]
            return dets, intent, [[(q.x, q.y) for q in hand]], timing
        feats = landmarks_to_features(hand).reshape(1, -1).astype(np.float32)
        t1 = time.perf_counter()
        proba = self._predict_proba(feats)
        timing["svm_us"] = (time.perf_counter() - t1) * 1e6
        top = int(proba.argmax())
        cls, p = self._classes[top], float(proba[top])
        xs, ys = [q.x for q in hand], [q.y for q in hand]
        x1, y1 = max(0.0, min(xs) - 0.02), max(0.0, min(ys) - 0.02)
        x2, y2 = min(1.0, max(xs) + 0.02), min(1.0, max(ys) + 0.02)
        if p < SVM_CONF_MIN:
            label, intent = f"unsure ({cls} {p:.2f})", ""
        elif cls == "point":
            # shape says point; GEOMETRY decides the outcome. Vertical points
            # command nothing by design (the UX contract: only a clearly
            # horizontal index finger turns the robot).
            direction = point_direction(hand)
            intent = direction if direction in ("LEFT", "RIGHT") else ""
            label = f"point→{direction.lower()} {p:.2f}"
        else:
            label, intent = f"{cls} {p:.2f}", SVM_CLASS_TO_INTENT.get(cls, "")
        dets = [Detection(x1, y1, x2, y2, p, self._classes.index(cls), label)]
        return dets, intent, [[(q.x, q.y) for q in hand]], timing

    def _gated_idle(self, frame_bgr: np.ndarray) -> bool:
        """True when the landmark CNN can be skipped: no hand for a while AND
        the scene is static (32x32 grey delta below SVM_WAKE_DELTA)."""
        small = cv2.cvtColor(cv2.resize(frame_bgr, (32, 32)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        prev, self._prev_small = self._prev_small, small
        if self._miss_streak < SVM_IDLE_AFTER or prev is None:
            return False
        return float(np.abs(small - prev).mean()) < SVM_WAKE_DELTA

    def run(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                break
            while True:  # coalesce to the newest frame
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
            try:
                self._ensure_models()
                if self._gated_idle(job):
                    self.results_ready.emit([], "", {"lm_ms": 0.0, "svm_us": 0.0, "gated": True}, [])
                    continue
                dets, intent, hands, timing = self._detect(job)
                self._miss_streak = 0 if hands else self._miss_streak + 1
            except FileNotFoundError as exc:
                self.failed.emit(str(exc))
                break
            except Exception:  # noqa: BLE001 - transient detect error must not kill the thread
                import traceback
                traceback.print_exc()
                continue
            self.results_ready.emit(dets, intent, timing, hands)


# ── Distilled tiny gesture net (train/tiny_gesture.py) — the A/B variant ─────
# One ~99k-param CNN distilled FROM the MediaPipe+SVM pipeline: pixels ->
# outcome in a single pass, no landmark stage. ~1ms/frame vs ~5ms. The trade:
# it sees pixels, not geometry, so it only knows the world in train/crude —
# judge it against the baseline live (runtime dropdown) and via its eval stage.
_TINY_RUN_DIR = Path(__file__).resolve().parents[1] / "train" / "runs" / "tiny_gesture"
_LANDMARK_RUN_DIR = Path(__file__).resolve().parents[1] / "train" / "runs" / "landmark_tower"
_ORACLE_STUDENT_DIR = (Path(__file__).resolve().parents[1]
                       / "train" / "runs" / "monty_landmark_alignment" / "oracle_student")
TINY_OUTCOME_TO_INTENT = {
    "stop": "STOP", "fist": "HOME", "like": "TROT",
    "point_left": "LEFT", "point_right": "RIGHT",
    "point_vertical": "", "no_hand": "",           # no-ops by design
}


def discover_tiny_models() -> dict[str, str]:
    try:
        meta = json.loads((_TINY_RUN_DIR / "meta.json").read_text())
    except Exception:
        meta = {}
    params = f"{meta.get('params', 0) / 1e3:.0f}k"
    nc = len(meta.get("classes", []))
    if meta.get("version") == 3 and (_TINY_RUN_DIR / "trunk.onnx").exists():
        return {f"tinynet3 · {params} · crop+seq-{nc}c · fp32": str(_TINY_RUN_DIR / "trunk.onnx")}
    model_p = _TINY_RUN_DIR / "model.onnx"
    if not model_p.exists():
        return {}
    return {f"tinynet · {params} · distilled-{nc}c-{meta.get('input_px', '?')}px · fp32": str(model_p)}


class TinyGestureWorker(QThread):
    """Distilled single-pass gesture net on its own thread. Same signal
    contract as MediaPipeSvmWorker so the viewer wiring is shared; emits no
    hand skeleton (there are no landmarks at inference time — that is the
    entire point of the distillation)."""

    STATUS_PREFIX = "TINY"

    progress = pyqtSignal(str)
    ready = pyqtSignal(str)
    failed = pyqtSignal(str)
    results_ready = pyqtSignal(object, str, object, object)  # [Detection], intent, timing, hands

    def __init__(self, model_path: str | None = None) -> None:
        super().__init__()
        self._jobs: "queue.Queue[np.ndarray | None]" = queue.Queue()
        self._model_path = model_path or str(_TINY_RUN_DIR / "model.onnx")
        self._sess = None
        self._classes: list[str] = []
        self._img = 128

    def submit(self, frame_bgr: np.ndarray) -> None:
        self._jobs.put(frame_bgr)

    def stop(self) -> None:
        self._jobs.put(None)

    def _ensure_model(self):
        if self._sess is not None:
            return
        if not Path(self._model_path).exists():
            raise FileNotFoundError(
                f"no tiny net at {self._model_path!r} — run .venv/bin/python train/tiny_gesture.py train")
        self.progress.emit("loading tiny gesture net…")
        root = str(Path(__file__).resolve().parents[1])
        if root not in sys.path:
            sys.path.insert(0, root)
        if "landmark_tower" in self._model_path:
            from train.landmark_tower import LandmarkerGestureRuntime
            self._rt = LandmarkerGestureRuntime(conf=SVM_CONF_MIN if SVM_CONF_MIN < 0.84 else 0.6)
            self._v3 = "landmarks"
            self.STATUS_PREFIX = "LAND"
            self._classes = ["palm", "fist", "like", "point_right", "point_left", "noop"]
            self.ready.emit("landmarker tower loaded (21 points + frozen crop/sequence towers, CPU fp32)")
            return
        meta = json.loads((_TINY_RUN_DIR / "meta.json").read_text())
        self._classes = meta["classes"]
        self._img = int(meta.get("input_px", 128))
        self._v3 = meta.get("version") == 3
        if self._v3:
            from train.gesture_lab import TinyV3Runtime
            self._rt = TinyV3Runtime(conf=SVM_CONF_MIN)
            self.ready.emit(f"tinynet v3 loaded (detect->crop->classify + seq, "
                            f"{meta.get('params', 0) / 1e3:.0f}k params, CPU fp32)")
            return
        import onnxruntime as ort
        self._sess = ort.InferenceSession(self._model_path, providers=["CPUExecutionProvider"])
        self.ready.emit(f"tinynet loaded ({len(self._classes)} outcomes, CPU, fp32, "
                        f"{meta.get('params', 0) / 1e3:.0f}k params)")

    def _detect(self, frame_bgr: np.ndarray) -> tuple[list, str, dict]:
        if getattr(self, "_v3", False) == "landmarks":
            from train.gesture_lab import V3_INTENT, _lm_crop_box
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            t0 = time.perf_counter()
            tower, p, lm42 = self._rt.step(rgb)
            timing = {"lm_ms": (time.perf_counter() - t0) * 1000.0, "svm_us": 0.0, "gated": False}
            if tower == "noop" or lm42 is None:
                return [], "", timing
            cx, cy, side = _lm_crop_box(lm42)
            x1, y1, x2, y2 = max(0.,cx-side/2),max(0.,cy-side/2),min(1.,cx+side/2),min(1.,cy+side/2)
            dets = [Detection(x1, y1, x2, y2, p, self._classes.index(tower), f"{tower} {p:.2f}")]
            return dets, V3_INTENT.get(tower, ""), timing
        if getattr(self, "_v3", False):
            from train.gesture_lab import V3_INTENT, _lm_crop_box
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            t0 = time.perf_counter()
            tower, p, lm42 = self._rt.step(rgb)
            timing = {"lm_ms": (time.perf_counter() - t0) * 1000.0, "svm_us": 0.0, "gated": False}
            if tower == "noop" or lm42 is None:
                return [], "", timing
            cx, cy, side = _lm_crop_box(lm42)
            dets = [Detection(max(0.0, cx - side / 2), max(0.0, cy - side / 2),
                              min(1.0, cx + side / 2), min(1.0, cy + side / 2),
                              p, self._classes.index(tower), f"{tower} {p:.2f}")]
            return dets, V3_INTENT.get(tower, ""), timing
        rgb = cv2.cvtColor(cv2.resize(frame_bgr, (self._img, self._img)), cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) / 255.0 - 0.5).transpose(2, 0, 1)[None]
        t0 = time.perf_counter()
        proba = self._sess.run(["probabilities"], {"frames": x})[0][0]
        timing = {"lm_ms": (time.perf_counter() - t0) * 1000.0, "svm_us": 0.0, "gated": False}
        p, cls = float(proba.max()), self._classes[int(proba.argmax())]
        if cls == "no_hand" or p < SVM_CONF_MIN:
            return [], "", timing
        label = f"{cls} {p:.2f}"
        dets = [Detection(0.02, 0.04, 0.30, 0.16, p, self._classes.index(cls), label)]
        return dets, TINY_OUTCOME_TO_INTENT.get(cls, ""), timing

    def run(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                break
            while True:  # coalesce to the newest frame
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
            try:
                self._ensure_model()
                dets, intent, timing = self._detect(job)
            except FileNotFoundError as exc:
                self.failed.emit(str(exc))
                break
            except Exception:  # noqa: BLE001 - transient error must not kill the thread
                import traceback
                traceback.print_exc()
                continue
            lm = getattr(getattr(self, "_rt", None), "last_lm", None)
            hands = [[tuple(x) for x in lm.reshape(21, 2)]] if lm is not None else []
            self.results_ready.emit(dets, intent, timing, hands)


# ── Monty (3D evidence) gesture recognition — monty_lab task #1 ──────────────
# Landmarker in front (as ever); recognition = few-shot 3D constellation
# evidence (train/monty_lab). Learned in seconds, extended in seconds.
# PAVE_MONTY_RUN_DIR lets an external Monty object memory (e.g. the
# HAND POSE SENSORIMOTOR LAB's objects.npz) be tested without touching the
# incumbent artifacts: PAVE_MONTY_RUN_DIR=/path/to/dir ./mlx-runtime.sh
_MONTY_RUN_DIR = Path(
    os.environ.get(
        "PAVE_MONTY_RUN_DIR",
        str(Path(__file__).resolve().parents[1] / "train" / "runs" / "monty_gestures"),
    )
).expanduser()
# The HAND POSE SENSORIMOTOR LAB's export is auto-discovered as its own
# dropdown variant when present (no env var needed).
_HAND_POSE_LAB_DIR = Path(
    os.environ.get(
        "PAVE_HAND_POSE_LAB_DIR",
        "~/Documents/GitHub/monty/tutorials/results/hand_pose_lab/pretrained",
    )
).expanduser()
_HAND_POSE_LAB_RUNS_DIR = Path(
    os.environ.get(
        "PAVE_HAND_POSE_LAB_RUNS_DIR",
        str(_HAND_POSE_LAB_DIR.parent / "runs"),
    )
).expanduser()


class LandmarkerMontyWorker(TinyGestureWorker):
    """Student landmark tower -> lifted 3D points -> frozen Monty evidence.

    The tower predicts MediaPipe-compatible x/y coordinates. A class-neutral
    per-joint depth prior, calculated from all frozen Monty exemplars, lifts
    them into EvidenceLM's 3D contract without MediaPipe at runtime. The
    original MediaPipe+Monty worker and artifact remain untouched.
    """

    STATUS_PREFIX = "LMNT"

    def _ensure_model(self):
        if self._sess is not None:
            return
        # A "?objects=" suffix binds this variant to a specific Monty object
        # memory (e.g. the HAND POSE SENSORIMOTOR LAB export) without touching
        # the incumbent _MONTY_RUN_DIR artifacts.
        landmark_value = str(self._model_path)
        if "?objects=" in landmark_value:
            landmark_value, objects_value = landmark_value.split("?objects=", 1)
            monty_path = Path(objects_value).expanduser()
        else:
            monty_path = _MONTY_RUN_DIR / "objects.npz"
        landmark_path = Path(landmark_value)
        if not landmark_path.exists():
            raise FileNotFoundError(
                f"no landmark tower at {landmark_path!s} — run ./train/landmark-tower.sh train")
        if not monty_path.exists():
            raise FileNotFoundError(
                "no Monty evidence — run: cd train && ../.venv/bin/python "
                "-m monty_lab.runner learn --task gestures")
        self.progress.emit("loading landmark tower + Monty evidence…")
        root = str(Path(__file__).resolve().parents[1])
        if root not in sys.path:
            sys.path.insert(0, root)
        from train.monty_lab import EvidenceLM

        try:
            landmark_meta = json.loads((landmark_path.parent / "meta.json").read_text())
        except Exception:
            landmark_meta = {}
        acquisition_matched = str(landmark_meta.get("contract", "")).startswith(
            ("openpave.acquisition-matched-landmarker", "openpave.oracle-roi-landmarker"))
        if acquisition_matched or "oracle_student" in str(landmark_path):
            # Oracle-ROI candidate: detector global search -> oriented crop
            # cold start -> landmark-derived ROI tracking (state machine from
            # docs/training-with-monty.md §7). Same step() contract.
            from train.monty_lab.tbp_adapter.oracle_runtime import OracleLandmarkerRuntime
            self._landmark_runtime = OracleLandmarkerRuntime(
                presence_gate=float(os.environ.get("PAVE_LANDMARK_PRESENCE", "0.50")),
                model_dir=landmark_path.parent)
        else:
            from train.landmark_tower import LandmarkerRuntime
            self._landmark_runtime = LandmarkerRuntime(
                presence_gate=float(os.environ.get("PAVE_LANDMARK_PRESENCE", "0.50")),
                quality_gate=float(os.environ.get("PAVE_LANDMARK_QUALITY", "0.15")))
        self._monty = EvidenceLM.load(monty_path)
        if not landmark_meta:
            try:
                landmark_meta = json.loads((_LANDMARK_RUN_DIR / "meta.json").read_text())
            except Exception:
                landmark_meta = {}
        self._input_px = int(landmark_meta.get("input_px", 96))

        stored = np.load(monty_path, allow_pickle=True)
        examples = np.concatenate([stored[k] for k in stored.files]).astype(np.float32)
        examples -= examples[:, :1]
        scale = np.maximum(np.abs(examples).max(axis=(1, 2), keepdims=True), 1e-6)
        # One neutral anatomical profile across every class: no answer leakage.
        self._depth_prior = np.median(examples / scale, axis=0)[:, 2]
        self._presence_gate = float(os.environ.get("PAVE_LANDMARK_PRESENCE", "0.50"))
        self._quality_gate = float(os.environ.get("PAVE_LANDMARK_QUALITY", "0.15"))
        self._classes = ["palm", "fist", "like", "point", "noop"]
        self.last_lm = None
        self._rt = self                    # inherited run() emits these overlay points
        self._sess = True                  # TinyGestureWorker loaded sentinel
        self.ready.emit("landmark tower + Monty 3D evidence loaded (CPU fp32, no MediaPipe)")

    def _detect(self, frame_bgr: np.ndarray) -> tuple[list, str, dict]:
        from train.monty_lab.protocol import Episode, Observation
        from train.monty_lab.tasks.gestures import INTENT as MONTY_INTENT

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        lm, presence, quality = self._landmark_runtime.step(rgb)
        timing = {"lm_ms": (time.perf_counter() - t0) * 1000.0,
                  "svm_us": 0.0, "gated": False}
        if lm is None:
            self.last_lm = None
            return [], "", timing

        self.last_lm = lm
        xy = lm.reshape(21, 2)
        joint_mask = getattr(self._landmark_runtime, "last_joint_mask", None)
        if joint_mask is None:
            joint_mask = np.isfinite(xy).all(axis=1)
        else:
            joint_mask = np.asarray(joint_mask, bool) & np.isfinite(xy).all(axis=1)
        joint_ids = np.flatnonzero(joint_mask)
        if len(joint_ids) < 3:
            return [], "", timing
        observed_xy = xy[joint_ids]
        centred = observed_xy - observed_xy[:1]
        xy_scale = max(float(np.abs(centred).max()), 1e-6)
        xyz = np.column_stack(
            (centred / xy_scale, self._depth_prior[joint_ids])).astype(np.float32)
        ep = Episode([Observation(location=p) for p in xyz])
        t1 = time.perf_counter()
        if len(joint_ids) == 21:
            obj, evidence, _pose = self._monty.infer(ep)
        else:
            obj, evidence, _pose = self._monty.infer_partial(ep, joint_ids)
        timing["svm_us"] = (time.perf_counter() - t1) * 1e6

        intent = MONTY_INTENT.get(obj, "")
        label = obj
        if obj == "point" and joint_mask[5] and joint_mask[8]:
            dx, dy = xy[8] - xy[5]
            angle = abs(float(np.degrees(np.arctan2(-dy, dx))))
            cone = float(os.environ.get("PAVE_POINT_CONE_DEG", "35"))
            flip = os.environ.get("PAVE_POINT_MIRROR", "").lower() in ("1", "true", "yes")
            direction = ""
            if angle <= cone:
                direction = "RIGHT" if flip else "LEFT"
            elif angle >= 180.0 - cone:
                direction = "LEFT" if flip else "RIGHT"
            intent = direction
            label = f"point→{direction.lower()}" if direction else "point→vertical"
        elif obj == "point":
            intent = ""
            label = "point→uncertain"

        if obj == "noop":
            return [], "", timing
        xs, ys = observed_xy[:, 0], observed_xy[:, 1]
        dets = [Detection(max(0.0, float(xs.min()) - .02),
                          max(0.0, float(ys.min()) - .02),
                          min(1.0, float(xs.max()) + .02),
                          min(1.0, float(ys.max()) + .02),
                          evidence, self._classes.index(obj), f"{label} {evidence:.2f}")]
        return dets, intent, timing


class HanCoTargetWorker(TinyGestureWorker):
    """Frozen legacy 71k acquirer plus the binary HanCo_tester geometry gate."""

    STATUS_PREFIX = "HNCO"

    def _ensure_model(self):
        if self._sess is not None:
            return
        model_path = Path(self._model_path)
        meta_path = model_path.with_name("meta.json")
        if not model_path.is_file() or not meta_path.is_file():
            raise FileNotFoundError(
                "no HanCo target proof-of-concept — run .venv/bin/python "
                "train/hanco_target_poc.py")
        self.progress.emit("loading 71k + HanCo_tester geometry gate…")
        from train.landmark_tower import LandmarkerRuntime

        self._meta = json.loads(meta_path.read_text())
        gates = self._meta["gates"]
        self._landmark_runtime = LandmarkerRuntime(
            presence_gate=float(gates["presence"]),
            quality_gate=float(gates["quality"]),
        )
        stored = np.load(model_path)
        self._mean = stored["mean"]
        self._scale = stored["scale"]
        self._coefficients = stored["coefficients"]
        self._intercept = float(stored["intercept"])
        self._target_gate = float(stored["threshold"])
        self._classes = ["HanCo_tester", "no_hand"]
        self.last_lm = None
        self._rt = self
        self._sess = True
        self.ready.emit("71k + HanCo_tester target gate loaded (CPU fp32, commands off)")

    def _detect(self, frame_bgr: np.ndarray) -> tuple[list, str, dict]:
        from train.hanco_target_poc import feature

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        landmarks, _presence, _quality = self._landmark_runtime.step(rgb)
        timing = {"lm_ms": (time.perf_counter() - t0) * 1000.0,
                  "svm_us": 0.0, "gated": False}
        if landmarks is None:
            self.last_lm = None
            return [], "", timing
        self.last_lm = landmarks
        xy = landmarks.reshape(21, 2)
        value = (feature(xy) - self._mean) / self._scale
        logit = float(value @ self._coefficients + self._intercept)
        probability = 1.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0)))
        if probability < self._target_gate:
            return [], "", timing
        xs, ys = xy[:, 0], xy[:, 1]
        detection = Detection(
            max(0.0, float(xs.min()) - 0.02),
            max(0.0, float(ys.min()) - 0.02),
            min(1.0, float(xs.max()) + 0.02),
            min(1.0, float(ys.max()) + 0.02),
            probability,
            0,
            f"HanCo_tester {probability:.2f}",
        )
        # Proof-of-concept recognition is display-only: it cannot issue a
        # robot command until a user explicitly maps this new gesture.
        return [detection], "", timing


class HanCoGestureWorker(TinyGestureWorker):
    """Frozen 71k acquirer plus reviewed multiclass HanCo geometry diagnostic."""

    STATUS_PREFIX = "HNGS"

    def _ensure_model(self):
        if self._sess is not None:
            return
        model_path = Path(self._model_path)
        meta_path = model_path.with_name("meta.json")
        if not model_path.is_file() or not meta_path.is_file():
            raise FileNotFoundError(
                "no HanCo gesture proof-of-concept — run ./train/monty-landmarks.sh "
                "hanco-gestures")
        self.progress.emit("loading 71k + reviewed HanCo gesture mix…")
        from train.landmark_tower import LandmarkerRuntime

        stored = np.load(model_path)
        self._landmark_runtime = LandmarkerRuntime(
            presence_gate=float(stored["presence_gate"]),
            quality_gate=float(stored["quality_gate"]),
        )
        self._mean = stored["mean"]
        self._scale = stored["scale"]
        self._coefficients = stored["coefficients"]
        self._intercept = stored["intercept"]
        self._classes = stored["classes"].astype(str).tolist()
        self._gesture_gate = float(stored["confidence_gate"])
        self.last_lm = None
        self._rt = self
        self._sess = True
        self.ready.emit("71k + HanCo gestures loaded (CPU diagnostic, commands off)")

    def _detect(self, frame_bgr: np.ndarray) -> tuple[list, str, dict]:
        from train.hanco_target_poc import feature

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        landmarks, _presence, _quality = self._landmark_runtime.step(rgb)
        timing = {"lm_ms": (time.perf_counter() - t0) * 1000.0,
                  "svm_us": 0.0, "gated": False}
        if landmarks is None:
            self.last_lm = None
            return [], "", timing
        self.last_lm = landmarks
        xy = landmarks.reshape(21, 2)
        value = (feature(xy) - self._mean) / self._scale
        logits = self._coefficients @ value + self._intercept
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        top = int(probabilities.argmax())
        probability = float(probabilities[top])
        label = self._classes[top]
        if label == "no_hand" or probability < self._gesture_gate:
            return [], "", timing
        display = label
        if label == "point":
            dx, dy = xy[8] - xy[5]
            angle = abs(float(np.degrees(np.arctan2(-dy, dx))))
            if angle <= 35:
                display = "point→left"
            elif angle >= 145:
                display = "point→right"
            else:
                display = "point→vertical"
        xs, ys = xy[:, 0], xy[:, 1]
        detection = Detection(
            max(0.0, float(xs.min()) - 0.02), max(0.0, float(ys.min()) - 0.02),
            min(1.0, float(xs.max()) + 0.02), min(1.0, float(ys.max()) + 0.02),
            probability, top, f"{display} {probability:.2f}")
        return [detection], "", timing


class HanCoCropGestureWorker(TinyGestureWorker):
    """Frozen 71k acquisition plus the HanCo-only reviewed RGB crop head."""

    STATUS_PREFIX = "HNCP"

    def _ensure_model(self):
        if self._sess is not None:
            return
        model_path = Path(self._model_path)
        meta_path = model_path.with_name("meta.json")
        if not model_path.is_file() or not meta_path.is_file():
            raise FileNotFoundError(
                "no HanCo crop model — run ./train/monty-landmarks.sh hanco-gestures")
        self.progress.emit("loading 71k + HanCo-only crop classifier…")
        import onnxruntime as ort
        from train.landmark_tower import LandmarkerRuntime

        meta = json.loads(meta_path.read_text())
        self._landmark_runtime = LandmarkerRuntime(
            presence_gate=float(os.environ.get("PAVE_LANDMARK_PRESENCE", "0.50")),
            quality_gate=float(os.environ.get("PAVE_LANDMARK_QUALITY", "0.15")))
        self._crop_session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"])
        self._classes = list(meta["classes"])
        self.last_lm = None
        self._rt = self
        self._sess = True
        self.ready.emit("71k + HanCo-only RGB crop head loaded (CPU diagnostic, commands off)")

    def _detect(self, frame_bgr: np.ndarray) -> tuple[list, str, dict]:
        from train.hanco_crop_gesture import take_crop

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        landmarks, _presence, _quality = self._landmark_runtime.step(rgb)
        landmark_ms = (time.perf_counter() - t0) * 1000.0
        if landmarks is None:
            self.last_lm = None
            return [], "", {"lm_ms": landmark_ms, "svm_us": 0.0, "gated": False}
        self.last_lm = landmarks
        xy = landmarks.reshape(21, 2)
        crop = take_crop(rgb, xy)
        values = (crop.astype(np.float32) / 255.0 - 0.5).transpose(2, 0, 1)[None]
        t1 = time.perf_counter()
        probabilities = self._crop_session.run(["probabilities"], {"crops": values})[0][0]
        classifier_us = (time.perf_counter() - t1) * 1e6
        top = int(probabilities.argmax())
        probability = float(probabilities[top])
        label = self._classes[top]
        timing = {"lm_ms": landmark_ms, "svm_us": classifier_us, "gated": False}
        if label == "no_hand":
            return [], "", timing
        display = label
        if label == "point":
            dx, dy = xy[8] - xy[5]
            angle = abs(float(np.degrees(np.arctan2(-dy, dx))))
            display = "point→left" if angle <= 35 else "point→right" if angle >= 145 else "point→vertical"
        xs, ys = xy[:, 0], xy[:, 1]
        detection = Detection(
            max(0.0, float(xs.min()) - 0.02), max(0.0, float(ys.min()) - 0.02),
            min(1.0, float(xs.max()) + 0.02), min(1.0, float(ys.max()) + 0.02),
            probability, top, f"{display} {probability:.2f}")
        return [detection], "", timing


def discover_landmarker_models() -> dict[str, str]:
    """Standalone 21-point student landmarker + frozen gesture towers."""
    model = _LANDMARK_RUN_DIR / "model.onnx"
    if (not model.exists() or not (_LANDMARK_RUN_DIR / "detector.onnx").exists()
            or not (_TINY_RUN_DIR / "crop.onnx").exists()):
        return {}
    try:
        meta = json.loads((_LANDMARK_RUN_DIR / "meta.json").read_text())
    except Exception:
        meta = {}
    state = "accepted" if meta.get("accepted") else "REJECTED candidate"
    models = {
        f"landmarks · {meta.get('params', 0)/1e3:.0f}k · 21pt+crop+seq · {state} · fp32": str(model)
    }
    target = REPO_ROOT / "train/runs/hanco_target_poc/model.npz"
    try:
        target_meta = json.loads(target.with_name("meta.json").read_text())
        metrics = target_meta["metrics"]
        title = (f"hanco-target · HanCo_tester · 71k+geometry · "
                 f"{metrics['presence_f1']:.1%} P-F1 · "
                 f"{metrics['target_acquisition_rate']:.1%} acq · fp32")
        models[title] = str(target)
    except Exception:
        pass
    gestures = REPO_ROOT / "train/runs/hanco_crop_gesture/crop.onnx"
    try:
        gesture_meta = json.loads(gestures.with_name("meta.json").read_text())
        metrics = gesture_meta["metrics"]
        title = (f"hanco-crop · reviewed full sequences · HanCo-only · "
                 f"{gesture_meta['params']/1e3:.0f}k · "
                 f"{metrics['correct_gesture_acquisition_rate']:.1%} correct · "
                 "DIAGNOSTIC · fp32")
        models[title] = str(gestures)
    except Exception:
        pass
    curriculum = REPO_ROOT / "train/runs/hanco_crop_curriculum/crop.onnx"
    try:
        curriculum_meta = json.loads(curriculum.with_name("meta.json").read_text())
        metrics = curriculum_meta["metrics"]
        title = (f"hanco-crop-ssl · monty-propagated + view-consistent · HanCo-only · "
                 f"{curriculum_meta['params']/1e3:.0f}k · "
                 f"{metrics['correct_gesture_acquisition_rate']:.1%} correct · "
                 "DIAGNOSTIC · fp32")
        models[title] = str(curriculum)
    except Exception:
        pass
    return models


def discover_landmarker_monty_models() -> dict[str, str]:
    """Combined student landmarker + frozen Monty artifact pair."""
    landmark = _LANDMARK_RUN_DIR / "model.onnx"
    monty = _MONTY_RUN_DIR / "objects.npz"
    if not landmark.exists() or not (_LANDMARK_RUN_DIR / "detector.onnx").exists() or not monty.exists():
        return {}
    try:
        lm_meta = json.loads((_LANDMARK_RUN_DIR / "meta.json").read_text())
        mo_meta = json.loads((_MONTY_RUN_DIR / "meta.json").read_text())
    except Exception:
        lm_meta, mo_meta = {}, {}
    objects = mo_meta.get("objects", {})
    exemplars = sum(objects.values()) if isinstance(objects, dict) else "?"
    models = {}
    # Completed acquisition-matched runs are listed first. Named runs (for
    # example the bounded HanCo revision) precede the historical `alternation`
    # run. Keep rejected rounds visible for empirical GUI comparison, but make
    # their provenance and live-gate status impossible to miss in the title.
    reports = sorted(
        _ORACLE_STUDENT_DIR.glob("alternation*/alternation_report.json"),
        key=lambda path: (path.parent.name == "alternation", path.parent.name),
    )
    for alternation in reports:
        try:
            report = json.loads(alternation.read_text())
            rounds = list(report.get("rounds", []))
            selected = report.get("selected_round")
            run_name = alternation.parent.name
            if run_name.startswith("alternation_hanco_seed"):
                run_label = f"HanCo seed{run_name.removeprefix('alternation_hanco_seed')}"
            elif run_name == "alternation":
                run_label = "MATCHED"
            else:
                run_label = run_name.removeprefix("alternation_").replace("_", " ")
            rounds.sort(key=lambda r: (r.get("round") != selected, r.get("round", 0)))
            for item in rounds:
                number = item.get("round", "?")
                directory = Path(item.get("directory", ""))
                if not directory.is_absolute():
                    directory = REPO_ROOT / directory
                candidate = directory / "landmarker.onnx"
                meta_path = directory / "meta.json"
                if not candidate.exists() or not meta_path.exists():
                    continue
                candidate_meta = json.loads(meta_path.read_text())
                gate = item.get("selection_gate", {})
                gate_state = "LIVE PASS" if gate.get("passed") else "LIVE REJECT"
                live = (candidate_meta.get("evaluation_columns", {})
                        .get("proposed_roi_deployment_truth", {})
                        .get("live_replay") or {})
                acquisition = (live.get("candidate", {}).get("summary", {})
                               .get("acquisition_rate"))
                acquisition_text = (f"{acquisition:.1%} acq" if acquisition is not None
                                    else "no live metric")
                chosen = " · SELECTED" if number == selected else ""
                title = (f"landmark+monty · {run_label} R{number}{chosen} · "
                         f"{candidate_meta.get('params', 0)/1e3:.0f}k · "
                         f"{acquisition_text} · {gate_state} · fp32")
                models[title] = str(candidate)
        except Exception:
            pass
    # Keep the previous oracle-ROI candidate available as the historical
    # comparison after any completed acquisition-matched rounds.
    oracle = _ORACLE_STUDENT_DIR / "landmarker.onnx"
    if oracle.exists():
        try:
            o_meta = json.loads((_ORACLE_STUDENT_DIR / "meta.json").read_text())
        except Exception:
            o_meta = {}
        models[(f"landmark+monty · NEW oracle-roi · {o_meta.get('params', 0)/1e3:.0f}k "
                f"· 21pt+{exemplars}ex-3D-evidence · candidate · fp32")] = str(oracle)
    state = "accepted" if lm_meta.get("accepted") else "REJECTED candidate"
    models[(f"landmark+monty · legacy tower · {lm_meta.get('params', 0)/1e3:.0f}k "
            f"· 21pt+{exemplars}ex-3D-evidence · {state} · fp32")] = str(landmark)
    # Every completed HAND POSE SENSORIMOTOR LAB run is immutable and
    # independently purgeable. Newest runs are listed first. The fixed
    # pretrained/ artifact remains a legacy fallback until versioned runs exist.
    lab_run_dirs = (
        sorted(
            (path for path in _HAND_POSE_LAB_RUNS_DIR.iterdir() if path.is_dir()),
            reverse=True,
        )
        if _HAND_POSE_LAB_RUNS_DIR.is_dir() else []
    )
    # Once the versioned runs directory exists it is authoritative, even when
    # empty after the user purges every run. Do not resurrect the fixed latest
    # working artifact as an apparently undeleted dropdown entry.
    lab_candidates = (
        lab_run_dirs
        if _HAND_POSE_LAB_RUNS_DIR.is_dir()
        else [_HAND_POSE_LAB_DIR]
    )
    lab_models = {}
    for run_dir in lab_candidates:
        lab_objects = run_dir / "objects.npz"
        if not lab_objects.is_file():
            continue
        try:
            lab_meta = json.loads((run_dir / "meta.json").read_text())
            if lab_meta.get("objects_convention") != "evidence-lm-normalised.v1":
                continue
            lab_count = sum(lab_meta.get("objects", {}).values())
        except Exception:
            continue
        run_id = str(lab_meta.get("run_id", "legacy-undated"))
        mode = str(lab_meta.get("training_mode", "legacy")).upper()
        episodes = lab_meta.get("episodes", "?")
        title = (
            f"landmark+monty · HAND POSE LAB · {run_id} · {mode} · "
            f"{episodes}ep · 71k · 21pt+{lab_count}ex-3D-evidence · "
            "DIAGNOSTIC · fp32"
        )
        lab_models[title] = f"{landmark}?objects={lab_objects}"
    # Put the newest lab run first so selecting this CPU runtime loads it by
    # default; retain every historical/oracle comparison after the run history.
    return {**lab_models, **models}


def discover_monty_models() -> dict[str, str]:
    model_p = _MONTY_RUN_DIR / "objects.npz"
    if not model_p.exists():
        return {}
    n_ex, n_obj = "?", "?"
    try:
        meta = json.loads((_MONTY_RUN_DIR / "meta.json").read_text())
        objs = meta.get("objects", {})
        n_ex, n_obj = sum(objs.values()), len(objs)
    except Exception:
        pass
    return {f"monty · {n_ex}ex · 3D-evidence-{n_obj}obj · fp32": str(model_p)}
