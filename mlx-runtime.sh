#!/bin/bash
# mlx-runtime.sh — Launch MLX based OpenPAVE
#
#   chmod +x mlx-runtime.sh
#   ./mlx-runtime.sh
#
# Single command. Brings up the whole local MLX stack through the PyQt6 operator
# console (pave_ui.viewer), which spawns intent_ingress + control_daemon
# (ROBOT_ADAPTER=mock), runs the streaming state server, embeds the visualiser,
# captures the camera, and routes all process logs into its Console panel. No
# Docker, no ROS2, no vLLM. Close the window (or Ctrl+C) to stop everything.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT/.venv"
MIN_PY_MINOR="10"

export ROBOT_ADAPTER="${ROBOT_ADAPTER:-mock}"
export INTENT_PATH="${INTENT_PATH:-/tmp/vla_intent.json}"
export COMMAND_RESULT_PATH="${COMMAND_RESULT_PATH:-/tmp/vla_command_result.json}"
export ROBOT_STATE_PATH="${ROBOT_STATE_PATH:-/tmp/vla_robot_state.json}"
export ROBOT_HEARTBEAT_SEC="${ROBOT_HEARTBEAT_SEC:-1.0}"
export PAVE_DEFAULT_MODEL="${PAVE_DEFAULT_MODEL:-No model}"
export PAVE_ALLOW_MODEL_DOWNLOADS="${PAVE_ALLOW_MODEL_DOWNLOADS:-1}"
export PAVE_MODEL_PREFLIGHT="${PAVE_MODEL_PREFLIGHT:-1}"

step() { printf '\n -> %s\n' "$*"; }
note() { printf '      %s\n' "$*"; }
die()  { printf 'Error: %s\n' "$*" >&2; exit 1; }

printf '===========================================\n'
printf '  OpenPAVE  ·  native MLX runtime\n'
printf '===========================================\n'

cd "$ROOT"

# ── interpreter (3.10+; the bare python3 may be 3.9) ─────────────────
BASE_PY=""
for cand in python3.12 python3.11 python3.10 python3; do
    bin="$(command -v "$cand" 2>/dev/null)" || continue
    if [ "$("$bin" -c 'import sys;print(sys.version_info[0])' 2>/dev/null)" = "3" ] \
       && [ "$("$bin" -c 'import sys;print(sys.version_info[1])' 2>/dev/null || echo 0)" -ge "$MIN_PY_MINOR" ]; then
        BASE_PY="$bin"; break
    fi
done
[ -n "$BASE_PY" ] || die "need Python 3.${MIN_PY_MINOR}+ (install python3.12)."

# ── virtual environment ─────────────────────────────────────────────
if [ "${RECREATE_VENV:-0}" = "1" ] && [ -d "$VENV_DIR" ]; then rm -rf "$VENV_DIR"; fi
if [ -x "$VENV_DIR/bin/python" ]; then
    vmin="$("$VENV_DIR/bin/python" -c 'import sys;print(sys.version_info[1])' 2>/dev/null || echo 0)"
    [ "$vmin" -lt "$MIN_PY_MINOR" ] && { note "recreating stale .venv (was 3.$vmin)"; rm -rf "$VENV_DIR"; }
fi
if [ ! -x "$VENV_DIR/bin/python" ]; then
    step "Creating virtual environment (.venv) on $("$BASE_PY" --version 2>&1)"
    "$BASE_PY" -m venv "$VENV_DIR"
else
    step "Using existing .venv ($("$VENV_DIR/bin/python" --version 2>&1))"
fi
PYBIN="$VENV_DIR/bin/python"

if [ "${PAVE_KEEP_MLX_IMAGE:-0}" != "1" ]; then
    if "$PYBIN" - <<'PY' >/dev/null 2>&1
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("mlxim") or importlib.util.find_spec("mlx_image") else 1)
PY
    then
        "$PYBIN" -m pip uninstall -y mlx-image >/dev/null 2>&1 || true
        note "removed mlx-image: its pinned deps conflict with current MLX/VLM runtime"
    fi
fi

# ── dependencies (announced per group; pip output kept quiet) ────────
step "Installing dependencies (first run pulls PyQt6/WebEngine; please wait)"
"$PYBIN" -m pip install -U pip >/dev/null
note "control plane + perception : flask, numpy, pillow"
"$PYBIN" -m pip install -r "$ROOT/intent_ingress/requirements.txt" >/dev/null
"$PYBIN" -m pip install -U numpy pillow >/dev/null
note "operator console           : PyQt6 (+ WebEngine)"
"$PYBIN" -m pip install -U "PyQt6==6.7.1" >/dev/null || die "PyQt6 is required for the console."
"$PYBIN" -m pip install -U "PyQt6-WebEngine>=6.7.0,<6.8" >/dev/null 2>&1 || note "(WebEngine optional: external browser fallback)"
note "camera + MLX               : opencv-python, mlx"
"$PYBIN" -m pip install -U opencv-python >/dev/null 2>&1 || note "(opencv optional: camera preview disabled)"
"$PYBIN" -m pip install -U mlx >/dev/null 2>&1 || note "(mlx optional: NumPy fallback)"
note "vision-language models     : mlx-vlm (Qwen3-VL / Gemma)"
"$PYBIN" -m pip install -U mlx-vlm >/dev/null 2>&1 || note "(mlx-vlm optional: VLMs disabled; encoders + safe-default still work)"
if [ "${PAVE_VLM_RUNTIME:-mlx-vlm}" = "vllm-mlx" ] || [ "${PAVE_VLM_RUNTIME:-mlx-vlm}" = "vllm" ]; then
    note "vLLM-MLX server adapter   : vllm-mlx"
    "$PYBIN" -m pip install -U vllm-mlx >/dev/null 2>&1 || note "(vllm-mlx missing: set PAVE_VLM_RUNTIME=mlx-vlm or install it manually)"
fi
note "feature overlay (encoders) : scikit-learn; mlx-image opt-in only"
"$PYBIN" -m pip install -U scikit-learn >/dev/null 2>&1 || note "(scikit-learn missing: encoder feature overlay disabled)"
if [ "${PAVE_KEEP_MLX_IMAGE:-0}" = "1" ]; then
    "$PYBIN" -m pip install -U --no-deps mlx-image >/dev/null 2>&1 || note "(mlx-image unavailable; continuing without it)"
fi

# ── system state (clear, no flooding) ───────────────────────────────
step "System state"
"$PYBIN" - <<'PY'
import importlib.util as u
def yn(mod): return "available" if u.find_spec(mod) else "not installed"
try:
    import mlx.core as mx
    compute = f"MLX · {mx.default_device()}"
except Exception:
    compute = "NumPy (Metal unavailable)"
print(f"      compute       : {compute}")
print(f"      camera        : opencv-python {yn('cv2')}")
print(f"      web engine    : PyQt6-WebEngine {yn('PyQt6.QtWebEngineWidgets')}")
print(f"      VLM runtime   : mlx-vlm {yn('mlx_vlm')} · vllm-mlx {yn('vllm_mlx')}")
print(f"      feature maps  : scikit-learn {yn('sklearn')} · DINOv3 mlx-image {yn('mlxim')}")
PY
note "robot adapter : ${ROBOT_ADAPTER} · heartbeat ${ROBOT_HEARTBEAT_SEC}s"
note "default model : ${PAVE_DEFAULT_MODEL}"
if [ "${PAVE_ALLOW_MODEL_DOWNLOADS}" = "1" ]; then
    note "VLM weights   : downloads enabled by PAVE_ALLOW_MODEL_DOWNLOADS=1"
else
    note "VLM weights   : cache-only; set PAVE_ALLOW_MODEL_DOWNLOADS=1 to fetch missing shards"
fi
if [ "${PAVE_MODEL_PREFLIGHT}" = "1" ]; then
    "$PYBIN" -m pave_mlx.downloads || true
fi

step "Launching operator console"
export PYTHONPATH=""
exec "$PYBIN" -m pave_ui.viewer
