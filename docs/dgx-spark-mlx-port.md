# OpenPAVE: DGX Spark → Native MLX Port

A migration guide for continuing OpenPAVE development on an Apple Silicon MacBook,
replacing the DGX Spark edge-inference runtime with a native **MLX** runtime and a
native **PyQt6** operator console.

The target architecture follows the reference patterns in
` GitHub/jepa/support/WORKING/13` (the "template"):
its PyQt6 layout, its strict file separation of concerns, and the way its
placeholder **Physics Simulator** drives an MLX runtime in the background while
the GUI only observes.

---

## 1. Repo analysis

### 1.1 What OpenPAVE is today

OpenPAVE is a **local-first Physical AI reference workflow**. It is split into three
replaceable roles (see [README.md](../README.md) and [docs/architecture.md](architecture.md)):

| Role | Current implementation | Where it runs |
|------|------------------------|---------------|
| **ROS2 Robot / Sensor Endpoint** | PuppyPi quadruped + camera + `puppy_control` | Robot (RPi-class ARM) |
| **Edge Inference / Observability Node** | vLLM serving `llava-v1.6-mistral-7b` via an OpenAI-compatible API + modified `live-vlm-webui` + `/pave` console | **DGX Spark** |
| **OpenPAVE Runtime Control Layer** | Intent Ingress (Flask) + Control Daemon + Robot Adapters + JSON file bus | Control machine (currently DGX Spark) |

### 1.2 Component inventory and portability

To port it's important to state that **OpenPAVE is already platform-neutral pure Python** — only two things are tied to DGX Spark /
NVIDIA / ROS hardware.

| Component | Files | Portable to Mac as-is? |
|-----------|-------|------------------------|
| Intent schema / normalization | [pave_runtime/intent_schema.py](../pave_runtime/intent_schema.py) | ✅ Pure Python, no deps |
| Intent Ingress HTTP API | [intent_ingress/server.py](../intent_ingress/server.py) | ✅ Flask only |
| Control Daemon (file-bus poller) | [control_daemon/daemon.py](../control_daemon/daemon.py) | ✅ Pure Python |
| Feedback writers | [control_daemon/feedback.py](../control_daemon/feedback.py) | ✅ Pure Python |
| **Robot adapters** | [control_daemon/adapters.py](../control_daemon/adapters.py) | ⚠️ `PuppyPiAdapter` shells out to `docker run ... ros2` (needs Docker + ROS2 + DDS multicast + robot). `MockAdapter` is fully portable. |
| Launcher | [scripts/start_all_dgx.sh](../scripts/start_all_dgx.sh) | ⚠️ Boots vLLM-pointing UI + ROS env |
| **Edge inference** | vLLM + LLaVA (external service at `localhost:8000/v1`) | ❌ vLLM has no CUDA on Apple Silicon |
| **Web UI** | `ui/` submodule `live-vlm-webui` | ⚠️ Not checked out locally; web frontend tied to the VLM backend |

### 1.3 The two hard blockers on a MacBook

1. **vLLM does not run on Apple Silicon.** The "Edge Inference Node" assumes an
   NVIDIA GPU. The architecture's saving grace is that the inference contract is
   just an *OpenAI-compatible VLM API* (`UI_API_BASE=http://localhost:8000/v1`),
   so the backend is swappable without touching the control plane.

2. **The PuppyPi path needs Docker + ROS2 + DDS multicast + the physical robot.**
   On a Mac with no robot, you use `ROBOT_ADAPTER=mock` (already supported) and a
   **digital-twin physics simulator** stands in for the robot so STOP/TROT/MOVE
   are still observable.

Everything else (intent ingress, daemon, schema, feedback file bus) runs unchanged
on macOS today.

---

## 2. The template: what we are copying

Template `WORKING/13` is a V-JEPA 2.1 / DINO dense-feature analyzer that already
solved the exact problem we now face: **run MLX inference locally and present it in
a native PyQt6 app, with a swappable headless physics runtime as a second
"experience".** Three properties matter for the port.

### 2.1 PyQt6 GUI shape — [viewer.py](  )

- A single `QWidget` (`VJepaViewer`) owns the window.
- A top **`QComboBox` "Experience" selector** switches between modes
  (`EXPERIENCES = ["Camera MLX Inference", "Physics Simulator"]`, `viewer.py:514`).
- A **`QStackedWidget`** holds two views: a `QLabel` for the camera/feature display
  and a **`QWebEngineView`** for the embedded browser visualizer.
- Heavy work never runs on the UI thread. **`QThread` subclasses** do it:
  `EngineBuilder`, `DetectorBuilder`, `InferenceWorker`, `DetectorWorker`, all
  communicating back via `pyqtSignal`.
- **`QTimer`s** drive the loop: `capture_timer` (camera grab), `infer_timer`,
  `det_timer`. Switching experience *stops the timers* and releases the camera.

### 2.2 Separation of concerns — one responsibility per file

```
viewer.py              PyQt6 UI execution loop ONLY (widgets, timers, threads)
engine.py              MLX weight + device loader (model lifecycle)
spatio_temporal.py     Decoder: model output → renderable feature map
dino_engine.py         Alternate (faster) engine, same interface as engine.py
yolo26.py              Optional detector runtime, built/queried via its own thread
cartpole_mlx_runtime.py   Headless physics + policy + HTTP service (NO UI imports)
cartpole_viewer.html      Three.js observer (renders state, never computes it)
requirements.txt       One unified dependency file
```

The rule the template enforces (see
[CARTPOLE_MLX_RUNTIME.md]( WORKING/13/CARTPOLE_MLX_RUNTIME.md)):
**the GUI and the browser viewer are observation-only. They never own state or
compute.** They call an API and render what comes back.

### 2.3 The placeholder "Physics Simulator" + MLX runtime

`cartpole_mlx_runtime.py` is the model to copy. It has three clean layers:

1. **`PhysicsBackend`** (`CartPoleAnalyticBackend`) — owns state integration via
   `reset()` / `step(action)` / `info()`. The docstring explicitly frames it as a
   *stand-in* for the real (AVBD (2025 Augmented Block Descent)/WASM, or here PuppyPi) backend exposing the same
   methods.
2. **`PolicyRuntime`** — owns inference + training. It tries **MLX first
   (`MlxPolicy`) and falls back to NumPy (`NumpyPolicy`)** when Metal is
   unavailable (`cartpole_mlx_runtime.py:241-272`). Same `act(obs)` /
   `train(...)` interface either way. This is the key MLX pattern: a try/except
   around `import mlx.core` with a working NumPy fallback so headless/CI runs
   never break.
3. **`CartPoleService`** — glues physics + policy to a tiny `ThreadingHTTPServer`
   (`/api/state`, `/api/train`, `/api/mode`, `/api/command`, ...). It runs in a
   **daemon thread**, started/stopped by the GUI.

In `viewer.py`, selecting "Physics Simulator" (`_enter_physics_simulator`,
`viewer.py:771`):

- stops the camera/inference/detector timers and releases the camera (so GPU/Metal
  work is not contended),
- lazily constructs `CartPoleService()`, calls `.start()` (returns a URL),
- points the embedded `QWebEngineView` at that URL,
- and `_leave_physics_simulator` (`viewer.py:805`) tears it all down and resumes
  the camera.

That start/stop symmetry, the lazy service construction, and the MLX-with-fallback
policy are exactly what OpenPAVE's port will reuse.

---

## 3. Target architecture on the MacBook

We keep OpenPAVE's three-role split and its intent file bus **unchanged**. We only
replace the **Edge Inference / Observability Node** (DGX Spark + vLLM +
live-vlm-webui) with a **native PyQt6 + MLX** node, and we add a **digital-twin
physics simulator** to stand in for the PuppyPi.

```
            ┌─────────────────────────── Native MLX Edge Node (PyQt6) ───────────────────────────┐
 MacBook    │  pave_ui/viewer.py   (PyQt6 console — Experience selector, QStackedWidget)          │
 camera ──► │     ├─ Experience "Camera VLA Inference"  → pave_mlx/engine.py (MLX VLM/V-JEPA)      │
            │     │                                        → intent text (STOP/TROT/MOVE)          │
            │     └─ Experience "Physics Simulator"      → pave_sim/runtime.py (PhysicsBackend +   │
            │                                               MLX PolicyRuntime + HTTP) ─► QWebEngine │
            └───────────────────────────────┬───────────────────────────────────────────────────-┘
                                            │ POST /intent (HTTP :7071)         ▲ reads JSON feedback
                                            ▼                                   │
            intent_ingress/server.py ─► /tmp/vla_intent.json ─► control_daemon/daemon.py
                                            │
                                            ▼
                       ROBOT_ADAPTER=mock   (PuppyPiAdapter unused on Mac)
                                            │
                                            ▼
            /tmp/vla_command_result.json , /tmp/vla_robot_state.json  ──► back to the PyQt console
```

The **only contract between the new GUI and the existing control plane** is the same
one the web UI used:

- the GUI **POSTs** `{"text": "STOP"}` / `{"intent":"MOVE","params":{...}}` to
  `http://127.0.0.1:7071/intent` (the same call `live-vlm-webui` made, see
  [docs/live-vlm-webui-hook.md](live-vlm-webui-hook.md)), and
- the GUI **reads** `/tmp/vla_command_result.json` and `/tmp/vla_robot_state.json`
  for display.

So the control daemon, intent ingress, schema, and feedback files do not change.

### 3.1 Inference backend: two options

| | Tier A — keep the API contract | Tier B — native in-process MLX (template-aligned) |
|--|-------------------------------|---------------------------------------------------|
| VLM serving | Swap vLLM for **`mlx-vlm`'s OpenAI-compatible server** (`python -m mlx_vlm.server`) serving e.g. Qwen2-VL / LLaVA on Metal | Load the MLX model **inside** the PyQt process via `pave_mlx/engine.py`, exactly like the template's `engine.py` |
| UI | Keep `live-vlm-webui`, just point `UI_API_BASE` at the MLX server | New native PyQt6 console (`pave_ui/viewer.py`) |
| Effort | Low — one env var + one pip install | Higher — but this is the template's design and the real destination |
| Why | Fastest way to get *something* running on the Mac today | Matches template separation of concerns, no web stack, no extra process |

**Recommendation:** do Tier A first to unblock (one afternoon), then build Tier B as
the real port. The rest of this guide details Tier B.

### 3.2 Tier A with local perception encoders (DINOv3 / V-JEPA 2.1 / LingBot-Map)

A tempting shortcut is to reuse the MLX perception models from the sibling app
(` jepa/`: `dino_engine.py`, `engine.py` /
`vljepa_engine.py`, `lingbot_pointcloud_engine.py`) as the Tier A backend. **None of
them is an OpenAI-compatible VLM** — they are vision *encoders / geometry
extractors*, not language models:

| Model | File | `process_sequence()` returns | Emits text? |
|-------|------|------------------------------|-------------|
| **DINOv3** | `dino_engine.py` → `DinoV3InferenceEngine` | CLS/patch **embeddings**, 384-d (single frame) | ❌ |
| **V-JEPA 2.1** | `engine.py` / `vljepa_engine.py` → `VJepaInferenceEngine` | spatiotemporal feature **cube**, 768-d (frame window) | ❌ |
| **LingBot-Map** | `lingbot_pointcloud_engine.py` → `LingBotPointCloudEngine` | `PointCloudResult` (xyz/rgb/depth/pose) — and a self-described *placeholder* | ❌ |

The OpenAI client in `live-vlm-webui` expects `choices[].message.content` **text**.
There is no path from "feature tensor" to "the word STOP" without adding a small
**intent head**. So to use these models while *keeping the API contract*, wrap each
behind a thin OpenAI shim whose `content` is a discrete intent token
(`STOP`/`TROT`/`MOVE`). OpenPAVE itself stays unchanged.

For a true zero-OpenPAVE-change Tier A drop-in that needs *no* head, use a real MLX
VLM instead (`mlx-vlm`'s `python -m mlx_vlm.server`). The shim below is only for
reusing the jepa encoders.

#### Three responsibilities, kept separate

The design splits along what is shared versus what genuinely varies per model:

- **A — One universal shim (`pave_mlx/openai_shim.py`).** The OpenAI envelope —
  `POST /v1/chat/completions`, decode the inbound image, return
  `choices[].message.content` — is identical for every backend. It lives in exactly
  **one file** and selects a backend + head by name (`--backend dino|vjepa|lingbot`).
  This is the only component OpenPAVE/`live-vlm-webui` talks to.

- **B — Each trained head needs a serialised artifact.** A head is useless without
  its trained **weights** *and* a **manifest**: feature dim, pooling, the label set
  — `STOP/TROT/HOME/LEFT/RIGHT`, the text aliases
  [intent_schema.py](../pave_runtime/intent_schema.py) already accepts, so `LEFT`/
  `RIGHT` normalize to `MOVE`+yaw — plus normalization stats, source model id, and
  intent-schema version. One **manifest schema**, **three serialised instances**
  (`heads/configs/{dino,vjepa,lingbot}.json` + matching `.npz` weights). This is the
  per-model "serialised preferences" interface.

- **C — Backend adapter is the per-model code; the head and trainer are not.** The
  thing that genuinely differs per model is the **featurizer** (the three engines
  return different tensor shapes), so there are **three backend adapters**. But:
  - the **intent head collapses to two types**, not three — DINOv3 and V-JEPA both
    produce dense embeddings, so they share one parameterized `EmbeddingProbe`
    (`pool → linear → softmax`, differing only in `in_dim`/pooling); LingBot is a
    different modality and gets a `geometry_head`;
  - the **trainer collapses to one** parameterized CLI — the loop *(collect labeled
    frames → run the backend featurizer → fit the head → save weights+manifest)* is
    identical, only the featurizer changes;
  - the **labeler is shared** and backend-agnostic, with a per-backend sampling flag
    (V-JEPA needs a frame *window* per label, DINOv3 a single frame).

> What does this mean? do **not** write three heads and three trainers. Write **3 backend adapters + 2 head types + 1 trainer + 1 labeler + 3 config files**. Same capability, no triplication. The hardest shared work is producing labeled `(frame → intent)` data — that is a shared tool, not per-model.

#### File Composition

```
pave_mlx/
├─ openai_shim.py          # (A) 1× OpenAI-compatible server; --backend picks perception + head
├─ backends.py             # (C) PerceptionBackend protocol + 3 adapters wrapping the real engines
│                          #     dino → DinoV3InferenceEngine | vjepa → VJepaInferenceEngine
│                          #     lingbot → LingBotPointCloudEngine
├─ heads/
│  ├─ base.py              # IntentHead protocol + shared save/load of the manifest
│  ├─ embedding_probe.py   # (C) shared probe for DINOv3 AND V-JEPA (param: in_dim, pooling)
│  └─ geometry_head.py     # (C) LingBot only — occupancy / PointNet-style, different modality
├─ train_intent_head.py    # (C) 1× training CLI, --backend {dino,vjepa,lingbot}
├─ label_frames.py         # shared labeler (the real bottleneck); --backend flag controls sampling
├─ intent_decode.py        # logits → STOP/TROT/HOME/LEFT/RIGHT, pinned to intent_schema vocab
└─ heads/
   ├─ configs/             # (B) 1 manifest schema, 3 serialised instances
   │  ├─ dino.json
   │  ├─ vjepa.json
   │  └─ lingbot.json
   └─ weights/             # (B) trained head weights (.npz; git-ignored build artifacts)
      ├─ dino.npz
      ├─ vjepa.npz
      └─ lingbot.npz
```

> **Weights format:** `.npz` (NumPy, stdlib — no extra dependency), not
> `.safetensors`. The probe is tiny, so this keeps `pave_mlx` dependency-light; swap
> to safetensors only if a head ever grows large enough to warrant it.

Request flow (DINOv3 example), with OpenPAVE untouched:

```
live-vlm-webui ─POST /v1/chat/completions {image}─► openai_shim.py
   └─ backends.dino.embed(frame)            → 384-d patch-pooled embedding   (C)
        └─ heads.embedding_probe(features)  → logits over intents
             └─ intent_decode               → "TROT"
                  └─ {choices:[{message:{role:"assistant", content:"TROT"}}]}  (A)
```

Because DINOv3/V-JEPA are **self-supervised**, the `EmbeddingProbe` must be trained
as a small linear probe on a handful of labeled frames first (`train_intent_head.py
--backend dino`); LingBot's `geometry_head` is heuristic/occupancy rather than a
learned probe. Until a head is trained, the shim should return a safe default
(`STOP`) so the control path stays well-defined.

**Status: scaffolded and verified.** `pave_mlx/` exists with the DINOv3 path wired
end to end and V-JEPA/LingBot stubbed. The `DinoBackend` attempts the real
`DinoV3InferenceEngine` (via `JEPA_APP_DIR`) and falls back to a deterministic NumPy
featurizer when the MLX/`mlxim` stack is absent, so the pipeline runs anywhere —
the same MLX-or-fallback philosophy used elsewhere in this port. Verified: untrained
shim returns `STOP`; after `train_intent_head --backend dino` the reloaded shim
returns a learned intent token. See §6 Step 7 to operate it.

---

## 4. Propose a CUDA pipeline improvement – DGXSpark req.

This space has been intentionally left blank.

---

## 5. Proposed file layout (Tier B)

Mirror the template's one-responsibility-per-file rule. Add three new packages and
leave the existing control plane untouched.

```
openpave/
├─ pave_runtime/          (unchanged — schema)
├─ intent_ingress/        (unchanged — HTTP ingress)
├─ control_daemon/        (unchanged — daemon, adapters, feedback)
│
├─ pave_mlx/                       NEW — MLX inference, no UI imports
│  ├─ engine.py                    MLX device + VLM/V-JEPA weight loader   (≈ template engine.py)
│  ├─ intent_decode.py             VLM text/feature → OpenPAVE intent      (≈ template spatio_temporal.py)
│  └─ requirements.txt             mlx, mlx-vlm / mlx-image, numpy, opencv-python
│
├─ pave_sim/                       NEW — placeholder digital twin, no UI imports
│  ├─ runtime.py                   PhysicsBackend + PolicyRuntime + HTTP   (≈ cartpole_mlx_runtime.py)
│  └─ robot_viewer.html            Three.js quadruped observer             (≈ cartpole_viewer.html)
│
└─ pave_ui/                        NEW — PyQt6 console (replaces /pave web console)
   ├─ viewer.py                    Experience selector, QStackedWidget, QThreads (≈ template viewer.py)
   └─ requirements.txt             PyQt6, PyQt6-WebEngine
```

### 5.1 `pave_sim/runtime.py` — the placeholder physics simulator

Copy `cartpole_mlx_runtime.py` almost verbatim and re-skin it as a **PuppyPi digital
twin**. Keep the three layers and the MLX-with-NumPy-fallback policy. The crucial
addition for OpenPAVE: the simulator **consumes OpenPAVE intents** so the same
STOP/TROT/MOVE that would drive the robot now drives the twin.

```python
# pave_sim/runtime.py  (skeleton — mirrors cartpole_mlx_runtime.py)
from __future__ import annotations
import json, math, os, threading, time
from dataclasses import dataclass, field
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


class QuadrupedBackend:
    """Stand-in for the PuppyPi. Same reset()/step()/info() shape a real
    backend (ROS2 state bridge) would expose later."""
    def __init__(self):
        self.gait = "idle"      # idle | trot
        self.yaw_rate = 0.0
        self.vx = 0.0
        self.heading = 0.0
        self.x = 0.0
        self.y = 0.0
        self.t = 0.0

    def apply_intent(self, intent: str, params: dict):
        if intent == "TROT":
            self.gait, self.vx, self.yaw_rate = "trot", 0.0, 0.0
        elif intent == "STOP" or intent == "HOME":
            self.gait, self.vx, self.yaw_rate = "idle", 0.0, 0.0
        elif intent == "MOVE":
            self.gait = "trot"
            self.vx = float(params.get("vx", 0.0))
            self.yaw_rate = float(params.get("yaw", 0.0))

    def step(self, dt: float = 1.0 / 60.0):
        if self.gait == "trot":
            self.heading += self.yaw_rate * dt
            self.x += self.vx * math.cos(self.heading) * dt
            self.y += self.vx * math.sin(self.heading) * dt
        self.t += dt

    def info(self) -> dict:
        return {"gait": self.gait, "vx": self.vx, "yaw_rate": self.yaw_rate,
                "heading": self.heading, "x": self.x, "y": self.y, "t": self.t}


# PolicyRuntime: copy template MlxPolicy/NumpyPolicy verbatim. Optional for the
# twin (only needed if you want a learned gait controller); keep it for parity
# and to prove MLX owns policy weights on-device.

@dataclass
class RobotTwinService:
    host: str = "127.0.0.1"
    port: int = 8770
    backend: QuadrupedBackend = field(default_factory=QuadrupedBackend)
    # ... ThreadingHTTPServer in a daemon thread, identical to CartPoleService:
    #   GET /              -> robot_viewer.html
    #   GET /api/state     -> {"state": backend.info()}; backend.step()
    #   GET /api/intent    -> backend.apply_intent(...); used if you drive the
    #                          twin directly instead of via the file bus
```

> Two ways to feed the twin, pick one:
> - **File-bus mirror (recommended):** the twin's `tick()` reads
>   `/tmp/vla_robot_state.json` / `/tmp/vla_command_result.json` and animates from
>   the *actual* daemon output — so the simulator visualizes the real control path.
> - **Direct:** the GUI calls `/api/intent` on the twin. Simpler, but bypasses the
>   daemon and proves less.

Keep `robot_viewer.html` observation-only, exactly like `cartpole_viewer.html`: it
polls `/api/state` and renders; it computes nothing.

### 5.2 `pave_ui/viewer.py` — the PyQt6 console

Start from the template's `viewer.py` and strip it to the OpenPAVE experiences.

```python
# pave_ui/viewer.py  (skeleton — mirrors template viewer.py)
EXPERIENCES = ["Camera VLA Inference", "Physics Simulator"]

class PaveConsole(QWidget):
    def __init__(self):
        super().__init__()
        self.experience_name = EXPERIENCES[0]
        self.sim_service = None
        # QStackedWidget: [camera QLabel, QWebEngineView]   (template viewer.py:585)
        # QComboBox experience selector                     (template viewer.py:606)
        # QTimers: capture / infer / feedback-poll          (template viewer.py:731)

    def _enter_physics_simulator(self):       # cf. template viewer.py:771
        self.capture_timer.stop(); self.infer_timer.stop()
        if self.cap: self.cap.release(); self.cap = None
        from pave_sim.runtime import RobotTwinService
        if self.sim_service is None:
            self.sim_service = RobotTwinService()
        url = self.sim_service.start()
        self.web_view.setUrl(QUrl(url))
        self.content_stack.setCurrentWidget(self.web_view)

    def _leave_physics_simulator(self):       # cf. template viewer.py:805
        self.web_view.setUrl(QUrl("about:blank"))
        if self.sim_service: self.sim_service.stop()
        self.content_stack.setCurrentWidget(self.display)
        self.capture_timer.start(33)

    def _on_vlm_intent(self, text: str):      # camera experience → control plane
        # exactly what live-vlm-webui's maybe_post_intent did:
        requests.post("http://127.0.0.1:7071/intent", json={"text": text}, timeout=0.2)

    def _poll_feedback(self):                 # observability panel
        cmd = _read_json("/tmp/vla_command_result.json")
        state = _read_json("/tmp/vla_robot_state.json")
        self.info.setText(f"{state.get('status')} | {cmd.get('intent')} -> {cmd.get('status')}")
```

The camera experience uses MLX inference via `pave_mlx/engine.py` and a
`QThread`-based `InferenceWorker` (copy the template's), then maps the model's
verdict to a STOP/TROT/MOVE intent through `pave_mlx/intent_decode.py` and posts it
to Intent Ingress — same wire format the web UI used.

### 5.3 `pave_mlx/engine.py` — MLX loader

Reuse the template's `engine.py` pattern: the environment sandbox guard at the top
(`engine.py:8-13` strips stray `PYTHONPATH` / system `site-packages` so the MLX venv
wins), then a single class that loads weights once and exposes one inference method.
For the VLM, wrap `mlx-vlm`; for a lighter "is-something-there" trigger, reuse the
V-JEPA/DINO engines directly.

---

## 6. Step-by-step migration

### Step 0 — Mac dev environment

```bash
cd  GitHub/openpave
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r intent_ingress/requirements.txt
# control plane has no extra deps beyond Flask; verify it runs:
python3 -B -m unittest discover
```

Use Python **3.10–3.12** (the README's constraint for prebuilt wheels still
applies, and MLX wheels are happiest there).

### Step 1 — Prove the control plane runs on macOS unchanged

```bash
# Terminal 1
source .venv/bin/activate && python3 -m intent_ingress.server
# Terminal 2
source .venv/bin/activate && ROBOT_ADAPTER=mock python3 -m control_daemon.daemon
# Terminal 3
curl -s -X POST http://127.0.0.1:7071/intent -H 'Content-Type: application/json' -d '{"text":"TROT"}'
cat /tmp/vla_command_result.json
```

You should see `MOCK ACTION=TROT` and `status":"completed"`. **Nothing here is
DGX-specific.** This is your stable foundation.

### Step 2 — Tier A inference swap (fast unblock, optional)

```bash
python3 -m pip install mlx-vlm
python3 -m mlx_vlm.server --model mlx-community/llava-v1.6-mistral-7b-hf-4bit --port 8000
```

If you keep `live-vlm-webui`, point it at the MLX server and use the existing
launcher with a Mac profile (`UI_API_BASE=http://localhost:8000/v1`,
`ROBOT_ADAPTER=mock`). This reuses `start_all_dgx.sh` logic minus ROS. Confirm
`http://localhost:8000/v1/models` is reachable.

### Step 3 — Build `pave_sim/runtime.py`

Copy `cartpole_mlx_runtime.py` from the template into `pave_sim/runtime.py`, rename
`CartPoleService` → `RobotTwinService`, replace the cart-pole physics with the
`QuadrupedBackend` above, and copy `cartpole_viewer.html` → `robot_viewer.html`
(swap the cart/pole meshes for a simple quadruped/heading marker). Keep the
`MlxPolicy`/`NumpyPolicy` fallback verbatim. Test headless first:

```bash
python3 -c "from pave_sim.runtime import RobotTwinService; s=RobotTwinService(); print(s.start()); import time; time.sleep(1); s.stop()"
```

### Step 4 — Build `pave_ui/viewer.py`

Copy the template `viewer.py`, delete the DINO/YOLO/feature-map machinery you don't
need yet, keep: the `QStackedWidget`, the Experience `QComboBox`, the
camera-capture `QTimer`, the `QWebEngineView`, and the `_enter/_leave` lifecycle.
Wire:

- **Physics Simulator experience** → `RobotTwinService` (Step 3).
- **Camera experience** → MLX engine (Step 5) → `_on_vlm_intent` → POST to
  `:7071/intent`.
- **Feedback poll `QTimer`** → reads the two feedback JSON files for the status
  panel.

```bash
python3 -m pip install -r pave_ui/requirements.txt   # PyQt6, PyQt6-WebEngine
python3 -m pave_ui.viewer
```

### Step 5 — Build `pave_mlx/engine.py` + `intent_decode.py`

Bring the MLX VLM in-process (Tier B). Reuse the template `engine.py` sandbox guard
and one-load-one-infer shape. `intent_decode.py` maps model output to the intent
vocabulary — the same `STOP/TROT/MOVE` set the daemon already understands
(`pave_runtime/intent_schema.py:16`). Reuse the dedup idea from
[live-vlm-webui-hook.md](live-vlm-webui-hook.md) so you don't spam identical
intents.

### Step 6 — Retire the DGX launcher for Mac

Add a `configs/mac.env` profile (`ROBOT_ADAPTER=mock`, no ROS images, MLX endpoint)
and a `scripts/run_mac_demo.sh` that starts intent ingress + control daemon +
`python -m pave_ui.viewer` (no Docker, no ROS, no vLLM). The existing
`start_all_dgx.sh` stays for the DGX reference path.

### Step 7 — DINOv3 Tier A shim (`pave_mlx/`)

Keep the OpenAI contract but serve it from a local DINOv3 probe instead of a VLM
(see §3.2). The `pave_mlx/` package is already scaffolded; operating it is three
commands.

```bash
python3 -m pip install -r pave_mlx/requirements.txt   # numpy, pillow

# 1. label real frames the daemon's vocabulary understands (STOP/TROT/HOME/LEFT/RIGHT)
python3 -m pave_mlx.label_frames ingest --label TROT --src ~/clips/trot
#    (or: python3 -m pave_mlx.label_frames capture   # camera + key-to-label)

# 2. train the DINOv3 intent probe on those frames
python3 -m pave_mlx.train_intent_head --backend dino --data data
#    quick wiring check with no images/model:  --synthetic

# 3. serve it as OpenPAVE's Tier A backend — contract unchanged
python3 -m pave_mlx.openai_shim --backend dino --port 8000
#    then point OpenPAVE at it:  UI_API_BASE=http://localhost:8000/v1
```

Notes:

- **Real DINOv3 vs fallback.** `DinoBackend` loads the real `DinoV3InferenceEngine`
  from `JEPA_APP_DIR` (default ` jepa`) when `mlx` +
  `mlx-image` are present; otherwise it uses a deterministic NumPy featurizer so the
  shim still runs. Install `mlx-image` into the venv to flip `backend_mode` from
  `fallback` to `dinov3` with no code change. Inspect the active mode in the
  response's `x_pave` field.
- **Safe by default.** The committed `heads/configs/dino.json` is `trained: false`,
  so an untrained shim returns `STOP`. Training writes `.npz` weights (git-ignored)
  and flips the manifest to `trained: true`.
- **V-JEPA / LingBot** are stubbed: `train_intent_head --backend vjepa|lingbot` and
  the shim dispatch exist, but the backends raise `NotImplementedError` until wired.

---

## 7. Component mapping (template → OpenPAVE port)

| Template file | Responsibility | OpenPAVE port file |
|---------------|----------------|--------------------|
| `viewer.py` | PyQt6 UI loop, Experience selector, QThreads, QStackedWidget | `pave_ui/viewer.py` |
| `engine.py` | MLX device + weight loader | `pave_mlx/engine.py` |
| `spatio_temporal.py` | Model output → renderable/decodable map | `pave_mlx/intent_decode.py` |
| `cartpole_mlx_runtime.py` | PhysicsBackend + MLX PolicyRuntime + HTTP service | `pave_sim/runtime.py` |
| `cartpole_viewer.html` | Three.js observation-only viewer | `pave_sim/robot_viewer.html` |
| `requirements.txt` | Unified deps | `pave_ui/` + `pave_mlx/` requirements |

And the OpenPAVE pieces that **do not move** (already portable):
`pave_runtime/intent_schema.py`, `intent_ingress/server.py`,
`control_daemon/daemon.py`, `control_daemon/feedback.py`, and the
`/tmp/vla_*.json` file bus.

---

## 8. Compatibility & design notes

- **MLX is the Apple Silicon training/inference path; keep the logical interfaces
  identical to the DGX path.** Like the template (CARTPOLE_MLX_RUNTIME.md), the DGX
  Spark / ROS2 deployment should keep the same `PhysicsBackend` / `PolicyRuntime` /
  adapter contracts and only swap the tensor backend and deployment wrapper. Do not
  let MLX-specifics leak into the control plane.
- **Always ship the NumPy fallback.** The template's `PolicyRuntime` tries
  `MlxPolicy` and falls back to `NumpyPolicy` on any import/Metal failure
  (`cartpole_mlx_runtime.py:250-255`). Mirror this so headless CI and non-Metal
  machines still run.
- **GUI and viewer are observation-only.** The PyQt console and `robot_viewer.html`
  must consume APIs / feedback files and render — never own simulation state or
  call ROS directly. Hardware execution stays behind the intent bus + adapters.
- **Do not bake PuppyPi numbers into the simulator.** Keep gait limits, action
  scaling, and observation normalization in `pave_sim` config so the real PuppyPi
  model can replace the placeholder without rewriting the twin (same rule the
  template states for cart-pole vs. the real robot).
- **Keep the OpenAI-compatible inference contract.** It is OpenPAVE's stated "first
  inference backend contract" (architecture.md). Swapping vLLM→mlx-vlm preserves it;
  going fully in-process (Tier B) is an optimization, not a contract change.
- **Robot safety semantics still apply.** The launcher's shutdown-STOP behavior
  (`start_all_dgx.sh:145`) only matters for `ROBOT_ADAPTER=puppypi`; on the Mac
  mock/sim path there is no physical robot, but keep the STOP-on-exit habit when you
  later bridge the twin to real hardware.
- **The `ui/` submodule is not checked out locally.** If you stay on Tier A you must
  `git submodule update --init ui` and reinstall it; Tier B removes that dependency
  entirely.

---

## 9. Definition of done

1. `python3 -B -m unittest discover` passes on macOS (control plane intact).
2. Intent Ingress + Control Daemon (`ROBOT_ADAPTER=mock`) accept curl intents and
   write the two feedback files — no Docker/ROS.
3. `python -m pave_ui.viewer` opens, the **Experience** selector switches between
   **Camera VLA Inference** and **Physics Simulator**, and switching stops/starts
   the camera and the twin service cleanly (template's `_enter`/`_leave` symmetry).
4. In **Physics Simulator**, the embedded `QWebEngineView` shows the quadruped twin
   animating STOP/TROT/MOVE driven by the real daemon feedback files.
5. In **Camera VLA Inference**, an MLX model runs on the MacBook camera and posts
   `STOP`/`TROT` intents to `:7071`, which the daemon executes via the mock adapter.
6. MLX is active when Metal is present; the NumPy fallback runs everywhere else.

---

## 10. Near-term priorities (short-list)

Three concrete gaps to close next, in priority order. Each names the current file,
what's missing, and the smallest change that closes the gap. None of this requires
`pave_sim`, Docker, ROS2, or a physical robot.

Status tags used below: ✅ Landed · ⚠️ Partial/blocked · ❌ Not started.

### 10.1 Make `MockAdapter` a faithful ROS2/DDS stand-in — ✅ Landed (2026-07-01)

Today `MockAdapter` (`control_daemon/adapters.py:210-229`) just prints one line and
returns `AdapterActionResult.ok(...)` per call — it doesn't mirror the *shape* of what
`PuppyPiAdapter` actually does (named steps, per-step return codes, settle delays on
`stop`/`move`), and the ThreeJS visualiser (`visualiser/index.html:312-342`) only
animates a single-rigid-body yaw + bob derived from `command_result` params, because
`control_daemon/feedback.py`'s `robot_state()` carries no pose/joint telemetry and
there is no heartbeat/liveness signal independent of commands (a real DDS node
publishes liveliness on its own cadence).

To close the gap without hardware:

- **Step-shaped mock actions.** ✅ Landed — `MockAdapter` (`control_daemon/adapters.py:211-309`)
  now emits the same `steps=[{"name": ..., "return_code": ...}]` shape as
  `PuppyPiAdapter` (`set_running`, `set_mark_time`, `go_home`, `velocity_move`), with
  matching `time.sleep(0.3)` settle delays (skippable via `fast=`/`MOCK_ADAPTER_FAST`
  for tests).
- **Fault injection.** ✅ Landed — `MockAdapter.__init__` takes `fail_step`/`fail_rate`
  (env: `MOCK_ADAPTER_FAIL_STEP`, `MOCK_ADAPTER_FAIL_RATE`); `_step()` returns a
  non-zero `return_code` on a named-step match or random draw, so the daemon's and
  feedback files' failure-handling paths are now exercised.
- **Heartbeat / liveness.** ✅ Landed — `control_daemon/daemon.py:127-140`
  `start_robot_heartbeat()` runs a daemon thread that writes `robot_state.json` every
  `ROBOT_HEARTBEAT_SEC` (default **30.0s** as of §10.4's revision — deliberately
  occasional, not per-second — `daemon.py:69`) independent of commands, incrementing
  `heartbeat_seq` — closing the "heartbeat and liveness feedback" future target in
  `docs/architecture.md`.
- **Richer `robot_state`.** ✅ Landed — `robot_state()` (`control_daemon/feedback.py:54-74`)
  now carries `pose`, `joint_state`, and optional `heartbeat_seq`; `MockAdapter.get_state()`
  (`adapters.py:261-265`) exposes a live `x, y, heading` pose + 4-leg `joint_state`, wired
  through `daemon.py:97-124` (`adapter_telemetry()` / `write_robot_state()`). The
  visualiser (`visualiser/index.html:312-344`) now animates from real `robot_state.pose`
  when present, falling back to command-derived yaw/bob only when it's absent.
  *(Minor, acceptable simplification: `joint_state` is a flat 4-value dict, not a real
  gait signal — fine for a mock; revisit only if a gait-aware twin is ever built.)*

**Net: §10.1 is functionally closed.** No further action needed unless a richer digital
twin (`pave_sim`) is prioritized later.

### 10.2 Freeze Camera VLA Inference to two lightweight, swappable VLMs (iPhone-camera path) — ⚠️ Partial

`pave_ui/viewer.py` already lists `Qwen3-VL` and `Gemma 4 E4B` as swappable backends
(`MODEL_BACKEND`, `viewer.py:104-108`), backed by `QwenVLMBackend` / `GemmaVLMBackend`
(`pave_mlx/backends.py:319-326`, both 4-bit MLX quantizations), and the camera
enumeration code already anticipates Continuity Camera (`viewer.py:133-175`) — which
is how an iPhone 13 mini shows up as a capture source. MLX/`mlx-vlm` run on the Mac;
the phone is a camera, not a compute node.

- **Gemma 4 E4B backend implementation.** ✅ Landed — `pave_mlx/backends.py` implements
  `GemmaVLMBackend` on the shared `MlxVlmBackend` loader, closely following the
  upstream [`lmstudio-community/gemma-4-E4B-it-MLX-4bit`](https://huggingface.co/lmstudio-community/gemma-4-E4B-it-MLX-4bit)
  `config.json` (4-bit, `group_size=64`, `image_token_id=258880`). It also carries a
  targeted compatibility patch, `_patch_gemma4_shared_kv_sanitize()`
  (`backends.py:74-120`), for a known shared-KV checkpoint mismatch, local-HF-snapshot
  detection (`_local_hf_snapshot`), and load-error messages that name the specific
  failure mode (shape mismatch vs. missing Metal device vs. incomplete cache).
  - ⚠️ **Audio tower not wired (accepted, deferred).** The checkpoint carries an
    `audio_tower` (`gemma4_audio`) and `audio_token_id=258881`, but OpenPAVE only
    builds an image+text request in `generate()` (`backends.py:308-316`) — no
    microphone capture, no audio tensor path anywhere in `pave_ui`/`pave_mlx`. This is
    an intentional scope cut for now, not a regression; revisit only when
    microphone-driven intent becomes a priority.
- **Scope the model row to just these two for now.** ❌ Not started — `MODELS`
  (`viewer.py:103`) still lists DINOv3/V-JEPA/LingBot alongside the VLMs with no
  "deferred" marking in the UI.
- **Gemma 4 E4B is the startup default.** ✅ Landed (2026-07-01) —
  `DEFAULT_MODEL = os.environ.get("PAVE_DEFAULT_MODEL", "Gemma 4 E4B")`
  (`viewer.py:102-105`). If Gemma isn't cached yet, `_start_default_model` still
  falls back safely to a model that fits rather than triggering a surprise ~7GB
  download (unchanged pre-existing behavior).
- **Verify Continuity Camera end-to-end on an actual iPhone 13 mini.** ❌ Not yet
  confirmed in this round of changes.
- **Confirm clean model-swap teardown under memory pressure.** ❌ Not yet verified.
- **Quantify "light-weight."** ❌ No documented min-RAM/latency figures yet.

### 10.3 Camera VLA Inference: prompt-driven + gesture control — ✅ Landed (2026-07-01)

`ROBOT_PROMPT` was a single hardcoded string, overridable only via an env var — there
was no runtime prompt UI, and `prompts/robot-commander-gesture.json` was never loaded
by any code path. Raw VLM text was discarded before reaching the UI. The prompt-probe
buttons were also wired but blocked by a real MLX/Metal threading bug.

- **The `Stream(gpu, 1)` bug is fixed.** ✅ Landed — root cause confirmed: the old
  design loaded the model on one `QThread` (`EngineBuilder`) and ran every inference
  call on a **brand-new native thread** (Qt spins up a fresh OS thread each time
  `start()` is called on a finished `QThread`), so MLX's Metal stream context — bound
  to whichever thread first touched it — was never valid on any of those per-call
  threads. `EngineBuilder` + `InferenceWorker` were replaced with a single persistent
  `perception.EngineWorker` (`perception.py`) that loads the model *and* serves every
  inference job for that model's lifetime from the same native thread. `viewer.py`'s
  `_build_engine`/`_teardown_engine`/`_on_engine_ready`/`_on_engine_failed` were
  updated to match (one worker object instead of two, `self._loading` replaces the
  `isRunning()` re-entrancy check).
- **Command-stream ordering: gesture ticks and prompt-probe taps no longer compete.**
  ✅ Landed by design, not just as a side effect of the bug fix. `EngineWorker` holds
  a `queue.Queue` (`perception.py`); both `_infer_tick` (periodic camera/gesture
  inference) and `_run_prompt_probe` (button taps) call `submit()`, which only ever
  *enqueues* — it never blocks, drops, or lets one cancel the other. Jobs are
  processed strictly FIFO and each `done` signal now carries its own
  `(result, ms, source, label)` (`perception.py`), so results can never be misattributed
  to the wrong job even when several are queued back-to-back. `_infer_tick` is
  self-throttled (`self._camera_job_pending`) so a slow model can't pile up stale
  camera frames, but this throttle **only applies to camera ticks** — prompt-probe
  submissions always go straight into the queue immediately, regardless of what the
  camera tick is doing.
- **(A) One-off / free-text prompt plumbing.** ✅ Landed and now verifiable —
  `pave_ui/viewer.py` has five prompt-probe buttons under the camera preview
  (`PROMPT_BUTTONS`; UI wiring `_build_ui`; handler `_run_prompt_probe`) that post the
  model's answer to `intent_ingress` (source `"prompt-probe"`). `perception.infer()`
  and `EngineWorker.submit()` take an explicit `prompt` argument instead of always
  using the module-level `ROBOT_PROMPT`. Raw model text is surfaced via
  `InferResult.raw_text` in both the console log and the `prompt_status` label.
- **(B) Gesture recognition — explicitly full-vocabulary, not restricted.** ✅ Landed,
  and deliberately different from the original §10.3 plan. The plan had proposed
  loading `prompts/robot-commander-gesture.json` and *restricting*
  `clamp_to_intent` to its narrow `["STOP","TROT"]` `allowed_intents` while that
  preset was active. **That idea was rejected.** Instead, the default
  `ROBOT_PROMPT` (`pave_ui/perception.py`, mirrored in `pave_mlx/openai_shim.py`)
  was rewritten to be gesture-aware across the *full* five-word vocabulary
  (thumbs-up→`TROT`, open palm→`STOP`, closed fist→`HOME`, point-left→`LEFT`,
  point-right→`RIGHT`), and `clamp_to_intent` is untouched — it still matches
  against all five `INTENT_LABELS`. `prompts/robot-commander-gesture.json` itself is
  still not programmatically loaded; it was superseded by folding its intent
  directly into the default prompt rather than being wired as a selectable,
  vocabulary-restricting preset.
- **Visual feedback for what the model "sees."** ⚠️ Landed, then superseded —
  this originally shipped as a single HSV-skin-blob "hand ROI" box
  (`find_hand_roi()`/`draw_gesture_box()`), was expanded to a multi-class
  Haar-cascade face/hand/body overlay in §10.5, and was then **removed entirely**
  in §10.6 after testing showed it produced frequent false positives ("utter
  garbage"). See §10.6 for what replaced it (the model's own raw text, drawn as
  a caption).
- **Respect the preset's own safety note.** ✅ Landed — `TROT_CONFIRMATIONS` /
  `TROT_CONFIRMATION_WINDOW_MS` (`viewer.py`, defaults `2` / `1500`, matching
  `scenarios/puppypi-gesture-stop-trot.json`'s `recommended_env`) are read at
  import time, and `_gate_trot()` requires TROT to be seen that many times in a
  row within that window — from *either* the gesture tick or a prompt-probe tap,
  since they're the same evidence stream — before it's forwarded to
  `intent_ingress`. Every other intent (including STOP) is never gated, matching
  the scenario's "open palm or uncertainty emits STOP" success criterion, which
  must stay immediate.

### 10.4 Idle sends nothing; the daemon heartbeat is occasional, not continuous — ✅ Landed (2026-07-01, revised later same day)

First pass (earlier 2026-07-01): the fake `DEMO_STEPS` schedule (full
TROT/turn/TROT/STOP every 2.5s) was scaled down to `KEEPALIVE_STEPS` — tiny
yaw-only nudges every 20s — on the theory that *some* periodic liveness signal
was still wanted. **That was revised the same day**, after feedback that idle
should not synthesize *any* intent traffic: the only thing idle needs to do is
let the already-existing, independent `control_daemon` heartbeat keep the
"ROBOT STATE — live data time series" panel showing occasional updates.

- `pave_ui/viewer.py`: `KEEPALIVE_STEPS`/`keepalive_timer`/`_keepalive_tick` were
  removed entirely (not just slowed down). The source-selector option is now
  `"Idle"` (was `"Demo schedule"`, then briefly `"Keep-alive (idle)"`), and
  selecting it posts **nothing** — `_select_source`'s else-branch only stops
  `infer_timer` and clears `self._hand_roi`. Prompt-probe buttons still work in
  Idle (they're independent of the source selector).
  `mlx-runtime/main.py`'s standalone `DEMO_STEPS`/`run_demo_schedule` (a separate
  smoke-test entrypoint, not launched by `mlx-runtime.sh`) was *not* reverted the
  same way — it still posts tiny nudges every 20s, since disabling it would defeat
  its own stated purpose of proving the control path end-to-end without a UI. It
  remains opt-outable via `DEMO_SCHEDULE=0`.
- `control_daemon/daemon.py`: `ROBOT_HEARTBEAT_SEC` default changed from `1.0` to
  `30.0` — this heartbeat's only job is to keep the live time-series panel
  non-empty during idle development, not to generate steady data; every real
  command still writes `robot_state.json` immediately regardless of this interval
  (`execute_intent`'s own `write_robot_state` calls are unaffected).

**Net for this round: §10.3 and §10.4 are both closed; §10.2 has one more landed
item (Gemma default) but the model-row scoping / iPhone verification / memory /
latency items are still open.**

### 10.5 Overlay quality, idempotent posting, true idle silence — ✅ Landed (2026-07-01, third pass)

Real bugs from actually running the console: the §10.3 visual-feedback overlay
rendered one giant gray box with its label painted over; nothing stopped a
continuously-recognized gesture from re-POSTing the same intent every ~1.5s
forever; and the ROBOT STATE panel was still visibly filling up during idle
despite §10.4's changes.

- **The overlay's "one gray box, label covered" bug — root-caused and fixed.**
  Two separate bugs, not one: (1) the single skin-color hand heuristic had no
  upper size bound, so ordinary lighting/background regularly filled most of
  the frame with "skin-toned" pixels and produced one huge box; (2) `_draw_overlay`
  (`viewer.py`) drew the full-width status band **after** the detection box, so
  whenever the box/label sat near the top of the frame the band painted over it
  — that's the "label is covered" symptom. Fixed both: `detect_regions()`
  (`perception.py`) now bounds candidate regions to `_MIN_ROI_AREA_FRAC`–
  `_MAX_ROI_AREA_FRAC` (~1%–35% of the frame) and a plausible aspect ratio, and
  `_draw_overlay` now draws the status band **first**, detection boxes **last**
  — boxes/labels are always the topmost layer and can never be covered again.
- **Face / hand / body detection, not just one ad hoc hand blob.** ⚠️ Landed
  here, then **removed in §10.6** — synthetic-frame tests below passed, but
  real webcam footage produced frequent false positives ("utter garbage" per
  testing) and was pulled entirely. Kept below for the record of what was tried
  and why it didn't survive contact with a real camera.
  `pave_ui/perception.py` adds `detect_regions()`, combining two OpenCV Haar
  cascades that ship inside `opencv-python` (`cv2.data.haarcascades`) —
  `haarcascade_frontalface_default.xml` for `face`, `haarcascade_upperbody.xml`
  for `body` — with a refined skin-color contour heuristic for `hand` that now
  **excludes the detected face region** first (a face is skin-toned too) before
  looking for hand-sized blobs, capped at 2 candidates. Each gets its own small,
  tight box, its own fixed color (`REGION_COLORS`), and its class name as the
  label drawn inside the box — this is genuinely multiple small labelled boxes,
  not one intent-colored blob. Verified against synthetic frames: random noise
  → zero regions; a small localized skin-toned patch → one tight `hand` box
  matching the patch; a full-frame uniform color → correctly rejected (too big).
  Still approximate/best-effort by construction (documented in the module
  docstring) — there's no real ML detector backing this, on purpose (§10.3
  already ruled that out: no weight-conversion pipeline in this repo).
- **Idempotent posting — critical flood guard.** ✅ Landed — `viewer.py`'s
  `_on_engine_infer` now tracks `self._pending_intent`/`self._pending_since` and
  calls `_action_in_flight()` (reads `COMMAND_RESULT_PATH`, true while status is
  outside `_TERMINAL_COMMAND_STATUSES`, with a `_PENDING_FALLBACK_MS` safety
  valve so it can never get stuck) before posting. A repeated identical intent —
  from a gesture held in frame across multiple ticks, or a prompt-probe button
  tapped twice — is now a silent no-op while the previous instance of that same
  intent is still executing. **This is commented in the code as critical, not
  incidental**: without it, a continuously-recognized gesture floods
  intent_ingress, the daemon, and (transitively) the ROBOT STATE feed with the
  same command on every single camera tick.
- **The ROBOT STATE panel is now genuinely silent while idle.** ✅ Landed — the
  real flood source was `mlx-runtime/state_server.py`'s SSE `_stream()`, which
  pushed a new frame every `STREAM_INTERVAL_S` (0.2s) **unconditionally**,
  and the browser's `render(frame)` (`visualiser/index.html`) appended a new row
  on every single message — 5 rows/sec, forever, even at rest. Fixed at the
  source: `_stream()` now computes `_significant(frame)` — the parts of a frame
  that represent an actual change (`robot_state.status`/`pose`/`joint_state`/
  `last_command` and all of `command_result`) — and only sends (so the browser
  only renders) when that differs from the last frame sent. `heartbeat_seq`/
  `updated_at`/`server_time` are deliberately excluded from the comparison, so
  the daemon's own liveness heartbeat (§10.1) no longer causes any visible
  panel/console churn — it still writes to `robot_state.json` for any other
  consumer, it just no longer reaches the browser as a "new" event. Net: the
  panel now only grows when a real command is received, accepted, executed, or
  completes.

### 10.6 Deferred default load, gated prompt buttons, and dropping the Haar-cascade overlay — ✅ Landed (2026-07-01, fourth pass)

Two more issues surfaced from actually running the console.

- **Default model load deferred ~100ms past window paint.** ✅ Landed —
  `PaveConsole.__init__` (`viewer.py`) used to call `_start_default_model()`
  synchronously before the window was ever shown, so the console could appear
  to hang before its first paint. It now schedules
  `QTimer.singleShot(100, self._start_default_model)` instead, so the window is
  on-screen first and the (still-default) Gemma load visibly starts just after.
- **Prompt-probe buttons are now actually disabled, not just soft-guarded.**
  ✅ Landed — the five buttons (`PROMPT_BUTTONS`) are created `setEnabled(False)`
  and kept in `self.prompt_buttons`; a new `_set_prompt_buttons_enabled()` is
  called `False` at the start of every `_set_model_backend()` (covers "No
  model", switching models, and the load window) and `True` from
  `_on_engine_ready` only when `handle.kind == "vlm"` **and** the load didn't
  fall back with an error. Previously clicking them with no model ready just
  produced an error message in `prompt_status`; they were never actually
  greyed out or unclickable.
- **The Haar-cascade/skin-blob overlay from §10.5 was removed entirely.** ⚠️ In
  practice it produced frequent, confident-looking false positives — a classical
  CV guess is not better visual feedback than the model's own answer.
  `detect_regions()` / `draw_regions()` / `Region` / `REGION_COLORS` / the
  face+body cascades and skin-contour hand heuristic are all gone from
  `pave_ui/perception.py`. **This step over-corrected** — it replaced the boxes
  with a text-only caption and dropped annotated boxes entirely, which was not
  what was asked for; see §10.7, landed immediately after, which brings
  annotated boxes back sourced from the model's own reported features instead
  of a classical-CV guess.

### 10.7 Correction: annotated boxes are back, sourced from Gemma itself — ✅ Landed (2026-07-01, fifth pass)

§10.6 over-corrected: dropping the Haar-cascade overlay was right, but it also
dropped the annotated-box format entirely in favor of a plain text caption. The
actual ask was to keep small annotated boxes with labels — just have the
*content* come from Gemma/Qwen instead of a generic classical-CV guess. Fixed:

- **`ROBOT_PROMPT` now asks for structured output** (`pave_ui/perception.py`):
  an `INTENT: <word>` line (unchanged semantics/vocabulary from §10.3) plus 0–3
  `FEATURE: <label> <grid-cell>` lines for whatever the model actually sees
  (face, hand, body, or a notable object), where grid-cell is one of a named
  3×3 grid (`top-left` … `bottom-right`). A small on-device model can't reliably
  regress precise pixel coordinates, so it only has to pick one of 9 cells,
  which maps deterministically to a fixed box (`_grid_box`) — coarse by
  construction, but every box is real model output, never a fabrication.
- **Parsing degrades gracefully.** `parse_model_response()` extracts the
  `INTENT:`/`FEATURE:` lines it can find; if the model ignores the format
  (e.g. the prompt-probe buttons deliberately ask for a bare one-word reply,
  no structure at all), `intent_text` falls back to the whole response for
  `clamp_to_intent` exactly as before, and the feature list is simply empty —
  no boxes that tick, no crash, no special-casing needed by the caller.
  Verified directly: a well-formed multi-line reply extracts intent + valid
  features and silently drops a malformed grid-cell name; a bare `"STOP"`
  reply falls back cleanly with zero features.
  `features_to_boxes()` converts the surviving (label, grid-cell) pairs into
  pixel boxes, and `draw_features()` draws them — same "small box + label chip
  inside it, drawn after the status band so it's never covered" convention as
  the removed Haar-cascade version, just fed from `self._features`
  (`viewer.py`, populated in `_on_engine_infer`) instead of a detector.
- **The caption band stays too**, showing the model's full raw reply
  alongside the boxes rather than instead of them.
- `generate()`'s `max_tokens` was bumped from 12 to 80 for this call (enough
  for `INTENT:` + 3 `FEATURE:` lines) — a bare one-word prompt-probe reply
  still stops at its own EOS well under that cap, so this doesn't add latency
  to that path.

(§10.6's other two items — deferred default-model load and genuinely-disabled
prompt-probe buttons — are unchanged by this pass and still stand as landed.)
