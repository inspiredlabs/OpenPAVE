# PAVE Physical AI MVP (PuppyPi Remote Control) Version #1

This repo is a GitHub-ready MVP that connects **VLA/VLM inference results** to **ROS 2 robot control** on **PuppyPi** using a simple, reproducible pipeline:

**live-vlm-webui (Observability UI)** → **Intent Ingress (HTTP)** → **Intent File Bus** → **Control Daemon** → **ROS 2 commands** → **PuppyPi**

## Repository layout

- `ui/live-vlm-webui/` (**git submodule**)  
  Observability UI for model/prompt iteration and performance monitoring.  
  Fork: `https://github.com/odincodeshen/live-vlm-webui/`

- `intent-ingress/`  
  Tiny HTTP service (`/intent`, default port **7071**) mapping `STOP/TROT/...` into JSON and writing atomically to `/tmp/vla_intent.json`.

- `control-daemon/`  
  Watches `/tmp/vla_intent.json` (mtime de-dupe) and emits ROS 2 commands (dockerized ROS2 CLI) to the robot.

- `docs/`  
  Architecture, runbook, and live-vlm-webui integration notes.

- `scripts/`  
  Start scripts and helper commands.

## Clone (with submodule)

```bash
git clone --recurse-submodules <YOUR_MAIN_REPO_URL>
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

## Quick start (DGX)

### 0) Prereqs
- Docker installed
- Python 3.10+
- Network access to PuppyPi (same LAN; ROS 2 multicast works)
- `ROS_DOMAIN_ID` and `RMW_IMPLEMENTATION` must match between DGX and PuppyPi

### 1) Start Intent Ingress
```bash
cd intent-ingress
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 intent_ingress.py
```

Health check:
```bash
curl -s http://127.0.0.1:7071/healthz
```

### 2) Start Control Daemon
In another terminal:
```bash
cd control-daemon
python3 pave_control_daemon_mvp.py
```

### 3) Start live-vlm-webui (Observability UI)
```bash
cd ui/live-vlm-webui
# follow upstream/fork instructions to start the webui + vLLM backend
```

### 4) Manual end-to-end test (writes to file bus)
```bash
curl -s -X POST http://127.0.0.1:7071/intent \
  -H 'Content-Type: application/json' \
  -d '{"text":"TROT"}'

curl -s -X POST http://127.0.0.1:7071/intent \
  -H 'Content-Type: application/json' \
  -d '{"text":"STOP"}'
```

## Quick start (PuppyPi)

Inside the robot-side ROS 2 container (example):
```bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
source /opt/ros/humble/setup.bash
ros2 launch puppy_control puppy_control.launch.py
```

## Documentation
- `docs/architecture.md`
- `docs/runbook.md`
- `docs/live-vlm-webui_hook.md`

## Status (MVP)
- ✅ STOP/TROT working end-to-end
- ✅ TURN (yaw) working through `/puppy_control/velocity_move` (platform-dependent)
- ⚠️ Straight walking (vx) is still debug / platform-dependent

## License
MIT (see `LICENSE`)
