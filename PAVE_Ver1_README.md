# PAVE Physical AI MVP (PuppyPi Remote Control)

This repository contains a GitHub-ready MVP that connects **VLA/VLM inference results** to **ROS 2 robot control** on **PuppyPi** via a simple, reproducible pipeline:

**live-vlm-webui (Observability UI)** → **Intent Ingress (HTTP)** → **Intent File Bus** → **Control Daemon** → **ROS 2 commands** → **PuppyPi**

## What’s in this repo

- `intent-ingress/`  
  A tiny HTTP service (`/intent`, default port **7071**) that maps `STOP/TROT/...` into a JSON intent and writes it atomically to `/tmp/vla_intent.json`.

- `control-daemon/`  
  A minimal control daemon that watches `/tmp/vla_intent.json` (mtime de-dupe) and emits ROS 2 commands (via dockerized ROS2 CLI) to the robot.

- `docs/`  
  Architecture and runbook documentation.

- `scripts/`  
  Start scripts for the DGX-side services.

> Note: The **live-vlm-webui** codebase is typically kept as a separate repo (fork/submodule).  
> This repo documents a small server-side hook you can apply to `server.py` to POST `STOP/TROT` into Intent Ingress.

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

### 3) Manual test (writes to file bus)
```bash
curl -s -X POST http://127.0.0.1:7071/intent   -H 'Content-Type: application/json'   -d '{"text":"TROT"}'

curl -s -X POST http://127.0.0.1:7071/intent   -H 'Content-Type: application/json'   -d '{"text":"STOP"}'
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
- `docs/architecture.md` — system architecture (roles, flow, interfaces)
- `docs/runbook.md` — stable runbook / troubleshooting (FastDDS RTPS issue etc.)
- `docs/live-vlm-webui_hook.md` — minimal server-side hook to forward STOP/TROT to ingress

## Status (MVP)
- ✅ STOP/TROT working end-to-end
- ✅ TURN (yaw) working through `/puppy_control/velocity_move` (platform-dependent)
- ⚠️ Straight walking (vx) is still **debug** (controller/firmware dependent)

## License
Choose a license appropriate for your distribution (MIT/Apache-2.0 recommended). A placeholder is provided in `LICENSE`.
