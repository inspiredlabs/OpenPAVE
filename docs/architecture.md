# Architecture

## Roles
1) **VLA Observability UI / Debug UI**
- Human-facing interface for model/prompt iteration and performance observation.
- Implemented via `ui/live-vlm-webui` (submodule) + vLLM backend.

2) **VLA Control Daemon**
- Headless control core consuming intents and issuing ROS 2 commands.
- Decoupled from UI internals (uses a stable Intent Bus).

## Data flow
RTSP/Camera → WebUI inference → (server-side) POST STOP/TROT → Intent Ingress → `/tmp/vla_intent.json` → Control Daemon → ROS 2 → PuppyPi

## Components
- **PuppyPi (robot)**
  - Runs `puppy_control.launch.py` (ROS 2) in a container.
  - Exposes services/topics for motion control.

- **DGX (edge)**
  - vLLM backend (e.g., `llava-v1.6-mistral-7b-hf`)
  - `live-vlm-webui` (observability)
  - `intent_ingress.py` (HTTP → file bus)
  - `pave_control_daemon_mvp.py` (file bus → ROS 2 commands)

## Interfaces

### Intent Ingress API
`POST /intent`:
```json
{ "text": "STOP" }
```

### Intent file bus
`/tmp/vla_intent.json`:
```json
{"intent":"STOP"}
{"intent":"TROT"}
{"intent":"MOVE","vx":0.0,"yaw":0.6,"duration_ms":600}
```

## Notes
- `ROS_DOMAIN_ID` and `RMW_IMPLEMENTATION` must match on both sides.
- DDS multicast must work on the LAN.
