# Lightweight OpenPAVE Console

## Purpose

Stage 2 adds a lightweight OpenPAVE console at:

```text
/pave
```

The console is designed for Physical AI experiments, not as a general-purpose VLM web UI. The original Live VLM WebUI remains available at `/`.

## Backend Reuse

The console reuses existing `live-vlm-webui` backend capabilities:

- WebSocket updates from `/ws`
- WebRTC offer handling through `/offer`
- RTSP stream handling through the existing video backend
- GPU/system metrics from the existing monitor loop
- prompt updates through the existing websocket command path
- VLM responses from the existing `vlm_response` websocket messages

The Stage 2 implementation does not rewrite the video backend.

## Stage 1 Feedback Integration

The server exposes the latest Stage 1 feedback through:

```text
GET /api/pave/runtime
```

The endpoint reads:

```text
/tmp/vla_command_result.json
/tmp/vla_robot_state.json
```

or the paths configured with:

```bash
export COMMAND_RESULT_PATH=/tmp/vla_command_result.json
export ROBOT_STATE_PATH=/tmp/vla_robot_state.json
```

## Console Sections

The first version includes:

- live stream view
- stream connection status
- CPU usage
- memory usage
- GPU usage when available
- GPU memory usage when available
- prompt input
- active model and backend endpoint
- raw VLM output
- parsed intent summary
- command result JSON
- robot state JSON

## Running

Start the existing UI server as usual, then open:

```text
https://<host>:8090/pave
```

or the host and port printed by the server.

The full original UI remains at:

```text
https://<host>:8090/
```
