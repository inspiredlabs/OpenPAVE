# OpenPAVE Stage 3A Developer Runtime Launcher

This guide validates the Stage 3A developer runtime launcher.

Stage 3A does not replace the distributed OpenPAVE architecture. It provides one command that starts the local developer-facing runtime processes that are already part of Stage 1 and Stage 2.

## Managed Services

The launcher starts and supervises:

- Intent Ingress
- Control Daemon
- OpenPAVE UI server / `live-vlm-webui`

The launcher writes logs to:

```text
.openpave/logs/
```

## External Dependencies

These are still external dependencies in Stage 3A:

- robot-side ROS2 controller
- robot/sensor stream source
- vLLM or another OpenAI-compatible VLM backend
- Docker images used by `PuppyPiAdapter`, such as `ros:humble` and `puppy-ros2-cli:humble`

This keeps Stage 3A focused on developer runtime orchestration without hiding hardware or inference setup problems.

## Basic Usage

From the repo root:

```bash
cd /path/to/OpenPAVE
source .venv/bin/activate

./scripts/run_stage3_demo.sh
```

To use an explicit software-only profile:

```bash
OPENPAVE_CONFIG=configs/mock.env ./scripts/run_stage3_demo.sh
```

The launcher defaults to:

```text
ROBOT_ADAPTER=mock
UI_PORT=8090
UI_API_BASE=http://localhost:8000/v1
UI_MODEL=llava-hf/llava-v1.6-mistral-7b-hf
ROBOT_IP_ADDRESS=192.168.0.8
```

`ROBOT_ADAPTER=mock` is the default to avoid accidental physical robot motion.

## Physical PuppyPi Validation

Start the PuppyPi ROS2 controller and vLLM backend first, then run:

```bash
cd /path/to/OpenPAVE
source .venv/bin/activate

OPENPAVE_CONFIG=configs/puppypi.env ./scripts/run_stage3_demo.sh
```

`configs/puppypi.env` contains the PuppyPi adapter, robot IP address, VLM model/backend, ROS domain, RMW implementation, and ROS CLI Docker image settings.

You can still override a profile value from the shell:

```bash
OPENPAVE_CONFIG=configs/puppypi.env ROBOT_IP_ADDRESS=192.168.0.42 ./scripts/run_stage3_demo.sh
```

## Open URLs

The launcher prints both UI entry points:

```text
http://127.0.0.1:8090/
http://127.0.0.1:8090/pave
```

Use `/` for the full upstream `live-vlm-webui`.

Use `/pave` for the lightweight OpenPAVE console.

## Runtime Files

The launcher uses these default runtime files:

```text
/tmp/vla_intent.json
/tmp/vla_command_result.json
/tmp/vla_robot_state.json
```

Override them with:

```bash
export INTENT_PATH=/tmp/vla_intent.json
export COMMAND_RESULT_PATH=/tmp/vla_command_result.json
export ROBOT_STATE_PATH=/tmp/vla_robot_state.json
```

## Logs

Inspect logs with:

```bash
tail -f .openpave/logs/intent-ingress.log
tail -f .openpave/logs/control-daemon.log
tail -f .openpave/logs/openpave-ui.log
```

## Health Checks

The launcher checks:

```text
http://127.0.0.1:7071/healthz
http://127.0.0.1:8090/pave
```

It also probes the configured OpenAI-compatible VLM endpoint:

```text
http://localhost:8000/v1/models
```

If the VLM endpoint is not reachable, the launcher still starts the UI. Layout, WebSocket connectivity, runtime feedback, and manual intent testing can still be validated without model inference.

## Stop

Press `Ctrl+C` in the launcher terminal.

The launcher shuts down the managed child processes:

- Intent Ingress
- Control Daemon
- OpenPAVE UI server

It does not stop external dependencies such as vLLM or the robot-side ROS2 controller.
