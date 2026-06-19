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

## Install live-vlm-webui Observability UI

The Stage 3 launcher starts the OpenPAVE console through the `ui/` submodule, which points to the OpenPAVE-maintained `live-vlm-webui` fork.

After cloning or pulling OpenPAVE, initialize the submodule and install it into the repo-level virtual environment:

```bash
cd /path/to/OpenPAVE

git submodule update --init --recursive

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r intent_ingress/requirements.txt
python3 -m pip install -e ui
```

Confirm the installed module resolves to the checked-out submodule:

```bash
python3 -c "import live_vlm_webui.server as s; print(s.__file__)"
```

Expected path:

```text
/path/to/OpenPAVE/ui/src/live_vlm_webui/server.py
```

Confirm the OpenPAVE console route and text-inference endpoint exist in the submodule checkout:

```bash
grep -n 'app.router.add_get("/pave"' ui/src/live_vlm_webui/server.py
grep -n 'app.router.add_post("/api/pave/infer"' ui/src/live_vlm_webui/server.py
```

If `/` works but `/pave` returns `404`, update the submodule:

```bash
git submodule update --init --recursive
python3 -m pip install -e ui
```

## Build the Custom ROS 2 CLI Docker Image

The PuppyPi adapter uses two ROS2 CLI Docker images:

```text
ROS_SVC_IMAGE=ros:humble
ROS_PUB_IMAGE=puppy-ros2-cli:humble
```

`ros:humble` is enough for standard ROS2 service calls such as `SetBool` and `Empty`.
`puppy-ros2-cli:humble` is required for PuppyPi custom message publishing, including:

```text
puppy_control_msgs/msg/Velocity
```

Build the custom image on the DGX/control machine before running physical PuppyPi `MOVE` validation or any scenario that publishes PuppyPi custom messages:

```bash
cd /path/to/OpenPAVE

./scripts/build_puppy_ros2_cli.sh
```

The script expects this repo path to exist:

```text
third_party/puppy_control_msgs
```

It builds and verifies:

```text
puppy-ros2-cli:humble
```

You can verify the image manually:

```bash
docker run -it --rm puppy-ros2-cli:humble bash -lc \
"source /opt/ros/humble/setup.bash && source /ws/install/setup.bash && ros2 interface show puppy_control_msgs/msg/Velocity"
```

This custom image is not required for `ROBOT_ADAPTER=mock`.
For PuppyPi `STOP`, `TROT`, and `HOME`, the adapter primarily uses `ros:humble` service calls.
For PuppyPi `MOVE`, the adapter uses `ROS_PUB_IMAGE`, which defaults to `puppy-ros2-cli:humble`.

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

## Start a vLLM Backend

Stage 3 can start the OpenPAVE runtime without vLLM, but real VLM inference requires an OpenAI-compatible backend at the configured `UI_API_BASE`.

Use a separate vLLM virtual environment to avoid dependency conflicts with the OpenPAVE repo-level `.venv`:

```bash
cd /path/to/OpenPAVE

python3 -m venv .venv-vllm
source .venv-vllm/bin/activate
python3 -m pip install -U pip
python3 -m pip install -U vllm
python3 -m pip check
```

Start vLLM with the default Stage 3 model:

```bash
source .venv-vllm/bin/activate

vllm serve llava-hf/llava-v1.6-mistral-7b-hf \
  --port 8000 \
  --dtype auto
```

For older vLLM installs that do not provide `vllm serve`, use the OpenAI-compatible API server entry point:

```bash
source .venv-vllm/bin/activate

python3 -m vllm.entrypoints.openai.api_server \
  --model llava-hf/llava-v1.6-mistral-7b-hf \
  --port 8000 \
  --dtype auto
```

Verify the endpoint from another terminal:

```bash
curl -s http://localhost:8000/v1/models
```

If vLLM fails with an `openai.types.responses` import error, the vLLM environment has an incompatible `openai` package version. Recreate `.venv-vllm` or reinstall vLLM and OpenAI SDK together in that separate environment.

## Physical PuppyPi Validation

Start the PuppyPi ROS2 controller on the PuppyPi side:

```bash
cd /path/to/OpenPAVE

export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

./scripts/start_puppypi_controller.sh
```

The script uses these defaults:

```text
PUPPYPI_CONTAINER=puppypi_ros2
PUPPYPI_USER=ubuntu
PUPPYPI_WORKDIR=/home/ubuntu
ROS_DOMAIN_ID=0
RMW_IMPLEMENTATION=rmw_fastrtps_cpp
PUPPYPI_RESTART_ROS_DAEMON=1
PUPPYPI_LAUNCH_CMD=ros2 launch puppy_control puppy_control.launch.py
```

Start the vLLM backend, then run the OpenPAVE runtime on the DGX/control side:

```bash
cd /path/to/OpenPAVE
source .venv/bin/activate

OPENPAVE_CONFIG=configs/puppypi.env ./scripts/run_stage3_demo.sh
```

`configs/puppypi.env` contains the PuppyPi adapter, robot IP address, VLM model/backend, ROS domain, RMW implementation, and ROS CLI Docker image settings.

For physical robot safety, VLM-driven `TROT` forwarding requires repeated confirmation by default:

```text
INTENT_FORWARDING_ENABLED=1
TROT_CONFIRMATIONS=2
TROT_CONFIRMATION_WINDOW_MS=1500
```

`STOP` is still forwarded immediately. To disable VLM-to-robot forwarding during debugging:

```bash
OPENPAVE_CONFIG=configs/puppypi.env INTENT_FORWARDING_ENABLED=0 ./scripts/run_stage3_demo.sh
```

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

Use `/pave` for the lightweight OpenPAVE console when the checked-out `ui` submodule includes the Stage 2 console patch. If `/` works but `/pave` returns 404, the current `live-vlm-webui` checkout does not include the OpenPAVE console route.

The `/pave` console can also run a text-only prompt inference through:

```text
POST /api/pave/infer
```

Use this to validate the vLLM/OpenAI-compatible backend and UI prompt/result path before connecting a live camera stream or physical robot.

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

## Debug Unexpected Robot Motion

If the robot executes `TROT` when you did not intentionally send a new command, first determine whether it came from a stale intent file or from a fresh VLM/UI output.

Inspect the current intent:

```bash
cat /tmp/vla_intent.json
```

Inspect command lifecycle feedback:

```bash
cat /tmp/vla_command_result.json
cat /tmp/vla_robot_state.json
```

Watch the managed runtime logs:

```bash
tail -f .openpave/logs/intent_ingress.log
tail -f .openpave/logs/control_daemon.log
tail -f .openpave/logs/openpave-ui.log
```

The control daemon ignores an intent file that already exists before daemon startup. It only processes that file after it changes. This prevents a stale `/tmp/vla_intent.json` containing `TROT` from being replayed when the runtime restarts.

To intentionally replay an existing intent file during development:

```bash
PROCESS_EXISTING_INTENT_ON_START=1 ./scripts/run_stage3_demo.sh
```

If `/tmp/vla_intent.json` changes to `TROT` while the UI is running, the likely source is a VLM output being forwarded by `live-vlm-webui`. In that case, check `openpave-ui.log` and the UI prompt/result panel. For physical robot validation, keep the prompt strict and prefer `STOP` as the fallback output.

## Logs

Inspect logs with:

```bash
tail -f .openpave/logs/intent_ingress.log
tail -f .openpave/logs/control_daemon.log
tail -f .openpave/logs/openpave-ui.log
```

## Health Checks

The launcher checks:

```text
http://127.0.0.1:7071/healthz
http://127.0.0.1:8090/
```

It also probes the configured OpenAI-compatible VLM endpoint:

```text
http://localhost:8000/v1/models
```

If the VLM endpoint is not reachable, the launcher still starts the UI. Layout, WebSocket connectivity, runtime feedback, and manual intent testing can still be validated without model inference.

## Stage 3C Benchmark Validation

The benchmark harness runs from a second terminal while the Stage 3 runtime is already running.

### Mock Control-Path Benchmark

Terminal 1:

```bash
cd /path/to/OpenPAVE
source .venv/bin/activate

OPENPAVE_CONFIG=configs/mock.env ./scripts/run_stage3_demo.sh
```

Confirm the launcher prints:

```text
ROBOT_ADAPTER=mock
```

Terminal 2:

```bash
cd /path/to/OpenPAVE
source .venv/bin/activate

python3 scripts/run_benchmark.py scenarios/mock-intent-stop-trot.json
```

Expected summary:

```text
summary=total=2 passed=2 failed=0 avg_latency_ms=...
```

Benchmark results are written to:

```text
benchmark-results/
```

Summarize one or more benchmark result files:

```bash
python3 scripts/summarize_benchmarks.py benchmark-results/*.jsonl
```

Compare model, endpoint, or inference-node dimensions by changing the grouping.
This compares benchmark result files by their scenario metadata; Stage 3C.1 does not replay camera frames or measure VLM output quality yet.

```bash
python3 scripts/summarize_benchmarks.py benchmark-results/*.jsonl \
  --group-by scenario.id \
  --group-by inference_node.default_model

python3 scripts/summarize_benchmarks.py benchmark-results/*.jsonl \
  --group-by scenario.id \
  --group-by runtime_env.UI_MODEL

python3 scripts/summarize_benchmarks.py benchmark-results/*.jsonl \
  --group-by scenario.id \
  --group-by robot_sensor_endpoint.validated_target

python3 scripts/summarize_benchmarks.py benchmark-results/*.jsonl \
  --group-by scenario.id \
  --group-by inference_node.validated_target
```

Use threshold gates when validating a release candidate:

```bash
python3 scripts/summarize_benchmarks.py benchmark-results/*.jsonl \
  --min-pass-rate 1.0 \
  --max-avg-latency-ms 1500
```

The command returns a non-zero exit code if any grouped result violates the gate.

### Physical PuppyPi Benchmark

Only run this after the PuppyPi ROS2 controller is running and the robot is in a safe test area.

Terminal 1:

```bash
cd /path/to/OpenPAVE
source .venv/bin/activate

OPENPAVE_CONFIG=configs/puppypi.env ./scripts/run_stage3_demo.sh
```

Terminal 2:

```bash
cd /path/to/OpenPAVE
source .venv/bin/activate

python3 scripts/run_benchmark.py scenarios/puppypi-gesture-stop-trot.json --allow-physical
```

Physical-motion scenarios are blocked unless `--allow-physical` is passed.

### STOP-Only Physical Check

Use this when you want to validate the physical control path without sending `TROT`:

```bash
python3 scripts/run_benchmark.py scenarios/puppypi-gesture-stop-trot.json \
  --allow-physical \
  --intent STOP
```

Inspect the latest command result:

```bash
cat /tmp/vla_command_result.json
```

## Stop

Press `Ctrl+C` in the launcher terminal.

The launcher shuts down the managed child processes:

- Intent Ingress
- Control Daemon
- OpenPAVE UI server

When `ROBOT_ADAPTER=puppypi`, the launcher sends a final `STOP` intent before stopping managed processes. This helps return PuppyPi to a safe state if the demo is interrupted while the robot is moving.

Disable this only for debugging:

```bash
STOP_ROBOT_ON_EXIT=0 OPENPAVE_CONFIG=configs/puppypi.env ./scripts/run_stage3_demo.sh
```

It does not stop external dependencies such as vLLM or the robot-side ROS2 controller.
