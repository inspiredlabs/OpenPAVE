# Benchmark Harness

Stage 3C adds a small control-path benchmark harness for OpenPAVE.

The first benchmark runner does not benchmark VLM inference. It validates the runtime control path:

```text
scenario expected intent
-> Intent Ingress
-> intent file bus
-> Control Daemon
-> Robot Adapter
-> command result feedback
```

This keeps the first benchmark repeatable and safe before adding camera frame datasets or VLM replay inputs.

## Start Runtime

Start the Stage 3 runtime first.

For software-only validation:

```bash
OPENPAVE_CONFIG=configs/mock.env ./scripts/run_stage3_demo.sh
```

For physical PuppyPi validation, start the PuppyPi controller and then:

```bash
OPENPAVE_CONFIG=configs/puppypi.env ./scripts/run_stage3_demo.sh
```

## Run Mock Benchmark

From another terminal:

```bash
source .venv/bin/activate

python3 scripts/run_benchmark.py scenarios/mock-intent-stop-trot.json
```

The runner writes JSONL output under:

```text
benchmark-results/
```

## Run Physical PuppyPi Benchmark

Physical scenarios are blocked unless explicitly allowed:

```bash
python3 scripts/run_benchmark.py scenarios/puppypi-gesture-stop-trot.json --allow-physical
```

Use this only when the PuppyPi side controller is running and the robot is in a safe test area.

## Result Format

Each JSONL row includes:

- scenario id, title, version, and runtime profile
- prompt id, title, version, and file reference
- expected intent
- observed intent
- observed command status
- pass/fail
- control-path latency in milliseconds
- Intent Ingress HTTP response
- command result feedback
- robot state feedback
- robot/sensor endpoint assumptions
- inference node assumptions
- adapter assumptions
- safety constraints

## Useful Options

Override expected intents:

```bash
python3 scripts/run_benchmark.py scenarios/mock-intent-stop-trot.json --intent STOP --intent TROT
```

Override runtime files:

```bash
python3 scripts/run_benchmark.py scenarios/mock-intent-stop-trot.json \
  --command-result-path /tmp/vla_command_result.json \
  --robot-state-path /tmp/vla_robot_state.json
```

Use a different Intent Ingress URL:

```bash
python3 scripts/run_benchmark.py scenarios/mock-intent-stop-trot.json \
  --intent-url http://127.0.0.1:7071/intent
```

## Current Scope

Stage 3C.1 measures control-path behavior. VLM inference latency, camera frame replay, FPS, and GPU usage should be added in a later Stage 3C slice once benchmark input datasets are defined.
