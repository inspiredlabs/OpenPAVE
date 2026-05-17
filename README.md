# OpenPAVE: Open Physical-AI VLA Experimentation

## A developer-friendly open edge workflow for local VLA experimentation on Arm computing platforms

OpenPAVE is a local-first, cloud-free reference workflow for validating Physical AI experiments across ROS2 robot/sensor endpoints, Arm-based edge inference nodes, VLM/VLA reasoning, adapter-based runtime control, and observability tooling.

This project is intended for developers, hobbyists, and small teams who want to experiment with upper-layer VLA / VLM applications without relying on expensive commercial robotics platforms.

## Project positioning

OpenPAVE is positioned as a local, developer-controlled Physical AI reference workflow built on open-source software commonly used across the Arm/Linux robotics ecosystem. It is intended to show how ROS2 robot/sensor endpoints, Arm-based edge inference nodes, prompt-driven VLM/VLA logic, adapter-based runtime control, and observability tooling can be composed into an end-to-end workflow for demos, research, PoCs, and architecture validation.

This is a reference workflow perspective, not an official Arm position or product statement. The current DGX Spark + PuppyPi setup is a validated reference implementation, not the product boundary of OpenPAVE. PuppyPi is the first validated ROS2 robot endpoint, and DGX Spark is the first validated Arm-based edge inference node. The architecture is expected to expand toward additional robot/sensor endpoints and edge inference hardware through clearer runtime contracts and robot adapters.


## Project highlights

- Provides a lower-cost and more accessible path for developers to prototype Physical AI applications
- Uses edge-side compute for local AI inference, reducing dependence on cloud connectivity
- Leverages an open-source Live WebUI to accelerate PoC integration and visualisation
- Uses standard ROS2 communication between robot/sensor endpoints and the edge inference node for interoperability and extensibility
- Uses prompt-driven task customisation to explore how general-purpose VLMs can adapt to robotics workflows
- Treats the current PuppyPi setup as the first validated adapter target, not the final hardware boundary
- Aims to support future experiments across multiple local inference nodes, robot endpoints, and sensor inputs
- Splits the system into replaceable roles:
   - ROS2 Robot / Sensor Endpoint: Provides sensor streams and accepts ROS2 command interfaces through an adapter.
   - Edge Inference / Observability Node: Runs local VLM/VLA inference, displays live streams, and exposes developer-facing runtime feedback.
   - OpenPAVE Runtime Control Layer: Normalizes intent, dispatches through robot adapters, and records command/state feedback.

## What this project is

OpenPAVE is a lightweight experimentation project, not a full commercial robotics software stack.

Its value is in showing that developers can use affordable hardware and open-source components to build and test Physical AI workflows in a practical and visible way. Rather than competing with mature one-stop robotics platforms, OpenPAVE focuses on lowering the barrier to experimentation and making upper-layer VLA / VLM PoCs easier to prototype, understand, and extend.

## What this project is not

OpenPAVE is not intended to be:

- a complete autonomous robotics platform
- a production-ready robot control stack
- a replacement for commercial end-to-end robotics software workflows

Instead, it is a developer-oriented PoC framework for experimentation, integration, and learning.

## Hardware scope

The current DGX Spark + PuppyPi setup is a validation target, not a product scope definition.

OpenPAVE is designed around replaceable roles:

1. **ROS2 Robot / Sensor Endpoint**
   - provides camera, depth, audio, lidar, robot state, or other ROS2 sensor streams
   - accepts ROS2 service/topic commands through an OpenPAVE adapter
   - current validation target: PuppyPi camera and `puppy_control`

2. **Edge Inference / Observability Node**
   - runs local inference through an OpenAI-compatible VLM API
   - displays live streams, prompts, inference output, resource metrics, and runtime feedback
   - current validation target: DGX Spark running vLLM and LLaVA

3. **OpenPAVE Runtime Control Layer**
   - receives high-level intent from the UI, VLM output, scripts, or tests
   - validates intent schema and dispatches commands through adapter contracts
   - writes command result and robot/sensor state feedback for observability and future benchmarks

Robot adapters are intended to become the main contribution boundary for supporting new ROS2 robot/sensor endpoints and hardware platforms. The first inference backend contract is an OpenAI-compatible VLM API, which keeps the workflow simple while allowing different local serving stacks and edge inference runtimes.

## Current validated implementation

The current validated implementation uses:

- PuppyPi as the first ROS2 robot endpoint
- DGX Spark as the first Arm-based edge inference node
- ROS2 for communication and streaming integration
- a modified Live WebUI as the interactive frontend
- edge-side VLA / VLM inference running on a separate compute platform
- prompt-driven tasks such as scene understanding, object recognition, and navigation suggestion

The robot and inference platforms used in the demo are validation vehicles. The core idea of the project is the software workflow itself, which can be adapted to other ROS2 robot/sensor endpoints and Arm-based edge inference nodes.

## System overview

OpenPAVE currently consists of three replaceable roles:

1. **ROS2 Robot / Sensor Endpoint**
   - provides sensor streams such as camera input
   - runs robot-side ROS2 control interfaces
   - current implementation: PuppyPi

2. **Edge Inference / Observability Node**
   - runs the VLM/VLA inference backend through an OpenAI-compatible API
   - provides video display, prompt interaction, result visualisation, and resource metrics
   - current implementation: DGX Spark + vLLM + modified `live-vlm-webui`

3. **OpenPAVE Runtime Control Layer**
   - receives and validates high-level intent
   - dispatches commands through robot adapters
   - records command result and robot/sensor state feedback

### High-level flow

ROS2 robot/sensor stream  
→ edge-side VLA / VLM inference  
→ OpenPAVE UI / observability  
→ intent schema  
→ runtime control daemon  
→ robot adapter  
→ ROS2 command interface  
→ command result and robot/sensor state feedback

## Development environment

For current OpenPAVE development, use one shared repo-level Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r intent-ingress/requirements.txt
python3 -m pip install -r ui/requirements.txt
python3 -m pip install -e ui
```

Run the current Python tests from the repo root:

```bash
python3 -B -m unittest discover
```

The repo-level `.venv` is intended for local development across `intent-ingress`, `control-daemon`, shared runtime helpers, tests, and the Stage 2 lightweight console. Module-level requirements files should remain available for service-specific runtime or future deployment packaging. As OpenPAVE grows, individual services may later get separate dependency sets or containers, but the current Stage 1 and Stage 2 development workflow assumes a single shared virtual environment.

For the UI dependencies, prefer Python 3.10 to 3.12. Very new Python versions may not have prebuilt wheels for packages such as `av`.

After updating the `ui` submodule to a new commit or tag, rerun the UI install commands so the repo-level `.venv` uses the current local `live-vlm-webui` code:

```bash
python3 -m pip install -r ui/requirements.txt
python3 -m pip install -e ui
```

## Stage 1 runtime

The Stage 1 runtime still follows the Ver1 multi-terminal flow:

```text
Terminal 1: Intent Ingress
Terminal 2: Control Daemon
Terminal 3: manual intent test or WebUI output
```

Stage 1B adds a robot adapter boundary. The default adapter is `puppypi`, which preserves the existing PuppyPi ROS2 command behavior. For local development without Docker, ROS2, or robot hardware, use the `mock` adapter.

### Terminal 1: Intent Ingress

```bash
cd /path/to/OpenPAVE
source .venv/bin/activate
python3 intent-ingress/intent_ingress.py
```

Health check:

```bash
curl -s http://127.0.0.1:7071/healthz
```

### Terminal 2A: Control Daemon with PuppyPi

Use this mode for the physical PuppyPi robot path:

```bash
cd /path/to/OpenPAVE
source .venv/bin/activate

export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROBOT_ADAPTER=puppypi
export ROS_SVC_IMAGE=ros:humble
export ROS_PUB_IMAGE=puppy-ros2-cli:humble
export COMMAND_RESULT_PATH=/tmp/vla_command_result.json
export ROBOT_STATE_PATH=/tmp/vla_robot_state.json

python3 control-daemon/pave_control_daemon_mvp.py
```

`ROBOT_ADAPTER=puppypi` is the default, but setting it explicitly makes the runtime path clear.

### Terminal 2B: Control Daemon Dry Run

Use this mode to validate the intent pipeline without robot hardware:

```bash
cd /path/to/OpenPAVE
source .venv/bin/activate

export ROBOT_ADAPTER=mock
python3 control-daemon/pave_control_daemon_mvp.py
```

The mock adapter prints actions such as `MOCK ACTION=STOP` or `MOCK ACTION=MOVE` instead of calling Docker or ROS2.

### Terminal 3: Manual Intent Test

Send a simple text intent:

```bash
curl -s -X POST http://127.0.0.1:7071/intent \
  -H 'Content-Type: application/json' \
  -d '{"text":"TROT"}'
```

Stop:

```bash
curl -s -X POST http://127.0.0.1:7071/intent \
  -H 'Content-Type: application/json' \
  -d '{"text":"STOP"}'
```

Turn right through the text alias:

```bash
curl -s -X POST http://127.0.0.1:7071/intent \
  -H 'Content-Type: application/json' \
  -d '{"text":"RIGHT"}'
```

Send the normalized schema shape directly:

```bash
curl -s -X POST http://127.0.0.1:7071/intent \
  -H 'Content-Type: application/json' \
  -d '{"intent":"MOVE","params":{"vx":0.0,"yaw":0.6,"duration_ms":600},"source":"manual"}'
```

The intent written to `/tmp/vla_intent.json` is normalized to intent schema v0.1 before the control daemon dispatches it to the selected robot adapter.

The control daemon also writes lightweight feedback files:

```text
/tmp/vla_command_result.json
/tmp/vla_robot_state.json
```

These files expose the latest command lifecycle state, adapter result, return codes, robot status, and last command summary for Stage 2 UI and future benchmark tooling.

## Stage 2 lightweight console

Stage 2 adds a lightweight OpenPAVE console while reusing the existing video, VLM, WebSocket, and GPU monitoring backend paths.

The original Live VLM WebUI remains available at:

```text
/
```

The lightweight OpenPAVE console is available at:

```text
/pave
```

The console focuses on:

- live stream
- CPU, GPU, and memory usage
- prompt input
- active model and backend endpoint
- raw VLM result
- parsed intent
- Stage 1 command result
- Stage 1 robot state

The console reads Stage 1 feedback through:

```text
GET /api/pave/runtime
```

## Why edge matters here

Many PoCs can be demonstrated using cloud inference. OpenPAVE specifically focuses on edge-side execution because it is closer to how real-world robotics systems are often evaluated and deployed.

Using edge-side compute helps:

- reduce dependence on cloud connectivity
- improve responsiveness and deployment flexibility
- provide a more realistic setup for Physical AI experimentation
- make it easier to study trade-offs across different edge computing platforms

## Why RPi is mentioned

RPi is used here because it is highly recognisable in the developer community and represents an accessible entry point for robotics and edge experimentation.

However, OpenPAVE is not limited to Raspberry Pi alone. The same workflow can be adapted to other Arm computing platforms and edge devices, depending on the hardware and deployment goals.

## Repository structure

The repository root keeps only the main `README.md` as the entry point. Supporting project documents are kept under `docs/`.

```text
.
├─ README.md
├─ docs/
│  ├─ architecture.md
│  ├─ arm-physical-ai-ref-workflow.md
│  ├─ intent_schema.md
│  ├─ live-vlm-webui_hook.md
│  ├─ pave_console.md
│  ├─ pave_ver1_readme.md
│  ├─ robot_adapters.md
│  ├─ robot_feedback.md
│  ├─ runbook.md
│  ├─ stage1.step.md
│  ├─ stage2.step.md
│  ├─ third_party_notices.md
│  └─ todo.md
├─ control-daemon/
├─ intent-ingress/
├─ pave_runtime/
├─ third_party/
└─ ui/
```

## Documentation guide

- `docs/architecture.md`: Current high-level architecture notes and role split across robot/sensor endpoints, edge inference/observability, and runtime control.
- `docs/arm-physical-ai-ref-workflow.md`: Reference workflow framing for local edge Physical AI on Arm/Linux ecosystems. This is not an official Arm position.
- `docs/intent_schema.md`: Stage 1 intent schema v0.1, including supported intent types, metadata, validation rules, and examples.
- `docs/live-vlm-webui_hook.md`: Notes for the server-side hook that forwards selected VLM outputs such as `STOP` and `TROT` to Intent Ingress.
- `docs/pave_console.md`: Stage 2 lightweight OpenPAVE console design and backend reuse notes.
- `docs/pave_ver1_readme.md`: Preserved Ver1 MVP README for the original demo/runtime flow.
- `docs/robot_adapters.md`: Robot Adapter interface contract, current PuppyPi/mock adapters, and guidance for adding future hardware targets.
- `docs/robot_feedback.md`: Robot state and command result feedback model used by Stage 1 and surfaced in Stage 2.
- `docs/runbook.md`: Stable MVP runbook for operating the PuppyPi-side and control-side flow.
- `docs/stage1.step.md`: End-to-end Stage 1 validation procedure for DGX/control machine plus PuppyPi.
- `docs/stage2.step.md`: Stage 2 lightweight console installation and validation procedure.
- `docs/third_party_notices.md`: Third-party component notices and attribution details.
- `docs/todo.md`: Roadmap and staged execution checklist for current and upcoming OpenPAVE work.

## Typical use cases

OpenPAVE is currently intended to support exploration of use cases such as:

* real-time scene and sensor understanding
* object recognition in live robot/sensor streams
* navigation suggestion based on visual context
* prompt-driven task switching for robotics scenarios
* edge-side benchmarking of VLA / VLM workflows

## Future directions

* Benchmark different VLA / VLM models under the same live robotics scenario across edge computing platforms
* Investigate how real-time communication can be improved for lower latency, higher reliability, and better fault tolerance
* Extend the workflow into additional Physical AI PoCs to validate reuse across different application scenarios
* Add additional ROS2 robot/sensor endpoints and Arm-based edge inference nodes to validate the workflow beyond the initial PuppyPi and DGX Spark setup
* Formalise robot adapters, intent schemas, and command feedback so the workflow can be reused across different local Physical AI experiments

## Target users

OpenPAVE is intended for:

* developers exploring Physical AI workflows
* robotics and edge AI hobbyists
* software ecosystem teams
* small teams looking for a lower-cost PoC path
* anyone interested in combining open-source software, edge inference, and robotics experimentation

## Open-source components and attribution

This project builds on open-source components and adapts them for a broader Physical AI experimentation workflow.

In particular, the web UI component is based on NVIDIA-AI-IOT’s live-vlm-webui, which is licensed under the Apache License 2.0. Any modified components should retain the original licence and attribution requirements where applicable.

***Suggested attribution text:
This project includes or references components derived from NVIDIA-AI-IOT/live-vlm-webui, licensed under the Apache License 2.0.***

## Positioning summary

OpenPAVE is best understood as:

* a lightweight experimentation project
* a developer-friendly edge VLA workflow
* a reproducible PoC path for Physical AI exploration
* a practical way to study how open-source software and affordable edge hardware can support real-world AI robotics use cases
* a non-official Arm/Linux ecosystem reference workflow for local Physical AI validation across ROS2 robot/sensor endpoints and Arm-based edge inference nodes

## Status


## Demo:
Ver 0.9:
Real-time VLA on DGX Spark: RPi Quadruped with LLaVA-7B
https://youtu.be/kRiXri0te0g?si=iOhW0d2SSSP6zT4V

Ver 1.0:


This project is currently an active PoC and experimentation framework. The scope, structure, and supported workflows may continue to evolve as new hardware targets, communication methods, and Physical AI scenarios are explored.
