# OpenPAVE: Open Physical-AI VLA Experimentation

## A developer-friendly open edge workflow for local VLA experimentation on Arm computing platforms

OpenPAVE is a local-first, cloud-free reference workflow for connecting robot endpoints, edge inference devices, ROS2 communication, VLM/VLA reasoning, and observability tooling into an end-to-end Physical AI experimentation path.

This project is intended for developers, hobbyists, and small teams who want to experiment with upper-layer VLA / VLM applications without relying on expensive commercial robotics platforms.

## Project positioning

OpenPAVE is positioned as a local, developer-controlled Physical AI reference workflow built on open-source software commonly used across the Arm/Linux robotics ecosystem. It is intended to show how robot endpoints, edge inference devices, ROS2 communication, prompt-driven VLM/VLA logic, and observability tooling can be composed into an end-to-end workflow for demos, research, PoCs, and architecture validation.

This is a reference workflow perspective, not an official Arm position or product statement. PuppyPi is the first physical robot target used to validate the workflow, but the architecture is expected to expand toward additional robot targets and compute devices through clearer runtime contracts and robot adapters.


## Project highlights

- Provides a lower-cost and more accessible path for developers to prototype Physical AI applications
- Uses edge-side compute for local AI inference, reducing dependence on cloud connectivity
- Leverages an open-source Live WebUI to accelerate PoC integration and visualisation
- Uses standard ROS2 communication between the robot and the edge server for interoperability and extensibility
- Uses prompt-driven task customisation to explore how general-purpose VLMs can adapt to robotics workflows
- Treats the current PuppyPi setup as the first adapter target, not the final hardware boundary
- Aims to support future experiments across multiple local compute devices and robot endpoints
- Split the system cleanly into two roles:
   - VLA Observability UI: Enables users to easily switch between models, try different prompts, and monitor both inference outputs and performance metrics—speeding up debugging and validation.
   - VLA Control Daemon: Translates VLA/VLM outputs into executable ROS 2 commands to reliably control the PuppyPi robot dog, and serves as a deployable, reusable control core.

## What this project is

OpenPAVE is a lightweight experimentation project, not a full commercial robotics software stack.

Its value is in showing that developers can use affordable hardware and open-source components to build and test Physical AI workflows in a practical and visible way. Rather than competing with mature one-stop robotics platforms, OpenPAVE focuses on lowering the barrier to experimentation and making upper-layer VLA / VLM PoCs easier to prototype, understand, and extend.

## What this project is not

OpenPAVE is not intended to be:

- a complete autonomous robotics platform
- a production-ready robot control stack
- a replacement for commercial end-to-end robotics software workflows

Instead, it is a developer-oriented PoC framework for experimentation, integration, and learning.

## Current showcase

The current showcase uses:

- an RPi-based robot endpoint
- ROS2 for communication and streaming integration
- a modified Live WebUI as the interactive frontend
- edge-side VLA / VLM inference running on a separate compute platform
- prompt-driven tasks such as scene understanding, object recognition, and navigation suggestion

The robot platform used in the demo is only a showcase vehicle. The core idea of the project is the software workflow itself, which can be adapted to other RPi-based or Arm-based Linux systems.

## System overview

OpenPAVE currently consists of three main parts:

1. **Live WebUI**
   - modified from the open-source `live-vlm-webui` project
   - used for video display, prompt interaction, and model output visualisation

2. **ROS2 communication**
   - connects the robot side and the edge inference side
   - handles real-time message exchange and integration between components

3. **Robot-side integration**
   - captures and streams robot camera data
   - supports robot control and future extensions for action generation or closed-loop interaction

### High-level flow

Robot-side camera stream  
→ ROS2 communication  
→ edge-side VLA / VLM inference  
→ output shown in the web UI  
→ optional task-specific prompt response or navigation suggestion

## Development environment

For current OpenPAVE development, use one shared repo-level Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r intent-ingress/requirements.txt
```

Run the current Python tests from the repo root:

```bash
python3 -B -m unittest discover
```

The repo-level `.venv` is intended for local development across `intent-ingress`, `control-daemon`, shared runtime helpers, and tests. Module-level requirements files should remain available for service-specific runtime or future deployment packaging. As OpenPAVE grows, individual services may later get separate dependency sets or containers, but the current Stage 1 development workflow assumes a single shared virtual environment.

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

Suggested top-level structure:

```text
.
├─ README.md
├─ docs/
│  ├─ architecture.md
│  ├─ demo-flow.md
│  └─ troubleshooting.md
├─ live_webui/
│  └─ notes about the modified web UI component
├─ ros2_bridge/
│  └─ ROS2 communication and integration code
├─ robot_control/
│  └─ robot-side scripts and hardware control logic
├─ prompts/
│  └─ task presets and prompt templates
├─ scripts/
│  └─ setup and launch helpers
└─ configs/
   └─ runtime configuration files
```

## Typical use cases

OpenPAVE is currently intended to support exploration of use cases such as:

* real-time scene understanding
* object recognition in live robot streams
* navigation suggestion based on visual context
* prompt-driven task switching for robotics scenarios
* edge-side benchmarking of VLA / VLM workflows

## Future directions

* Benchmark different VLA / VLM models under the same live robotics scenario across edge computing platforms
* Investigate how real-time communication can be improved for lower latency, higher reliability, and better fault tolerance
* Extend the workflow into additional Physical AI PoCs to validate reuse across different application scenarios
* Add additional hardware targets and compute devices to validate the workflow beyond the initial PuppyPi setup
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
* a non-official Arm/Linux ecosystem reference workflow for local Physical AI demos, research, PoCs, and architecture validation

## Status


## Demo:
Ver 0.9:
Real-time VLA on DGX Spark: RPi Quadruped with LLaVA-7B
https://youtu.be/kRiXri0te0g?si=iOhW0d2SSSP6zT4V

Ver 1.0:


This project is currently an active PoC and experimentation framework. The scope, structure, and supported workflows may continue to evolve as new hardware targets, communication methods, and Physical AI scenarios are explored.
