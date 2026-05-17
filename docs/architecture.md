# OpenPAVE Reference Architecture

## Purpose

- Headless control core consuming intents and issuing ROS 2 commands.
OpenPAVE is a local-first Physical AI reference workflow for validating how ROS2 robot/sensor endpoints, Arm-based edge inference nodes, VLM/VLA reasoning, and runtime control feedback can be composed into a repeatable experimentation platform.

The current DGX Spark + PuppyPi setup is a validated reference implementation, not the product boundary of OpenPAVE.

## Replaceable Roles

### 1. ROS2 Robot / Sensor Endpoint

This role provides physical-world observations and accepts control commands.

Responsibilities:

- expose camera, depth, audio, lidar, robot state, or other ROS2 sensor streams
- run robot-side ROS2 controllers, services, topics, or bridges
- receive commands from an OpenPAVE robot adapter
- execute physical or simulated robot actions

Current validation:

- PuppyPi
- PuppyPi camera stream
- `puppy_control` ROS2 services and topics

Future targets:

- any ROS2 robot endpoint with an OpenPAVE adapter
- mobile robots
- robot arms
- quadrupeds
- sensor-only endpoints
- simulation endpoints

### 2. Edge Inference / Observability Node

This role runs local inference and provides developer-facing observability.

Responsibilities:

- run VLM/VLA inference through an OpenAI-compatible VLM API
- display live robot/sensor streams
- manage prompts and observe model outputs
- show CPU, GPU, memory, and runtime feedback
- provide an operator/debug UI for experiments

Current validation:

- DGX Spark
- vLLM
- `llava-hf/llava-v1.6-mistral-7b-hf`
- modified `live-vlm-webui`
- OpenPAVE `/pave` console

Future targets:

- Arm-based edge inference servers
- Jetson or other Arm Linux systems
- local VLM serving stacks compatible with OpenAI-style APIs
- future local GPU/NPU/VPU inference runtimes

### 3. OpenPAVE Runtime Control Layer

This role turns high-level model or user intent into validated robot commands.

Responsibilities:

- accept high-level intent from WebUI/VLM output, manual `curl`, scripts, tests, or future scenario runners
- normalize intent into a versioned schema
- dispatch commands through robot adapters
- isolate robot-specific execution details from the control daemon core
- write command result and robot/sensor state feedback
- provide stable contracts for future benchmark and scenario tooling

Current validation:

- Intent Ingress API
- `/tmp/vla_intent.json` file bus
- Control Daemon
- `PuppyPiAdapter`
- `MockAdapter`
- Dockerized ROS2 CLI command path
- `/tmp/vla_command_result.json`
- `/tmp/vla_robot_state.json`

Future targets:

- additional robot adapters
- richer capability contracts
- ROS2-native bridge implementations
- non-file message bus options
- heartbeat and liveness feedback
- benchmark harness integration

## Data Flow

```text
ROS2 robot/sensor stream
-> OpenPAVE UI / observability
-> OpenAI-compatible VLM API
-> VLM/VLA output
-> Intent Ingress
-> /tmp/vla_intent.json
-> Control Daemon
-> Robot Adapter
-> ROS2 command interface
-> robot action

Control Daemon
-> /tmp/vla_command_result.json
-> /tmp/vla_robot_state.json
-> OpenPAVE UI runtime feedback
```

## Current Implementation Mapping

```text
ROS2 Robot / Sensor Endpoint
  Current: PuppyPi

Edge Inference / Observability Node
  Current: DGX Spark + vLLM + live-vlm-webui + /pave

OpenPAVE Runtime Control Layer
  Current: Intent Ingress + Control Daemon + PuppyPiAdapter
```

## Interfaces

### Intent Ingress API

`POST /intent`:

```json
{ "text": "STOP" }
```

or:

```json
{
  "intent": "MOVE",
  "params": {
    "vx": 0.0,
    "yaw": 0.6,
    "duration_ms": 600
  },
  "source": "manual"
}
```

### Intent File Bus

Default path:

```text
/tmp/vla_intent.json
```

### Runtime Feedback

Default paths:

```text
/tmp/vla_command_result.json
/tmp/vla_robot_state.json
```

### Inference Backend

The first inference backend contract is an OpenAI-compatible VLM API.

Current default:

```text
http://localhost:8000/v1
```

## Design Notes

- PuppyPi and DGX Spark are validation targets, not the final project boundary.
- Robot adapters are the intended contribution surface for new hardware.
- Sensor assumptions should be explicit in future prompts, scenarios, and benchmarks.
- `ROS_DOMAIN_ID` and `RMW_IMPLEMENTATION` must match across ROS2 participants.
- DDS multicast must work on the LAN for the current Dockerized ROS2 CLI path.
