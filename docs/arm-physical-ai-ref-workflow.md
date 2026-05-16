# Arm Physical AI Reference Workflow

## Purpose

This document describes the PAVE reference workflow from an Arm/Linux Physical AI ecosystem perspective. It explains how PAVE is intended to connect local robot endpoints, edge inference devices, ROS2 communication, VLM/VLA logic, and observability tooling into a repeatable workflow for demos, research, PoCs, and architecture validation.

This is a project-level reference workflow note. It is not an official Arm position, product statement, endorsement, or architecture specification.

## Positioning

PAVE focuses on a local, developer-controlled Physical AI workflow:

```text
robot camera / RTSP stream
-> local edge VLM/VLA inference
-> normalized intent
-> robot adapter
-> ROS2 command path
-> physical robot action
-> state, command result, and benchmark feedback
```

The current implementation uses PuppyPi as the first physical robot target. PuppyPi is a validation target for the workflow, not the final hardware boundary. Future hardware targets and compute devices are expected to be added as the runtime contracts become clearer.

## Why This Matters for the Arm Ecosystem

Arm-based Linux systems are widely used as robot endpoints, embedded controllers, edge gateways, and developer-accessible robotics platforms. PAVE aims to show how these systems can participate in an end-to-end Physical AI workflow using open-source software and local inference infrastructure.

The value of the project is not only that a robot can be controlled by a VLM/VLA output. The value is that the full workflow can become understandable, replaceable, and repeatable:

- robot endpoint integration
- camera or RTSP streaming
- local edge inference
- intent normalization
- ROS2 command execution
- robot state and command feedback
- lightweight observability UI
- benchmark and scenario replay

## Workflow Layers

### 1. Robot Endpoint Layer

The robot endpoint provides sensors, motion control, and ROS2 services or topics. In the current project, PuppyPi is the first endpoint.

Expected evolution:

- keep PuppyPi as the first working adapter
- add additional robot targets
- avoid embedding PuppyPi-specific assumptions in the control daemon core
- expose common capabilities through robot adapters

### 2. Edge Inference Layer

The edge inference layer runs local VLM/VLA inference through an OpenAI-compatible backend such as vLLM or another local service.

Expected evolution:

- compare different inference devices
- compare different model backends
- track latency, throughput, GPU usage, and command-path timing
- keep cloud inference optional rather than required

### 3. Intent Contract Layer

The intent contract translates model output into a stable runtime command format.

Expected evolution:

- define an intent schema
- validate and normalize incoming intent payloads
- include metadata such as source, timestamp, confidence, request ID, and schema version
- keep unsafe or unknown outputs mapped to safe behavior, such as `STOP`

### 4. Robot Adapter Layer

The robot adapter maps normalized intents to robot-specific control calls.

Expected evolution:

- define a common adapter interface
- implement `PuppyPiAdapter`
- add mock or dry-run adapters for local development
- add future hardware adapters without rewriting the control daemon core

### 5. Observability Layer

The observability layer helps developers see what the system is doing.

Expected evolution:

- replace the general VLM web UI frontend with a lightweight PAVE console
- reuse the existing video backend at first
- show live stream, CPU/GPU/memory usage, prompt, model result, parsed intent, robot state, and command result

### 6. Experimentation Layer

The experimentation layer makes demos and research runs repeatable.

Expected evolution:

- add prompt presets
- add demo scenarios
- add benchmark harness
- compare model, prompt, hardware, and scenario combinations
- store structured results for later analysis

## Relationship to Roadmap

The current roadmap follows this dependency order:

```text
Stage 1: make the runtime portable
-> Stage 2: make the workflow visible
-> Stage 3: make experiments repeatable
```

Stage 1 focuses on intent schema, robot adapters, and robot state or command feedback. This is the foundation that allows PAVE to move beyond a PuppyPi-specific demo.

Stage 2 creates a lightweight PAVE UI while reusing the existing video backend. This keeps the UI work focused on workflow visibility instead of rebuilding the streaming stack too early.

Stage 3 adds prompt presets, demo scenarios, and benchmark harnesses so the workflow can be reused across models, hardware, and Physical AI tasks.

## Non-Goals

PAVE is not intended to be:

- an official Arm reference design
- a commercial robot control stack
- a complete autonomous robotics platform
- a replacement for vendor-specific robotics SDKs

PAVE is intended to be a practical, open, and adaptable workflow reference for developers exploring local Physical AI systems.
