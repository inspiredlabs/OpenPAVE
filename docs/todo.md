# PAVE Roadmap TODO

## Goal

Move PAVE from a working demo pipeline toward a repeatable Physical AI experimentation framework for:

- edge-side VLM/VLA inference
- ROS2 robot/sensor endpoints
- PuppyPi as the first validated physical robot target
- DGX Spark as the first validated Arm-based edge inference node
- future reusable robot adapters, sensor integrations, lightweight UI, benchmarks, prompts, and scenarios

## Background Assumptions

- PAVE is a local Physical AI reference workflow built on open-source software commonly used across the Arm/Linux robotics ecosystem.
- The Arm-related framing is a reference workflow perspective, not an official Arm position or product statement.
- The current DGX Spark + PuppyPi setup is a validated reference implementation, not the product boundary.
- PuppyPi is the first ROS2 robot endpoint, but additional robot/sensor endpoints are expected.
- DGX Spark is the first Arm-based edge inference node, but additional Arm-based inference hardware is expected.
- The first inference backend contract is an OpenAI-compatible VLM API.
- Runtime contracts, adapters, UI, and benchmarks should avoid PuppyPi-only or camera-only assumptions where practical.

## Stage 1: Core Runtime Maturity

### Stage 1A: Intent Schema v0.1

- [x] Define a versioned intent schema document.
- [x] Specify supported intent types, including `STOP`, `TROT`, `MOVE`, and `HOME`.
- [x] Define required and optional fields.
- [x] Add metadata fields such as source, timestamp, confidence, schema version, and request ID.
- [x] Define validation rules for numeric parameters such as velocity, yaw, and duration.
- [x] Add a shared Python intent model or schema helper used by both `intent-ingress` and `control-daemon`.
- [x] Update `intent-ingress` to validate requests against the schema.
- [x] Update the control daemon to consume the normalized schema instead of ad hoc JSON.
- [x] Add examples for valid and invalid intent payloads.

### Stage 1B: Robot Adapter Interface

- [x] Introduce a robot adapter interface for common robot capabilities.
- [x] Define capability methods such as `stop`, `trot`, `home`, and `move`.
- [x] Implement a `PuppyPiAdapter` as the first adapter.
- [x] Move PuppyPi-specific ROS 2 service and topic calls out of the control daemon core.
- [x] Keep the control daemon responsible for intent handling and adapter orchestration.
- [x] Make adapter selection configurable through environment variables or config files.
- [x] Add a simple mock or dry-run adapter for local testing without robot hardware.
- [x] Document how future robot adapters should be added.

### Stage 1C: Robot State and Command Result Feedback

- [x] Define a robot state model.
- [x] Define command lifecycle states such as received, accepted, executing, completed, failed, and rejected.
- [x] Add command result output from the control daemon.
- [x] Record command execution status, return codes, timestamps, and failure reasons.
- [x] Add a state/result file, HTTP endpoint, ROS topic, or other lightweight feedback channel.
- [x] Add basic daemon state updates for `idle`, `received`, `accepted`, `executing`, and `error`.
- [ ] Add periodic robot heartbeat or liveness tracking when a reliable robot-side signal is available.
- [x] Surface command result feedback in logs first.
- [x] Prepare the WebUI or observability layer to display command feedback later.

## Stage 2: Lightweight PAVE UI

### Stage 2A: Lightweight UI Prototype

- [x] Design a lightweight PAVE console focused on Physical AI experiments.
- [x] Keep the first version small and operational, not a general-purpose VLM UI.
- [x] Reuse the existing video backend instead of rewriting the live-stream pipeline.
- [x] Reuse existing VLM and GPU monitoring backend capabilities first where possible.
- [x] Avoid rewriting the live-stream, VLM, and GPU monitoring backend paths until the UI shape is validated.
- [x] Decide whether the first implementation is a simplified static frontend or a small frontend app.
- [x] Document which parts of the current `live-vlm-webui` are still reused.
- [x] Show the robot camera or RTSP stream as the primary view.
- [x] Display stream connection state, such as connected, disconnected, or reconnecting.
- [x] Keep the stream layout stable across desktop and smaller screens.
- [x] Preserve room for prompt/result panels without hiding the live view.
- [x] Show CPU usage.
- [x] Show memory usage.
- [x] Show GPU utilization when available.
- [x] Show GPU memory usage when available.
- [x] Prepare space for FPS and inference latency metrics.
- [x] Keep metrics compact enough for repeated experiment monitoring.
- [x] Provide prompt input.
- [x] Show the active model and backend endpoint.
- [x] Show raw VLM output.
- [x] Show parsed intent.
- [x] Include timestamps and basic latency information where available.

### Stage 2B: Runtime Feedback Integration

- [x] Connect the UI to the Stage 1 intent schema.
- [x] Connect the UI to robot state and command result feedback.
- [x] Show robot state from Stage 1 feedback.
- [x] Show command result from Stage 1 feedback.
- [x] Show parsed intent and command lifecycle side by side with raw VLM output.
- [x] Keep UI actions routed through stable backend APIs rather than direct robot-specific logic.
- [x] Keep PuppyPi visible as the first robot target, but avoid baking PuppyPi assumptions into the UI.

## Stage 3: Experimentation Framework

### Stage 3A: Developer Runtime Launcher

- [ ] Add a one-command Stage 2/Stage 3 demo launcher.
- [ ] Start Intent Ingress, Control Daemon, and OpenPAVE UI server from one command.
- [ ] Keep vLLM backend and robot-side ROS2 controller as external dependencies first.
- [ ] Support configuration for robot adapter, robot IP address, model, API base, and runtime file paths.
- [ ] Write logs to `.openpave/logs/`.
- [ ] Add health checks for Intent Ingress and OpenPAVE UI.
- [ ] Print the full UI URLs, including `/` and `/pave`.
- [ ] Print runtime debug files such as `/tmp/vla_intent.json`, `/tmp/vla_command_result.json`, and `/tmp/vla_robot_state.json`.
- [ ] Provide a clean shutdown path for child processes.
- [ ] Document which services are managed by the launcher and which remain external.

### Stage 3B: Prompt Presets and Demo Scenarios

- [ ] Add a `prompts/` directory.
- [ ] Add baseline prompt presets for stable intent output.
- [ ] Add prompts for scene understanding, object recognition, navigation suggestion, and stop/trot control.
- [ ] Add a `scenarios/` directory.
- [ ] Define demo scenarios in a structured format such as YAML or JSON.
- [ ] Include expected intents, safety constraints, model prompt, robot/sensor endpoint assumptions, and success criteria.
- [ ] Include sensor input assumptions such as camera, depth, lidar, audio, robot state, or other ROS2 streams.
- [ ] Include inference node assumptions such as OpenAI-compatible API endpoint, model, and serving backend.
- [ ] Include adapter assumptions such as robot target, supported capabilities, and command interface.
- [ ] Document how to add a new prompt or scenario.

### Stage 3C: Benchmark Harness

- [ ] Define benchmark dimensions: inference latency, intent correctness, command latency, action success rate, recovery behavior, FPS, and GPU usage.
- [ ] Add a repeatable benchmark runner.
- [ ] Capture model, prompt, scenario, robot/sensor endpoint, adapter, inference node, hardware, and runtime configuration for each run.
- [ ] Store benchmark results in a structured format such as JSONL or CSV.
- [ ] Add summary reporting for benchmark runs.
- [ ] Support comparing different VLM/VLA models under the same scenario.
- [ ] Support comparing different robot/sensor endpoints under the same prompt and scenario contract.
- [ ] Support comparing different Arm-based inference nodes under the same prompt and scenario contract.
- [ ] Include control-path latency from intent emission to ROS 2 command execution.
- [ ] Connect benchmark runs to prompt presets and demo scenarios.
- [ ] Document how to add a new prompt or scenario.

## Stage 4: UI Independence and live-vlm-webui Decoupling

### Stage 4A: Licensing and Attribution Boundary

- [ ] Keep Apache-2.0 license and attribution notices for code derived from `NVIDIA-AI-IOT/live-vlm-webui`.
- [ ] Clearly document which UI/runtime components are derived from `live-vlm-webui`.
- [ ] Avoid implying NVIDIA endorsement or official product alignment.
- [ ] Maintain a third-party notice path for the current UI component.
- [ ] Track OpenPAVE-specific modifications separately from upstream-derived behavior.

### Stage 4B: OpenPAVE Native Console

- [ ] Move the `/pave` console into an OpenPAVE-owned frontend module.
- [ ] Define stable OpenPAVE backend APIs for runtime state, prompt control, stream configuration, and experiment metadata.
- [ ] Keep `live-vlm-webui` available as an optional full VLM debugging UI during the transition.
- [ ] Replace direct dependency on `live-vlm-webui` static assets for the default OpenPAVE workflow.
- [ ] Keep the lightweight console focused on Physical AI experiment operation rather than general VLM UI features.

### Stage 4C: Backend Decoupling

- [ ] Extract or wrap the required video, WebSocket, VLM, and metrics backend capabilities behind OpenPAVE-owned interfaces.
- [ ] Make stream backend selection configurable, so RTSP/WebRTC implementation details are not tied to one upstream UI project.
- [ ] Preserve compatibility with the existing `live-vlm-webui` backend until the OpenPAVE-native backend is validated.
- [ ] Add migration documentation for moving from `live-vlm-webui` mode to OpenPAVE-native console mode.
- [ ] Validate the decoupled UI against PuppyPi and at least one additional hardware target.

## Notes

- Stage 1 should be completed before the lightweight UI depends on command feedback or robot state.
- Stage 2 should create the PAVE-specific console while reusing the existing video backend.
- Stage 3 should first make the runtime repeatable to launch, then make prompts/scenarios repeatable to describe, and then make benchmarks repeatable to measure.
- Stage 3 prompts, scenarios, and benchmarks should use explicit robot/sensor endpoint and inference node assumptions.
- Stage 4 should reduce long-term coupling to `live-vlm-webui` while keeping Apache-2.0 attribution and compatibility during migration.
- The immediate maturity target is to move from a demo pipeline to a repeatable runtime foundation.
- PuppyPi and DGX Spark remain the first validation targets, but the architecture should stop assuming them everywhere.
