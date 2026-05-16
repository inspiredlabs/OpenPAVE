# PAVE Roadmap TODO

## Goal

Move PAVE from a working demo pipeline toward a repeatable Physical AI experimentation framework for:

- edge-side VLM/VLA inference
- ROS 2 robot control
- PuppyPi as the first physical robot target
- future reusable robot adapters, lightweight UI, benchmarks, prompts, and scenarios

## Background Assumptions

- PAVE is a local Physical AI reference workflow built on open-source software commonly used across the Arm/Linux robotics ecosystem.
- The Arm-related framing is a reference workflow perspective, not an official Arm position or product statement.
- PuppyPi is the first robot target, but additional hardware targets and compute devices are expected.
- Runtime contracts, adapters, UI, and benchmarks should avoid PuppyPi-only assumptions where practical.

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

### Stage 3A: Prompt Presets and Demo Scenarios

- [ ] Add a `prompts/` directory.
- [ ] Add baseline prompt presets for stable intent output.
- [ ] Add prompts for scene understanding, object recognition, navigation suggestion, and stop/trot control.
- [ ] Add a `scenarios/` directory.
- [ ] Define demo scenarios in a structured format such as YAML or JSON.
- [ ] Include expected intents, safety constraints, model prompt, camera/source assumptions, and success criteria.
- [ ] Document how to add a new prompt or scenario.

### Stage 3B: Benchmark Harness

- [ ] Define benchmark dimensions: inference latency, intent correctness, command latency, action success rate, recovery behavior, FPS, and GPU usage.
- [ ] Add a repeatable benchmark runner.
- [ ] Capture model, prompt, scenario, hardware, and runtime configuration for each run.
- [ ] Store benchmark results in a structured format such as JSONL or CSV.
- [ ] Add summary reporting for benchmark runs.
- [ ] Support comparing different VLM/VLA models under the same scenario.
- [ ] Include control-path latency from intent emission to ROS 2 command execution.
- [ ] Connect benchmark runs to prompt presets and demo scenarios.
- [ ] Document how to add a new prompt or scenario.

## Notes

- Stage 1 should be completed before the lightweight UI depends on command feedback or robot state.
- Stage 2 should create the PAVE-specific console while reusing the existing video backend.
- Stage 3 should expand the benchmark and scenario system after the runtime contract and UI feedback loop exist.
- The immediate maturity target is to move from a demo pipeline to a repeatable runtime foundation.
- PuppyPi remains the first robot target, but the architecture should stop assuming PuppyPi everywhere.
