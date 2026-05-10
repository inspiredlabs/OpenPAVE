# Lightweight Physical-AI Edge VLA Experimentation

## A developer-friendly workflow for edge VLA experimentation on RPi and Cortex-X/A edge platforms

PAVE is a personal project that explores how low-cost hardware, open-source software, and edge-side AI inference can be combined into a practical workflow for Physical AI experimentation. It connects robot-side streaming, ROS2 communication, an open-source web UI, and edge-side inference into a reproducible path for prototyping VLA-driven applications on real robotic systems.

This project is intended for developers, hobbyists, and small teams who want to experiment with upper-layer VLA / VLM applications without relying on expensive commercial robotics platforms.


## Project highlights

- Provides a lower-cost and more accessible path for developers to prototype Physical AI applications
- Uses edge-side compute for local AI inference, reducing dependence on cloud connectivity
- Leverages an open-source Live WebUI to accelerate PoC integration and visualisation
- Uses standard ROS2 communication between the robot and the edge server for interoperability and extensibility
- Uses prompt-driven task customisation to explore how general-purpose VLMs can adapt to robotics workflows
- Split the system cleanly into two roles:
   - VLA Observability UI: Enables users to easily switch between models, try different prompts, and monitor both inference outputs and performance metrics—speeding up debugging and validation.
   - VLA Control Daemon: Translates VLA/VLM outputs into executable ROS 2 commands to reliably control the PuppyPi robot dog, and serves as a deployable, reusable control core.

## What this project is

PAVE is a lightweight experimentation project, not a full commercial robotics software stack.

Its value is in showing that developers can use affordable hardware and open-source components to build and test Physical AI workflows in a practical and visible way. Rather than competing with mature one-stop robotics platforms, PAVE focuses on lowering the barrier to experimentation and making upper-layer VLA / VLM PoCs easier to prototype, understand, and extend.

## What this project is not

PAVE is not intended to be:

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

PAVE currently consists of three main parts:

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

## Why edge matters here

Many PoCs can be demonstrated using cloud inference. PAVE specifically focuses on edge-side execution because it is closer to how real-world robotics systems are often evaluated and deployed.

Using edge-side compute helps:

- reduce dependence on cloud connectivity
- improve responsiveness and deployment flexibility
- provide a more realistic setup for Physical AI experimentation
- make it easier to study trade-offs across different edge computing platforms

## Why RPi is mentioned

RPi is used here because it is highly recognisable in the developer community and represents an accessible entry point for robotics and edge experimentation.

However, PAVE is not limited to Raspberry Pi alone. The same workflow can be adapted to other Arm-based Linux systems and Cortex-X/A-based edge platforms, depending on the hardware and deployment goals.

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

PAVE is currently intended to support exploration of use cases such as:

* real-time scene understanding
* object recognition in live robot streams
* navigation suggestion based on visual context
* prompt-driven task switching for robotics scenarios
* edge-side benchmarking of VLA / VLM workflows

## Future directions

* Benchmark different VLA / VLM models under the same live robotics scenario across edge computing platforms
* Investigate how real-time communication can be improved for lower latency, higher reliability, and better fault tolerance
* Extend the workflow into additional Physical AI PoCs to validate reuse across different application scenarios

## Target users

PAVE is intended for:

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

PAVE is best understood as:

* a lightweight experimentation project
* a developer-friendly edge VLA workflow
* a reproducible PoC path for Physical AI exploration
* a practical way to study how open-source software and affordable edge hardware can support real-world AI robotics use cases

## Status


## Demo:
Ver 1: 
https://youtu.be/kRiXri0te0g?si=iOhW0d2SSSP6zT4V

This project is currently an active PoC and experimentation framework. The scope, structure, and supported workflows may continue to evolve as new hardware targets, communication methods, and Physical AI scenarios are explored.
