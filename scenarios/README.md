# OpenPAVE Demo Scenarios

This directory stores reusable demo scenarios for OpenPAVE experiments.

Scenarios are structured JSON files that bind together:

- prompt preset reference
- expected intents or expected observation behavior
- safety constraints
- robot/sensor endpoint assumptions
- inference node assumptions
- adapter assumptions
- success criteria

## Current Scenarios

- `mock-intent-stop-trot.json`: Software-only control path validation with `MockAdapter`.
- `puppypi-gesture-stop-trot.json`: Physical PuppyPi gesture control validation.
- `scene-understanding-camera.json`: Camera-based scene observation scenario.
- `object-recognition-camera.json`: Camera-based object recognition scenario.
- `navigation-suggestion-camera.json`: Camera-based navigation suggestion scenario.

## Design Rule

Scenarios should describe assumptions explicitly. Avoid hiding PuppyPi-only, DGX-only, or camera-only assumptions in prompt text.
