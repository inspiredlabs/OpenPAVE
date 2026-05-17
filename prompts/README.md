# OpenPAVE Prompt Presets

This directory stores reusable prompt presets for OpenPAVE experiments.

Prompt presets are versioned JSON files so scenarios and future benchmark runs can reference an exact prompt instead of copying prompt text into multiple places.

## Format

Each prompt file uses this shape:

```json
{
  "id": "robot_commander_gesture_v0",
  "version": "0.1",
  "title": "Robot Commander Gesture Control",
  "task_type": "intent_control",
  "output_contract": {
    "format": "single_intent",
    "allowed_intents": ["STOP", "TROT"],
    "fallback_intent": "STOP"
  },
  "prompt": "..."
}
```

## Current Presets

- `intent-stop-trot.json`: Minimal `STOP` / `TROT` output contract.
- `robot-commander-gesture.json`: Camera gesture control prompt for thumbs-up and open-palm gestures.
- `scene-understanding.json`: General scene and sensor observation prompt.
- `object-recognition.json`: Object recognition prompt for live robot/sensor streams.
- `navigation-suggestion.json`: Navigation suggestion prompt that avoids direct robot control.

## Safety Notes

Prompts that can drive physical robot intent must use a safe fallback. For the current PuppyPi validation path, unknown or ambiguous visual evidence should map to `STOP`.
