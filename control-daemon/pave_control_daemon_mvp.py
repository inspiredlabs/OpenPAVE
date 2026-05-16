"""
VLA Control Daemon (MVP) — Intent File Bus → ROS 2 Commands (PuppyPi)

Goal
- Provide a minimal, reproducible “control daemon” that turns high-level intents
  (produced by a VLA/VLM pipeline) into ROS 2 commands for the PuppyPi robot dog.
- The daemon watches a single JSON file (intent bus) and triggers robot actions
  when the file changes (mtime-based de-duplication).
- This keeps the control path headless and decoupled from any UI.

How to use
1) Start the robot-side controller first (on PuppyPi):
   - ros2 launch puppy_control puppy_control.launch.py

2) Start this daemon on the control machine (e.g., DGX):
   - export ROS_DOMAIN_ID=0
   - export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
   - python3 vla_control_daemon_mvp.py

3) Write intents into the intent file (default: /tmp/vla_intent.json).
   Examples (write ONE JSON object each time). Legacy flat payloads are still
   accepted and normalized to intent schema v0.1 internally:

   Forward (currently debug / may behave like in-place stepping depending on firmware/controller):
     {"intent":"MOVE","vx":0.1,"yaw":0.0,"duration_ms":800}

   Turn right:
     {"intent":"MOVE","vx":0.0,"yaw":0.6,"duration_ms":600}

   Turn left:
     {"intent":"MOVE","vx":0.0,"yaw":-0.4,"duration_ms":600}

   Stop + reset posture (STOP triggers a go_home reset in this build):
     {"intent":"STOP"}

   Normalized schema v0.1:
     {"schema_version":"0.1","intent":"MOVE","params":{"vx":0.0,"yaw":0.6,"duration_ms":600},"source":"manual","request_id":"demo","timestamp":"2026-05-16T00:00:00+00:00"}

Notes
- Turning is implemented via /puppy_control/velocity_move (yaw_rate).
- Straight walking (vx) is still under debugging on this platform; different PuppyPi
  builds may require gait/pose/velocity pipelines instead of velocity_move.x.
"""

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pave_runtime.intent_schema import (
    IntentValidationError,
    intent_action_key,
    normalize_intent_payload,
    now_iso,
)
from control_daemon.adapters import create_robot_adapter

INTENT_PATH = os.environ.get("INTENT_PATH", "/tmp/vla_intent.json")
POLL_SEC = float(os.environ.get("POLL_SEC", "0.2"))

def load_intent():
    try:
        with open(INTENT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def get_mtime(path: str):
    try:
        return os.path.getmtime(path)
    except Exception:
        return None

def main():
    adapter = create_robot_adapter()

    print(f"[daemon] INTENT_PATH={INTENT_PATH} POLL_SEC={POLL_SEC}")
    print(f"[daemon] ROBOT_ADAPTER={adapter.name}")

    last_mtime = None
    last_action_key = None  # de-dupe repeated identical actions

    while True:
        mtime = get_mtime(INTENT_PATH)
        if mtime is not None and mtime != last_mtime:
            last_mtime = mtime
            evt = load_intent()

            if not isinstance(evt, dict):
                time.sleep(POLL_SEC)
                continue

            try:
                normalized = normalize_intent_payload(
                    evt,
                    default_source="file-bus",
                    safe_default=True,
                )
            except IntentValidationError as exc:
                print(f"[{now_iso()}] WARN: invalid intent payload: {exc}")
                time.sleep(POLL_SEC)
                continue

            # Optional: simple per-action de-dupe (prevents repeated identical write spam)
            action_key = intent_action_key(normalized)
            if action_key == last_action_key:
                time.sleep(POLL_SEC)
                continue
            last_action_key = action_key

            intent = normalized["intent"]
            if intent == "TROT":
                adapter.trot()
            elif intent == "HOME":
                adapter.home()
            elif intent == "MOVE":
                params = normalized.get("params", {})
                vx = float(params.get("vx", 0.0))
                yaw = float(params.get("yaw", 0.0))
                duration_ms = int(params.get("duration_ms", 500))
                adapter.move(vx=vx, yaw=yaw, duration_ms=duration_ms)
            else:
                adapter.stop()

        time.sleep(POLL_SEC)

if __name__ == "__main__":
    main()
