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

2) Start this daemon on the control machine:
   - export ROS_DOMAIN_ID=0
   - export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
   - python3 -m control_daemon.daemon

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
import threading
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
from control_daemon.feedback import atomic_write_json, command_result, robot_state

INTENT_PATH = os.environ.get("INTENT_PATH", "/tmp/vla_intent.json")
# Intent-file poll cadence. 50ms (was 200ms) so a posted intent is picked up and
# executed near-instantly — localhost file stat is cheap; this is pure pickup
# latency between intent_ingress writing and the daemon acting on it.
POLL_SEC = float(os.environ.get("POLL_SEC", "0.05"))
COMMAND_RESULT_PATH = os.environ.get("COMMAND_RESULT_PATH", "/tmp/vla_command_result.json")
ROBOT_STATE_PATH = os.environ.get("ROBOT_STATE_PATH", "/tmp/vla_robot_state.json")
# Occasional on purpose: this heartbeat's only job is to keep the ROBOT STATE
# live time-series panel showing *something* while nothing real is happening —
# not to generate a steady stream of data. 30s is "very occasionally during
# development"; every real command still writes robot_state immediately on its
# own (see execute_intent), so this interval never affects actual responsiveness.
ROBOT_HEARTBEAT_SEC = float(os.environ.get("ROBOT_HEARTBEAT_SEC", "30.0"))
PROCESS_EXISTING_INTENT_ON_START = os.environ.get(
    "PROCESS_EXISTING_INTENT_ON_START", "0"
).lower() in {"1", "true", "yes"}
_STATE_LOCK = threading.Lock()
_CURRENT_ROBOT_STATUS = "starting"
_CURRENT_LAST_COMMAND = None

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

def write_command_feedback(result: dict):
    atomic_write_json(COMMAND_RESULT_PATH, result)
    print(
        f"[{now_iso()}] COMMAND status={result.get('status')} "
        f"intent={result.get('intent')} request_id={result.get('request_id')}"
    )

def adapter_telemetry(adapter) -> dict:
    getter = getattr(adapter, "get_state", None)
    if callable(getter):
        try:
            state = getter()
            if isinstance(state, dict):
                return state
        except Exception:
            pass
    return {}


def write_robot_state(status: str, adapter, last_command: dict | None = None, heartbeat_seq: int | None = None):
    global _CURRENT_LAST_COMMAND, _CURRENT_ROBOT_STATUS
    telemetry = adapter_telemetry(adapter)
    payload = robot_state(
        adapter_name=adapter.name,
        status=status,
        last_command=last_command,
        pose=telemetry.get("pose"),
        joint_state=telemetry.get("joint_state"),
        heartbeat_seq=heartbeat_seq,
    )
    with _STATE_LOCK:
        _CURRENT_ROBOT_STATUS = status
        if last_command is not None:
            _CURRENT_LAST_COMMAND = last_command
        atomic_write_json(ROBOT_STATE_PATH, payload)


def start_robot_heartbeat(adapter):
    def loop():
        seq = 0
        while True:
            time.sleep(ROBOT_HEARTBEAT_SEC)
            seq += 1
            with _STATE_LOCK:
                status = _CURRENT_ROBOT_STATUS
                last_command = _CURRENT_LAST_COMMAND
            write_robot_state(status, adapter, last_command, heartbeat_seq=seq)

    thread = threading.Thread(target=loop, name="robot-state-heartbeat", daemon=True)
    thread.start()
    return thread

def execute_intent(normalized: dict, adapter):
    started_at = now_iso()
    executing = command_result(
        intent=normalized,
        adapter_name=adapter.name,
        status="executing",
        started_at=started_at,
    )
    write_command_feedback(executing)
    write_robot_state("executing", adapter, executing)

    try:
        intent = normalized["intent"]
        if intent == "TROT":
            adapter_result = adapter.trot()
        elif intent == "HOME":
            adapter_result = adapter.home()
        elif intent == "MOVE":
            params = normalized.get("params", {})
            vx = float(params.get("vx", 0.0))
            yaw = float(params.get("yaw", 0.0))
            duration_ms = int(params.get("duration_ms", 500))
            adapter_result = adapter.move(vx=vx, yaw=yaw, duration_ms=duration_ms)
        else:
            adapter_result = adapter.stop()
    except Exception as exc:
        failed = command_result(
            intent=normalized,
            adapter_name=adapter.name,
            status="failed",
            started_at=started_at,
            completed_at=now_iso(),
            error=str(exc),
        )
        write_command_feedback(failed)
        write_robot_state("error", adapter, failed)
        return failed

    status = "completed" if adapter_result.success else "failed"
    result = command_result(
        intent=normalized,
        adapter_name=adapter.name,
        status=status,
        started_at=started_at,
        completed_at=now_iso(),
        steps=adapter_result.steps,
        error=adapter_result.error,
    )
    write_command_feedback(result)
    write_robot_state("idle" if adapter_result.success else "error", adapter, result)
    return result

def main():
    adapter = create_robot_adapter()

    print(f"[daemon] INTENT_PATH={INTENT_PATH} POLL_SEC={POLL_SEC}")
    print(f"[daemon] ROBOT_ADAPTER={adapter.name}")
    print(f"[daemon] COMMAND_RESULT_PATH={COMMAND_RESULT_PATH}")
    print(f"[daemon] ROBOT_STATE_PATH={ROBOT_STATE_PATH}")
    print(f"[daemon] ROBOT_HEARTBEAT_SEC={ROBOT_HEARTBEAT_SEC}")
    print(f"[daemon] PROCESS_EXISTING_INTENT_ON_START={PROCESS_EXISTING_INTENT_ON_START}")
    write_robot_state("idle", adapter)
    start_robot_heartbeat(adapter)

    last_mtime = None if PROCESS_EXISTING_INTENT_ON_START else get_mtime(INTENT_PATH)
    if last_mtime is not None:
        print(f"[daemon] ignoring existing intent file until it changes: {INTENT_PATH}")
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
                rejected = command_result(
                    intent=None,
                    adapter_name=adapter.name,
                    status="rejected",
                    completed_at=now_iso(),
                    error=str(exc),
                )
                write_command_feedback(rejected)
                write_robot_state("idle", adapter, rejected)
                time.sleep(POLL_SEC)
                continue

            # Optional: simple per-action de-dupe (prevents repeated identical write
            # spam). MOVE is exempt: each MOVE is an incremental nudge (~5° per
            # held pointing gesture, see intent_schema), so repeats are the
            # commanded behaviour — continuous turning — not spam. Mode commands
            # (TROT/STOP/HOME) stay de-duped; repeating those is a no-op.
            action_key = intent_action_key(normalized)
            if normalized.get("intent") != "MOVE" and action_key == last_action_key:
                time.sleep(POLL_SEC)
                continue
            last_action_key = action_key

            received = command_result(
                intent=normalized,
                adapter_name=adapter.name,
                status="received",
            )
            write_command_feedback(received)
            write_robot_state("received", adapter, received)
            accepted = command_result(
                intent=normalized,
                adapter_name=adapter.name,
                status="accepted",
            )
            write_command_feedback(accepted)
            write_robot_state("accepted", adapter, accepted)
            execute_intent(normalized, adapter)

        time.sleep(POLL_SEC)

if __name__ == "__main__":
    main()
