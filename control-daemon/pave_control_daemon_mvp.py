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
import subprocess
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

INTENT_PATH = os.environ.get("INTENT_PATH", "/tmp/vla_intent.json")
POLL_SEC = float(os.environ.get("POLL_SEC", "0.2"))

ROS_DOMAIN_ID = os.environ.get("ROS_DOMAIN_ID", "0")
RMW = os.environ.get("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")

# For ROS2 services (std_srvs) we can use plain ros:humble
ROS_SVC_IMAGE = os.environ.get("ROS_SVC_IMAGE", "ros:humble")

# For publishing puppy_control_msgs/msg/Velocity we need an image that contains puppy_control_msgs
# e.g. your built image: puppy-ros2-cli:humble
ROS_PUB_IMAGE = os.environ.get("ROS_PUB_IMAGE", "puppy-ros2-cli:humble")

def sh(cmd: str) -> int:
    """Run shell command inside bash -lc."""
    p = subprocess.run(cmd, shell=True)
    return p.returncode

def ros2_service_call(service: str, srv_type: str, payload: str) -> int:
    cmd = (
        f"docker run --rm --net=host "
        f"-e ROS_DOMAIN_ID={ROS_DOMAIN_ID} "
        f"-e RMW_IMPLEMENTATION={RMW} "
        f"{ROS_SVC_IMAGE} bash -lc "
        f"\"source /opt/ros/humble/setup.bash && "
        f"ros2 service call {service} {srv_type} '{payload}' >/dev/null 2>&1\""
    )
    return sh(cmd)

def ros2_topic_pub_velocity_move(vx: float, yaw: float) -> int:
    cmd = (
        f"docker run --rm --net=host "
        f"-e ROS_DOMAIN_ID={ROS_DOMAIN_ID} "
        f"-e RMW_IMPLEMENTATION={RMW} "
        f"{ROS_PUB_IMAGE} bash -lc "
        f"\"source /opt/ros/humble/setup.bash && source /ws/install/setup.bash && "
        f"ros2 topic pub -1 /puppy_control/velocity_move puppy_control_msgs/msg/Velocity "
        f"'{{x: {vx}, y: 0.0, yaw_rate: {yaw}}}'\""
    )
    return sh(cmd)

def do_trot():
    print(f"[{now_iso()}] ACTION=TROT")
    ros2_service_call("/puppy_control/set_running", "std_srvs/srv/SetBool", "{data: true}")
    ros2_service_call("/puppy_control/set_mark_time", "std_srvs/srv/SetBool", "{data: true}")

def do_stop():
    print(f"[{now_iso()}] ACTION=STOP")
    ros2_service_call("/puppy_control/set_mark_time", "std_srvs/srv/SetBool", "{data: false}")
    ros2_service_call("/puppy_control/set_running", "std_srvs/srv/SetBool", "{data: false}")
    # --- add reset to normal standing posture ---
    ros2_service_call("/puppy_control/go_home", "std_srvs/srv/Empty", "{}")
    time.sleep(0.3)

def do_home():
    print(f"[{now_iso()}] ACTION=HOME")
    ros2_service_call("/puppy_control/go_home", "std_srvs/srv/Empty", "{}")

def do_move(vx: float, yaw: float, duration_ms: int):
    print(f"[{now_iso()}] ACTION=MOVE vx={vx} yaw={yaw} duration_ms={duration_ms}")
    ros2_service_call("/puppy_control/go_home", "std_srvs/srv/Empty", "{}")
    ros2_service_call("/puppy_control/set_mark_time", "std_srvs/srv/SetBool", "{data: false}")
    ros2_service_call("/puppy_control/set_running", "std_srvs/srv/SetBool", "{data: true}")
    time.sleep(0.3)  # give the robot a moment to stand stably
    # ---------------------------------------------------------------
    rc = ros2_topic_pub_velocity_move(vx=vx, yaw=yaw)
    if rc != 0:
        print(f"[{now_iso()}] WARN: velocity_move pub rc={rc}")

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
    print(f"[daemon] INTENT_PATH={INTENT_PATH} POLL_SEC={POLL_SEC}")
    print(f"[daemon] ROS_DOMAIN_ID={ROS_DOMAIN_ID} RMW={RMW}")
    print(f"[daemon] ROS_SVC_IMAGE={ROS_SVC_IMAGE}")
    print(f"[daemon] ROS_PUB_IMAGE={ROS_PUB_IMAGE}")

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
                do_trot()
            elif intent == "HOME":
                do_home()
            elif intent == "MOVE":
                params = normalized.get("params", {})
                vx = float(params.get("vx", 0.0))
                yaw = float(params.get("yaw", 0.0))
                duration_ms = int(params.get("duration_ms", 500))
                do_move(vx=vx, yaw=yaw, duration_ms=duration_ms)
            else:
                do_stop()

        time.sleep(POLL_SEC)

if __name__ == "__main__":
    main()
