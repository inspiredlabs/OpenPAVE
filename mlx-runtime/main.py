"""OpenPAVE MLX runtime — initial cutover entrypoint.

This is the *first slice* of the DGX Spark -> native MLX port described in
`docs/dgx-spark-mlx-port.md`. It is intentionally small and headless:

  1. Confirm MLX is available on this machine (NumPy fallback if not), mirroring
     the template's `PolicyRuntime` try/except pattern.
  2. Prove the OpenPAVE control plane works end to end on macOS by sending a
     demo intent to Intent Ingress and reading back the feedback files.

The launcher `mlx-runtime.sh` starts Intent Ingress + the Control Daemon
(`ROBOT_ADAPTER=mock`) before running this file. No Docker, no ROS2, no vLLM.

The TODO seams at the bottom mark exactly where the next pieces from the port
guide plug in: the PyQt6 console (pave_ui), the physics-simulator digital twin
(pave_sim), and the in-process MLX VLA engine (pave_mlx).
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from state_server import StateServer

INGRESS_URL = os.environ.get("INTENT_INGRESS_URL", "http://127.0.0.1:7071/intent")
HEALTH_URL = os.environ.get("INTENT_HEALTH_URL", "http://127.0.0.1:7071/healthz")
COMMAND_RESULT_PATH = os.environ.get("COMMAND_RESULT_PATH", "/tmp/vla_command_result.json")
ROBOT_STATE_PATH = os.environ.get("ROBOT_STATE_PATH", "/tmp/vla_robot_state.json")


def check_mlx() -> str:
    """Report the active compute backend. MLX when Metal is present, else NumPy."""
    try:
        import mlx.core as mx

        a = mx.array([1.0, 2.0, 3.0])
        mx.eval(a * 2)
        print(f" -> MLX backend active (default device: {mx.default_device()})")
        return "mlx"
    except Exception as exc:  # noqa: BLE001 - any import/Metal failure falls back
        print(f" -> MLX unavailable ({type(exc).__name__}: {exc}); using NumPy fallback")
        return "numpy"


def wait_for_health(timeout_s: float = 10.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=1.0) as resp:
                if 200 <= resp.status < 300:
                    return True
        except Exception:
            time.sleep(0.25)
    return False


def send_intent(payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        INGRESS_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=2.0) as resp:
        return json.loads(resp.read().decode("utf-8"))


def read_json(path: str) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


# Robot "I'm alive" heartbeat — NOT a movement demo. Cycles tiny, infrequent
# intents through the full pipeline (ingress -> file bus -> daemon -> mock
# adapter -> feedback) so the visualiser visibly ticks and the path is proven
# end to end, without looking like real navigation. Keep vx/yaw tiny and the
# period long ON PURPOSE — see pave_ui/viewer.py's KEEPALIVE_STEPS for the same
# convention. Disable with DEMO_SCHEDULE=0.
DEMO_STEPS = [
    {"intent": "MOVE", "params": {"vx": 0.0, "yaw": 0.05, "duration_ms": 200}},
    {"intent": "MOVE", "params": {"vx": 0.0, "yaw": -0.05, "duration_ms": 200}},
]


def run_demo_schedule(period_s: float = 20.0) -> None:
    time.sleep(2.0)  # let the browser connect to the stream first
    i = 0
    while True:
        step = DEMO_STEPS[i % len(DEMO_STEPS)]
        try:
            send_intent({**step, "source": "keepalive"})
        except Exception:
            pass
        i += 1
        time.sleep(period_s)


def main() -> None:
    print("=== OpenPAVE MLX Runtime (initial cutover) ===")

    backend = check_mlx()

    if not wait_for_health():
        raise SystemExit(
            f"Intent Ingress not reachable at {HEALTH_URL}. "
            "Start it via ./mlx-runtime.sh (it launches ingress + daemon for you)."
        )
    print(f" -> Intent Ingress healthy at {HEALTH_URL}")

    # End-to-end control-path smoke test: text intents the daemon already speaks.
    for text in ("TROT", "STOP"):
        ack = send_intent({"text": text, "source": "mlx-runtime"})
        print(f" -> sent intent {text}: ok={ack.get('ok')}")
        time.sleep(0.5)  # let the daemon poll the file bus and write feedback

    print("\n--- latest command result ---")
    print(json.dumps(read_json(COMMAND_RESULT_PATH), indent=2, sort_keys=True))
    print("\n--- latest robot state ---")
    print(json.dumps(read_json(ROBOT_STATE_PATH), indent=2, sort_keys=True))

    print(f"\n[ok] control plane verified on this machine (compute backend: {backend}).")

    # Streaming robot-state viewer (Three.js + SSE), served WITHOUT Flask so the
    # browser gets a consistent time series. This is the observation half of the
    # port; Intent Ingress (Flask, :7071) stays POST-only for control.
    server = StateServer(port=int(os.environ.get("STATE_SERVER_PORT", "7080")))
    url = server.start()
    print(f" -> robot-state viewer streaming at {url}")
    print(f"    SSE endpoint: {url}api/stream   STL model: {url}static/base_link.STL")
    if os.environ.get("OPEN_BROWSER", "1") not in {"0", "false", "no"}:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    demo_on = os.environ.get("DEMO_SCHEDULE", "1") not in {"0", "false", "no"}
    if demo_on:
        threading.Thread(target=run_demo_schedule, name="keepalive", daemon=True).start()
        print(" -> keep-alive heartbeat running (tiny yaw nudges, every ~20s)")

    print(
        "\nNext cutover steps (see docs/dgx-spark-mlx-port.md):\n"
        "  - pave_sim/  : physics-simulator digital twin + MLX PolicyRuntime\n"
        "  - pave_mlx/  : in-process MLX VLA engine -> STOP/TROT/MOVE intents\n"
        "  - pave_ui/   : PyQt6 console with the Experience selector\n"
        "\nStreaming… a keep-alive heartbeat ticks in the background; set DEMO_SCHEDULE=0 to drive it yourself:\n"
        "  curl -s -X POST http://127.0.0.1:7071/intent -H 'Content-Type: application/json' -d '{\"text\":\"TROT\"}'\n"
        "Press Ctrl+C to stop.\n"
    )
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
