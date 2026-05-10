#!/usr/bin/env bash
set -euo pipefail

ROOT="$(dirname "$0")/.."
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

echo "[start_all] ROS_DOMAIN_ID=$ROS_DOMAIN_ID RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION"

python3 "$ROOT/intent-ingress/intent_ingress.py" &
INGRESS_PID=$!
echo "[start_all] intent_ingress pid=$INGRESS_PID"

sleep 0.5
python3 "$ROOT/control-daemon/pave_control_daemon_mvp.py"

kill "$INGRESS_PID" 2>/dev/null || true
