#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

echo "[start_all] ROS_DOMAIN_ID=$ROS_DOMAIN_ID RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION"

cd "$ROOT"

python3 -m intent_ingress.server &
INGRESS_PID=$!
echo "[start_all] intent_ingress pid=$INGRESS_PID"

sleep 0.5
python3 -m control_daemon.daemon

kill "$INGRESS_PID" 2>/dev/null || true
