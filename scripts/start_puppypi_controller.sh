#!/usr/bin/env bash
set -euo pipefail

PUPPYPI_CONTAINER="${PUPPYPI_CONTAINER:-puppypi_ros2}"
PUPPYPI_USER="${PUPPYPI_USER:-ubuntu}"
PUPPYPI_WORKDIR="${PUPPYPI_WORKDIR:-/home/ubuntu}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
PUPPYPI_RESTART_ROS_DAEMON="${PUPPYPI_RESTART_ROS_DAEMON:-1}"
PUPPYPI_LAUNCH_CMD="${PUPPYPI_LAUNCH_CMD:-ros2 launch puppy_control puppy_control.launch.py}"

log() {
  printf '[puppypi] %s\n' "$*"
}

if ! command -v docker >/dev/null 2>&1; then
  log "docker command not found"
  exit 1
fi

if ! docker inspect "$PUPPYPI_CONTAINER" >/dev/null 2>&1; then
  log "container not found: $PUPPYPI_CONTAINER"
  log "set PUPPYPI_CONTAINER=<container-name> if your PuppyPi ROS2 container has a different name"
  exit 1
fi

log "starting container: $PUPPYPI_CONTAINER"
docker start "$PUPPYPI_CONTAINER" >/dev/null

log "ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
log "RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION"
log "launch: $PUPPYPI_LAUNCH_CMD"

docker exec \
  -it \
  -u "$PUPPYPI_USER" \
  -w "$PUPPYPI_WORKDIR" \
  -e ROS_DOMAIN_ID="$ROS_DOMAIN_ID" \
  -e RMW_IMPLEMENTATION="$RMW_IMPLEMENTATION" \
  -e PUPPYPI_RESTART_ROS_DAEMON="$PUPPYPI_RESTART_ROS_DAEMON" \
  -e PUPPYPI_LAUNCH_CMD="$PUPPYPI_LAUNCH_CMD" \
  "$PUPPYPI_CONTAINER" \
  bash -lc '
    set -euo pipefail
    source /opt/ros/humble/setup.bash

    if [[ "$PUPPYPI_RESTART_ROS_DAEMON" == "1" || "$PUPPYPI_RESTART_ROS_DAEMON" == "true" ]]; then
      ros2 daemon stop >/dev/null 2>&1 || true
      ros2 daemon start
    fi

    exec $PUPPYPI_LAUNCH_CMD
  '
