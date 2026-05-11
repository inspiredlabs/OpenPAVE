#!/usr/bin/env bash
set -euo pipefail

# Build a ROS2 CLI Docker image that includes PuppyPi custom message types:
# - puppy_control_msgs/msg/Velocity
#
# Image output:
#   puppy-ros2-cli:humble
#
# Repo requirement:
#   third_party/puppy_control_msgs must exist in this repo.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$ROOT/third_party/puppy_control_msgs"
DOCKER_DIR="$ROOT/docker/puppy_ros2_cli"
CTX_SRC_DIR="$DOCKER_DIR/src"

IMAGE_TAG="${IMAGE_TAG:-puppy-ros2-cli:humble}"

if [[ ! -d "$SRC_DIR" ]]; then
  echo "[ERROR] Missing $SRC_DIR"
  echo "Make sure third_party/puppy_control_msgs exists in the repo."
  exit 1
fi

echo "[INFO] Repo root: $ROOT"
echo "[INFO] Using source: $SRC_DIR"
echo "[INFO] Build context: $DOCKER_DIR"
echo "[INFO] Image tag: $IMAGE_TAG"

mkdir -p "$CTX_SRC_DIR"

# Copy fresh (avoid stale files if upstream changes)
rm -rf "$CTX_SRC_DIR/puppy_control_msgs"
cp -r "$SRC_DIR" "$CTX_SRC_DIR/puppy_control_msgs"

# Write Dockerfile (idempotent)
cat > "$DOCKER_DIR/Dockerfile" <<'EOF'
FROM ros:humble

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-colcon-common-extensions \
    ros-humble-rosidl-default-generators \
    ros-humble-rosidl-default-runtime \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ws
COPY src/ /ws/src/
RUN bash -lc "source /opt/ros/humble/setup.bash && colcon build --symlink-install"
EOF

echo "[INFO] Building docker image..."
docker build -t "$IMAGE_TAG" "$DOCKER_DIR"

echo "[INFO] Verifying custom msg exists in image..."
docker run -it --rm "$IMAGE_TAG" bash -lc \
"source /opt/ros/humble/setup.bash && source /ws/install/setup.bash && ros2 interface show puppy_control_msgs/msg/Velocity"

echo "[OK] Built and verified: $IMAGE_TAG"
