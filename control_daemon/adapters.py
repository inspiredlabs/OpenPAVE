"""Robot adapters for the OpenPAVE control daemon."""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from typing import Callable, Protocol

from pave_runtime.intent_schema import now_iso


CommandRunner = Callable[[str], int]


class RobotAdapter(Protocol):
    """Common robot capability interface consumed by the control daemon."""

    name: str

    def stop(self) -> None:
        """Stop robot motion and return to a safe posture when supported."""

    def trot(self) -> None:
        """Start the adapter's trot or mark-time behavior."""

    def home(self) -> None:
        """Return the robot to its home posture."""

    def move(self, vx: float, yaw: float, duration_ms: int) -> None:
        """Run a short velocity-style movement command."""


@dataclass(frozen=True)
class RosCliConfig:
    ros_domain_id: str
    rmw_implementation: str
    ros_svc_image: str
    ros_pub_image: str

    @classmethod
    def from_env(cls) -> "RosCliConfig":
        return cls(
            ros_domain_id=os.environ.get("ROS_DOMAIN_ID", "0"),
            rmw_implementation=os.environ.get("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp"),
            ros_svc_image=os.environ.get("ROS_SVC_IMAGE", "ros:humble"),
            ros_pub_image=os.environ.get("ROS_PUB_IMAGE", "puppy-ros2-cli:humble"),
        )


def default_runner(cmd: str) -> int:
    return subprocess.run(cmd, shell=True).returncode


class PuppyPiAdapter:
    """PuppyPi robot adapter backed by Dockerized ROS2 CLI calls."""

    name = "puppypi"

    def __init__(self, config: RosCliConfig | None = None, runner: CommandRunner | None = None):
        self.config = config or RosCliConfig.from_env()
        self.runner = runner or default_runner

    def _run(self, cmd: str) -> int:
        return self.runner(cmd)

    def _ros2_service_call(self, service: str, srv_type: str, payload: str) -> int:
        cmd = (
            f"docker run --rm --net=host "
            f"-e ROS_DOMAIN_ID={self.config.ros_domain_id} "
            f"-e RMW_IMPLEMENTATION={self.config.rmw_implementation} "
            f"{self.config.ros_svc_image} bash -lc "
            f"\"source /opt/ros/humble/setup.bash && "
            f"ros2 service call {service} {srv_type} '{payload}' >/dev/null 2>&1\""
        )
        return self._run(cmd)

    def _ros2_topic_pub_velocity_move(self, vx: float, yaw: float) -> int:
        cmd = (
            f"docker run --rm --net=host "
            f"-e ROS_DOMAIN_ID={self.config.ros_domain_id} "
            f"-e RMW_IMPLEMENTATION={self.config.rmw_implementation} "
            f"{self.config.ros_pub_image} bash -lc "
            f"\"source /opt/ros/humble/setup.bash && source /ws/install/setup.bash && "
            f"ros2 topic pub -1 /puppy_control/velocity_move puppy_control_msgs/msg/Velocity "
            f"'{{x: {vx}, y: 0.0, yaw_rate: {yaw}}}'\""
        )
        return self._run(cmd)

    def trot(self) -> None:
        print(f"[{now_iso()}] ACTION=TROT adapter={self.name}")
        self._ros2_service_call("/puppy_control/set_running", "std_srvs/srv/SetBool", "{data: true}")
        self._ros2_service_call(
            "/puppy_control/set_mark_time",
            "std_srvs/srv/SetBool",
            "{data: true}",
        )

    def stop(self) -> None:
        print(f"[{now_iso()}] ACTION=STOP adapter={self.name}")
        self._ros2_service_call(
            "/puppy_control/set_mark_time",
            "std_srvs/srv/SetBool",
            "{data: false}",
        )
        self._ros2_service_call("/puppy_control/set_running", "std_srvs/srv/SetBool", "{data: false}")
        self._ros2_service_call("/puppy_control/go_home", "std_srvs/srv/Empty", "{}")
        time.sleep(0.3)

    def home(self) -> None:
        print(f"[{now_iso()}] ACTION=HOME adapter={self.name}")
        self._ros2_service_call("/puppy_control/go_home", "std_srvs/srv/Empty", "{}")

    def move(self, vx: float, yaw: float, duration_ms: int) -> None:
        print(f"[{now_iso()}] ACTION=MOVE adapter={self.name} vx={vx} yaw={yaw} duration_ms={duration_ms}")
        self._ros2_service_call("/puppy_control/go_home", "std_srvs/srv/Empty", "{}")
        self._ros2_service_call(
            "/puppy_control/set_mark_time",
            "std_srvs/srv/SetBool",
            "{data: false}",
        )
        self._ros2_service_call("/puppy_control/set_running", "std_srvs/srv/SetBool", "{data: true}")
        time.sleep(0.3)
        rc = self._ros2_topic_pub_velocity_move(vx=vx, yaw=yaw)
        if rc != 0:
            print(f"[{now_iso()}] WARN: velocity_move pub rc={rc}")


class MockAdapter:
    """Dry-run adapter for local development without robot hardware."""

    name = "mock"

    def stop(self) -> None:
        print(f"[{now_iso()}] MOCK ACTION=STOP")

    def trot(self) -> None:
        print(f"[{now_iso()}] MOCK ACTION=TROT")

    def home(self) -> None:
        print(f"[{now_iso()}] MOCK ACTION=HOME")

    def move(self, vx: float, yaw: float, duration_ms: int) -> None:
        print(f"[{now_iso()}] MOCK ACTION=MOVE vx={vx} yaw={yaw} duration_ms={duration_ms}")


def create_robot_adapter(name: str | None = None) -> RobotAdapter:
    adapter_name = (name or os.environ.get("ROBOT_ADAPTER", "puppypi")).strip().lower()

    if adapter_name in {"mock", "dry-run", "dry_run"}:
        return MockAdapter()
    if adapter_name == "puppypi":
        return PuppyPiAdapter()

    raise ValueError(f"unsupported ROBOT_ADAPTER: {adapter_name}")
