"""Robot adapters for the OpenPAVE control daemon."""

from __future__ import annotations

import os
import random
import subprocess
import time
from dataclasses import dataclass
from typing import Callable, Protocol

from pave_runtime.intent_schema import now_iso


CommandRunner = Callable[[str], int]


@dataclass(frozen=True)
class AdapterActionResult:
    success: bool
    steps: list[dict[str, object]]
    error: str | None = None

    @classmethod
    def ok(cls, steps: list[dict[str, object]] | None = None) -> "AdapterActionResult":
        return cls(success=True, steps=steps or [])

    @classmethod
    def failed(
        cls,
        error: str,
        steps: list[dict[str, object]] | None = None,
    ) -> "AdapterActionResult":
        return cls(success=False, steps=steps or [], error=error)


class RobotAdapter(Protocol):
    """Common robot capability interface consumed by the control daemon."""

    name: str

    def stop(self) -> AdapterActionResult:
        """Stop robot motion and return to a safe posture when supported."""

    def trot(self) -> AdapterActionResult:
        """Start the adapter's trot or mark-time behavior."""

    def home(self) -> AdapterActionResult:
        """Return the robot to its home posture."""

    def move(self, vx: float, yaw: float, duration_ms: int) -> AdapterActionResult:
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

    def _step(self, name: str, rc: int) -> dict[str, object]:
        return {"name": name, "return_code": rc}

    def _result(self, steps: list[dict[str, object]]) -> AdapterActionResult:
        failed_steps = [step for step in steps if step.get("return_code") != 0]
        if failed_steps:
            return AdapterActionResult.failed("one or more adapter steps failed", steps)
        return AdapterActionResult.ok(steps)

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

    def trot(self) -> AdapterActionResult:
        print(f"[{now_iso()}] ACTION=TROT adapter={self.name}")
        steps = [
            self._step(
                "set_running:true",
                self._ros2_service_call(
                    "/puppy_control/set_running",
                    "std_srvs/srv/SetBool",
                    "{data: true}",
                ),
            ),
            self._step(
                "set_mark_time:true",
                self._ros2_service_call(
                    "/puppy_control/set_mark_time",
                    "std_srvs/srv/SetBool",
                    "{data: true}",
                ),
            ),
        ]
        return self._result(steps)

    def stop(self) -> AdapterActionResult:
        print(f"[{now_iso()}] ACTION=STOP adapter={self.name}")
        steps = [
            self._step(
                "set_mark_time:false",
                self._ros2_service_call(
                    "/puppy_control/set_mark_time",
                    "std_srvs/srv/SetBool",
                    "{data: false}",
                ),
            ),
            self._step(
                "set_running:false",
                self._ros2_service_call(
                    "/puppy_control/set_running",
                    "std_srvs/srv/SetBool",
                    "{data: false}",
                ),
            ),
            self._step(
                "go_home",
                self._ros2_service_call("/puppy_control/go_home", "std_srvs/srv/Empty", "{}"),
            ),
        ]
        time.sleep(0.3)
        return self._result(steps)

    def home(self) -> AdapterActionResult:
        print(f"[{now_iso()}] ACTION=HOME adapter={self.name}")
        steps = [
            self._step(
                "go_home",
                self._ros2_service_call("/puppy_control/go_home", "std_srvs/srv/Empty", "{}"),
            )
        ]
        return self._result(steps)

    def move(self, vx: float, yaw: float, duration_ms: int) -> AdapterActionResult:
        print(f"[{now_iso()}] ACTION=MOVE adapter={self.name} vx={vx} yaw={yaw} duration_ms={duration_ms}")
        steps = [
            self._step(
                "go_home",
                self._ros2_service_call("/puppy_control/go_home", "std_srvs/srv/Empty", "{}"),
            ),
            self._step(
                "set_mark_time:false",
                self._ros2_service_call(
                    "/puppy_control/set_mark_time",
                    "std_srvs/srv/SetBool",
                    "{data: false}",
                ),
            ),
            self._step(
                "set_running:true",
                self._ros2_service_call(
                    "/puppy_control/set_running",
                    "std_srvs/srv/SetBool",
                    "{data: true}",
                ),
            ),
        ]
        time.sleep(0.3)
        rc = self._ros2_topic_pub_velocity_move(vx=vx, yaw=yaw)
        steps.append(self._step("velocity_move", rc))
        if rc != 0:
            print(f"[{now_iso()}] WARN: velocity_move pub rc={rc}")
        return self._result(steps)


class MockAdapter:
    """Dry-run adapter for local development without robot hardware.

    The mock intentionally mirrors PuppyPiAdapter's step names and settle delays so
    local tests exercise the same command-result shape as the ROS2 path.
    """

    name = "mock"

    def __init__(
        self,
        *,
        fail_step: str | None = None,
        fail_rate: float | None = None,
        fast: bool | None = None,
    ) -> None:
        self.fail_step = fail_step if fail_step is not None else os.environ.get("MOCK_ADAPTER_FAIL_STEP", "")
        self.fail_rate = (
            float(fail_rate)
            if fail_rate is not None
            else float(os.environ.get("MOCK_ADAPTER_FAIL_RATE", "0") or 0)
        )
        self.fast = (
            fast
            if fast is not None
            else os.environ.get("MOCK_ADAPTER_FAST", "0").lower() in {"1", "true", "yes"}
        )
        # per-command settle time (the "command cycle"): tunable so the UI can
        # trade realism (PuppyPi-ish settle) against gesture responsiveness
        self.settle_s = float(os.environ.get("MOCK_ADAPTER_SETTLE_MS", "300")) / 1000.0
        self.pose = {"x": 0.0, "y": 0.0, "heading": 0.0}
        self.joint_state = {
            "lf": 0.0,
            "rf": 0.0,
            "lb": 0.0,
            "rb": 0.0,
        }

    def _sleep(self, seconds: float) -> None:
        if not self.fast:
            time.sleep(seconds)

    def _step(self, name: str) -> dict[str, object]:
        fail_named = bool(self.fail_step) and self.fail_step == name
        fail_random = self.fail_rate > 0 and random.random() < self.fail_rate
        return {"name": name, "return_code": 9 if fail_named or fail_random else 0}

    def _result(self, steps: list[dict[str, object]]) -> AdapterActionResult:
        failed_steps = [step for step in steps if step.get("return_code") != 0]
        if failed_steps:
            return AdapterActionResult.failed("one or more adapter steps failed", steps)
        return AdapterActionResult.ok(steps)

    def get_state(self) -> dict[str, object]:
        return {
            "pose": dict(self.pose),
            "joint_state": dict(self.joint_state),
        }

    def stop(self) -> AdapterActionResult:
        print(f"[{now_iso()}] MOCK ACTION=STOP")
        steps = [
            self._step("set_mark_time:false"),
            self._step("set_running:false"),
            self._step("go_home"),
        ]
        self.joint_state = {key: 0.0 for key in self.joint_state}
        self._sleep(self.settle_s)
        return self._result(steps)

    def trot(self) -> AdapterActionResult:
        print(f"[{now_iso()}] MOCK ACTION=TROT")
        steps = [
            self._step("set_running:true"),
            self._step("set_mark_time:true"),
        ]
        self.joint_state = {key: 0.5 for key in self.joint_state}
        return self._result(steps)

    def home(self) -> AdapterActionResult:
        print(f"[{now_iso()}] MOCK ACTION=HOME")
        steps = [self._step("go_home")]
        self.pose = {"x": 0.0, "y": 0.0, "heading": 0.0}
        self.joint_state = {key: 0.0 for key in self.joint_state}
        self._sleep(self.settle_s)
        return self._result(steps)

    def move(self, vx: float, yaw: float, duration_ms: int) -> AdapterActionResult:
        print(f"[{now_iso()}] MOCK ACTION=MOVE vx={vx} yaw={yaw} duration_ms={duration_ms}")
        steps = [
            self._step("go_home"),
            self._step("set_mark_time:false"),
            self._step("set_running:true"),
        ]
        self._sleep(self.settle_s)
        steps.append(self._step("velocity_move"))
        if all(step.get("return_code") == 0 for step in steps):
            seconds = max(0.0, duration_ms / 1000.0)
            self.pose["x"] += vx * seconds
            self.pose["heading"] += yaw * seconds
            # velocity_move includes set_mark_time:false — trot stops, like the
            # real PuppyPi path; joint_state is how downstream (the visualiser's
            # bob) knows whether the robot is currently trotting.
            self.joint_state = {key: 0.0 for key in self.joint_state}
        return self._result(steps)


def create_robot_adapter(name: str | None = None) -> RobotAdapter:
    adapter_name = (name or os.environ.get("ROBOT_ADAPTER", "puppypi")).strip().lower()

    if adapter_name in {"mock", "dry-run", "dry_run"}:
        return MockAdapter()
    if adapter_name == "puppypi":
        return PuppyPiAdapter()

    raise ValueError(f"unsupported ROBOT_ADAPTER: {adapter_name}")
