"""Robot state and command result feedback helpers."""

from __future__ import annotations

import json
import os
from typing import Any

from pave_runtime.intent_schema import now_iso


COMMAND_STATES = {"received", "accepted", "executing", "completed", "failed", "rejected"}


def atomic_write_json(path: str, payload: dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, path)


def command_result(
    *,
    intent: dict[str, Any] | None,
    adapter_name: str,
    status: str,
    started_at: str | None = None,
    completed_at: str | None = None,
    steps: list[dict[str, Any]] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    if status not in COMMAND_STATES:
        raise ValueError(f"unsupported command status: {status}")

    intent = intent or {}
    result: dict[str, Any] = {
        "schema_version": "0.1",
        "request_id": intent.get("request_id"),
        "intent": intent.get("intent"),
        "params": intent.get("params", {}),
        "source": intent.get("source"),
        "adapter": adapter_name,
        "status": status,
        "updated_at": now_iso(),
        "started_at": started_at,
        "completed_at": completed_at,
        "steps": steps or [],
    }
    if error:
        result["error"] = error
    return result


def robot_state(
    *,
    adapter_name: str,
    status: str,
    last_command: dict[str, Any] | None = None,
    pose: dict[str, Any] | None = None,
    joint_state: dict[str, Any] | None = None,
    heartbeat_seq: int | None = None,
) -> dict[str, Any]:
    state = {
        "schema_version": "0.1",
        "adapter": adapter_name,
        "status": status,
        "updated_at": now_iso(),
        "last_command": last_command,
        "pose": pose or {"x": 0.0, "y": 0.0, "heading": 0.0},
        "joint_state": joint_state or {},
    }
    if heartbeat_seq is not None:
        state["heartbeat_seq"] = heartbeat_seq
    return state
