#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../control-daemon"
python3 pave_control_daemon_mvp.py
