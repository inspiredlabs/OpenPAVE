#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../intent-ingress"
python3 intent_ingress.py
