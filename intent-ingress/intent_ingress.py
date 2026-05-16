import json
import os
import sys
from pathlib import Path

from flask import Flask, request, jsonify

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pave_runtime.intent_schema import IntentValidationError, normalize_intent_payload

INTENT_PATH = os.environ.get("INTENT_PATH", "/tmp/vla_intent.json")

app = Flask(__name__)

def atomic_write(obj: dict):
    tmp = INTENT_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, INTENT_PATH)

@app.post("/intent")
def intent():
    data = request.get_json(force=True, silent=True) or {}
    try:
        payload = normalize_intent_payload(data, default_source="webui", safe_default=True)
    except IntentValidationError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    atomic_write(payload)
    return jsonify({"ok": True, "written": payload})

@app.get("/healthz")
def healthz():
    return "ok\n", 200

if __name__ == "__main__":
    # Bind localhost only for MVP (safer). Change to 0.0.0.0 if needed.
    app.run(host="127.0.0.1", port=7071)
