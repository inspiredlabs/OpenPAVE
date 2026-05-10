import json
import os
from datetime import datetime, timezone
from flask import Flask, request, jsonify

INTENT_PATH = os.environ.get("INTENT_PATH", "/tmp/vla_intent.json")

app = Flask(__name__)

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def atomic_write(obj: dict):
    tmp = INTENT_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, INTENT_PATH)

@app.post("/intent")
def intent():
    data = request.get_json(force=True, silent=True) or {}
    text = (data.get("intent") or data.get("text") or "").strip().upper()

    if text == "TROT":
        payload = {"intent": "TROT"}
    elif text == "STOP":
        payload = {"intent": "STOP"}
    elif text in ("TURN_LEFT", "LEFT"):
        payload = {"intent":"MOVE","vx":0.0,"yaw":-0.4,"duration_ms":600}
    elif text in ("TURN_RIGHT", "RIGHT"):
        payload = {"intent":"MOVE","vx":0.0,"yaw":0.6,"duration_ms":600}
    else:
        payload = {"intent": "STOP"}  # safe default

    payload["_ts"] = now_iso()
    payload["_src"] = "webui"

    atomic_write(payload)
    return jsonify({"ok": True, "written": payload})

@app.get("/healthz")
def healthz():
    return "ok\n", 200

if __name__ == "__main__":
    # Bind localhost only for MVP (safer). Change to 0.0.0.0 if needed.
    app.run(host="127.0.0.1", port=7071)
