# PAVE Ver1 (MVP) — Lightweight Physical-AI Edge VLA Experimentation

This repository is the **first MVP** of PAVE: a lightweight, reproducible workflow for **Physical-AI experimentation on edge platforms**.
The key idea is to decouple:
- **VLA Observability UI / Debug UI**: choose models, tune prompts, observe outputs and performance
- **VLA Control Daemon**: convert VLA/VLM outputs into **real ROS 2 commands** to control a robot

---
## 0) MVP Pipeline (Ver1)
**live-vlm-webui (Observability UI)** → **Intent Ingress (HTTP)** → **Intent File Bus** → **Control Daemon** → **ROS 2 commands** → **PuppyPi**

In Ver1, we validate that a VLM can output a small, stable “intent” (e.g., `STOP` / `TROT`) and reliably drive a physical robot via ROS 2.
---

## 1) Repos & Submodule

You have two repos:
- **Main MVP repo**:  
  `https://github.com/odincodeshen/PAVE`
- **Updated WebUI repo (already modified)**:  
  `https://github.com/odincodeshen/live-vlm-webui`

This repo includes the WebUI under:
- `ui/live-vlm-webui` (committed in this repo)
> If you are using git submodules instead of vendoring, ensure `.gitmodules` points to your updated WebUI repo.

---

## 2) Prerequisites
### Hardware / Network
- PuppyPi (or any Arm-based Linux robotic) and DGX are on the **same LAN**
- ROS 2 DDS multicast must work (home router LAN is OK, but VLANs can break it)

### Software (DGX)
- Docker installed
- Python 3.10+
- vLLM (OpenAI-compatible server) running on DGX
- `ui/live-vlm-webui` runnable via pip / python

### Software (PuppyPi)
- ROS 2 Humble (in Docker container image)
- `puppy_control.launch.py` available inside the container

---

## 3) Required Environment Variables (Must Match)
On **both PuppyPi and DGX**:
```bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

> If these don’t match, you may see `/puppy` sometimes but service/topic calls will fail or hang.

---

## 4) Step-by-Step Demo (Copy/Paste)

### Step 1 — Start PuppyPi controller (robot-side)

On PuppyPi (inside your ROS 2 container / shell):

Launch ROS2 in robotic side:

```bash
docker start puppypi_ros2
docker exec -it -u ubuntu -w /home/ubuntu puppypi_ros2 /bin/zsh
```

Inside the docker, 

```bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
source /opt/ros/humble/setup.bash
ros2 launch puppy_control puppy_control.launch.py
```

**Verify robot node is alive:**

```bash
ros2 node list | grep -E "^/puppy$" || echo "NO /puppy"
ros2 service list | grep puppy_control
```

### Step 2 — Build a ROS 2 CLI Docker Image with ROS2 Custom Messages

Robotics may have dirreent customer message can use, if you want DGX to publish following command in PuppyPI, 

- `/puppy_control/velocity_move`
- message type: `puppy_control_msgs/msg/Velocity`

you must use a ROS 2 CLI environment that includes the **custom message package** `puppy_control_msgs`.
The base image `ros:humble` does not include it.

This repo vendors the message package here:

- `third_party/puppy_control_msgs`

```bash
chmod +x scripts/build_puppy_ros2_cli.sh
./scripts/build_puppy_ros2_cli.sh
```

This produces:
* Docker image: puppy-ros2-cli:humble

Verify the image can resolve the custom message:

```bash
docker run -it --rm puppy-ros2-cli:humble bash -lc \
"source /opt/ros/humble/setup.bash && source /ws/install/setup.bash && ros2 interface show puppy_control_msgs/msg/Velocity"
```

Expected output will be:
```bash
* float32 x
* float32 y
* float32 yaw_rate
```

You can also publish a safe TURN command to verifiy as well:

```bash
docker run -it --rm --net=host \
  -e ROS_DOMAIN_ID=0 \
  -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  puppy-ros2-cli:humble bash -lc \
"source /opt/ros/humble/setup.bash && source /ws/install/setup.bash && \
(ros2 topic pub -r 10 /puppy_control/velocity_move puppy_control_msgs/msg/Velocity '{x: 0.0, y: 0.0, yaw_rate: 0.3}' & PID=\$!; sleep 2; kill \$PID >/dev/null 2>&1 || true) && \
ros2 topic pub -1 /puppy_control/velocity_move puppy_control_msgs/msg/Velocity '{x: 0.0, y: 0.0, yaw_rate: 0.0}'"
```

---

### Step 3 — Start Intent Ingress on DGX (HTTP → File Bus)

On DGX:

```bash

cd intent-ingress
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 intent_ingress.py

```

**Health check:**

```bash
curl -s http://127.0.0.1:7071/healthz
```

This service writes intents atomically to:

- `/tmp/vla_intent.json`

---

### Step 4 — Start Control Daemon on DGX (File Bus → ROS 2)

In a new DGX terminal:

```bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
cd control-daemon
python3 pave_control_daemon_mvp.py
```

---

### Step 5 — Start vLLM (DGX inference backend)

Start your vLLM server (example — adjust to your environment):

- Model: `llava-hf/llava-v1.6-mistral-7b-hf`
- Must expose an OpenAI-compatible endpoint (e.g., `http://localhost:8000/v1`)
> Your exact vLLM command depends on DGX and local deployment policy.  
> Confirm vLLM is reachable from live-vlm-webui before proceeding.

For example:

```bash
vllm serve llava-hf/llava-v1.6-mistral-7b-hf --port 8000 --dtype auto
```


---

### Step 6 — Start live-vlm-webui (Observability UI)

On DGX:

```bash
cd ui/live-vlm-webui
```

**Important:** this updated WebUI includes a server-side hook that POSTs intents to ingress.

Ensure the env is set:

```bash
export INTENT_INGRESS_URL="http://127.0.0.1:7071/intent"
```

Then start the WebUI using your existing method (pip/python entrypoint).

**Expected behavior:**

- WebUI shows inference output (`STOP` / `TROT`)
- `intent_ingress.py` logs: `POST /intent 200`
- `/tmp/vla_intent.json` updates
- Control daemon triggers robot motion

---

## 5) Manual End-to-End Test (Without WebUI)

You can validate the control path even without VLM:
### TROT

```bash
curl -s -X POST http://127.0.0.1:7071/intent \\
  -H 'Content-Type: application/json' \\
  -d '{"text":"TROT"}'
```

### STOP (includes posture reset to standing)

```bash
curl -s -X POST http://127.0.0.1:7071/intent \\
  -H 'Content-Type: application/json' \\
  -d '{"text":"STOP"}'
```

---

## 6) Recommended MVP Prompt (Stable Output)

To maximize downstream reliability, the prompt should force a single-token intent:

**Example prompt (MVP):**

> Output exactly ONE word: `TROT` or `STOP`. No other text.

**Optional gesture-based demo:**

- Thumbs-up gesture → `TROT`
- Open hand (five fingers) → `STOP`

---

## 7) Current MVP Status (Ver1)

✅ Working:

- STOP / TROT end-to-end control
- Remote control via DGX using ROS 2 over LAN
- Server-side webui → ingress integration

⚠️ Known limitations:

- `vx` straight walking remains debug / platform-dependent
- Turning via `velocity_move yaw` can be unstable at high yaw values (may cause imbalance)
- FastDDS can occasionally enter a bad state (see runbook)

---

## 8) Runbook (Most Common Failure)

### Symptom

Robot suddenly stops responding even though messages appear delivered.

### Common log

`RTPS_READER_HISTORY Error ... cannot be resized`

### Fix (fastest)

- Restart PuppyPi `puppy_control.launch.py`

- Optionally:

  ```bash

  ros2 daemon stop || true
  ros2 daemon start

  ```

---

## 9) Why this MVP matters

This MVP is not trying to replace commercial robotics platforms.  

Its value is in **lowering the barrier** for Physical-AI experimentation:

- cheap/accessible robot hardware
- edge inference (reduced cloud dependence)
- open tooling and reproducible workflow
- clear separation between Observability UI and Control Daemon

---

## License

MIT (or your preferred OSS license)