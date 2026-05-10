# Stable Runbook (MVP)

## PuppyPi side
1) Export environment:
```bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

2) Start ROS 2 daemon (optional, helps stale state):
```bash
ros2 daemon stop || true
ros2 daemon start
```

3) Launch controller:
```bash
source /opt/ros/humble/setup.bash
ros2 launch puppy_control puppy_control.launch.py
```

## DGX side
1) Export environment:
```bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

2) Start Intent Ingress:
```bash
cd intent-ingress
python3 intent_ingress.py
```

3) Start Control Daemon:
```bash
cd control-daemon
python3 pave_control_daemon_mvp.py
```

## Common failure: FastDDS RTPS history error
Symptom: Robot stops responding even though messages appear to be delivered.
Log example:
`RTPS_READER_HISTORY Error ... cannot be resized`

Fix:
- Restart `puppy_control.launch.py`
- (Optional) `ros2 daemon stop/start`
