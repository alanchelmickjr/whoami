# K-1 Arm Control Guide

Complete guide for using the K-1 Booster's arm control system with face recognition and gestures.

## Overview

The K-1 Booster arm control system provides:
- **ROS2 Integration**: Direct control of K-1's force-controlled dual-encoder joints
- **Predefined Gestures**: Wave, point, and custom arm movements
- **Face Recognition Integration**: Automatic gestures during greetings
- **Simulation Mode**: Test without ROS2 for development

## Hardware Specifications

The K-1 Booster has:
- **22 DOF total**: Legs (6×2), Arms (4×2), Head (2)
- **4-DOF Arms**: Shoulder pitch/roll, elbow, wrist per arm
- **Force-Controlled Joints**: Dual-encoder joints with torque feedback
- **ROS2 Communication**: Standard ROS2 trajectory control

## Quick Start

### 1. Basic Usage

```python
from whoami.yolo_face_recognition import K1FaceRecognitionSystem

# Create system with arm control enabled
system = K1FaceRecognitionSystem(
    enable_arm_control=True,  # Enable arm gestures
    enable_voice=True         # Enable voice interaction
)

# Start system
system.start()

# Process frames with wave gesture
results = system.process_frame(
    frame=camera_frame,
    greet_known=True,
    wave_on_greet=True  # Wave when greeting known people
)

# Manual wave
system.wave()

# Cleanup
system.stop()
```

### 2. Using the Demo

```bash
# Run with wave gestures
python3 examples/k1_yolo_demo.py --wave

# Run in simulation mode (no ROS2)
python3 examples/k1_yolo_demo.py --wave --no-display
```

## Configuration

### ROS2 Topic Names

The default configuration uses placeholder topic names. You need to configure these to match your K-1's actual ROS2 topics:

```python
from whoami.k1_arm_controller import K1ArmController, ArmConfig

# Create custom configuration
config = ArmConfig(
    # Update these with your K-1's actual ROS2 topics
    joint_state_topic="/k1/joint_states",
    trajectory_topic_left="/k1/left_arm_controller/joint_trajectory",
    trajectory_topic_right="/k1/right_arm_controller/joint_trajectory",

    # Update these with your K-1's actual joint names
    left_arm_joints=[
        "left_shoulder_pitch",
        "left_shoulder_roll",
        "left_elbow",
        "left_wrist"
    ],
    right_arm_joints=[
        "right_shoulder_pitch",
        "right_shoulder_roll",
        "right_elbow",
        "right_wrist"
    ]
)

# Create controller with custom config
controller = K1ArmController(config=config)
```

### Finding Your K-1's Topics

To find the actual ROS2 topics on your K-1:

```bash
# List all active topics
ros2 topic list

# Show joint state message structure
ros2 topic echo /joint_states

# Show trajectory topics
ros2 topic list | grep trajectory
```

### Safety Limits

Configure safe range limits for each joint:

```python
config = ArmConfig(
    # Shoulder pitch: forward/back (degrees)
    shoulder_pitch_limits=(-90, 180),

    # Shoulder roll: out/in (degrees)
    shoulder_roll_limits=(-45, 180),

    # Elbow: straight to fully bent (degrees)
    elbow_limits=(0, 150),

    # Wrist: rotation (degrees)
    wrist_limits=(-90, 90)
)
```

## Gestures

### Wave Gesture

The wave gesture:
1. Raises arm to shoulder height
2. Rotates wrist left-right 3 times
3. Returns arm to rest position

```python
from whoami.k1_arm_controller import create_k1_arm_controller, ArmSide

controller = create_k1_arm_controller()
controller.start()

# Wave with right arm (default)
controller.wave(ArmSide.RIGHT)

# Wave with left arm
controller.wave(ArmSide.LEFT)

# Wave with both arms
controller.wave(ArmSide.BOTH)
```

### Customizing Wave Parameters

```python
from whoami.k1_arm_controller import ArmConfig

config = ArmConfig(
    wave_duration=3.0,    # Total duration in seconds
    wave_cycles=3,        # Number of left-right cycles
    default_duration=2.0  # Duration for other movements
)
```

### Point Gesture

Point at a specific 3D coordinate:

```python
# Point at coordinates (x, y, z) in meters
controller.point(
    direction=(1.0, 0.5, 2.0),  # Point forward and slightly up
    arm=ArmSide.RIGHT
)
```

### Rest Position

Return arms to rest position (at sides):

```python
controller.move_to_rest(ArmSide.BOTH)
```

## Simulation Mode

For development without ROS2 or on non-Jetson systems:

```python
# Force simulation mode
controller = create_k1_arm_controller(simulate=True)

# Gestures will log actions without actual movement
controller.wave(ArmSide.RIGHT)
# Output: [SIMULATED] Moving right arm through 6 waypoints
```

The system automatically uses simulation mode if ROS2 is not installed.

## Integration with Face Recognition

### Automatic Wave on Greeting

```python
system = K1FaceRecognitionSystem(
    enable_voice=True,
    enable_arm_control=True
)

# Process frame with wave
results = system.process_frame(
    frame=frame,
    greet_known=True,
    wave_on_greet=True  # Wave when greeting known people
)
```

### Custom Greeting Behavior

```python
for result in results:
    if result.name != "Unknown":
        # Get time since last seen
        time_since = system.recognizer.get_time_since_last_seen(result.name)

        # Wave only if haven't seen them in a while
        if time_since > 3600:  # 1 hour
            system.wave()
            system.voice.say(f"Hi {result.name}! Long time no see!")
```

## Advanced Usage

### Custom Arm Trajectories

Create custom multi-waypoint movements:

```python
from whoami.k1_arm_controller import JointPosition

# Define waypoints
positions = [
    JointPosition(
        shoulder_pitch=0,
        shoulder_roll=90,
        elbow=90,
        wrist=0
    ),
    JointPosition(
        shoulder_pitch=45,
        shoulder_roll=90,
        elbow=120,
        wrist=-30
    ),
    # ... more waypoints
]

# Define timing (seconds from start for each waypoint)
durations = [1.0, 2.0, 3.0]

# Execute trajectory
controller._send_trajectory(ArmSide.RIGHT, positions, durations)
```

### Checking Arm Position

```python
# Get current position
position = controller.get_current_position(ArmSide.RIGHT)
if position:
    print(f"Shoulder pitch: {position.shoulder_pitch}°")
    print(f"Shoulder roll: {position.shoulder_roll}°")
    print(f"Elbow: {position.elbow}°")
    print(f"Wrist: {position.wrist}°")

# Check if at rest
if controller.is_at_rest(ArmSide.RIGHT, tolerance=5.0):
    print("Right arm is at rest")
```

## Installation

### On K-1 Booster (Jetson Orin NX)

The `k1_setup.sh` script automatically installs ROS2:

```bash
./k1_setup.sh
```

This installs:
- ROS2 Humble
- `sensor_msgs` package
- `trajectory_msgs` package
- Python ROS2 bindings

### Manual ROS2 Installation

```bash
# Add ROS2 repository
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository universe

# Install ROS2 Humble
sudo apt install -y \
    ros-humble-ros-base \
    ros-humble-sensor-msgs \
    ros-humble-trajectory-msgs \
    python3-colcon-common-extensions

# Source ROS2 in your shell
echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc
source ~/.bashrc
```

### Testing Installation

```bash
# Check ROS2
ros2 --version

# Test arm controller
python3 -c "
from whoami.k1_arm_controller import create_k1_arm_controller
controller = create_k1_arm_controller()
print('Arm controller created successfully')
"
```

## Troubleshooting

### ROS2 Not Found

If you see "K-1 arm control not available":

```bash
# Check if ROS2 is sourced
which ros2

# Source ROS2 manually
source /opt/ros/humble/setup.bash

# Add to .bashrc permanently
echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc
```

### Topics Not Found

If arm control can't find topics:

```bash
# List all topics to find the correct names
ros2 topic list

# Update your ArmConfig with the actual topic names
```

### Joint Names Mismatch

If you get errors about joint names:

```bash
# Check joint state message
ros2 topic echo /joint_states

# Look for the actual joint names in the output
# Update ArmConfig.left_arm_joints and right_arm_joints
```

### Simulation Mode Always Active

If arm control is always in simulation mode:

```python
# Check if ROS2 is available
import rclpy
print(rclpy.__version__)  # Should print ROS2 version

# Force real mode (will fail if ROS2 not available)
controller = K1ArmController(simulate=False)
```

## Performance

### Latency

- **Gesture start**: < 50ms (ROS2 message publish)
- **Wave duration**: ~3 seconds (configurable)
- **Point duration**: ~2 seconds (configurable)

### Resource Usage

- **CPU**: Minimal (<5% on Orin NX)
- **Memory**: ~50MB for ROS2 node
- **Network**: Local ROS2 messages only

## Safety

### Built-in Safety Features

1. **Joint Limits**: All movements clamped to safe ranges
2. **Smooth Trajectories**: No sudden jerky movements
3. **Simulation Mode**: Safe testing without hardware

### Best Practices

- Always test new gestures in simulation mode first
- Configure conservative joint limits initially
- Monitor arm movements during first tests
- Keep emergency stop accessible
- Don't modify safety limits without understanding K-1 mechanics

## Example: Full Integration

Complete example with face recognition, voice, and arm control:

```python
#!/usr/bin/env python3
from whoami.yolo_face_recognition import K1FaceRecognitionSystem
import cv2

# Create system
system = K1FaceRecognitionSystem(
    yolo_model='yolov8n.pt',
    deepface_model='Facenet',
    enable_voice=True,
    enable_arm_control=True,
    camera_resolution=(1280, 720)
)

# Start
if not system.start():
    print("Failed to start system")
    exit(1)

print("K-1 Face Recognition System running!")
print("- Will greet known people")
print("- Will wave when greeting")
print("- Press 'q' to quit")

# Main loop
try:
    while True:
        # Get frame
        frame = system.camera.get_frame()
        if frame is None:
            continue

        # Process with gestures
        results = system.process_frame(
            frame=frame,
            ask_unknown=True,
            greet_known=True,
            wave_on_greet=True
        )

        # Display (optional)
        for result in results:
            x1, y1, x2, y2 = result.bbox
            color = (0, 255, 0) if result.name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, result.name, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow('K-1 Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    system.stop()
    cv2.destroyAllWindows()
    print("System stopped")
```

## API Reference

See `whoami/k1_arm_controller.py` for complete API documentation.

### Key Classes

- `K1ArmController`: Main arm controller class
- `ArmConfig`: Configuration dataclass
- `JointPosition`: Joint angle container
- `ArmSide`: Enum for left/right/both

### Key Methods

- `controller.start()`: Initialize ROS2 connection
- `controller.stop()`: Cleanup and shutdown
- `controller.wave(arm)`: Wave gesture
- `controller.point(direction, arm)`: Point gesture
- `controller.move_to_rest(arm)`: Return to rest position

## See Also

- [K-1 Booster Setup Guide](K1_BOOSTER_SETUP.md)
- [Voice Interaction Guide](VOICE_INTERACTION_GUIDE.md)
- [Face Recognition API](API_REFERENCE.md)
- [ROS2 Documentation](https://docs.ros.org/en/humble/)

## Support

For issues with K-1 arm control:
1. Check ROS2 is properly installed and sourced
2. Verify topic names match your K-1
3. Test in simulation mode first
4. Check joint state messages are publishing
5. Open GitHub issue with `k1-arm-control` tag
