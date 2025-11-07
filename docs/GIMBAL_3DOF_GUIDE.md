# 3DOF Gimbal System for Orbital 3D Scanning

Complete guide to the 3-degree-of-freedom coordinated gimbal system for WhoAmI robot.

## Table of Contents

- [Overview](#overview)
- [Mechanical Design](#mechanical-design)
- [Kinematics](#kinematics)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Scanning Strategies](#scanning-strategies)
- [Calibration](#calibration)
- [Troubleshooting](#troubleshooting)

---

## Overview

The 3DOF gimbal system enables the WhoAmI robot to perform **complete 360° orbital scanning** of objects. Unlike traditional gimbals that only rotate in place, this system creates circular paths in 3D space with the camera always facing inward toward the scan target.

### Key Capabilities

- ✅ **Orbital Motion**: Camera orbits around objects in any plane
- ✅ **Inward-Facing**: Camera always points at scan target
- ✅ **Full 360° Coverage**: Complete spherical scanning capability
- ✅ **Coordinated 3-Axis Motion**: All servos work together for smooth paths
- ✅ **Rolling Scanner**: Camera can spin during scanning for enhanced depth data

### Applications

- **3D Object Scanning**: Complete geometry capture from all angles
- **Face Recognition**: Multi-angle face capture for better recognition
- **Tool Creation**: Scan objects to design custom grippers and tools
- **Environment Mapping**: 3D mapping of surrounding workspace
- **Object Inspection**: Detailed examination from multiple viewpoints

---

## Mechanical Design

### Kinematic Chain

```
                    ┌─────────┐
                    │ Servo 1 │  ← Base/Yaw (rotates around Z-axis)
                    └────┬────┘
                         │ (base_height)
                         │
                    ┌────┴────┐
                    │ Servo 2 │  ← Shoulder/Pitch (rotates around Y-axis)
                    └────┬────┘
                         │
                         ├──────────┐ (arm_length)
                         │          │
                    ┌────┴────┐     │
                    │ Servo 3 │     │  ← Wrist/Roll (rotates around X-axis)
                    └────┬────┘     │
                         │          │
                         │    ┌─────┴──────┐
                         └───→│  OAK-D     │  ← Camera
                              │  Camera    │
                              └────────────┘
```

### Three Axes Explained

1. **Servo 1 (Yaw/Base)**
   - Rotates entire assembly around vertical axis
   - 360° continuous rotation
   - Controls horizontal orientation

2. **Servo 2 (Pitch/Shoulder)**
   - Extends arm horizontally from base
   - Swings arm up/down
   - Controls elevation angle

3. **Servo 3 (Roll/Wrist)**
   - Rotates camera around optical axis
   - Spins camera "like a fan"
   - Captures depth data from different roll angles

### Physical Parameters

Configure in `config/gimbal_3dof_config.json`:

```json
{
  "kinematics": {
    "base_height": 0.10,      // 10cm from Servo 1 to Servo 2
    "arm_length": 0.15,       // 15cm from Servo 2 to Servo 3
    "camera_offset": 0.05,    // 5cm from Servo 3 to camera center
    "yaw_limits": [-180, 180],
    "pitch_limits": [-90, 90],
    "roll_limits": [-180, 180]
  }
}
```

**Adjust these values for your physical build!**

### Hardware Requirements

- **3x Feetech SCS/STS Servos** (continuous rotation capable)
- **Serial Communication**: USB-to-Serial adapter or Jetson UART
- **Power Supply**: 6-8V for servos (separate from Jetson)
- **Mechanical Frame**: 3D printed or machined
- **OAK-D S3 Camera**: Depth-sensing RGB camera

---

## Kinematics

### Forward Kinematics

Computes camera pose from servo angles:

**Given**: Joint angles (yaw, pitch, roll)
**Returns**: Camera position + orientation in world frame

```python
from whoami.gimbal_3dof import Kinematics3DOF, JointAngles

# Create kinematics solver
kinematics = Kinematics3DOF(params)

# Compute camera pose
angles = JointAngles(yaw=45, pitch=30, roll=0)
pose = kinematics.forward_kinematics(angles)

print(f"Camera position: {pose.position}")
print(f"Camera looking: {pose.forward}")
print(f"Camera up vector: {pose.up}")
```

### Inverse Kinematics

Computes servo angles to achieve desired camera pose:

**Given**: Target point + camera position
**Returns**: Required joint angles (or None if unreachable)

```python
# Where you want the camera to look
target = np.array([0.3, 0.0, 0.0])

# Where you want the camera to be
camera_pos = np.array([0.2, 0.2, 0.1])

# Solve for angles
angles = kinematics.inverse_kinematics(
    target_position=target,
    camera_position=camera_pos,
    camera_roll=0.0
)

if angles:
    print(f"Move to: {angles}")
else:
    print("Position unreachable")
```

### Reachable Workspace

The camera can reach points within a **spherical shell**:

- **Inner radius**: `base_height`
- **Outer radius**: `base_height + arm_length + camera_offset`
- **Height range**: Depends on pitch limits

```python
# Check if point is reachable
point = np.array([0.25, 0.15, 0.05])
reachable = kinematics.check_reachability(point)
print(f"Reachable: {reachable}")
```

---

## Configuration

### Configuration File

Location: `config/gimbal_3dof_config.json`

#### Key Sections:

**1. Kinematics** - Physical dimensions
```json
{
  "base_height": 0.10,
  "arm_length": 0.15,
  "camera_offset": 0.05
}
```

**2. Servo IDs** - Feetech servo assignments
```json
{
  "servo_ids": {
    "yaw": 1,
    "pitch": 2,
    "roll": 3
  }
}
```

**3. Motion** - Speed and smoothness
```json
{
  "motion": {
    "max_velocities": [60, 60, 120],  // deg/s per axis
    "interpolation_steps": 10
  }
}
```

**4. Scanning** - Default scan parameters
```json
{
  "scanning": {
    "default_radius": 0.3,  // 30cm orbital radius
    "default_num_points": 36  // 36 positions (10° steps)
  }
}
```

### Calibration

Must calibrate before first use:

```python
from whoami.gimbal_3dof_controller import Gimbal3DOFController

gimbal = Gimbal3DOFController()
gimbal.connect()
gimbal.calibrate()  # Moves through test positions
gimbal.home()       # Return to center
```

This verifies:
- ✅ All servos respond correctly
- ✅ Movement directions are correct
- ✅ Angle calculations match physical motion
- ✅ Home position is accurate

---

## Usage Examples

### Basic Movement

```python
from whoami.gimbal_3dof_controller import Gimbal3DOFController
from whoami.gimbal_3dof import JointAngles
import numpy as np

# Initialize
gimbal = Gimbal3DOFController()
gimbal.connect()

# Move to specific angles
angles = JointAngles(yaw=45, pitch=30, roll=0)
gimbal.move_to_angles(angles)

# Move to look at specific point
target = np.array([0.3, 0.0, 0.0])  # 30cm forward
camera_pos = np.array([0.2, 0.2, 0.1])  # Orbit position
gimbal.move_to_pose(target, camera_pos)

# Return home
gimbal.home()
gimbal.disconnect()
```

### Horizontal Orbital Scan

Scan object from all sides (equatorial orbit):

```python
from whoami.gimbal_3dof_controller import Gimbal3DOFController
import numpy as np

gimbal = Gimbal3DOFController()
gimbal.connect()

# Object on desk 30cm away
object_center = np.array([0.3, 0.0, 0.0])

# Orbit 20cm around object
scan_radius = 0.20

# Capture at 36 positions (every 10°)
def capture_callback(position_index, angles):
    print(f"Capturing at position {position_index}")
    # Your capture code here

gimbal.scan_horizontal_orbit(
    center=object_center,
    radius=scan_radius,
    num_points=36,
    scan_callback=capture_callback
)

gimbal.disconnect()
```

### Spherical Scan

Complete spherical coverage with multiple elevation rings:

```python
gimbal.scan_spherical(
    center=object_center,
    radius=scan_radius,
    num_rings=5,           # 5 elevation rings
    points_per_ring=36,    # 36 points each
    scan_callback=capture_callback
)
```

Creates orbits at different heights to cover entire sphere.

### Rolling Camera Scan

Spin camera while maintaining position (enhanced depth capture):

```python
# Position camera 20cm from object
camera_pos = np.array([0.3, 0.0, 0.0])
object_center = np.array([0.2, 0.0, 0.0])

# Spin camera 360° over 12 seconds
def capture_during_roll(step, roll_angle):
    print(f"Roll: {roll_angle:.1f}°")
    # Capture depth frame

gimbal.continuous_roll_scan(
    target_point=object_center,
    camera_position=camera_pos,
    roll_speed=30.0,      # 30 deg/s
    duration=12.0,        // 12 seconds = 360°
    capture_callback=capture_during_roll
)
```

### Combined Orbital + Roll Scan

Ultra-detailed: orbit with multiple roll angles at each position:

```python
from whoami.gimbal_3dof import OrbitalPathGenerator

path_gen = OrbitalPathGenerator(gimbal.kinematics)

# Generate 12 orbital positions
angles_list = []
for angle in range(0, 360, 30):  # Every 30°
    angle_rad = np.deg2rad(angle)

    # Position on orbit
    position = object_center + np.array([
        scan_radius * np.cos(angle_rad),
        scan_radius * np.sin(angle_rad),
        0
    ])

    # Solve for multiple roll angles
    for roll in [0, 45, 90, 135]:
        angles = gimbal.kinematics.inverse_kinematics(
            target_position=object_center,
            camera_position=position,
            camera_roll=roll
        )
        if angles:
            angles_list.append(angles)

# Execute trajectory
gimbal.execute_trajectory(angles_list, smooth=True)
```

---

## Scanning Strategies

### 1. Quick Scan (12 positions, horizontal)

**Use when:**
- Quick preview needed
- Limited time
- Simple objects

**Coverage:**
- Horizontal ring only
- 30° angular spacing
- ~1-2 minutes

```python
gimbal.scan_horizontal_orbit(
    center=object_center,
    radius=0.20,
    num_points=12
)
```

### 2. Standard Scan (36 positions, horizontal)

**Use when:**
- Good quality needed
- Symmetrical objects
- Standard workflow

**Coverage:**
- Horizontal ring
- 10° angular spacing
- ~3-5 minutes

```python
gimbal.scan_horizontal_orbit(
    center=object_center,
    radius=0.20,
    num_points=36
)
```

### 3. Detailed Scan (spherical, 3-5 rings)

**Use when:**
- Complete coverage needed
- Complex geometry
- Top/bottom details important

**Coverage:**
- Multiple elevation rings
- Full spherical coverage
- ~10-15 minutes

```python
gimbal.scan_spherical(
    center=object_center,
    radius=0.20,
    num_rings=5,
    points_per_ring=24
)
```

### 4. Ultra-Detailed (orbit + roll)

**Use when:**
- Maximum detail needed
- Creating tools/molds
- High-precision models

**Coverage:**
- Orbital positions
- Multiple roll angles each
- ~20-30 minutes

```python
# 12 orbital positions × 4 roll angles = 48 total captures
scan_with_rotating_camera(
    object_center=object_center,
    orbital_positions=12,
    roll_angles=[0, 45, 90, 135]
)
```

### Scan Parameter Guidelines

| Scenario | Positions | Roll Angles | Time | Quality |
|----------|-----------|-------------|------|---------|
| Preview | 12 | 1 | 1-2 min | Low |
| Standard | 36 | 1 | 3-5 min | Medium |
| Detailed | 5 rings × 24 | 1 | 10-15 min | High |
| Ultra | 12 × 4 rolls | 4 | 20-30 min | Maximum |

---

## Calibration

### Initial Calibration Procedure

1. **Home Position**
   ```python
   gimbal.home()  # All servos to 0°
   ```

2. **Verify Home**
   - Camera should be pointing forward
   - Arm should be horizontal
   - No mechanical binding

3. **Test Each Axis**
   ```python
   # Test yaw (should rotate base)
   gimbal.move_to_angles(JointAngles(45, 0, 0))

   # Test pitch (should swing arm up)
   gimbal.move_to_angles(JointAngles(0, 45, 0))

   # Test roll (should spin camera)
   gimbal.move_to_angles(JointAngles(0, 0, 90))
   ```

4. **Verify Directions**
   - Positive yaw: counter-clockwise (viewed from above)
   - Positive pitch: arm swings up
   - Positive roll: camera rotates... (define your convention)

5. **Set Zero Offsets**
   If servos are misaligned, adjust in config:
   ```json
   "zero_positions": [5, -3, 2]  // [yaw_offset, pitch_offset, roll_offset]
   ```

6. **Measure Actual Dimensions**
   Measure your physical build:
   ```json
   "base_height": 0.12,    // Measured: 12cm
   "arm_length": 0.14,     // Measured: 14cm
   "camera_offset": 0.06   // Measured: 6cm
   ```

7. **Test Kinematics**
   ```python
   # Place object at known position
   object_pos = np.array([0.25, 0.0, 0.0])

   # Generate orbit
   waypoints = path_gen.generate_horizontal_orbit(
       center=object_pos,
       radius=0.10,
       num_points=4  // Just 4 positions for test
   )

   # Execute and verify camera always faces object
   gimbal.execute_trajectory(waypoints)
   ```

8. **Mark as Calibrated**
   ```json
   "calibration": {
     "calibrated": true,
     "calibration_date": "2025-01-15T10:30:00Z"
   }
   ```

### Periodic Recalibration

Recalibrate when:
- ❌ Servos are replaced
- ❌ Mechanical adjustments made
- ❌ Camera aims incorrectly during scans
- ❌ Reach limits have changed
- ⚠ Every 3-6 months (preventive)

---

## Troubleshooting

### Camera Not Pointing At Target

**Symptoms:**
- During orbit, camera doesn't track object
- Camera points wrong direction

**Causes & Solutions:**

1. **Wrong servo directions**
   ```json
   // Try flipping directions
   "directions": [-1, 1, 1]  // Flip yaw
   ```

2. **Incorrect dimensions**
   - Measure actual arm lengths
   - Update config with real values

3. **Zero position offsets**
   ```json
   "zero_positions": [10, -5, 0]  // Adjust
   ```

### Position Unreachable

**Symptoms:**
```
WARNING: Position unreachable: needs 0.350m, have 0.300m
```

**Solutions:**

1. **Increase orbital radius**
   ```python
   scan_radius = 0.15  // Reduce from 0.25
   ```

2. **Move object closer**
   - Objects should be within reach envelope

3. **Check workspace limits**
   ```python
   # Verify max reach
   max_reach = (params.base_height +
                params.arm_length +
                params.camera_offset)
   print(f"Max reach: {max_reach}m")
   ```

### Jerky Motion

**Symptoms:**
- Servos move in stutters
- Not smooth motion

**Solutions:**

1. **Increase interpolation**
   ```json
   "interpolation_steps": 20  // Was 10
   ```

2. **Reduce speed**
   ```json
   "max_velocities": [30, 30, 60]  // Slower
   ```

3. **Check serial communication**
   - Use shorter USB cable
   - Check baudrate matches servos

### Servos Not Responding

**Symptoms:**
- `Servo not responding` errors
- Timeout during connection

**Solutions:**

1. **Check power**
   - Servos need 6-8V
   - Separate power supply from Jetson

2. **Verify serial port**
   ```bash
   ls /dev/ttyUSB*
   # or
   ls /dev/ttyTHS*  # Jetson UART
   ```

3. **Check servo IDs**
   - Use Feetech debugging tool
   - Verify IDs match config

4. **Test communication**
   ```python
   # Ping individual servo
   gimbal.servo_driver.ping(1)  # Should return True
   ```

### Mechanical Binding

**Symptoms:**
- Servos stall
- Position errors
- Temperature warnings

**Solutions:**

1. **Check for collisions**
   - Camera hitting base?
   - Cables wrapped around joints?

2. **Lubricate joints**
   - Use dry lubricant on pivots

3. **Adjust limits**
   ```json
   "pitch_limits": [-80, 80]  // Reduce from [-90, 90]
   ```

4. **Cable management**
   - Use slip rings for continuous rotation
   - Or limit rotation range to prevent wrapping

---

## Advanced Topics

### Custom Scanning Patterns

Create your own orbital patterns:

```python
from whoami.gimbal_3dof import OrbitalPathGenerator

path_gen = OrbitalPathGenerator(kinematics)

# Custom spiral scan
waypoints = path_gen.generate_spiral_scan(
    center=object_center,
    radius=0.20,
    height_range=(-0.10, 0.10),
    num_revolutions=3,
    points_per_revolution=24
)

gimbal.execute_trajectory(waypoints)
```

### Integration with 3D Scanner

Complete workflow:

```python
from whoami.scanner_3d import Scanner3D
from whoami.gimbal_3dof_controller import Gimbal3DOFController

# Initialize both systems
scanner = Scanner3D()
gimbal = Gimbal3DOFController()

scanner.initialize()
gimbal.connect()

# Scan with point cloud capture
point_clouds = []

def capture_at_position(idx, angles):
    cloud = scanner.capture_point_cloud()
    point_clouds.append(cloud)

gimbal.scan_horizontal_orbit(
    center=object_center,
    radius=0.20,
    num_points=36,
    scan_callback=capture_at_position
)

# Merge all point clouds
merged = scanner.merge_point_clouds(point_clouds)

# Generate mesh
mesh = scanner.generate_mesh(merged)

# Save
scanner.save_mesh(mesh, "scanned_object.stl")
```

### Multi-Object Scanning

Scan multiple objects in sequence:

```python
objects = [
    {"name": "mug", "center": [0.30, 0.10, 0.0], "radius": 0.15},
    {"name": "tool", "center": [0.35, -0.10, 0.0], "radius": 0.12},
    {"name": "part", "center": [0.25, 0.0, 0.05], "radius": 0.10}
]

for obj in objects:
    print(f"Scanning {obj['name']}...")

    gimbal.scan_horizontal_orbit(
        center=np.array(obj['center']),
        radius=obj['radius'],
        num_points=36,
        scan_callback=lambda idx, ang: capture_and_save(obj['name'], idx)
    )

    gimbal.home()  # Reset between objects
    time.sleep(1)
```

---

## Quick Reference

### Essential Commands

```python
# Connect
gimbal = Gimbal3DOFController()
gimbal.connect()

# Move to angles
gimbal.move_to_angles(JointAngles(45, 30, 0))

# Move to pose
gimbal.move_to_pose(target, camera_pos)

# Horizontal scan
gimbal.scan_horizontal_orbit(center, radius, num_points)

# Spherical scan
gimbal.scan_spherical(center, radius, num_rings, points_per_ring)

# Home
gimbal.home()

# Emergency stop
gimbal.emergency_stop()

# Disconnect
gimbal.disconnect()
```

### Configuration Files

- **Gimbal Config**: `config/gimbal_3dof_config.json`
- **Scanner Config**: `config/scanner_3d_config.json`
- **Examples**: `examples/orbital_3d_scanning.py`

### Key Parameters

- **Workspace**: ~0.05m to 0.30m radius
- **Scan Radius**: 0.15-0.25m typical
- **Scan Speed**: 200-300 servo speed units
- **Positions**: 12 (quick), 36 (standard), 72+ (detailed)

---

## Support

### Logs

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Diagnostics

Check gimbal status:
```python
status = gimbal.get_status()
print(json.dumps(status, indent=2))
```

### Help

- **Documentation**: `docs/`
- **Examples**: `examples/orbital_3d_scanning.py`
- **Issues**: GitHub repository

---

**Version**: 1.0
**Last Updated**: 2025-11-07
**For**: Jetson Orin Nano + Feetech Servos + OAK-D S3
