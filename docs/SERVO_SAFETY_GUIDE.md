# Servo Safety and Fault Tolerance Guide

**CRITICAL: "If the neck/head gimbal fails, the robot does not fail"**

## Overview

The servo safety system ensures gimbal servo failures won't crash the robot. It provides:

- Real-time health monitoring
- Automatic failure detection
- Safe fallback positions
- Error recovery
- Graceful degradation
- Emergency stop capability

## Quick Start

### Basic Safe Usage

```python
from whoami.gimbal_safe_controller import SafeGimbalController

# Create safe gimbal controller
with SafeGimbalController() as gimbal:
    # Check if operational
    if gimbal.operational:
        # Perform safe movement
        gimbal.move_to_angles(yaw=45.0, pitch=0.0, roll=0.0)
    else:
        print("Gimbal not available - using fallback")
```

### Calibration

```bash
# Run safe calibration utility
python tools/calibrate_gimbal.py

# Test safety only
python tools/calibrate_gimbal.py --test-only

# Simulate calibration
python tools/calibrate_gimbal.py --simulate
```

## Architecture

### Three-Layer Safety System

```
┌─────────────────────────────────────────────────┐
│ SafeGimbalController                            │
│ - Fault-tolerant operations                     │
│ - Graceful degradation                          │
│ - User-facing API                               │
└───────────────────┬─────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────┐
│ ServoSafetyMonitor                              │
│ - Real-time health monitoring                   │
│ - Failure detection                             │
│ - Recovery attempts                             │
└───────────────────┬─────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────┐
│ Gimbal3DOFController / FeetechServoController   │
│ - Low-level hardware control                    │
└─────────────────────────────────────────────────┘
```

## Components

### 1. ServoSafetyMonitor

Real-time servo health monitoring system.

**Features:**
- Monitors temperature, current, voltage, position
- Detects communication failures
- Tracks position errors
- Maintains failure history
- Provides safe fallback positions

**Usage:**

```python
from whoami.servo_safety import ServoSafetyMonitor, SafetyLimits

# Initialize
safety = ServoSafetyMonitor(
    servo_ids=[1, 2, 3],
    safety_limits=SafetyLimits(
        max_temperature=75.0,
        max_current=1000.0,
        max_position_error=10.0
    )
)

# Start monitoring
safety.start()

# Update status (from servo reads)
safety.update_servo_status(
    servo_id=1,
    position=45.0,
    temperature=55.0,
    current=450.0,
    communication_success=True
)

# Check health
health = safety.check_servo_health(1)
print(f"Servo 1 health: {health.value}")

# Get diagnostics
diag = safety.get_diagnostics()
safety.print_health_report()
```

### 2. SafeGimbalController

Fault-tolerant wrapper around Gimbal3DOFController.

**Features:**
- All gimbal operations with safety monitoring
- Automatic failure detection and recovery
- Graceful degradation
- Emergency stop
- Capability reporting

**Usage:**

```python
from whoami.gimbal_safe_controller import SafeGimbalController

# Initialize
gimbal = SafeGimbalController()

# Check capabilities
caps = gimbal.get_capabilities()
if caps['can_move']:
    # Safe movement
    gimbal.move_to_angles(0, 0, 0)

# Safe orbital scan
success = gimbal.scan_horizontal_orbit(
    center=np.array([0.25, 0.0, 0.0]),
    radius=0.15,
    num_points=36
)

if not success:
    print("Scan failed - but robot still operational")

# Check status
gimbal.print_status()
```

### 3. Calibrate Gimbal Utility

Safe step-by-step calibration tool.

**Features:**
- Center position calibration
- Range of motion testing
- Angle accuracy verification
- Safety checks at each step
- Emergency stop capability

**Usage:**

```bash
# Full calibration
python tools/calibrate_gimbal.py

# Specific servo only
python tools/calibrate_gimbal.py --servo 1

# Test without calibration
python tools/calibrate_gimbal.py --test-only
```

## Safety Limits

### Default Safety Limits

```python
SafetyLimits(
    max_temperature=75.0,        # Celsius
    max_current=1000.0,          # mA
    max_position_error=10.0,     # degrees
    max_velocity=300.0,          # deg/s
    max_consecutive_errors=3,
    communication_timeout=2.0,   # seconds

    # Warning thresholds
    warning_temperature=65.0,
    warning_current=800.0,
    warning_position_error=5.0
)
```

### Customizing Limits

```python
from whoami.servo_safety import SafetyLimits

# Custom limits for your setup
custom_limits = SafetyLimits(
    max_temperature=70.0,  # More conservative
    max_current=800.0,     # Lower current limit
    max_position_error=5.0 # Stricter positioning
)

gimbal = SafeGimbalController(safety_limits=custom_limits)
```

## Failure Detection

### Failure Types

```python
class FailureMode(Enum):
    COMMUNICATION_LOST  # Servo not responding
    POSITION_ERROR      # Can't reach target position
    OVERHEATING         # Temperature too high
    OVERCURRENT         # Drawing too much current
    STALLED             # Motor stalled (high load)
    MECHANICAL_JAM      # Mechanical obstruction
    CALIBRATION_ERROR   # Calibration issue
    TIMEOUT             # Operation timeout
```

### Automatic Detection

The safety monitor automatically detects:

1. **Communication failures** - Servo not responding
2. **Overheating** - Temperature > max_temperature
3. **Overcurrent** - Current > max_current
4. **Position errors** - Can't reach target position
5. **Stalls** - High load + position error
6. **Consecutive errors** - Too many errors in a row

### Callbacks

```python
def on_failure(servo_id, failures):
    print(f"Servo {servo_id} failed: {failures}")
    # Take action...

def on_recovery(servo_id):
    print(f"Servo {servo_id} recovered")

safety.on_failure_detected = on_failure
safety.on_recovery = on_recovery
```

## Recovery Strategies

### Automatic Recovery

```python
# Enable automatic recovery
gimbal = SafeGimbalController(enable_recovery=True)

# Recovery happens automatically:
# 1. Failure detected
# 2. Move to safe position
# 3. Clear error state
# 4. Test operation
# 5. Resume if successful
```

### Manual Recovery

```python
# Check for failures
failures = safety.detect_failures()

if failures:
    # Attempt recovery
    for servo_id in failures:
        success = safety.attempt_recovery(servo_id)
        if success:
            print(f"Servo {servo_id} recovered")
```

## Safe Positions

### Predefined Safe Positions

```python
# Home position (neutral)
safety.move_to_safe_position("home", servo_controller)

# Park position (tucked away)
safety.move_to_safe_position("park", servo_controller)

# Look forward (functional)
safety.move_to_safe_position("look_forward", servo_controller)
```

### Custom Safe Positions

```python
from whoami.servo_safety import SafePosition

# Define custom position
custom_pos = SafePosition(
    name="look_down",
    positions={1: 0.0, 2: -30.0, 3: 0.0},
    description="Looking down at workspace"
)

# Add to safety system
safety.add_safe_position(custom_pos)

# Use it
safety.move_to_safe_position("look_down", servo_controller)
```

## Graceful Degradation

### Degradation Levels

```python
gimbal = SafeGimbalController()

level = gimbal.get_degradation_level()

if level == 'fully_operational':
    # All capabilities available
    perform_advanced_scanning()

elif level == 'degraded':
    # Limited capabilities - robot still functional
    use_fixed_camera()

else:  # 'failed'
    # No gimbal - core systems still work
    request_manual_camera_adjustment()
```

### Capability Checking

```python
caps = gimbal.get_capabilities()

if caps['can_scan']:
    perform_orbital_scan()
elif caps['can_move']:
    simple_camera_positioning()
else:
    use_fallback_strategy()
```

### Fallback Strategies

When gimbal fails, robot can still:

✓ **Face recognition** (fixed camera)
✓ **Object detection** (limited FOV)
✓ **Spatial awareness** (reduced)
✓ **Brain/reasoning** (unaffected)
✓ **Genesis simulation** (unaffected)

## Emergency Stop

### Triggering Emergency Stop

```python
# Manual emergency stop
gimbal.emergency_stop()

# Automatic on critical failure
# (communication lost, overheating, jam)

# Via keyboard interrupt
# Press Ctrl+C during operation
```

### What Happens

1. All servo motion halted immediately
2. Move to safe position (when possible)
3. Gimbal operations disabled
4. Robot enters degraded mode
5. **Robot remains operational**

### Recovery from Emergency Stop

```python
# Reset after emergency stop
success = gimbal.reset()

if success:
    print("✓ Reset successful - resumed operation")
else:
    print("✗ Reset failed - manual intervention needed")
```

## Calibration Process

### Step-by-Step Calibration

**1. Center Position Calibration**

```
For each servo:
- Move to 0° position
- Verify mechanically centered
- Record actual position
```

**2. Range of Motion Calibration**

```
For each axis:
- Test minimum limit
- Test maximum limit
- Find safe limits if needed
- Record actual range
```

**3. Angle Accuracy Calibration**

```
For each axis:
- Test known angles (-45°, 0°, 45°)
- Measure actual positions
- Compute error statistics
- Record accuracy data
```

### Calibration Safety

Calibration includes safety checks:

- Temperature monitoring (won't calibrate if hot)
- Position error detection
- Communication verification
- Emergency stop capability (Ctrl+C)
- Safe fallback at each step

### Calibration Output

```json
{
  "timestamp": 1234567890.0,
  "servo_ids": {"yaw": 1, "pitch": 2, "roll": 3},
  "calibration": {
    "yaw": {
      "center_position": 0.0,
      "center_verified": true,
      "min_angle": -180.0,
      "max_angle": 180.0,
      "mean_error": 0.5,
      "std_error": 0.3
    },
    ...
  }
}
```

Saved to: `config/gimbal_calibration.json`

## Integration Examples

### With Spatial Awareness

```python
from whoami.gimbal_safe_controller import SafeGimbalController
from whoami.spatial_awareness import SpatialAwarenessSystem

spatial = SpatialAwarenessSystem()
gimbal = SafeGimbalController()

# If gimbal operational, use orbital scanning
if gimbal.operational:
    gimbal.scan_horizontal_orbit(...)
    # Build complete 3D spatial map
else:
    # Use fixed camera
    # Build limited spatial map

# Spatial awareness works either way
spatial.print_awareness_report()
```

### With Robot Brain

```python
from whoami.gimbal_safe_controller import SafeGimbalController
from whoami.robot_brain import RobotBrain

brain = RobotBrain()
gimbal = SafeGimbalController()

# Brain checks gimbal capabilities
caps = gimbal.get_capabilities()

if caps['can_move']:
    brain.plan_action_with_camera_control()
else:
    brain.plan_action_with_fixed_camera()

# Brain adapts to available capabilities
```

### With Genesis (Think Before Acting)

```python
from whoami.gimbal_safe_controller import SafeGimbalController
from whoami.genesis_bridge import GenesisSceneBuilder

gimbal = SafeGimbalController()
genesis = GenesisSceneBuilder()

# Think in Genesis
print("Thinking: what if I scan this object?")
result = simulate_scan_in_genesis(...)

# If safe, execute on real robot
if result.safe and gimbal.operational:
    gimbal.scan_horizontal_orbit(...)
elif not gimbal.operational:
    print("Gimbal failed - using alternative strategy")
```

## Best Practices

### 1. Always Use Safe Controller

```python
# ✗ DON'T use raw gimbal controller in production
gimbal = Gimbal3DOFController()  # No safety!

# ✓ DO use safe wrapper
gimbal = SafeGimbalController()  # With safety!
```

### 2. Check Capabilities

```python
# ✗ DON'T assume gimbal is available
gimbal.move_to_angles(0, 0, 0)  # May fail!

# ✓ DO check capabilities first
if gimbal.get_capabilities()['can_move']:
    gimbal.move_to_angles(0, 0, 0)
else:
    use_fallback()
```

### 3. Handle Failures Gracefully

```python
# ✗ DON'T crash on failure
success = gimbal.scan_horizontal_orbit(...)
# Assume success, continue...

# ✓ DO check result and adapt
success = gimbal.scan_horizontal_orbit(...)
if success:
    process_scan_data()
else:
    use_alternative_approach()
```

### 4. Use Context Manager

```python
# ✓ Ensures safe shutdown
with SafeGimbalController() as gimbal:
    # Operations...
    pass
# Automatically shutdown safely
```

### 5. Monitor Health

```python
# Periodically check status
gimbal.print_status()

# Check degradation
level = gimbal.get_degradation_level()
if level != 'fully_operational':
    alert_user()
```

## Troubleshooting

### Problem: Servo Not Responding

**Symptoms:** Communication timeout errors

**Solutions:**
1. Check USB connection
2. Verify servo power supply
3. Check servo ID configuration
4. Test with `--test-only` mode

### Problem: Position Errors

**Symptoms:** Large position_error values

**Solutions:**
1. Recalibrate servos
2. Check for mechanical binding
3. Verify load is not too high
4. Adjust safety limits if needed

### Problem: Overheating

**Symptoms:** Temperature warnings

**Solutions:**
1. Let servos cool down
2. Reduce operation speed
3. Check for mechanical friction
4. Improve ventilation

### Problem: Emergency Stop Won't Reset

**Symptoms:** Reset fails after emergency stop

**Solutions:**
1. Check for hardware issues
2. Power cycle servos
3. Manually verify servo positions
4. Check failure logs

### Getting Diagnostics

```python
# Full diagnostics
diag = safety.get_diagnostics()

# Save to file
safety.save_diagnostics(Path("diagnostics.json"))

# Print health report
safety.print_health_report()

# Check failure history
for failure in safety.failure_history:
    print(failure)
```

## Examples

See complete examples in:

- `examples/safe_gimbal_demo.py` - 7 safety demos
- `tools/calibrate_gimbal.py` - Safe calibration
- `whoami/gimbal_safe_controller.py` - Implementation

Run demos:

```bash
python examples/safe_gimbal_demo.py
```

## Summary

The servo safety system ensures **gimbal failures won't crash the robot**:

✓ **Real-time monitoring** - Continuous health tracking
✓ **Automatic detection** - Failures caught immediately
✓ **Safe fallbacks** - Move to safe positions
✓ **Error recovery** - Automatic recovery attempts
✓ **Graceful degradation** - Robot remains functional
✓ **Emergency stop** - Instant halt capability
✓ **Capability reporting** - Know what's available

**Philosophy: "If the neck/head gimbal fails, the robot does not fail"**

The robot adapts and continues operating with reduced capabilities rather than crashing.
