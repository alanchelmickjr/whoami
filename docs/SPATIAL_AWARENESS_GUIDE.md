# Spatial Awareness System Guide

**JOB #1: Give the robot awareness of what's around it so it can interact**

## Overview

The Spatial Awareness System provides the robot with real-time 3D understanding of its environment, enabling safe and intelligent interaction with objects.

## Quick Start

```python
from whoami.spatial_awareness import SpatialAwarenessSystem, ObjectCategory

# Initialize system
spatial = SpatialAwarenessSystem()
spatial.start()

# Detect an object
obj = spatial.detect_object(
    position=np.array([0.3, 0.0, 0.0]),  # 30cm forward
    category=ObjectCategory.CONTAINER,
    name="coffee_mug"
)

# Natural language query
reachable = spatial.query("what can I reach?")
for obj in reachable:
    print(f"{obj.name} at {obj.distance:.2f}m")

spatial.stop()
```

## Core Features

### 1. Object Detection and Tracking

```python
# Add object to spatial map
obj = spatial.detect_object(
    position=np.array([x, y, z]),
    category=ObjectCategory.TOOL,
    name="screwdriver",
    confidence=0.95
)

# Object properties
print(f"Distance: {obj.distance:.2f}m")
print(f"Reachable: {obj.reachable}")
print(f"Graspable: {obj.graspable}")
print(f"Moving: {obj.is_moving()}")
```

### 2. Spatial Relationships

The system automatically computes relationships between objects:

- `left_of` / `right_of`
- `in_front_of` / `behind`
- `near` / `far`
- `on_top_of`

```python
# Compute relationships
spatial.compute_relationships()

# Query relationships
left_objects = spatial.query_relationship("obj_1", "left_of")
```

### 3. Interaction Zones

Predefined zones for workspace organization:

- `front_workspace`: Primary interaction area (30cm forward, 30cm radius)
- `left_side`: Left reach zone
- `right_side`: Right reach zone

```python
# Get objects in zone
front_objects = spatial.get_objects_in_zone("front_workspace")

# Get all reachable objects
reachable = spatial.get_reachable_objects()

# Get graspable objects
graspable = spatial.get_graspable_objects()
```

### 4. Natural Language Queries

```python
# Simple queries
spatial.query("what's in front of me?")
spatial.query("what can I reach?")
spatial.query("what's closest?")
spatial.query("what can I grasp?")
```

## Integration with Existing Systems

### Integration with Vision Behaviors (Curiosity Mode)

```python
from whoami.vision_behaviors import VisionBehaviorController
from whoami.spatial_awareness import (
    SpatialAwarenessSystem,
    integrate_with_vision_behaviors
)

# Initialize systems
spatial = SpatialAwarenessSystem()
vision = VisionBehaviorController()

# Integrate
integrate_with_vision_behaviors(spatial, vision)

# Now when robot explores in curiosity mode,
# detected objects are added to spatial map
vision.curiosity_mode()
```

**How It Works:**
1. Robot explores using curiosity patterns (GRID, SPIRAL, etc.)
2. At each scan position, depth/RGB is captured
3. Objects are detected and added to spatial map
4. Spatial relationships are computed automatically
5. Robot builds understanding over time

### Integration with Robot Brain

```python
from whoami.robot_brain import RobotBrain
from whoami.spatial_awareness import (
    SpatialAwarenessSystem,
    integrate_with_robot_brain
)

# Initialize systems
spatial = SpatialAwarenessSystem()
brain = RobotBrain()

# Integrate
integrate_with_robot_brain(spatial, brain)

# Brain can now query spatial awareness
# brain.spatial_awareness.query("what can I reach?")
```

**How It Works:**
1. Brain has direct access to spatial awareness
2. Brain queries spatial map for decision making
3. Example: "I need to pick up tool" → brain checks if tool is reachable
4. Enables spatial reasoning in cognitive loop

### Integration with 3DOF Orbital Scanning

```python
from whoami.gimbal_3dof_controller import Gimbal3DOFController
from whoami.spatial_awareness import SpatialAwarenessSystem

spatial = SpatialAwarenessSystem()
spatial.start()

gimbal = Gimbal3DOFController()

# Scan callback
def on_scan_position(position_idx, camera_pose):
    # Capture depth/RGB at this position
    # Detect objects and add to spatial map
    pass

# Perform orbital scan
gimbal.scan_horizontal_orbit(
    center=np.array([0.25, 0.0, 0.0]),
    radius=0.15,
    num_points=36,
    scan_callback=on_scan_position
)

# Spatial map now has complete 3D understanding
spatial.print_awareness_report()
```

**How It Works:**
1. Gimbal orbits camera around target object
2. At each position, depth/RGB is captured
3. Objects detected from all angles
4. Complete 3D understanding of object and surroundings
5. Enables accurate interaction planning

### Integration with Genesis (Think Before Acting)

```python
from whoami.spatial_awareness import SpatialAwarenessSystem
from whoami.genesis_bridge import GenesisSceneBuilder, Sim2RealBridge

spatial = SpatialAwarenessSystem()
genesis = GenesisSceneBuilder()
sim2real = Sim2RealBridge()

# 1. Detect object
obj = spatial.detect_object(
    position=np.array([0.3, 0.0, 0.0]),
    category=ObjectCategory.CONTAINER,
    name="mug"
)

# 2. Think in Genesis
print("Thinking: what happens if I push the mug left?")
scene = genesis.create_scene("mental_sim", [obj])
# Simulate action...
result = simulate_push(scene, obj, direction="left")

# 3. If safe in simulation → execute on robot
if result.safe:
    execute_push_on_robot(obj, direction="left")
    print("✓ Action completed safely")
```

**How It Works:**
1. Robot detects objects using spatial awareness
2. Before acting, robot creates Genesis scene with detected objects
3. Robot simulates action in Genesis (ultra-fast!)
4. If simulation shows safe outcome → execute on real robot
5. **"Safety first reinvented"** - robot thinks before acting

## Complete Pipeline Example

```python
"""
Complete embodied AI learning pipeline
"""

# 1. Initialize all systems
spatial = SpatialAwarenessSystem()
vision = VisionBehaviorController()
brain = RobotBrain()
gimbal = Gimbal3DOFController()

# 2. Integrate systems
integrate_with_vision_behaviors(spatial, vision)
integrate_with_robot_brain(spatial, brain)

# 3. Start spatial awareness
spatial.start()

# 4. Explore environment (curiosity mode)
print("Exploring environment...")
vision.curiosity_mode()  # Robot looks around

# 5. Spatial map is populated as robot explores
time.sleep(10)

# 6. Query spatial understanding
print("What can I reach?")
reachable = spatial.query("what can I reach?")

# 7. Brain reasons about interaction
if reachable:
    target = reachable[0]

    # 8. Think in Genesis before acting
    print(f"Thinking about grasping {target.name}...")
    success = simulate_grasp_in_genesis(target)

    if success:
        # 9. Execute safely
        print("Executing grasp...")
        execute_grasp(target)

# 10. Continuous learning
spatial.on_new_object = lambda obj: print(f"Learned about: {obj.name}")
```

## Advanced Features

### Callbacks for Events

```python
# Called when new object discovered
spatial.on_new_object = lambda obj: print(f"Found: {obj.name}")

# Called when object lost
spatial.on_object_lost = lambda obj: print(f"Lost: {obj.name}")

# Called when relationship detected
spatial.on_relationship_detected = lambda rel: print(f"Relationship: {rel}")
```

### Persistence

```python
# Save spatial map
spatial.save_map(Path("data/spatial_maps/kitchen.json"))

# Load spatial map
spatial.load_map(Path("data/spatial_maps/kitchen.json"))
```

### Status Reports

```python
# Get summary
summary = spatial.get_spatial_summary()
print(f"Objects: {summary['num_objects']}")
print(f"Relationships: {summary['num_relationships']}")
print(f"Reachable: {summary['reachable_count']}")

# Print full report
spatial.print_awareness_report()
```

## Configuration

### Interaction Zones

```python
# Add custom zone
spatial.interaction_zones["overhead"] = InteractionZone(
    name="overhead",
    center=np.array([0.0, 0.0, 0.5]),  # 50cm up
    radius=0.2,
    reachable=False
)
```

### Detection Parameters

```python
spatial = SpatialAwarenessSystem(
    robot_frame_origin=np.zeros(3),
    max_detection_range=2.0,  # meters
    update_rate=10.0  # Hz
)
```

## Examples

See `examples/spatial_awareness_demo.py` for complete demonstrations:

- Demo 1: Basic spatial awareness
- Demo 2: Exploration and discovery
- Demo 3: Orbital scanning integration
- Demo 4: Brain integration (spatial reasoning)
- Demo 5: Genesis integration (think before acting)
- Demo 6: Continuous learning loop

## Running the Demo

```bash
# Activate environment
source ~/whoami_env/bin/activate

# Run demo
python examples/spatial_awareness_demo.py
```

The demo shows 6 scenarios demonstrating all integration points.

## Object Categories

Available categories:

```python
ObjectCategory.PERSON      # People
ObjectCategory.TOOL        # Tools and instruments
ObjectCategory.CONTAINER   # Cups, boxes, etc.
ObjectCategory.FURNITURE   # Tables, chairs, etc.
ObjectCategory.OBSTACLE    # Things to avoid
ObjectCategory.UNKNOWN     # Unclassified
```

## Best Practices

### 1. Continuous Updates

Let spatial awareness run in background:

```python
spatial.start()  # Starts background thread
# System continuously updates relationships and zones
```

### 2. Stale Object Removal

Objects not seen for 5 seconds are automatically removed:

```python
# Adjust timeout if needed
spatial._remove_stale_objects(timeout=10.0)
```

### 3. Thread Safety

All operations are thread-safe. Safe to call from multiple threads:

```python
# Thread 1: Adding objects
spatial.detect_object(...)

# Thread 2: Querying
reachable = spatial.query("what can I reach?")
```

### 4. Integration Order

Recommended initialization order:

```python
1. spatial = SpatialAwarenessSystem()
2. vision = VisionBehaviorController()
3. brain = RobotBrain()
4. Integrate: integrate_with_vision_behaviors(spatial, vision)
5. Integrate: integrate_with_robot_brain(spatial, brain)
6. spatial.start()
7. Begin exploration/interaction
```

## Troubleshooting

### Objects Not Detected

- Check max_detection_range setting
- Verify position is in robot frame (not world frame)
- Check if objects are being marked as stale (timeout too short)

### Relationships Not Computing

- Call `spatial.compute_relationships()` manually if needed
- Check if background thread is running (`spatial.start()`)
- Verify objects are within detection range

### Performance Issues

- Reduce update_rate if too high
- Increase stale object timeout to reduce churn
- Limit number of objects tracked

## Future Enhancements

Planned improvements:

1. Vision integration for automatic object detection from camera
2. More sophisticated reachability checking using kinematics
3. Object tracking with velocity prediction
4. Semantic understanding (object properties, affordances)
5. Multi-robot spatial awareness (shared map)

## Summary

The Spatial Awareness System provides the foundation for intelligent interaction:

✓ **Real-time 3D map** of environment
✓ **Object tracking** with properties
✓ **Spatial relationships** between objects
✓ **Interaction analysis** (reachable, graspable)
✓ **Natural language queries**
✓ **Integration** with curiosity, brain, scanning, Genesis

**JOB #1 COMPLETE**: Robot knows what's around it and can interact safely!
