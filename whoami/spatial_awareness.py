"""
Spatial Awareness System

JOB #1: Give the robot awareness of what's around it so it can interact

Integrates with existing:
- vision_behaviors.py (curiosity exploration)
- robot_brain.py (reasoning and learning)
- gimbal_3dof.py (orbital scanning)
- scanner_3d.py (3D perception)
- genesis_bridge.py (simulation for thinking)

Provides:
- Real-time 3D spatial map
- Object detection and tracking
- Spatial relationships ("cup is left of mug")
- Interaction zones (what's reachable)
- Live workspace visualization
"""

import numpy as np
import logging
import time
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import json
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

class ObjectCategory(Enum):
    """Categories of detected objects"""
    PERSON = "person"
    TOOL = "tool"
    CONTAINER = "container"
    FURNITURE = "furniture"
    UNKNOWN = "unknown"
    OBSTACLE = "obstacle"


@dataclass
class SpatialObject:
    """
    An object in the robot's spatial awareness map
    """
    id: str                              # Unique identifier
    category: ObjectCategory             # What it is
    position: np.ndarray                 # (x, y, z) in robot frame
    bounding_box: Dict[str, float]       # min/max in each axis
    confidence: float                    # Detection confidence

    # Tracking
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    last_seen: float = field(default_factory=time.time)
    observation_count: int = 0

    # Interaction
    graspable: bool = False
    reachable: bool = False
    distance: float = 0.0                # Distance from robot

    # Semantic
    name: Optional[str] = None           # "coffee mug", "alan", etc.
    properties: Dict[str, Any] = field(default_factory=dict)

    def update_position(self, new_position: np.ndarray):
        """Update object position and compute velocity"""
        dt = time.time() - self.last_seen
        if dt > 0:
            self.velocity = (new_position - self.position) / dt
        self.position = new_position
        self.last_seen = time.time()
        self.observation_count += 1
        self.distance = np.linalg.norm(self.position)

    def is_moving(self, threshold: float = 0.01) -> bool:
        """Check if object is moving"""
        return np.linalg.norm(self.velocity) > threshold

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'category': self.category.value,
            'position': self.position.tolist(),
            'bounding_box': self.bounding_box,
            'name': self.name,
            'graspable': self.graspable,
            'reachable': self.reachable,
            'distance': self.distance,
            'moving': self.is_moving()
        }


@dataclass
class SpatialRelationship:
    """
    Relationship between two objects
    """
    object_a_id: str
    object_b_id: str
    relation_type: str                   # "left_of", "on_top_of", "near", etc.
    confidence: float
    distance: float


@dataclass
class InteractionZone:
    """
    Zone in space where robot can interact
    """
    name: str
    center: np.ndarray
    radius: float
    reachable: bool
    objects_in_zone: List[str] = field(default_factory=list)


# ============================================================================
# Spatial Awareness System
# ============================================================================

class SpatialAwarenessSystem:
    """
    Complete spatial awareness system

    Maintains real-time 3D understanding of environment:
    - Where objects are
    - What they are
    - How they relate spatially
    - What robot can reach/interact with

    Integrates with Genesis for "thinking" about interactions
    """

    def __init__(
        self,
        robot_frame_origin: np.ndarray = np.zeros(3),
        max_detection_range: float = 2.0,  # meters
        update_rate: float = 10.0  # Hz
    ):
        """
        Initialize spatial awareness system

        Args:
            robot_frame_origin: Robot position in world frame
            max_detection_range: Maximum detection range
            update_rate: Update frequency
        """
        self.robot_origin = robot_frame_origin
        self.max_range = max_detection_range
        self.update_rate = update_rate

        # Spatial map
        self.objects: Dict[str, SpatialObject] = {}
        self.relationships: List[SpatialRelationship] = []
        self.interaction_zones: Dict[str, InteractionZone] = {}

        # Tracking
        self.next_object_id = 0
        self.observation_history = deque(maxlen=1000)

        # Thread safety
        self._lock = threading.Lock()

        # Background update
        self._running = False
        self._update_thread: Optional[threading.Thread] = None

        # Integration hooks
        self.on_new_object: Optional[callable] = None
        self.on_object_lost: Optional[callable] = None
        self.on_relationship_detected: Optional[callable] = None

        # Define interaction zones
        self._initialize_interaction_zones()

        logger.info("Spatial awareness system initialized")

    def _initialize_interaction_zones(self):
        """Initialize default interaction zones"""
        # Front workspace
        self.interaction_zones["front_workspace"] = InteractionZone(
            name="front_workspace",
            center=np.array([0.3, 0.0, 0.0]),  # 30cm forward
            radius=0.3,
            reachable=True
        )

        # Left side
        self.interaction_zones["left_side"] = InteractionZone(
            name="left_side",
            center=np.array([0.2, 0.3, 0.0]),
            radius=0.2,
            reachable=True
        )

        # Right side
        self.interaction_zones["right_side"] = InteractionZone(
            name="right_side",
            center=np.array([0.2, -0.3, 0.0]),
            radius=0.2,
            reachable=True
        )

    # ========================================================================
    # Object Detection and Tracking
    # ========================================================================

    def detect_object(
        self,
        position: np.ndarray,
        category: ObjectCategory = ObjectCategory.UNKNOWN,
        bounding_box: Optional[Dict[str, float]] = None,
        confidence: float = 1.0,
        name: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None
    ) -> SpatialObject:
        """
        Detect/add object to spatial awareness

        Args:
            position: (x, y, z) position in robot frame
            category: Object category
            bounding_box: Spatial bounds
            confidence: Detection confidence
            name: Optional semantic name
            properties: Additional properties

        Returns:
            SpatialObject (new or updated)
        """
        with self._lock:
            # Check if this matches existing object
            existing_obj = self._find_nearby_object(position, threshold=0.1)

            if existing_obj:
                # Update existing object
                existing_obj.update_position(position)
                existing_obj.confidence = max(existing_obj.confidence, confidence)
                if name:
                    existing_obj.name = name
                if properties:
                    existing_obj.properties.update(properties)
                return existing_obj

            # Create new object
            obj_id = f"obj_{self.next_object_id}"
            self.next_object_id += 1

            if not bounding_box:
                # Estimate bounding box
                size = 0.05  # 5cm default
                bounding_box = {
                    'min': (position - size).tolist(),
                    'max': (position + size).tolist()
                }

            obj = SpatialObject(
                id=obj_id,
                category=category,
                position=position,
                bounding_box=bounding_box,
                confidence=confidence,
                name=name,
                properties=properties or {}
            )

            # Compute interaction properties
            obj.distance = np.linalg.norm(position)
            obj.reachable = self._is_reachable(position)
            obj.graspable = self._is_graspable(obj)

            # Add to map
            self.objects[obj_id] = obj

            # Update zone memberships
            self._update_zone_memberships(obj)

            # Trigger callback
            if self.on_new_object:
                self.on_new_object(obj)

            logger.info(f"New object detected: {obj_id} at {position}")

            return obj

    def update_object(self, obj_id: str, **kwargs):
        """Update object properties"""
        with self._lock:
            if obj_id in self.objects:
                obj = self.objects[obj_id]

                for key, value in kwargs.items():
                    if hasattr(obj, key):
                        setattr(obj, key, value)

                obj.last_seen = time.time()

    def remove_object(self, obj_id: str):
        """Remove object from awareness"""
        with self._lock:
            if obj_id in self.objects:
                obj = self.objects.pop(obj_id)

                # Trigger callback
                if self.on_object_lost:
                    self.on_object_lost(obj)

                logger.info(f"Object lost: {obj_id}")

    def _find_nearby_object(
        self,
        position: np.ndarray,
        threshold: float = 0.1
    ) -> Optional[SpatialObject]:
        """Find object near given position"""
        for obj in self.objects.values():
            distance = np.linalg.norm(obj.position - position)
            if distance < threshold:
                return obj
        return None

    # ========================================================================
    # Spatial Relationships
    # ========================================================================

    def compute_relationships(self):
        """
        Compute spatial relationships between objects

        Detects: left_of, right_of, in_front_of, behind, near, far, on_top_of
        """
        with self._lock:
            self.relationships.clear()

            object_list = list(self.objects.values())

            for i, obj_a in enumerate(object_list):
                for obj_b in object_list[i+1:]:
                    relations = self._analyze_relationship(obj_a, obj_b)
                    self.relationships.extend(relations)

    def _analyze_relationship(
        self,
        obj_a: SpatialObject,
        obj_b: SpatialObject
    ) -> List[SpatialRelationship]:
        """Analyze spatial relationship between two objects"""
        relationships = []

        # Compute relative position
        delta = obj_b.position - obj_a.position
        distance = np.linalg.norm(delta)

        # Near/far
        if distance < 0.15:  # 15cm
            relationships.append(SpatialRelationship(
                object_a_id=obj_a.id,
                object_b_id=obj_b.id,
                relation_type="near",
                confidence=1.0,
                distance=distance
            ))

        # Left/right (Y-axis in robot frame)
        if abs(delta[1]) > abs(delta[0]) * 0.5:
            if delta[1] > 0:
                rel_type = "left_of"
            else:
                rel_type = "right_of"

            relationships.append(SpatialRelationship(
                object_a_id=obj_a.id,
                object_b_id=obj_b.id,
                relation_type=rel_type,
                confidence=0.8,
                distance=distance
            ))

        # Front/behind (X-axis)
        if abs(delta[0]) > abs(delta[1]) * 0.5:
            if delta[0] > 0:
                rel_type = "in_front_of"
            else:
                rel_type = "behind"

            relationships.append(SpatialRelationship(
                object_a_id=obj_a.id,
                object_b_id=obj_b.id,
                relation_type=rel_type,
                confidence=0.8,
                distance=distance
            ))

        # On top of (Z-axis, close XY)
        if delta[2] > 0.05 and np.linalg.norm(delta[:2]) < 0.1:
            relationships.append(SpatialRelationship(
                object_a_id=obj_a.id,
                object_b_id=obj_b.id,
                relation_type="on_top_of",
                confidence=0.9,
                distance=distance
            ))

        return relationships

    def get_relationships_for_object(self, obj_id: str) -> List[SpatialRelationship]:
        """Get all relationships involving an object"""
        return [
            rel for rel in self.relationships
            if rel.object_a_id == obj_id or rel.object_b_id == obj_id
        ]

    def query_relationship(
        self,
        obj_a_id: str,
        relation_type: str
    ) -> List[str]:
        """
        Query: "What objects are <relation> to obj_a?"

        Example: query_relationship("obj_1", "left_of") → ["obj_2", "obj_5"]
        """
        results = []
        for rel in self.relationships:
            if rel.object_a_id == obj_a_id and rel.relation_type == relation_type:
                results.append(rel.object_b_id)
        return results

    # ========================================================================
    # Interaction Analysis
    # ========================================================================

    def _is_reachable(self, position: np.ndarray) -> bool:
        """Check if position is reachable by robot"""
        distance = np.linalg.norm(position)

        # Simple reachability check
        # TODO: Use actual kinematics for precise check
        return distance <= 0.5  # 50cm max reach

    def _is_graspable(self, obj: SpatialObject) -> bool:
        """Check if object is graspable"""
        # Consider size, shape, reachability
        if not obj.reachable:
            return False

        # Check size (bounding box)
        bbox = obj.bounding_box
        size = np.array(bbox['max']) - np.array(bbox['min'])

        # Too big or too small
        if np.any(size > 0.3) or np.any(size < 0.01):
            return False

        return True

    def _update_zone_memberships(self, obj: SpatialObject):
        """Update which zones object belongs to"""
        for zone_name, zone in self.interaction_zones.items():
            distance_to_zone = np.linalg.norm(obj.position - zone.center)

            if distance_to_zone <= zone.radius:
                if obj.id not in zone.objects_in_zone:
                    zone.objects_in_zone.append(obj.id)
            else:
                if obj.id in zone.objects_in_zone:
                    zone.objects_in_zone.remove(obj.id)

    def get_objects_in_zone(self, zone_name: str) -> List[SpatialObject]:
        """Get all objects in an interaction zone"""
        if zone_name not in self.interaction_zones:
            return []

        zone = self.interaction_zones[zone_name]
        return [self.objects[obj_id] for obj_id in zone.objects_in_zone
                if obj_id in self.objects]

    def get_reachable_objects(self) -> List[SpatialObject]:
        """Get all objects robot can reach"""
        return [obj for obj in self.objects.values() if obj.reachable]

    def get_graspable_objects(self) -> List[SpatialObject]:
        """Get all objects robot can grasp"""
        return [obj for obj in self.objects.values() if obj.graspable]

    def get_closest_object(
        self,
        category: Optional[ObjectCategory] = None
    ) -> Optional[SpatialObject]:
        """Get closest object (optionally filtered by category)"""
        candidates = [
            obj for obj in self.objects.values()
            if category is None or obj.category == category
        ]

        if not candidates:
            return None

        return min(candidates, key=lambda obj: obj.distance)

    # ========================================================================
    # Queries and Natural Language Interface
    # ========================================================================

    def query(self, question: str) -> Any:
        """
        Natural language spatial query

        Examples:
        - "what's in front of me?"
        - "can I reach the mug?"
        - "what's closest?"
        - "what's left of the cup?"
        """
        question = question.lower().strip()

        # Simple pattern matching (could use NLP later)
        if "in front" in question or "ahead" in question:
            # Objects in front workspace
            return self.get_objects_in_zone("front_workspace")

        elif "reach" in question:
            # Reachable objects
            return self.get_reachable_objects()

        elif "closest" in question or "nearest" in question:
            return self.get_closest_object()

        elif "grasp" in question or "grab" in question or "pick" in question:
            return self.get_graspable_objects()

        elif "left" in question:
            # Parse: "what's left of X"
            # For now, return all objects
            # TODO: Parse specific object reference
            return list(self.objects.values())

        else:
            # Return all objects
            return list(self.objects.values())

    # ========================================================================
    # Status and Reporting
    # ========================================================================

    def get_spatial_summary(self) -> Dict[str, Any]:
        """Get summary of spatial awareness state"""
        with self._lock:
            return {
                'num_objects': len(self.objects),
                'num_relationships': len(self.relationships),
                'reachable_count': len(self.get_reachable_objects()),
                'graspable_count': len(self.get_graspable_objects()),
                'moving_count': sum(1 for obj in self.objects.values() if obj.is_moving()),
                'zones': {
                    name: len(zone.objects_in_zone)
                    for name, zone in self.interaction_zones.items()
                },
                'objects': [obj.to_dict() for obj in self.objects.values()]
            }

    def print_awareness_report(self):
        """Print human-readable awareness report"""
        print("\n" + "="*60)
        print("SPATIAL AWARENESS REPORT")
        print("="*60)

        summary = self.get_spatial_summary()

        print(f"\nObjects Detected: {summary['num_objects']}")
        print(f"Reachable: {summary['reachable_count']}")
        print(f"Graspable: {summary['graspable_count']}")
        print(f"Moving: {summary['moving_count']}")

        print("\nObjects by Zone:")
        for zone_name, count in summary['zones'].items():
            print(f"  {zone_name}: {count} objects")

        if len(self.objects) > 0:
            print("\nDetailed Objects:")
            for obj in self.objects.values():
                pos = obj.position
                print(f"  {obj.id}: {obj.category.value}")
                print(f"    Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})m")
                print(f"    Distance: {obj.distance:.2f}m")
                print(f"    Reachable: {obj.reachable}, Graspable: {obj.graspable}")
                if obj.name:
                    print(f"    Name: {obj.name}")

        if len(self.relationships) > 0:
            print("\nSpatial Relationships:")
            for rel in self.relationships[:10]:  # Show first 10
                obj_a = self.objects.get(rel.object_a_id)
                obj_b = self.objects.get(rel.object_b_id)
                if obj_a and obj_b:
                    name_a = obj_a.name or obj_a.id
                    name_b = obj_b.name or obj_b.id
                    print(f"  {name_a} is {rel.relation_type} {name_b}")

        print("="*60 + "\n")

    # ========================================================================
    # Persistence
    # ========================================================================

    def save_map(self, filepath: Path):
        """Save spatial map to file"""
        data = {
            'timestamp': time.time(),
            'robot_origin': self.robot_origin.tolist(),
            'objects': [obj.to_dict() for obj in self.objects.values()],
            'relationships': [
                {
                    'obj_a': rel.object_a_id,
                    'obj_b': rel.object_b_id,
                    'type': rel.relation_type,
                    'confidence': rel.confidence
                }
                for rel in self.relationships
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved spatial map to {filepath}")

    def load_map(self, filepath: Path):
        """Load spatial map from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        # TODO: Reconstruct objects and relationships
        logger.info(f"Loaded spatial map from {filepath}")

    # ========================================================================
    # Background Processing
    # ========================================================================

    def start(self):
        """Start background spatial awareness updates"""
        if self._running:
            return

        self._running = True
        self._update_thread = threading.Thread(target=self._update_loop)
        self._update_thread.daemon = True
        self._update_thread.start()

        logger.info("Spatial awareness system started")

    def stop(self):
        """Stop background updates"""
        self._running = False
        if self._update_thread:
            self._update_thread.join(timeout=2.0)

        logger.info("Spatial awareness system stopped")

    def _update_loop(self):
        """Background update loop"""
        update_interval = 1.0 / self.update_rate

        while self._running:
            try:
                # Remove stale objects
                self._remove_stale_objects()

                # Update relationships
                self.compute_relationships()

                # Update zone memberships
                for obj in self.objects.values():
                    self._update_zone_memberships(obj)

            except Exception as e:
                logger.error(f"Spatial awareness update error: {e}")

            time.sleep(update_interval)

    def _remove_stale_objects(self, timeout: float = 5.0):
        """Remove objects that haven't been seen recently"""
        current_time = time.time()
        stale_ids = []

        with self._lock:
            for obj_id, obj in self.objects.items():
                if current_time - obj.last_seen > timeout:
                    stale_ids.append(obj_id)

            for obj_id in stale_ids:
                self.remove_object(obj_id)


# ============================================================================
# Integration Helpers
# ============================================================================

def integrate_with_vision_behaviors(
    spatial_system: SpatialAwarenessSystem,
    vision_behaviors  # VisionBehaviorController from vision_behaviors.py
):
    """
    Integrate spatial awareness with vision behaviors

    As robot explores (curiosity mode), update spatial map
    """
    def on_scan_position(position_idx, pan_tilt):
        """Called when robot looks at new position during scan"""
        # TODO: Capture depth/RGB at this position
        # TODO: Detect objects and add to spatial map
        pass

    vision_behaviors.register_callback('on_scan_position', on_scan_position)

    logger.info("Integrated spatial awareness with vision behaviors")


def integrate_with_robot_brain(
    spatial_system: SpatialAwarenessSystem,
    robot_brain  # RobotBrain from robot_brain.py
):
    """
    Integrate spatial awareness with robot brain

    Brain can query spatial awareness for decision making
    """
    # Add spatial system to brain
    robot_brain.spatial_awareness = spatial_system

    logger.info("Integrated spatial awareness with robot brain")


# ============================================================================
# Main Module Interface
# ============================================================================

__all__ = [
    'SpatialAwarenessSystem',
    'SpatialObject',
    'SpatialRelationship',
    'InteractionZone',
    'ObjectCategory',
    'integrate_with_vision_behaviors',
    'integrate_with_robot_brain'
]
