"""
Vision Behaviors for Robot Gimbal Control
Provides high-level behaviors for camera movement based on vision input

This module implements:
- Face centering and tracking
- Object scanning patterns
- Environmental exploration
- Curiosity and search behaviors
- Smooth coordinated movements
"""

import time
import threading
import logging
import math
import numpy as np
from typing import Optional, Tuple, Dict, Any, Callable, List
from dataclasses import dataclass
from enum import Enum

from .gimbal_control import GimbalController, MovementProfile, create_gimbal_controller
from .face_recognition_api import RecognitionResult

logger = logging.getLogger(__name__)


# ============================================================================
# Behavior Types and Configurations
# ============================================================================

class BehaviorType(Enum):
    """Types of vision behaviors"""
    IDLE = "idle"
    CENTER_FACE = "center_face"
    TRACK_PERSON = "track_person"
    SCAN_OBJECT = "scan_object"
    SCAN_ENVIRONMENT = "scan_environment"
    SEARCH_PATTERN = "search_pattern"
    CURIOSITY_MODE = "curiosity_mode"
    LOOK_AT_POINT = "look_at_point"
    FOLLOW_MOTION = "follow_motion"


class ScanPattern(Enum):
    """Scanning patterns for exploration"""
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    SPIRAL = "spiral"
    GRID = "grid"
    RANDOM = "random"
    FIGURE_EIGHT = "figure_eight"


@dataclass
class BehaviorConfig:
    """Configuration for vision behaviors"""
    # Face tracking
    face_center_threshold: float = 50.0  # Pixels from center to trigger movement
    face_tracking_smoothing: float = 0.5  # Smoothing factor for face tracking
    face_lost_timeout: float = 2.0  # Seconds before searching for lost face
    
    # Object scanning
    scan_speed: MovementProfile = MovementProfile.SMOOTH
    scan_overlap: float = 0.2  # Overlap percentage for scanning
    scan_pause_duration: float = 0.5  # Pause at each scan position
    
    # Environmental scanning
    env_scan_range_pan: Tuple[float, float] = (-60, 60)
    env_scan_range_tilt: Tuple[float, float] = (-30, 30)
    env_scan_speed: MovementProfile = MovementProfile.SLOW
    
    # Search patterns
    search_spiral_increment: float = 5.0  # Degrees per spiral step
    search_grid_size: Tuple[int, int] = (5, 3)  # Grid divisions
    search_timeout: float = 30.0  # Maximum search duration
    
    # Curiosity mode
    curiosity_min_interval: float = 3.0  # Minimum time between curiosity movements
    curiosity_max_interval: float = 10.0  # Maximum time between curiosity movements
    curiosity_movement_range: float = 30.0  # Maximum random movement degrees
    
    # Motion detection
    motion_threshold: float = 20.0  # Minimum motion to trigger following
    motion_tracking_speed: MovementProfile = MovementProfile.FAST


@dataclass
class TrackingState:
    """Current tracking state"""
    target_id: Optional[str] = None
    last_position: Optional[Tuple[float, float]] = None
    last_seen_time: float = 0
    lost_count: int = 0
    confidence: float = 0.0


# ============================================================================
# Vision Behavior Controller
# ============================================================================

class VisionBehaviorController:
    """
    High-level vision behavior controller
    Coordinates gimbal movements based on vision input
    """
    
    def __init__(self, gimbal: GimbalController, config: Optional[BehaviorConfig] = None):
        """
        Initialize vision behavior controller
        
        Args:
            gimbal: Gimbal controller instance
            config: Behavior configuration
        """
        self.gimbal = gimbal
        self.config = config or BehaviorConfig()
        
        # Current behavior
        self.current_behavior = BehaviorType.IDLE
        self._behavior_thread: Optional[threading.Thread] = None
        self._stop_behavior = threading.Event()
        
        # Tracking state
        self.tracking_state = TrackingState()
        
        # Frame dimensions (will be set from camera)
        self.frame_width = 1280
        self.frame_height = 720
        
        # Callbacks
        self._behavior_callbacks: Dict[str, List[Callable]] = {
            'on_behavior_start': [],
            'on_behavior_complete': [],
            'on_target_found': [],
            'on_target_lost': [],
            'on_scan_position': []
        }
        
        # Behavior-specific state
        self._scan_positions: List[Tuple[float, float]] = []
        self._scan_index = 0
        self._curiosity_last_move = time.time()
        self._search_start_time = 0
        
        logger.info("Vision behavior controller initialized")
    
    def set_frame_dimensions(self, width: int, height: int) -> None:
        """Set camera frame dimensions for coordinate calculations"""
        self.frame_width = width
        self.frame_height = height
    
    # ========================================================================
    # Face Tracking Behaviors
    # ========================================================================
    
    def center_face(self, face_bbox: Tuple[int, int, int, int], 
                   frame_dimensions: Optional[Tuple[int, int]] = None) -> bool:
        """
        Center a detected face in the camera frame
        
        Args:
            face_bbox: Face bounding box (x, y, width, height)
            frame_dimensions: Optional (width, height) of frame
        
        Returns:
            True if movement initiated
        """
        if frame_dimensions:
            frame_width, frame_height = frame_dimensions
        else:
            frame_width = self.frame_width
            frame_height = self.frame_height
        
        # Calculate face center
        x, y, w, h = face_bbox
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        
        # Calculate offset from frame center
        frame_center_x = frame_width // 2
        frame_center_y = frame_height // 2
        
        offset_x = face_center_x - frame_center_x
        offset_y = face_center_y - frame_center_y
        
        # Check if movement needed
        if (abs(offset_x) < self.config.face_center_threshold and 
            abs(offset_y) < self.config.face_center_threshold):
            return False
        
        # Convert pixel offset to angles
        # Assuming ~68° horizontal FOV and ~55° vertical FOV
        h_fov = 68.0
        v_fov = 55.0
        
        pan_delta = -(offset_x / frame_width) * h_fov
        tilt_delta = (offset_y / frame_height) * v_fov
        
        # Apply smoothing
        pan_delta *= self.config.face_tracking_smoothing
        tilt_delta *= self.config.face_tracking_smoothing
        
        # Move gimbal
        return self.gimbal.move_relative(pan_delta, tilt_delta, MovementProfile.NORMAL)
    
    def track_person(self, recognition_results: List[RecognitionResult],
                    target_name: Optional[str] = None) -> None:
        """
        Track a specific person or the most prominent face
        
        Args:
            recognition_results: Face recognition results
            target_name: Optional specific person to track
        """
        if not recognition_results:
            # No faces detected
            if self.tracking_state.target_id:
                self._handle_target_lost()
            return
        
        # Find target face
        target_face = None
        
        if target_name:
            # Look for specific person
            for result in recognition_results:
                if result.name == target_name:
                    target_face = result
                    break
        else:
            # Track largest/closest face
            target_face = max(recognition_results, 
                            key=lambda r: r.bbox[2] * r.bbox[3])
        
        if target_face:
            # Update tracking state
            self.tracking_state.target_id = target_face.name
            self.tracking_state.last_seen_time = time.time()
            self.tracking_state.lost_count = 0
            self.tracking_state.confidence = target_face.confidence
            
            # Center the face
            self.center_face(target_face.bbox)
            
            # Trigger callback
            if self.tracking_state.lost_count == 0:
                self._trigger_callback('on_target_found', target_face)
        else:
            # Target not found
            if self.tracking_state.target_id:
                self._handle_target_lost()
    
    def _handle_target_lost(self) -> None:
        """Handle lost tracking target"""
        self.tracking_state.lost_count += 1
        
        # Check timeout
        time_lost = time.time() - self.tracking_state.last_seen_time
        
        if time_lost > self.config.face_lost_timeout:
            # Start search pattern
            self._trigger_callback('on_target_lost', self.tracking_state.target_id)
            self.tracking_state.target_id = None
            self.start_behavior(BehaviorType.SEARCH_PATTERN)
    
    # ========================================================================
    # Scanning Behaviors
    # ========================================================================
    
    def scan_object(self, num_positions: int = 8, radius: float = 30.0) -> None:
        """
        Scan around an object in a circular pattern
        
        Args:
            num_positions: Number of scan positions
            radius: Scanning radius in degrees
        """
        self._scan_positions = []
        
        # Generate circular scan positions
        for i in range(num_positions):
            angle = (2 * math.pi * i) / num_positions
            pan = radius * math.cos(angle)
            tilt = radius * math.sin(angle) * 0.5  # Less vertical movement
            self._scan_positions.append((pan, tilt))
        
        # Add center position
        self._scan_positions.append((0, 0))
        
        # Start scanning
        self._scan_index = 0
        self.start_behavior(BehaviorType.SCAN_OBJECT)
    
    def scan_environment(self, pattern: ScanPattern = ScanPattern.GRID) -> None:
        """
        Scan the environment using a specific pattern
        
        Args:
            pattern: Scanning pattern to use
        """
        self._scan_positions = self._generate_scan_pattern(pattern)
        self._scan_index = 0
        self.start_behavior(BehaviorType.SCAN_ENVIRONMENT)
    
    def _generate_scan_pattern(self, pattern: ScanPattern) -> List[Tuple[float, float]]:
        """Generate scan positions based on pattern"""
        positions = []
        
        pan_min, pan_max = self.config.env_scan_range_pan
        tilt_min, tilt_max = self.config.env_scan_range_tilt
        
        if pattern == ScanPattern.GRID:
            # Grid pattern
            cols, rows = self.config.search_grid_size
            for row in range(rows):
                for col in range(cols):
                    pan = pan_min + (pan_max - pan_min) * col / (cols - 1)
                    tilt = tilt_min + (tilt_max - tilt_min) * row / (rows - 1)
                    
                    # Zigzag pattern for efficiency
                    if row % 2 == 1:
                        pan = pan_max - (pan - pan_min)
                    
                    positions.append((pan, tilt))
        
        elif pattern == ScanPattern.SPIRAL:
            # Spiral pattern
            center_pan = (pan_min + pan_max) / 2
            center_tilt = (tilt_min + tilt_max) / 2
            max_radius = min(pan_max - center_pan, tilt_max - center_tilt)
            
            steps = 20
            for i in range(steps):
                t = i / steps
                radius = max_radius * t
                angle = 2 * math.pi * 3 * t  # 3 full rotations
                
                pan = center_pan + radius * math.cos(angle)
                tilt = center_tilt + radius * math.sin(angle)
                positions.append((pan, tilt))
        
        elif pattern == ScanPattern.HORIZONTAL:
            # Horizontal sweep
            rows = 3
            for row in range(rows):
                tilt = tilt_min + (tilt_max - tilt_min) * row / (rows - 1)
                
                # Left to right or right to left
                if row % 2 == 0:
                    positions.extend([
                        (pan_min, tilt),
                        (0, tilt),
                        (pan_max, tilt)
                    ])
                else:
                    positions.extend([
                        (pan_max, tilt),
                        (0, tilt),
                        (pan_min, tilt)
                    ])
        
        elif pattern == ScanPattern.FIGURE_EIGHT:
            # Figure-8 pattern
            steps = 16
            for i in range(steps):
                t = (2 * math.pi * i) / steps
                pan = 30 * math.sin(t)
                tilt = 15 * math.sin(2 * t)
                positions.append((pan, tilt))
        
        elif pattern == ScanPattern.RANDOM:
            # Random positions
            for _ in range(10):
                pan = np.random.uniform(pan_min, pan_max)
                tilt = np.random.uniform(tilt_min, tilt_max)
                positions.append((pan, tilt))
        
        return positions
    
    def _execute_scan(self) -> None:
        """Execute scanning behavior"""
        while not self._stop_behavior.is_set():
            if self._scan_index >= len(self._scan_positions):
                # Scan complete
                self._trigger_callback('on_behavior_complete', BehaviorType.SCAN_OBJECT)
                break
            
            # Move to next position
            pan, tilt = self._scan_positions[self._scan_index]
            self.gimbal.move_to(pan, tilt, self.config.scan_speed, wait=True)
            
            # Trigger callback for this position
            self._trigger_callback('on_scan_position', self._scan_index, (pan, tilt))
            
            # Pause at position
            time.sleep(self.config.scan_pause_duration)
            
            self._scan_index += 1
        
        # Return to center
        self.gimbal.center()
    
    # ========================================================================
    # Search and Exploration Behaviors
    # ========================================================================
    
    def search_pattern(self, pattern: ScanPattern = ScanPattern.SPIRAL) -> None:
        """
        Search for a lost target using a pattern
        
        Args:
            pattern: Search pattern to use
        """
        self._search_start_time = time.time()
        self._scan_positions = self._generate_scan_pattern(pattern)
        self._scan_index = 0
        self.start_behavior(BehaviorType.SEARCH_PATTERN)
    
    def _execute_search(self) -> None:
        """Execute search pattern"""
        while not self._stop_behavior.is_set():
            # Check timeout
            if time.time() - self._search_start_time > self.config.search_timeout:
                logger.info("Search timeout reached")
                break
            
            if self._scan_index >= len(self._scan_positions):
                # Loop back to start
                self._scan_index = 0
            
            # Move to next position
            pan, tilt = self._scan_positions[self._scan_index]
            self.gimbal.move_to(pan, tilt, MovementProfile.NORMAL, wait=True)
            
            # Brief pause to check for target
            time.sleep(0.5)
            
            self._scan_index += 1
        
        self.gimbal.center()
    
    def curiosity_mode(self) -> None:
        """
        Enable curiosity mode - random exploration when idle
        """
        self.start_behavior(BehaviorType.CURIOSITY_MODE)
    
    def _execute_curiosity(self) -> None:
        """Execute curiosity behavior"""
        while not self._stop_behavior.is_set():
            # Wait for random interval
            wait_time = np.random.uniform(
                self.config.curiosity_min_interval,
                self.config.curiosity_max_interval
            )
            
            if self._stop_behavior.wait(wait_time):
                break
            
            # Generate random look direction
            pan_range = self.config.curiosity_movement_range
            tilt_range = self.config.curiosity_movement_range * 0.6
            
            pan = np.random.uniform(-pan_range, pan_range)
            tilt = np.random.uniform(-tilt_range, tilt_range)
            
            # Move with natural speed
            self.gimbal.move_to(pan, tilt, MovementProfile.NATURAL)
            
            # Look for a moment
            time.sleep(np.random.uniform(1.0, 3.0))
    
    # ========================================================================
    # Point-of-Interest Behaviors
    # ========================================================================
    
    def look_at_point(self, x: float, y: float, z: float,
                     duration: Optional[float] = None) -> None:
        """
        Look at a specific 3D point
        
        Args:
            x, y, z: Target point coordinates (meters)
            duration: Optional duration to look at point
        """
        # Use gimbal's look_at_point method
        self.gimbal.look_at_point(x, y, z)
        
        if duration:
            time.sleep(duration)
            self.gimbal.center()
    
    def follow_motion(self, motion_vector: Tuple[float, float],
                     magnitude: float) -> None:
        """
        Follow detected motion
        
        Args:
            motion_vector: (x, y) motion direction
            magnitude: Motion magnitude
        """
        if magnitude < self.config.motion_threshold:
            return
        
        # Scale motion to gimbal movement
        pan_delta = motion_vector[0] * 0.1
        tilt_delta = motion_vector[1] * 0.1
        
        self.gimbal.move_relative(
            pan_delta, tilt_delta,
            self.config.motion_tracking_speed
        )
    
    # ========================================================================
    # Behavior Management
    # ========================================================================
    
    def start_behavior(self, behavior: BehaviorType) -> bool:
        """
        Start a vision behavior
        
        Args:
            behavior: Behavior type to start
        
        Returns:
            True if behavior started successfully
        """
        # Stop current behavior
        self.stop_current_behavior()
        
        self.current_behavior = behavior
        self._stop_behavior.clear()
        
        # Start appropriate behavior thread
        if behavior == BehaviorType.SCAN_OBJECT:
            self._behavior_thread = threading.Thread(target=self._execute_scan)
        elif behavior == BehaviorType.SCAN_ENVIRONMENT:
            self._behavior_thread = threading.Thread(target=self._execute_scan)
        elif behavior == BehaviorType.SEARCH_PATTERN:
            self._behavior_thread = threading.Thread(target=self._execute_search)
        elif behavior == BehaviorType.CURIOSITY_MODE:
            self._behavior_thread = threading.Thread(target=self._execute_curiosity)
        else:
            # Non-threaded behaviors
            logger.info(f"Started behavior: {behavior.value}")
            self._trigger_callback('on_behavior_start', behavior)
            return True
        
        if self._behavior_thread:
            self._behavior_thread.daemon = True
            self._behavior_thread.start()
            logger.info(f"Started behavior: {behavior.value}")
            self._trigger_callback('on_behavior_start', behavior)
            return True
        
        return False
    
    def stop_current_behavior(self) -> None:
        """Stop the current behavior"""
        if self._behavior_thread and self._behavior_thread.is_alive():
            self._stop_behavior.set()
            self._behavior_thread.join(timeout=2.0)
            self._behavior_thread = None
        
        self.current_behavior = BehaviorType.IDLE
    
    def is_active(self) -> bool:
        """Check if a behavior is currently active"""
        return self.current_behavior != BehaviorType.IDLE
    
    # ========================================================================
    # Presets and Patterns
    # ========================================================================
    
    def inspection_routine(self, num_angles: int = 6) -> None:
        """
        Perform a complete inspection routine
        
        Args:
            num_angles: Number of inspection angles
        """
        # Start with environment scan
        self.scan_environment(ScanPattern.GRID)
        time.sleep(1)
        
        # Then detailed object scan
        self.scan_object(num_angles)
    
    def greeting_gesture(self) -> None:
        """Perform a greeting gesture"""
        # Nod motion
        current_pan, current_tilt = self.gimbal.get_position()
        
        self.gimbal.move_to(current_pan, current_tilt - 10, MovementProfile.NORMAL, wait=True)
        time.sleep(0.2)
        self.gimbal.move_to(current_pan, current_tilt + 5, MovementProfile.NORMAL, wait=True)
        time.sleep(0.2)
        self.gimbal.move_to(current_pan, current_tilt, MovementProfile.NORMAL, wait=True)
    
    def alert_gesture(self) -> None:
        """Perform an alert/attention gesture"""
        # Quick left-right motion
        self.gimbal.move_to(-20, 0, MovementProfile.FAST, wait=True)
        self.gimbal.move_to(20, 0, MovementProfile.FAST, wait=True)
        self.gimbal.center()
    
    # ========================================================================
    # Callbacks
    # ========================================================================
    
    def register_callback(self, event: str, callback: Callable) -> None:
        """Register a callback for an event"""
        if event in self._behavior_callbacks:
            self._behavior_callbacks[event].append(callback)
    
    def _trigger_callback(self, event: str, *args, **kwargs) -> None:
        """Trigger callbacks for an event"""
        for callback in self._behavior_callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Behavior callback error: {e}")


# ============================================================================
# Integrated Vision Tracking System
# ============================================================================

class VisionTrackingSystem:
    """
    Complete vision tracking system combining gimbal and behaviors
    Ready for integration with robot vision API
    """
    
    def __init__(self, gimbal_config: Optional[Dict] = None,
                behavior_config: Optional[BehaviorConfig] = None):
        """
        Initialize complete vision tracking system
        
        Args:
            gimbal_config: Gimbal configuration parameters
            behavior_config: Behavior configuration
        """
        # Create gimbal controller
        if gimbal_config:
            self.gimbal = create_gimbal_controller(**gimbal_config)
        else:
            self.gimbal = create_gimbal_controller()
        
        # Create behavior controller
        self.behaviors = VisionBehaviorController(self.gimbal, behavior_config)
        
        # System state
        self._connected = False
        self._tracking_enabled = False
        
        logger.info("Vision tracking system initialized")
    
    def connect(self) -> bool:
        """Connect to gimbal hardware"""
        if self.gimbal.connect():
            self._connected = True
            logger.info("Vision tracking system connected")
            return True
        return False
    
    def disconnect(self) -> None:
        """Disconnect from gimbal hardware"""
        self.behaviors.stop_current_behavior()
        self.gimbal.disconnect()
        self._connected = False
        logger.info("Vision tracking system disconnected")
    
    def enable_tracking(self, mode: BehaviorType = BehaviorType.TRACK_PERSON) -> None:
        """Enable automatic tracking"""
        self._tracking_enabled = True
        self.behaviors.start_behavior(mode)
    
    def disable_tracking(self) -> None:
        """Disable automatic tracking"""
        self._tracking_enabled = False
        self.behaviors.stop_current_behavior()
    
    def process_vision_frame(self, recognition_results: List[RecognitionResult],
                           target_name: Optional[str] = None) -> None:
        """
        Process vision frame for tracking
        
        Args:
            recognition_results: Face recognition results
            target_name: Optional specific person to track
        """
        if self._tracking_enabled:
            self.behaviors.track_person(recognition_results, target_name)
    
    def center_on_face(self, face_bbox: Tuple[int, int, int, int]) -> bool:
        """Center gimbal on detected face"""
        return self.behaviors.center_face(face_bbox)
    
    def scan_for_3d(self, num_positions: int = 8) -> None:
        """Perform scanning pattern for 3D capture"""
        self.behaviors.scan_object(num_positions)
    
    def explore_environment(self) -> None:
        """Start environmental exploration"""
        self.behaviors.scan_environment(ScanPattern.GRID)
    
    def enable_curiosity(self) -> None:
        """Enable curiosity mode for idle exploration"""
        self.behaviors.curiosity_mode()
    
    def search_for_target(self) -> None:
        """Search for lost tracking target"""
        self.behaviors.search_pattern(ScanPattern.SPIRAL)
    
    def look_at(self, x: float, y: float, z: float) -> None:
        """Look at specific 3D point"""
        self.behaviors.look_at_point(x, y, z)
    
    def perform_greeting(self) -> None:
        """Perform greeting gesture"""
        self.behaviors.greeting_gesture()
    
    def perform_alert(self) -> None:
        """Perform alert gesture"""
        self.behaviors.alert_gesture()
    
    def home(self) -> None:
        """Return to home position"""
        self.gimbal.move_to_home()
    
    def get_position(self) -> Tuple[float, float]:
        """Get current gimbal position"""
        return self.gimbal.get_position()
    
    def is_connected(self) -> bool:
        """Check if system is connected"""
        return self._connected
    
    def is_tracking(self) -> bool:
        """Check if tracking is enabled"""
        return self._tracking_enabled


# ============================================================================
# Factory Function
# ============================================================================

def create_vision_tracking_system(**kwargs) -> VisionTrackingSystem:
    """
    Factory function to create vision tracking system
    
    Args:
        **kwargs: Configuration parameters
    
    Returns:
        Configured VisionTrackingSystem instance
    """
    gimbal_config = kwargs.get('gimbal_config', {})
    behavior_config = kwargs.get('behavior_config', BehaviorConfig())
    
    return VisionTrackingSystem(gimbal_config, behavior_config)


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    # Test vision tracking system
    system = create_vision_tracking_system()
    
    if system.connect():
        print("Vision tracking system connected!")
        
        # Test behaviors
        print("\nTesting greeting...")
        system.perform_greeting()
        time.sleep(2)
        
        print("\nTesting alert...")
        system.perform_alert()
        time.sleep(2)
        
        print("\nTesting environment scan...")
        system.explore_environment()
        time.sleep(10)  # Let it run for a bit
        
        print("\nEnabling curiosity mode...")
        system.enable_curiosity()
        time.sleep(15)  # Watch random movements
        
        print("\nReturning home...")
        system.home()
        
        system.disconnect()
        print("\nTest complete!")
    else:
        print("Failed to connect to vision tracking system")