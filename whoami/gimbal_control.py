"""
Gimbal Control System for Robot Vision
Provides 2-axis control for Feetech servos to move the OAK-D camera

This module implements:
- Feetech SCS/STS servo protocol driver
- 2-axis gimbal controller (pan/tilt)
- Smooth movement with acceleration control
- Safety limits and home position calibration
- Natural movement patterns for robot personality
"""

import serial
import struct
import time
import threading
import logging
import math
from typing import Optional, Tuple, Dict, Any, Callable, List
from dataclasses import dataclass
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Constants and Protocol Definitions
# ============================================================================

# Feetech SCS/STS Protocol Commands
class ServoCommand:
    """Feetech servo protocol commands"""
    PING = 0x01
    READ = 0x02
    WRITE = 0x03
    REG_WRITE = 0x04
    ACTION = 0x05
    RESET = 0x06
    SYNC_WRITE = 0x83
    
    # Control table addresses
    TORQUE_ENABLE = 0x28
    LED = 0x19
    GOAL_POSITION = 0x2A
    GOAL_SPEED = 0x2E
    GOAL_ACCELERATION = 0x29
    PRESENT_POSITION = 0x38
    PRESENT_SPEED = 0x3A
    PRESENT_LOAD = 0x3C
    PRESENT_VOLTAGE = 0x3E
    PRESENT_TEMPERATURE = 0x3F
    MOVING = 0x42
    LOCK = 0x30
    MIN_ANGLE_LIMIT = 0x09
    MAX_ANGLE_LIMIT = 0x0B
    
    # Special values
    BROADCAST_ID = 0xFE


class MovementProfile(Enum):
    """Movement speed profiles"""
    SLOW = "slow"           # Careful, precise movements
    NORMAL = "normal"       # Standard tracking speed
    FAST = "fast"          # Quick reactions
    SMOOTH = "smooth"      # Extra smooth for scanning
    NATURAL = "natural"    # Lifelike with micro-movements


@dataclass
class ServoConfig:
    """Configuration for a single servo"""
    servo_id: int
    min_angle: float  # degrees
    max_angle: float  # degrees
    home_position: float  # degrees
    max_speed: int  # 0-1023
    acceleration: int  # 0-254
    inverted: bool = False  # Reverse direction


@dataclass
class GimbalConfig:
    """Configuration for gimbal system"""
    pan_servo: ServoConfig
    tilt_servo: ServoConfig
    serial_port: str = "/dev/ttyUSB0"
    baudrate: int = 1000000  # 1Mbps default for Feetech
    movement_threshold: float = 1.0  # Minimum angle change to trigger movement
    smoothing_factor: float = 0.3  # For smooth tracking (0-1)
    safety_check_interval: float = 0.1  # Seconds between safety checks
    enable_micro_movements: bool = True  # Natural idle movements


# ============================================================================
# Feetech Servo Driver
# ============================================================================

class FeetechServoDriver:
    """
    Low-level driver for Feetech SCS/STS servos
    Implements the serial protocol for servo communication
    """
    
    def __init__(self, port: str, baudrate: int = 1000000):
        """
        Initialize servo driver
        
        Args:
            port: Serial port path
            baudrate: Communication speed
        """
        self.port = port
        self.baudrate = baudrate
        self.serial: Optional[serial.Serial] = None
        self._lock = threading.Lock()
        
    def connect(self) -> bool:
        """Connect to servo serial port"""
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=0.05,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            logger.info(f"Connected to servos on {self.port} at {self.baudrate} baud")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to servos: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from servo serial port"""
        if self.serial and self.serial.is_open:
            self.serial.close()
            logger.info("Disconnected from servos")
    
    def _calculate_checksum(self, packet: List[int]) -> int:
        """Calculate packet checksum"""
        return (~sum(packet[2:])) & 0xFF
    
    def _send_packet(self, servo_id: int, command: int, params: List[int] = None) -> bool:
        """
        Send a command packet to servo
        
        Args:
            servo_id: Target servo ID
            command: Command byte
            params: Optional parameters
        
        Returns:
            True if sent successfully
        """
        if not self.serial or not self.serial.is_open:
            return False
        
        params = params or []
        length = len(params) + 2
        
        # Build packet: [0xFF, 0xFF, ID, Length, Command, Params..., Checksum]
        packet = [0xFF, 0xFF, servo_id, length, command] + params
        packet.append(self._calculate_checksum(packet))
        
        with self._lock:
            try:
                self.serial.write(bytes(packet))
                self.serial.flush()
                return True
            except Exception as e:
                logger.error(f"Failed to send packet: {e}")
                return False
    
    def _read_response(self, expected_length: int = 6) -> Optional[List[int]]:
        """Read response packet from servo"""
        if not self.serial or not self.serial.is_open:
            return None
        
        with self._lock:
            try:
                response = self.serial.read(expected_length)
                if len(response) >= 6:
                    return list(response)
                return None
            except Exception as e:
                logger.error(f"Failed to read response: {e}")
                return None
    
    def ping(self, servo_id: int) -> bool:
        """Check if servo is responding"""
        self._send_packet(servo_id, ServoCommand.PING)
        response = self._read_response()
        return response is not None and len(response) >= 6
    
    def set_torque(self, servo_id: int, enabled: bool) -> bool:
        """Enable/disable servo torque"""
        value = 1 if enabled else 0
        return self._send_packet(
            servo_id,
            ServoCommand.WRITE,
            [ServoCommand.TORQUE_ENABLE, value]
        )
    
    def set_position(self, servo_id: int, position: int, speed: int = None) -> bool:
        """
        Set servo position
        
        Args:
            servo_id: Servo ID
            position: Target position (0-4095)
            speed: Movement speed (0-1023)
        """
        # Position is 2 bytes
        pos_low = position & 0xFF
        pos_high = (position >> 8) & 0xFF
        
        if speed is not None:
            # Set speed first
            speed_low = speed & 0xFF
            speed_high = (speed >> 8) & 0xFF
            self._send_packet(
                servo_id,
                ServoCommand.WRITE,
                [ServoCommand.GOAL_SPEED, speed_low, speed_high]
            )
        
        return self._send_packet(
            servo_id,
            ServoCommand.WRITE,
            [ServoCommand.GOAL_POSITION, pos_low, pos_high]
        )
    
    def set_speed(self, servo_id: int, speed: int) -> bool:
        """Set servo movement speed (0-1023)"""
        speed_low = speed & 0xFF
        speed_high = (speed >> 8) & 0xFF
        return self._send_packet(
            servo_id,
            ServoCommand.WRITE,
            [ServoCommand.GOAL_SPEED, speed_low, speed_high]
        )
    
    def set_acceleration(self, servo_id: int, acceleration: int) -> bool:
        """Set servo acceleration (0-254)"""
        return self._send_packet(
            servo_id,
            ServoCommand.WRITE,
            [ServoCommand.GOAL_ACCELERATION, acceleration & 0xFF]
        )
    
    def get_position(self, servo_id: int) -> Optional[int]:
        """Get current servo position"""
        self._send_packet(
            servo_id,
            ServoCommand.READ,
            [ServoCommand.PRESENT_POSITION, 2]
        )
        response = self._read_response(8)
        
        if response and len(response) >= 8:
            # Extract position from response
            pos_low = response[5]
            pos_high = response[6]
            return pos_low | (pos_high << 8)
        return None
    
    def is_moving(self, servo_id: int) -> bool:
        """Check if servo is currently moving"""
        self._send_packet(
            servo_id,
            ServoCommand.READ,
            [ServoCommand.MOVING, 1]
        )
        response = self._read_response(7)
        
        if response and len(response) >= 7:
            return response[5] == 1
        return False
    
    def sync_write_position(self, positions: Dict[int, int]) -> bool:
        """
        Write positions to multiple servos simultaneously
        
        Args:
            positions: Dict of servo_id -> position
        """
        if not positions:
            return False
        
        # Build sync write packet
        params = [ServoCommand.GOAL_POSITION, 2]  # Address and data length
        
        for servo_id, position in positions.items():
            params.append(servo_id)
            params.append(position & 0xFF)
            params.append((position >> 8) & 0xFF)
        
        return self._send_packet(
            ServoCommand.BROADCAST_ID,
            ServoCommand.SYNC_WRITE,
            params
        )


# ============================================================================
# Gimbal Controller
# ============================================================================

class GimbalController:
    """
    High-level gimbal controller for 2-axis camera movement
    Provides smooth movements, safety limits, and natural behaviors
    """
    
    def __init__(self, config: GimbalConfig):
        """
        Initialize gimbal controller
        
        Args:
            config: Gimbal configuration
        """
        self.config = config
        self.driver = FeetechServoDriver(config.serial_port, config.baudrate)
        
        # Current state
        self._pan_angle = config.pan_servo.home_position
        self._tilt_angle = config.tilt_servo.home_position
        self._target_pan = self._pan_angle
        self._target_tilt = self._tilt_angle
        
        # Movement control
        self._moving = False
        self._movement_thread: Optional[threading.Thread] = None
        self._stop_movement = threading.Event()
        
        # Callbacks
        self._movement_callbacks: List[Callable] = []
        
        # Safety
        self._safety_enabled = True
        self._emergency_stop = False
        
        # Smooth tracking
        self._smoothing_buffer_pan: List[float] = []
        self._smoothing_buffer_tilt: List[float] = []
        self._max_buffer_size = 5
        
        logger.info("Gimbal controller initialized")
    
    def connect(self) -> bool:
        """Connect to gimbal servos"""
        if not self.driver.connect():
            return False
        
        # Initialize servos
        if not self._initialize_servos():
            self.driver.disconnect()
            return False
        
        # Start movement monitoring thread
        self._stop_movement.clear()
        self._movement_thread = threading.Thread(target=self._movement_monitor)
        self._movement_thread.daemon = True
        self._movement_thread.start()
        
        logger.info("Gimbal connected and initialized")
        return True
    
    def disconnect(self) -> None:
        """Disconnect from gimbal servos"""
        # Stop movement thread
        self._stop_movement.set()
        if self._movement_thread:
            self._movement_thread.join(timeout=1.0)
        
        # Disable torque
        self.driver.set_torque(self.config.pan_servo.servo_id, False)
        self.driver.set_torque(self.config.tilt_servo.servo_id, False)
        
        # Disconnect driver
        self.driver.disconnect()
        logger.info("Gimbal disconnected")
    
    def _initialize_servos(self) -> bool:
        """Initialize servo settings"""
        try:
            # Check servo connectivity
            if not self.driver.ping(self.config.pan_servo.servo_id):
                logger.error(f"Pan servo {self.config.pan_servo.servo_id} not responding")
                return False
            
            if not self.driver.ping(self.config.tilt_servo.servo_id):
                logger.error(f"Tilt servo {self.config.tilt_servo.servo_id} not responding")
                return False
            
            # Set acceleration
            self.driver.set_acceleration(
                self.config.pan_servo.servo_id,
                self.config.pan_servo.acceleration
            )
            self.driver.set_acceleration(
                self.config.tilt_servo.servo_id,
                self.config.tilt_servo.acceleration
            )
            
            # Enable torque
            self.driver.set_torque(self.config.pan_servo.servo_id, True)
            self.driver.set_torque(self.config.tilt_servo.servo_id, True)
            
            # Move to home position
            self.move_to_home()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize servos: {e}")
            return False
    
    def _degrees_to_position(self, degrees: float, servo_config: ServoConfig) -> int:
        """Convert degrees to servo position (0-4095)"""
        if servo_config.inverted:
            degrees = -degrees
        
        # Map angle range to servo range (0-4095 = 0-360 degrees)
        position = int((degrees + 180) * 4095 / 360)
        return max(0, min(4095, position))
    
    def _position_to_degrees(self, position: int, servo_config: ServoConfig) -> float:
        """Convert servo position to degrees"""
        degrees = (position * 360 / 4095) - 180
        if servo_config.inverted:
            degrees = -degrees
        return degrees
    
    def _apply_safety_limits(self, pan: float, tilt: float) -> Tuple[float, float]:
        """Apply safety limits to angles"""
        # Clamp to configured limits
        pan = max(self.config.pan_servo.min_angle, 
                 min(self.config.pan_servo.max_angle, pan))
        tilt = max(self.config.tilt_servo.min_angle,
                  min(self.config.tilt_servo.max_angle, tilt))
        
        return pan, tilt
    
    def _smooth_movement(self, target: float, current: float, 
                        smoothing: float = None) -> float:
        """Apply smoothing to movement"""
        smoothing = smoothing or self.config.smoothing_factor
        return current + (target - current) * smoothing
    
    def _movement_monitor(self) -> None:
        """Background thread for movement monitoring"""
        while not self._stop_movement.is_set():
            try:
                # Check if we need to move
                if self._should_move():
                    self._execute_movement()
                
                # Add micro-movements if enabled and idle
                if self.config.enable_micro_movements and not self._moving:
                    self._add_micro_movement()
                
                time.sleep(0.02)  # 50Hz update rate
                
            except Exception as e:
                logger.error(f"Movement monitor error: {e}")
    
    def _should_move(self) -> bool:
        """Check if movement is needed"""
        pan_diff = abs(self._target_pan - self._pan_angle)
        tilt_diff = abs(self._target_tilt - self._tilt_angle)
        return (pan_diff > self.config.movement_threshold or 
                tilt_diff > self.config.movement_threshold)
    
    def _execute_movement(self) -> None:
        """Execute movement towards target"""
        if self._emergency_stop:
            return
        
        # Smooth movement
        new_pan = self._smooth_movement(self._target_pan, self._pan_angle)
        new_tilt = self._smooth_movement(self._target_tilt, self._tilt_angle)
        
        # Apply safety limits
        new_pan, new_tilt = self._apply_safety_limits(new_pan, new_tilt)
        
        # Convert to servo positions
        pan_pos = self._degrees_to_position(new_pan, self.config.pan_servo)
        tilt_pos = self._degrees_to_position(new_tilt, self.config.tilt_servo)
        
        # Send to servos
        self.driver.sync_write_position({
            self.config.pan_servo.servo_id: pan_pos,
            self.config.tilt_servo.servo_id: tilt_pos
        })
        
        # Update current position
        self._pan_angle = new_pan
        self._tilt_angle = new_tilt
        
        # Check if we've reached target
        if not self._should_move():
            self._moving = False
            self._trigger_movement_complete()
        else:
            self._moving = True
    
    def _add_micro_movement(self) -> None:
        """Add subtle random movements for natural appearance"""
        if np.random.random() < 0.01:  # 1% chance per cycle
            # Small random offset
            pan_offset = np.random.normal(0, 0.5)  # ±0.5 degrees
            tilt_offset = np.random.normal(0, 0.3)  # ±0.3 degrees
            
            self._target_pan += pan_offset
            self._target_tilt += tilt_offset
    
    # ========================================================================
    # Public Movement Methods
    # ========================================================================
    
    def move_to(self, pan: float, tilt: float, speed: MovementProfile = MovementProfile.NORMAL,
                wait: bool = False) -> bool:
        """
        Move gimbal to specific angles
        
        Args:
            pan: Pan angle in degrees
            tilt: Tilt angle in degrees
            speed: Movement speed profile
            wait: Wait for movement to complete
        
        Returns:
            True if movement started successfully
        """
        if self._emergency_stop:
            logger.warning("Emergency stop active, movement blocked")
            return False
        
        # Apply safety limits
        pan, tilt = self._apply_safety_limits(pan, tilt)
        
        # Set speed based on profile
        self._set_movement_speed(speed)
        
        # Update targets
        self._target_pan = pan
        self._target_tilt = tilt
        
        # Wait if requested
        if wait:
            self.wait_for_movement()
        
        return True
    
    def move_relative(self, pan_delta: float, tilt_delta: float,
                     speed: MovementProfile = MovementProfile.NORMAL) -> bool:
        """
        Move gimbal relative to current position
        
        Args:
            pan_delta: Pan change in degrees
            tilt_delta: Tilt change in degrees
            speed: Movement speed profile
        """
        new_pan = self._pan_angle + pan_delta
        new_tilt = self._tilt_angle + tilt_delta
        return self.move_to(new_pan, new_tilt, speed)
    
    def move_to_home(self, speed: MovementProfile = MovementProfile.NORMAL) -> bool:
        """Move to home position"""
        return self.move_to(
            self.config.pan_servo.home_position,
            self.config.tilt_servo.home_position,
            speed
        )
    
    def center(self) -> bool:
        """Center the gimbal (0, 0)"""
        return self.move_to(0, 0, MovementProfile.FAST)
    
    def look_at_point(self, x: float, y: float, z: float, 
                     camera_fov_h: float = 68.0, camera_fov_v: float = 55.0) -> bool:
        """
        Point gimbal at a 3D coordinate
        
        Args:
            x, y, z: Target point in camera space (meters)
            camera_fov_h: Horizontal field of view
            camera_fov_v: Vertical field of view
        """
        # Calculate angles to point
        pan_angle = math.degrees(math.atan2(x, z))
        tilt_angle = math.degrees(math.atan2(y, z))
        
        return self.move_to(pan_angle, tilt_angle, MovementProfile.NORMAL)
    
    def smooth_track(self, pan: float, tilt: float) -> None:
        """
        Smoothly track a target with filtering
        
        Args:
            pan: Target pan angle
            tilt: Target tilt angle
        """
        # Add to smoothing buffer
        self._smoothing_buffer_pan.append(pan)
        self._smoothing_buffer_tilt.append(tilt)
        
        # Limit buffer size
        if len(self._smoothing_buffer_pan) > self._max_buffer_size:
            self._smoothing_buffer_pan.pop(0)
        if len(self._smoothing_buffer_tilt) > self._max_buffer_size:
            self._smoothing_buffer_tilt.pop(0)
        
        # Calculate smoothed target
        smooth_pan = np.mean(self._smoothing_buffer_pan)
        smooth_tilt = np.mean(self._smoothing_buffer_tilt)
        
        self.move_to(smooth_pan, smooth_tilt, MovementProfile.SMOOTH)
    
    def emergency_stop(self) -> None:
        """Emergency stop - halt all movement"""
        self._emergency_stop = True
        self._target_pan = self._pan_angle
        self._target_tilt = self._tilt_angle
        logger.warning("Emergency stop activated")
    
    def resume(self) -> None:
        """Resume after emergency stop"""
        self._emergency_stop = False
        logger.info("Emergency stop deactivated")
    
    def wait_for_movement(self, timeout: float = 5.0) -> bool:
        """
        Wait for current movement to complete
        
        Args:
            timeout: Maximum wait time in seconds
        
        Returns:
            True if movement completed, False if timeout
        """
        start_time = time.time()
        while self._moving and (time.time() - start_time) < timeout:
            time.sleep(0.05)
        
        return not self._moving
    
    def is_moving(self) -> bool:
        """Check if gimbal is currently moving"""
        return self._moving
    
    def get_position(self) -> Tuple[float, float]:
        """Get current gimbal position (pan, tilt) in degrees"""
        return self._pan_angle, self._tilt_angle
    
    def get_target(self) -> Tuple[float, float]:
        """Get target position (pan, tilt) in degrees"""
        return self._target_pan, self._target_tilt
    
    # ========================================================================
    # Speed Control
    # ========================================================================
    
    def _set_movement_speed(self, profile: MovementProfile) -> None:
        """Set movement speed based on profile"""
        speed_map = {
            MovementProfile.SLOW: (200, 150),    # (pan_speed, tilt_speed)
            MovementProfile.NORMAL: (400, 300),
            MovementProfile.FAST: (800, 600),
            MovementProfile.SMOOTH: (300, 200),
            MovementProfile.NATURAL: (350, 250)
        }
        
        pan_speed, tilt_speed = speed_map.get(profile, (400, 300))
        
        self.driver.set_speed(self.config.pan_servo.servo_id, pan_speed)
        self.driver.set_speed(self.config.tilt_servo.servo_id, tilt_speed)
    
    def set_speed_limits(self, max_pan_speed: int, max_tilt_speed: int) -> None:
        """Set maximum movement speeds (0-1023)"""
        self.config.pan_servo.max_speed = max_pan_speed
        self.config.tilt_servo.max_speed = max_tilt_speed
    
    def set_acceleration(self, pan_accel: int, tilt_accel: int) -> None:
        """Set servo acceleration (0-254)"""
        self.driver.set_acceleration(self.config.pan_servo.servo_id, pan_accel)
        self.driver.set_acceleration(self.config.tilt_servo.servo_id, tilt_accel)
        self.config.pan_servo.acceleration = pan_accel
        self.config.tilt_servo.acceleration = tilt_accel
    
    # ========================================================================
    # Callbacks
    # ========================================================================
    
    def register_movement_callback(self, callback: Callable) -> None:
        """Register callback for movement completion"""
        self._movement_callbacks.append(callback)
    
    def _trigger_movement_complete(self) -> None:
        """Trigger movement complete callbacks"""
        for callback in self._movement_callbacks:
            try:
                callback(self._pan_angle, self._tilt_angle)
            except Exception as e:
                logger.error(f"Movement callback error: {e}")
    
    # ========================================================================
    # Calibration
    # ========================================================================
    
    def calibrate_home(self) -> None:
        """Calibrate current position as home"""
        self.config.pan_servo.home_position = self._pan_angle
        self.config.tilt_servo.home_position = self._tilt_angle
        logger.info(f"Home position calibrated: pan={self._pan_angle}, tilt={self._tilt_angle}")
    
    def set_limits(self, pan_min: float, pan_max: float, 
                  tilt_min: float, tilt_max: float) -> None:
        """Set movement limits"""
        self.config.pan_servo.min_angle = pan_min
        self.config.pan_servo.max_angle = pan_max
        self.config.tilt_servo.min_angle = tilt_min
        self.config.tilt_servo.max_angle = tilt_max
        logger.info(f"Limits set: pan=[{pan_min}, {pan_max}], tilt=[{tilt_min}, {tilt_max}]")


# ============================================================================
# Factory Function
# ============================================================================

def create_gimbal_controller(
    pan_servo_id: int = 1,
    tilt_servo_id: int = 2,
    serial_port: str = "/dev/ttyUSB0",
    pan_limits: Tuple[float, float] = (-90, 90),
    tilt_limits: Tuple[float, float] = (-45, 45)
) -> GimbalController:
    """
    Factory function to create gimbal controller
    
    Args:
        pan_servo_id: ID of pan servo
        tilt_servo_id: ID of tilt servo  
        serial_port: Serial port for servo communication
        pan_limits: (min, max) pan angles in degrees
        tilt_limits: (min, max) tilt angles in degrees
    
    Returns:
        Configured GimbalController instance
    """
    config = GimbalConfig(
        pan_servo=ServoConfig(
            servo_id=pan_servo_id,
            min_angle=pan_limits[0],
            max_angle=pan_limits[1],
            home_position=0,
            max_speed=500,
            acceleration=100
        ),
        tilt_servo=ServoConfig(
            servo_id=tilt_servo_id,
            min_angle=tilt_limits[0],
            max_angle=tilt_limits[1],
            home_position=0,
            max_speed=400,
            acceleration=100
        ),
        serial_port=serial_port
    )
    
    return GimbalController(config)


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    # Test gimbal controller
    gimbal = create_gimbal_controller()
    
    if gimbal.connect():
        print("Gimbal connected!")
        
        # Test movements
        print("Moving to home...")
        gimbal.move_to_home(wait=True)
        time.sleep(1)
        
        print("Testing pan...")
        gimbal.move_to(30, 0, wait=True)
        time.sleep(1)
        gimbal.move_to(-30, 0, wait=True)
        time.sleep(1)
        
        print("Testing tilt...")
        gimbal.move_to(0, 20, wait=True)
        time.sleep(1)
        gimbal.move_to(0, -20, wait=True)
        time.sleep(1)
        
        print("Returning home...")
        gimbal.move_to_home(wait=True)
        
        gimbal.disconnect()
        print("Test complete!")
    else:
        print("Failed to connect to gimbal")