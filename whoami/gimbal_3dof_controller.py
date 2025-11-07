"""
3DOF Gimbal Controller

Integrates 3DOF kinematics with Feetech servo control and 3D scanning.
Provides high-level interface for orbital scanning motions.

This controller:
- Manages three Feetech servos as coordinated system
- Executes orbital scan patterns
- Synchronizes gimbal motion with camera capture
- Provides smooth coordinated movement
"""

import numpy as np
import logging
import time
import threading
from typing import Optional, List, Dict, Any, Tuple, Callable
from dataclasses import dataclass
from pathlib import Path
import json

# Import existing modules
from .gimbal_control import FeetechServoDriver, ServoCommand
from .gimbal_3dof import (
    KinematicParameters, CameraPose, JointAngles,
    Kinematics3DOF, OrbitalPathGenerator, TrajectoryInterpolator,
    Gimbal3DOFConfig
)

logger = logging.getLogger(__name__)


# ============================================================================
# 3DOF Gimbal Controller
# ============================================================================

class Gimbal3DOFController:
    """
    High-level controller for 3DOF orbital scanning gimbal

    Coordinates three Feetech servos to create smooth orbital motions
    around objects for complete 3D scanning coverage
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        serial_port: str = "/dev/ttyUSB0",
        baudrate: int = 1000000
    ):
        """
        Initialize 3DOF gimbal controller

        Args:
            config_path: Path to configuration JSON file
            serial_port: Serial port for Feetech servos
            baudrate: Serial baudrate (default 1Mbps)
        """
        self.logger = logging.getLogger(f"{__name__}.Gimbal3DOFController")

        # Load configuration
        if config_path is None:
            config_path = Path("config/gimbal_3dof_config.json")
        self.config = Gimbal3DOFConfig.load(config_path)

        # Initialize kinematics
        self.params = KinematicParameters.from_dict(self.config['kinematics'])
        self.kinematics = Kinematics3DOF(self.params)

        # Initialize path generator
        self.path_generator = OrbitalPathGenerator(self.kinematics)

        # Initialize servo driver
        self.servo_driver = FeetechServoDriver(serial_port, baudrate)

        # Servo IDs
        self.servo_ids = self.config['servo_ids']
        self.yaw_servo_id = self.servo_ids['yaw']
        self.pitch_servo_id = self.servo_ids['pitch']
        self.roll_servo_id = self.servo_ids['roll']

        # Current state
        self.current_angles: Optional[JointAngles] = None
        self.is_moving = False
        self.is_connected = False

        # Motion parameters
        self.max_velocities = tuple(self.config['motion']['max_velocities'])
        self.acceleration = self.config['motion']['acceleration']
        self.interpolation_steps = self.config['motion']['interpolation_steps']

        # Thread safety
        self._lock = threading.Lock()

        # Scanning state
        self.scan_callback: Optional[Callable] = None
        self.scan_thread: Optional[threading.Thread] = None
        self.stop_scan_flag = False

    def connect(self) -> bool:
        """
        Connect to servos and initialize

        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info("Connecting to 3DOF gimbal servos...")

            # Open serial port
            if not self.servo_driver.connect():
                self.logger.error("Failed to open serial port")
                return False

            # Ping all servos
            for name, servo_id in self.servo_ids.items():
                if not self.servo_driver.ping(servo_id):
                    self.logger.error(f"Servo {name} (ID {servo_id}) not responding")
                    return False
                self.logger.info(f"Servo {name} (ID {servo_id}) connected")

            # Enable torque on all servos
            for servo_id in self.servo_ids.values():
                self.servo_driver.write(servo_id, ServoCommand.TORQUE_ENABLE, [1])

            # Read current positions
            self._update_current_position()

            self.is_connected = True
            self.logger.info("3DOF gimbal connected successfully")
            return True

        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            return False

    def disconnect(self):
        """Disconnect from servos"""
        try:
            # Disable torque
            for servo_id in self.servo_ids.values():
                self.servo_driver.write(servo_id, ServoCommand.TORQUE_ENABLE, [0])

            self.servo_driver.disconnect()
            self.is_connected = False
            self.logger.info("3DOF gimbal disconnected")

        except Exception as e:
            self.logger.error(f"Disconnect error: {e}")

    def _update_current_position(self):
        """Read current servo positions and update state"""
        try:
            yaw_pos = self.servo_driver.read(self.yaw_servo_id, ServoCommand.PRESENT_POSITION, 2)
            pitch_pos = self.servo_driver.read(self.pitch_servo_id, ServoCommand.PRESENT_POSITION, 2)
            roll_pos = self.servo_driver.read(self.roll_servo_id, ServoCommand.PRESENT_POSITION, 2)

            if yaw_pos and pitch_pos and roll_pos:
                yaw_deg = self._position_to_degrees(yaw_pos[0])
                pitch_deg = self._position_to_degrees(pitch_pos[0])
                roll_deg = self._position_to_degrees(roll_pos[0])

                self.current_angles = JointAngles(
                    yaw=yaw_deg,
                    pitch=pitch_deg,
                    roll=roll_deg
                )

        except Exception as e:
            self.logger.warning(f"Failed to update position: {e}")

    def _degrees_to_position(self, degrees: float) -> int:
        """
        Convert degrees to servo position value

        Feetech servos: 0-4095 corresponds to full rotation
        Assuming 300° range: position = (degrees + 150) / 300 * 4095
        """
        # Adjust based on your servo's actual range
        # Standard SCS servos: 0-1000 = 0-240°
        position = int((degrees + 150) / 300 * 4095)
        return np.clip(position, 0, 4095)

    def _position_to_degrees(self, position: int) -> float:
        """Convert servo position to degrees"""
        degrees = (position / 4095.0) * 300 - 150
        return degrees

    def move_to_angles(
        self,
        target_angles: JointAngles,
        speed: Optional[int] = None,
        blocking: bool = True
    ) -> bool:
        """
        Move to specified joint angles

        Args:
            target_angles: Desired joint configuration
            speed: Movement speed (0-1023), None = use max
            blocking: Wait for movement to complete

        Returns:
            True if successful
        """
        if not self.is_connected:
            self.logger.error("Not connected to gimbal")
            return False

        try:
            with self._lock:
                self.is_moving = True

                # Convert angles to servo positions
                yaw_pos = self._degrees_to_position(target_angles.yaw)
                pitch_pos = self._degrees_to_position(target_angles.pitch)
                roll_pos = self._degrees_to_position(target_angles.roll)

                # Set speed for all servos
                if speed is not None:
                    for servo_id in self.servo_ids.values():
                        self.servo_driver.write(servo_id, ServoCommand.GOAL_SPEED, [speed & 0xFF, (speed >> 8) & 0xFF])

                # Write positions to all servos (synchronized)
                self.servo_driver.write(self.yaw_servo_id, ServoCommand.GOAL_POSITION, [yaw_pos & 0xFF, (yaw_pos >> 8) & 0xFF])
                self.servo_driver.write(self.pitch_servo_id, ServoCommand.GOAL_POSITION, [pitch_pos & 0xFF, (pitch_pos >> 8) & 0xFF])
                self.servo_driver.write(self.roll_servo_id, ServoCommand.GOAL_POSITION, [roll_pos & 0xFF, (roll_pos >> 8) & 0xFF])

                self.logger.debug(f"Moving to: {target_angles}")

                # Wait for completion if blocking
                if blocking:
                    self._wait_for_movement_complete()

                self.current_angles = target_angles
                self.is_moving = False

                return True

        except Exception as e:
            self.logger.error(f"Move failed: {e}")
            self.is_moving = False
            return False

    def _wait_for_movement_complete(self, timeout: float = 10.0):
        """Wait for all servos to stop moving"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            all_stopped = True

            for servo_id in self.servo_ids.values():
                moving = self.servo_driver.read(servo_id, ServoCommand.MOVING, 1)
                if moving and moving[0] != 0:
                    all_stopped = False
                    break

            if all_stopped:
                return

            time.sleep(0.05)

        self.logger.warning("Movement timeout")

    def move_to_pose(
        self,
        target_point: np.ndarray,
        camera_position: np.ndarray,
        camera_roll: float = 0.0,
        speed: Optional[int] = None,
        blocking: bool = True
    ) -> bool:
        """
        Move camera to specific pose (position + look-at target)

        Args:
            target_point: (x, y, z) point to look at
            camera_position: (x, y, z) desired camera location
            camera_roll: camera roll angle (degrees)
            speed: movement speed
            blocking: wait for completion

        Returns:
            True if successful
        """
        # Solve inverse kinematics
        target_angles = self.kinematics.inverse_kinematics(
            target_position=target_point,
            camera_position=camera_position,
            camera_roll=camera_roll
        )

        if target_angles is None:
            self.logger.error("Pose unreachable")
            return False

        return self.move_to_angles(target_angles, speed, blocking)

    def execute_trajectory(
        self,
        waypoints: List[JointAngles],
        speed: Optional[int] = None,
        smooth: bool = True,
        callback: Optional[Callable[[int, JointAngles], None]] = None
    ) -> bool:
        """
        Execute trajectory through multiple waypoints

        Args:
            waypoints: List of joint angles to visit
            speed: movement speed
            smooth: enable interpolation between waypoints
            callback: function called at each waypoint (waypoint_index, angles)

        Returns:
            True if successful
        """
        if not self.is_connected:
            self.logger.error("Not connected")
            return False

        if len(waypoints) == 0:
            return True

        try:
            # Interpolate for smooth motion
            if smooth:
                trajectory = TrajectoryInterpolator.interpolate_path(
                    waypoints,
                    self.interpolation_steps
                )
            else:
                trajectory = waypoints

            self.logger.info(f"Executing trajectory: {len(trajectory)} points")

            # Execute each point
            for i, angles in enumerate(trajectory):
                if not self.move_to_angles(angles, speed, blocking=True):
                    self.logger.error(f"Failed at waypoint {i}")
                    return False

                # Call callback if provided
                if callback is not None and i % self.interpolation_steps == 0:
                    waypoint_idx = i // self.interpolation_steps
                    callback(waypoint_idx, angles)

            self.logger.info("Trajectory complete")
            return True

        except Exception as e:
            self.logger.error(f"Trajectory execution failed: {e}")
            return False

    # ========================================================================
    # Scanning Operations
    # ========================================================================

    def scan_horizontal_orbit(
        self,
        center: np.ndarray,
        radius: float,
        num_points: int = 36,
        camera_roll: float = 0.0,
        scan_callback: Optional[Callable] = None
    ) -> bool:
        """
        Perform horizontal orbital scan around object

        Args:
            center: (x, y, z) center of object
            radius: orbital radius
            num_points: number of scan positions
            camera_roll: camera roll angle
            scan_callback: function called at each position for capture

        Returns:
            True if successful
        """
        self.logger.info(f"Starting horizontal orbit scan: {num_points} points")

        # Generate orbital path
        waypoints = self.path_generator.generate_horizontal_orbit(
            center=center,
            radius=radius,
            num_points=num_points,
            camera_roll=camera_roll
        )

        if len(waypoints) == 0:
            self.logger.error("No reachable waypoints in orbit")
            return False

        # Execute trajectory
        return self.execute_trajectory(
            waypoints,
            speed=200,  # Moderate speed for scanning
            smooth=True,
            callback=scan_callback
        )

    def scan_vertical_orbit(
        self,
        center: np.ndarray,
        radius: float,
        axis_direction: str = 'x',
        num_points: int = 36,
        camera_roll: float = 0.0,
        scan_callback: Optional[Callable] = None
    ) -> bool:
        """
        Perform vertical orbital scan (meridian)

        Args:
            center: (x, y, z) center of object
            radius: orbital radius
            axis_direction: 'x' or 'y' axis
            num_points: number of scan positions
            camera_roll: camera roll angle
            scan_callback: function called at each position

        Returns:
            True if successful
        """
        self.logger.info(f"Starting vertical orbit scan: {num_points} points")

        # Generate vertical orbit
        waypoints = self.path_generator.generate_vertical_orbit(
            center=center,
            radius=radius,
            axis_direction=axis_direction,
            num_points=num_points,
            camera_roll=camera_roll
        )

        if len(waypoints) == 0:
            self.logger.error("No reachable waypoints in orbit")
            return False

        return self.execute_trajectory(
            waypoints,
            speed=200,
            smooth=True,
            callback=scan_callback
        )

    def scan_spherical(
        self,
        center: np.ndarray,
        radius: float,
        num_rings: int = 5,
        points_per_ring: int = 36,
        camera_roll: float = 0.0,
        scan_callback: Optional[Callable] = None
    ) -> bool:
        """
        Perform complete spherical scan with multiple orbital rings

        Args:
            center: (x, y, z) center of object
            radius: orbital radius
            num_rings: number of latitude rings
            points_per_ring: points per ring
            camera_roll: camera roll angle
            scan_callback: function called at each position

        Returns:
            True if successful
        """
        self.logger.info(f"Starting spherical scan: {num_rings} rings, {points_per_ring} points each")

        # Generate spherical scan pattern
        waypoints = self.path_generator.generate_spherical_scan(
            center=center,
            radius=radius,
            num_rings=num_rings,
            points_per_ring=points_per_ring,
            camera_roll=camera_roll
        )

        if len(waypoints) == 0:
            self.logger.error("No reachable waypoints in spherical scan")
            return False

        self.logger.info(f"Generated {len(waypoints)} waypoints")

        return self.execute_trajectory(
            waypoints,
            speed=200,
            smooth=True,
            callback=scan_callback
        )

    def continuous_roll_scan(
        self,
        target_point: np.ndarray,
        camera_position: np.ndarray,
        roll_speed: float = 30.0,  # deg/s
        duration: float = 12.0,    # seconds (full 360° at 30 deg/s)
        capture_callback: Optional[Callable] = None
    ) -> bool:
        """
        Perform continuous roll scan at fixed position

        Camera spins around its optical axis while capturing depth data
        Creates 360° coverage of object from one position

        Args:
            target_point: point to look at
            camera_position: camera location
            roll_speed: rotation speed (deg/s)
            duration: scan duration (seconds)
            capture_callback: called periodically during scan

        Returns:
            True if successful
        """
        self.logger.info(f"Starting continuous roll scan: {roll_speed}°/s for {duration}s")

        # Move to starting position (roll = 0)
        if not self.move_to_pose(target_point, camera_position, camera_roll=0.0):
            return False

        # Calculate number of steps
        time_step = 0.1  # Update every 100ms
        num_steps = int(duration / time_step)
        degrees_per_step = roll_speed * time_step

        try:
            for step in range(num_steps):
                current_roll = (step * degrees_per_step) % 360

                # Move to next roll angle
                if not self.move_to_pose(
                    target_point,
                    camera_position,
                    camera_roll=current_roll,
                    blocking=False
                ):
                    break

                # Call capture callback
                if capture_callback is not None:
                    capture_callback(step, current_roll)

                time.sleep(time_step)

            self.logger.info("Continuous roll scan complete")
            return True

        except Exception as e:
            self.logger.error(f"Roll scan failed: {e}")
            return False

    def home(self) -> bool:
        """
        Move to home position (all servos at 0°)

        Returns:
            True if successful
        """
        home_angles = JointAngles(yaw=0.0, pitch=0.0, roll=0.0)
        self.logger.info("Moving to home position")
        return self.move_to_angles(home_angles, speed=300, blocking=True)

    def emergency_stop(self):
        """Emergency stop - disable all servos immediately"""
        self.logger.warning("EMERGENCY STOP")
        try:
            for servo_id in self.servo_ids.values():
                self.servo_driver.write(servo_id, ServoCommand.TORQUE_ENABLE, [0])
            self.is_moving = False
            self.stop_scan_flag = True
        except Exception as e:
            self.logger.error(f"Emergency stop error: {e}")

    # ========================================================================
    # Status and Diagnostics
    # ========================================================================

    def get_current_pose(self) -> Optional[CameraPose]:
        """
        Get current camera pose in world frame

        Returns:
            CameraPose if available, None otherwise
        """
        if self.current_angles is None:
            return None

        return self.kinematics.forward_kinematics(self.current_angles)

    def get_status(self) -> Dict[str, Any]:
        """
        Get gimbal status information

        Returns:
            Dictionary with status information
        """
        status = {
            'connected': self.is_connected,
            'moving': self.is_moving,
            'current_angles': self.current_angles.to_dict() if self.current_angles else None,
            'current_pose': None
        }

        if self.current_angles is not None:
            pose = self.get_current_pose()
            if pose is not None:
                status['current_pose'] = {
                    'position': pose.position.tolist(),
                    'forward': pose.forward.tolist(),
                    'up': pose.up.tolist()
                }

        return status

    def calibrate(self) -> bool:
        """
        Run calibration routine

        Moves through known positions to verify kinematics
        """
        self.logger.info("Starting calibration...")

        # Move to home
        if not self.home():
            return False

        time.sleep(1)

        # Test each axis individually
        test_angles = [
            JointAngles(yaw=45, pitch=0, roll=0),
            JointAngles(yaw=0, pitch=45, roll=0),
            JointAngles(yaw=0, pitch=0, roll=90),
            JointAngles(yaw=0, pitch=0, roll=0)
        ]

        for angles in test_angles:
            self.logger.info(f"Testing: {angles}")
            if not self.move_to_angles(angles, speed=200, blocking=True):
                self.logger.error("Calibration failed")
                return False
            time.sleep(0.5)

        self.logger.info("Calibration complete")
        return True


# ============================================================================
# Helper Functions
# ============================================================================

def create_default_config(output_path: Path):
    """Create default configuration file"""
    config = Gimbal3DOFConfig.DEFAULT_CONFIG
    Gimbal3DOFConfig.save(config, output_path)
    logger.info(f"Created default config at {output_path}")


# ============================================================================
# Main Module Interface
# ============================================================================

__all__ = [
    'Gimbal3DOFController',
    'create_default_config'
]
