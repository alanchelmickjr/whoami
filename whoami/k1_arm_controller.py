"""
K-1 Booster Arm Controller

⚠️ IMPORTANT: This is a SIMPLIFIED interface for K-1 arm gestures.
For production use with real K-1 hardware, use the official Booster Robotics SDK:
https://github.com/BoosterRobotics/booster_robotics_sdk

This module provides basic gesture commands (wave, point) as a demonstration
of how to integrate arm control with face recognition. It uses the correct
joint IDs and limits from K-1 specifications, but full SDK integration
requires Fast-DDS message definitions and proper mode management.

Hardware Specifications:
- 22 DOF total (Legs: 6×2, Arms: 4×2, Head: 2)
- Force-controlled dual-encoder joints
- Fast-DDS communication (ROS2 compatible)
- Jetson Orin NX 8GB (117 TOPS)

SDK Topics (Fast-DDS):
- rt/low_state: Subscribe to joint feedback (IMU + MotorState)
  * struct MotorState { float q, dq, ddq, tau_est; }
- rt/low_cmd: Publish joint commands (MotorCmd)
  * struct MotorCmd { float q, dq, tau, kp, kd, weight; }
  * enum CmdType { PARALLEL, SERIAL }
- rt/robot_states: Subscribe to robot mode and status
- rt/odometer_state: Subscribe to odometry
- rt/button_event: Subscribe to backboard buttons
- rt/remote_controller_state: Subscribe to Xbox controller

This module provides:
- Gesture primitives (wave, point, rest)
- Correct K-1 joint IDs and limits
- Safety limits enforcement
- Simulation mode for testing
- Integration with face recognition

For full features (locomotion, balance, complex motions), use the official SDK.
"""

import logging
import time
import threading
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)

# Check if ROS2 is available
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from std_msgs.msg import Header
    from builtin_interfaces.msg import Duration
    ROS2_AVAILABLE = True
except ImportError:
    logger.warning("ROS2 not available - arm control will be simulated")
    ROS2_AVAILABLE = False


class ArmSide(Enum):
    """Which arm to control"""
    LEFT = "left"
    RIGHT = "right"
    BOTH = "both"


@dataclass
class JointPosition:
    """
    Joint position configuration for K-1 4-DOF arm

    Based on actual K-1 specifications:
    - Shoulder Pitch: Forward/back rotation
    - Shoulder Roll: Out/in rotation
    - Shoulder Yaw: Twist rotation
    - Elbow: Bend
    """
    shoulder_pitch: float  # degrees
    shoulder_roll: float   # degrees
    shoulder_yaw: float    # degrees
    elbow: float          # degrees


@dataclass
class ArmConfig:
    """
    Configuration for K-1 arm control

    Based on actual K-1 Booster specifications:
    - Left arm: Joint IDs 2-5
    - Right arm: Joint IDs 6-9
    - Each arm: Shoulder Pitch/Roll/Yaw + Elbow (4 DOF)
    """
    # K-1 SDK Fast-DDS topics
    low_state_topic: str = "rt/low_state"  # Subscribe to joint feedback
    low_cmd_topic: str = "rt/low_cmd"      # Publish joint commands
    robot_states_topic: str = "rt/robot_states"  # Robot mode/status

    # Joint IDs for K-1 motor control
    left_arm_joint_ids: List[int] = None   # Joints 2-5
    right_arm_joint_ids: List[int] = None  # Joints 6-9

    # Control mode
    cmd_type: str = "PARALLEL"  # or "SERIAL" - K-1 uses parallel structure

    # Safety limits (degrees) - from K-1 specifications
    # Left arm: IDs 2-5
    left_shoulder_pitch_limits: Tuple[float, float] = (-169, 69)   # Joint 2
    left_shoulder_roll_limits: Tuple[float, float] = (-99, 89)     # Joint 3
    left_shoulder_yaw_limits: Tuple[float, float] = (-109, 109)    # Joint 4
    left_elbow_limits: Tuple[float, float] = (-129, 39)            # Joint 5

    # Right arm: IDs 6-9
    right_shoulder_pitch_limits: Tuple[float, float] = (-169, 69)  # Joint 6
    right_shoulder_roll_limits: Tuple[float, float] = (-99, 89)    # Joint 7
    right_shoulder_yaw_limits: Tuple[float, float] = (-109, 109)   # Joint 8
    right_elbow_limits: Tuple[float, float] = (-39, 129)           # Joint 9

    # Control parameters (for K-1 MotorCmd)
    default_kp: float = 50.0   # Proportional gain
    default_kd: float = 2.0    # Derivative gain
    default_weight: float = 1.0  # Command weight (0=internal controller, 1=user control)

    # Movement parameters
    default_duration: float = 2.0  # seconds for gesture completion
    wave_duration: float = 3.0     # seconds for wave gesture
    wave_cycles: int = 3           # number of wave cycles

    def __post_init__(self):
        """Set default joint IDs if not provided"""
        if self.left_arm_joint_ids is None:
            self.left_arm_joint_ids = [2, 3, 4, 5]  # Left: Shoulder P/R/Y, Elbow
        if self.right_arm_joint_ids is None:
            self.right_arm_joint_ids = [6, 7, 8, 9]  # Right: Shoulder P/R/Y, Elbow


class K1ArmController:
    """
    High-level arm controller for K-1 Booster robot

    This controller provides simple gesture commands that can be integrated
    with the face recognition system for natural human-robot interaction.
    """

    def __init__(self, config: Optional[ArmConfig] = None, simulate: bool = False):
        """
        Initialize K-1 arm controller

        Args:
            config: Arm configuration (uses defaults if None)
            simulate: If True, run in simulation mode without ROS2
        """
        self.config = config or ArmConfig()
        self.simulate = simulate or not ROS2_AVAILABLE

        # Current joint states
        self._current_left_position: Optional[JointPosition] = None
        self._current_right_position: Optional[JointPosition] = None
        self._lock = threading.RLock()

        # ROS2 node
        self._node: Optional[Node] = None
        self._joint_state_sub = None
        self._left_arm_pub = None
        self._right_arm_pub = None
        self._spin_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()

        # Rest position (arms at sides)
        self._rest_position = JointPosition(
            shoulder_pitch=0,    # Neutral
            shoulder_roll=0,     # Arms at sides
            shoulder_yaw=0,      # Neutral (no twist)
            elbow=15             # Slightly bent
        )

        logger.info(f"K-1 Arm Controller initialized (simulate={self.simulate})")

    def start(self) -> bool:
        """
        Start the arm controller

        Returns:
            True if started successfully
        """
        if self.simulate:
            logger.info("Running in simulation mode - no actual arm movement")
            # Initialize simulated positions
            self._current_left_position = self._rest_position
            self._current_right_position = self._rest_position
            return True

        try:
            # Initialize ROS2
            if not rclpy.ok():
                rclpy.init()

            # Create node
            self._node = rclpy.create_node('k1_arm_controller')

            # Subscribe to joint states
            self._joint_state_sub = self._node.create_subscription(
                JointState,
                self.config.joint_state_topic,
                self._joint_state_callback,
                10
            )

            # Create publishers for arm trajectories
            self._left_arm_pub = self._node.create_publisher(
                JointTrajectory,
                self.config.trajectory_topic_left,
                10
            )

            self._right_arm_pub = self._node.create_publisher(
                JointTrajectory,
                self.config.trajectory_topic_right,
                10
            )

            # Start ROS2 spin thread
            self._shutdown_event.clear()
            self._spin_thread = threading.Thread(target=self._spin_ros2, daemon=True)
            self._spin_thread.start()

            logger.info("K-1 Arm Controller started with ROS2")
            return True

        except Exception as e:
            logger.error(f"Failed to start arm controller: {e}")
            return False

    def stop(self) -> None:
        """Stop the arm controller"""
        if self.simulate:
            return

        # Shutdown ROS2
        self._shutdown_event.set()
        if self._spin_thread:
            self._spin_thread.join(timeout=2.0)

        if self._node:
            self._node.destroy_node()

        logger.info("K-1 Arm Controller stopped")

    def _spin_ros2(self) -> None:
        """ROS2 spin thread"""
        while not self._shutdown_event.is_set() and rclpy.ok():
            try:
                rclpy.spin_once(self._node, timeout_sec=0.1)
            except Exception as e:
                logger.error(f"ROS2 spin error: {e}")

    def _joint_state_callback(self, msg: 'JointState') -> None:
        """Process joint state updates from ROS2"""
        with self._lock:
            try:
                # Extract left arm joints (Shoulder Pitch, Roll, Yaw, Elbow)
                left_indices = [msg.name.index(j) for j in self.config.left_arm_joints]
                self._current_left_position = JointPosition(
                    shoulder_pitch=np.degrees(msg.position[left_indices[0]]),
                    shoulder_roll=np.degrees(msg.position[left_indices[1]]),
                    shoulder_yaw=np.degrees(msg.position[left_indices[2]]),
                    elbow=np.degrees(msg.position[left_indices[3]])
                )

                # Extract right arm joints (Shoulder Pitch, Roll, Yaw, Elbow)
                right_indices = [msg.name.index(j) for j in self.config.right_arm_joints]
                self._current_right_position = JointPosition(
                    shoulder_pitch=np.degrees(msg.position[right_indices[0]]),
                    shoulder_roll=np.degrees(msg.position[right_indices[1]]),
                    shoulder_yaw=np.degrees(msg.position[right_indices[2]]),
                    elbow=np.degrees(msg.position[right_indices[3]])
                )

            except (ValueError, IndexError) as e:
                logger.warning(f"Failed to parse joint states: {e}")

    def _apply_limits(self, position: JointPosition, arm: ArmSide) -> JointPosition:
        """Apply safety limits to joint positions based on arm side"""
        # Use arm-specific limits
        if arm == ArmSide.LEFT:
            return JointPosition(
                shoulder_pitch=np.clip(position.shoulder_pitch,
                                      *self.config.left_shoulder_pitch_limits),
                shoulder_roll=np.clip(position.shoulder_roll,
                                     *self.config.left_shoulder_roll_limits),
                shoulder_yaw=np.clip(position.shoulder_yaw,
                                    *self.config.left_shoulder_yaw_limits),
                elbow=np.clip(position.elbow,
                             *self.config.left_elbow_limits)
            )
        else:  # RIGHT or BOTH (use right limits for both)
            return JointPosition(
                shoulder_pitch=np.clip(position.shoulder_pitch,
                                      *self.config.right_shoulder_pitch_limits),
                shoulder_roll=np.clip(position.shoulder_roll,
                                     *self.config.right_shoulder_roll_limits),
                shoulder_yaw=np.clip(position.shoulder_yaw,
                                    *self.config.right_shoulder_yaw_limits),
                elbow=np.clip(position.elbow,
                             *self.config.right_elbow_limits)
            )

    def _send_trajectory(self, arm: ArmSide, positions: List[JointPosition],
                        durations: List[float]) -> bool:
        """
        Send joint trajectory to arm(s)

        Args:
            arm: Which arm to control
            positions: List of joint positions for waypoints
            durations: Time to reach each waypoint (seconds from start)

        Returns:
            True if trajectory sent successfully
        """
        if self.simulate:
            logger.info(f"[SIMULATED] Moving {arm.value} arm through {len(positions)} waypoints")
            # Update simulated position to final waypoint
            final_pos = self._apply_limits(positions[-1], arm)
            with self._lock:
                if arm in [ArmSide.LEFT, ArmSide.BOTH]:
                    self._current_left_position = final_pos
                if arm in [ArmSide.RIGHT, ArmSide.BOTH]:
                    self._current_right_position = final_pos
            return True

        # Apply safety limits
        safe_positions = [self._apply_limits(p, arm) for p in positions]

        # Build trajectory message
        trajectory = JointTrajectory()
        trajectory.header = Header()
        trajectory.header.stamp = self._node.get_clock().now().to_msg()

        # Set joint names based on arm
        arms_to_move = []
        if arm == ArmSide.LEFT:
            trajectory.joint_names = self.config.left_arm_joints
            arms_to_move = [(self._left_arm_pub, self.config.left_arm_joints)]
        elif arm == ArmSide.RIGHT:
            trajectory.joint_names = self.config.right_arm_joints
            arms_to_move = [(self._right_arm_pub, self.config.right_arm_joints)]
        else:  # BOTH
            # Send separate trajectories to each arm
            arms_to_move = [
                (self._left_arm_pub, self.config.left_arm_joints),
                (self._right_arm_pub, self.config.right_arm_joints)
            ]

        # Create trajectory points
        for i, (pos, duration) in enumerate(zip(safe_positions, durations)):
            point = JointTrajectoryPoint()

            # Convert degrees to radians
            # K-1 arm joints: Shoulder Pitch, Roll, Yaw, Elbow
            point.positions = [
                np.radians(pos.shoulder_pitch),
                np.radians(pos.shoulder_roll),
                np.radians(pos.shoulder_yaw),
                np.radians(pos.elbow)
            ]

            # Set time from start
            point.time_from_start = Duration(sec=int(duration),
                                            nanosec=int((duration % 1) * 1e9))

            trajectory.points.append(point)

        # Publish trajectory
        for pub, joint_names in arms_to_move:
            traj = JointTrajectory()
            traj.header = trajectory.header
            traj.joint_names = joint_names
            traj.points = trajectory.points
            pub.publish(traj)

        logger.info(f"Sent trajectory to {arm.value} arm with {len(positions)} waypoints")
        return True

    # ========================================================================
    # Gesture Commands
    # ========================================================================

    def move_to_rest(self, arm: ArmSide = ArmSide.BOTH) -> bool:
        """
        Move arm(s) to rest position (at sides)

        Args:
            arm: Which arm(s) to move

        Returns:
            True if movement started successfully
        """
        return self._send_trajectory(
            arm,
            [self._rest_position],
            [self.config.default_duration]
        )

    def wave(self, arm: ArmSide = ArmSide.RIGHT) -> bool:
        """
        Perform a friendly wave gesture

        This creates a natural wave motion by:
        1. Raising the arm to shoulder height
        2. Rotating shoulder yaw left-right multiple times (twisting motion)
        3. Lowering arm back to rest

        Args:
            arm: Which arm to wave with (default: right)

        Returns:
            True if wave started successfully
        """
        # Define wave positions
        wave_height = JointPosition(
            shoulder_pitch=0,      # Level
            shoulder_roll=90,      # Arm out to side at shoulder height
            shoulder_yaw=0,        # Neutral (no twist)
            elbow=90               # Elbow bent 90 degrees (hand up)
        )

        # Create wave motion with shoulder yaw rotation (twisting)
        positions = [wave_height]  # Start: raise arm

        # Wave cycles (shoulder yaw rotation)
        cycle_duration = self.config.wave_duration / (self.config.wave_cycles * 2 + 2)
        current_time = self.config.default_duration

        for i in range(self.config.wave_cycles):
            # Wave left (internal rotation)
            positions.append(JointPosition(
                shoulder_pitch=0,
                shoulder_roll=90,
                shoulder_yaw=-40,  # Twist inward
                elbow=90
            ))
            # Wave right (external rotation)
            positions.append(JointPosition(
                shoulder_pitch=0,
                shoulder_roll=90,
                shoulder_yaw=40,   # Twist outward
                elbow=90
            ))

        # Return to neutral shoulder yaw
        positions.append(wave_height)

        # Lower arm back to rest
        positions.append(self._rest_position)

        # Calculate durations for each waypoint
        durations = [self.config.default_duration]  # Raise arm
        for i in range(self.config.wave_cycles * 2):
            current_time += cycle_duration
            durations.append(current_time)
        durations.append(current_time + cycle_duration)  # Neutral yaw
        durations.append(current_time + cycle_duration + self.config.default_duration)  # Lower

        logger.info(f"Waving with {arm.value} arm ({self.config.wave_cycles} cycles)")
        return self._send_trajectory(arm, positions, durations)

    def point(self, direction: Tuple[float, float, float],
              arm: ArmSide = ArmSide.RIGHT) -> bool:
        """
        Point in a specific direction

        Args:
            direction: (x, y, z) direction vector to point at
            arm: Which arm to use

        Returns:
            True if pointing gesture started
        """
        # Calculate angles to point in direction
        x, y, z = direction

        # Calculate shoulder angles
        horizontal_angle = np.degrees(np.arctan2(x, z))
        vertical_angle = np.degrees(np.arctan2(y, np.sqrt(x**2 + z**2)))

        point_position = JointPosition(
            shoulder_pitch=vertical_angle,
            shoulder_roll=horizontal_angle,
            shoulder_yaw=0,  # No twist when pointing
            elbow=180        # Arm straight (note: limited to 129° on right, 39° on left)
        )

        return self._send_trajectory(
            arm,
            [point_position, self._rest_position],
            [self.config.default_duration, self.config.default_duration * 2]
        )

    def nod_with_head(self) -> None:
        """
        Combine arm gesture with head nod for more expressive greeting

        Note: This requires integration with the gimbal controller.
        For now, this is a placeholder for future integration.
        """
        logger.info("Head nod gesture (requires gimbal integration)")
        # TODO: Integrate with whoami.gimbal_control.GimbalController
        # Example:
        #   from whoami.gimbal_control import create_gimbal_controller
        #   gimbal = create_gimbal_controller()
        #   gimbal.move_relative(0, -10)  # Nod down
        #   time.sleep(0.3)
        #   gimbal.move_relative(0, 10)   # Nod up

    def get_current_position(self, arm: ArmSide) -> Optional[JointPosition]:
        """
        Get current arm position

        Args:
            arm: Which arm to query

        Returns:
            Current joint position or None if unavailable
        """
        with self._lock:
            if arm == ArmSide.LEFT:
                return self._current_left_position
            elif arm == ArmSide.RIGHT:
                return self._current_right_position
            else:
                return None

    def is_at_rest(self, arm: ArmSide, tolerance: float = 5.0) -> bool:
        """
        Check if arm is at rest position

        Args:
            arm: Which arm to check
            tolerance: Position tolerance in degrees

        Returns:
            True if arm is within tolerance of rest position
        """
        pos = self.get_current_position(arm)
        if pos is None:
            return False

        rest = self._rest_position
        return (abs(pos.shoulder_pitch - rest.shoulder_pitch) < tolerance and
                abs(pos.shoulder_roll - rest.shoulder_roll) < tolerance and
                abs(pos.shoulder_yaw - rest.shoulder_yaw) < tolerance and
                abs(pos.elbow - rest.elbow) < tolerance)


# ============================================================================
# Factory Function
# ============================================================================

def create_k1_arm_controller(simulate: bool = None) -> K1ArmController:
    """
    Create K-1 arm controller with default configuration

    Args:
        simulate: Force simulation mode (default: auto-detect based on ROS2 availability)

    Returns:
        Configured K1ArmController
    """
    if simulate is None:
        simulate = not ROS2_AVAILABLE

    return K1ArmController(simulate=simulate)


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create controller in simulation mode for testing
    controller = create_k1_arm_controller(simulate=True)

    if controller.start():
        print("K-1 Arm Controller started!")

        # Test wave gesture
        print("\nTesting wave gesture...")
        controller.wave(ArmSide.RIGHT)
        time.sleep(4)

        # Test pointing
        print("\nTesting point gesture...")
        controller.point((1, 0.5, 2), ArmSide.RIGHT)
        time.sleep(3)

        # Return to rest
        print("\nReturning to rest...")
        controller.move_to_rest()
        time.sleep(2)

        controller.stop()
        print("\nTest complete!")
    else:
        print("Failed to start controller")
