"""
3DOF Coordinated Gimbal System for Orbital 3D Scanning

This module implements a 3-axis serial kinematic chain where:
- Servo 1 (Base/Yaw): Rotates entire assembly around vertical axis
- Servo 2 (Shoulder/Pitch): Extends horizontally to position camera offset
- Servo 3 (Wrist/Roll): Rotates camera for scanning

Key Feature: Camera orbits around objects with INWARD-FACING orientation
- All three servos coordinate to create circular paths in 3D space
- Camera always points toward scan target
- Roll axis can spin for 360° depth capture while orbiting

Mathematical Framework:
- Forward Kinematics: Servo angles → Camera pose (position + orientation)
- Inverse Kinematics: Desired camera pose → Servo angles
- Trajectory Generation: Creates smooth orbital paths around objects
- Synchronized Control: All servos move together for smooth motion
"""

import numpy as np
import logging
import time
import threading
from typing import Optional, Tuple, List, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes and Configuration
# ============================================================================

@dataclass
class KinematicParameters:
    """
    Physical parameters of the 3DOF gimbal kinematic chain

    Coordinate System:
        - Origin at Servo 1 (base)
        - Z-axis: vertical (up)
        - X-axis: forward
        - Y-axis: left (right-hand rule)

    Servo Chain:
        Servo 1 (Yaw) → rotates around Z-axis at origin
        Servo 2 (Pitch) → offset vertically, rotates around Y-axis
        Servo 3 (Roll) → offset horizontally along Servo 2 arm, rotates around arm axis
    """
    # Link lengths (meters) - adjust these for your physical build
    base_height: float = 0.10          # Height from Servo 1 to Servo 2 (10cm default)
    arm_length: float = 0.15           # Distance from Servo 2 to Servo 3 (15cm default)
    camera_offset: float = 0.05        # Distance from Servo 3 to camera optical center (5cm)

    # Servo limits (degrees)
    yaw_min: float = -180.0            # Servo 1 limits
    yaw_max: float = 180.0
    pitch_min: float = -90.0           # Servo 2 limits
    pitch_max: float = 90.0
    roll_min: float = -180.0           # Servo 3 limits
    roll_max: float = 180.0

    # Servo directions (1 or -1 to flip rotation direction)
    yaw_direction: int = 1
    pitch_direction: int = 1
    roll_direction: int = 1

    # Servo zero positions (degrees) - calibration offsets
    yaw_zero: float = 0.0
    pitch_zero: float = 0.0
    roll_zero: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'base_height': self.base_height,
            'arm_length': self.arm_length,
            'camera_offset': self.camera_offset,
            'yaw_limits': [self.yaw_min, self.yaw_max],
            'pitch_limits': [self.pitch_min, self.pitch_max],
            'roll_limits': [self.roll_min, self.roll_max],
            'directions': [self.yaw_direction, self.pitch_direction, self.roll_direction],
            'zero_positions': [self.yaw_zero, self.pitch_zero, self.roll_zero]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KinematicParameters':
        """Create from dictionary"""
        return cls(
            base_height=data.get('base_height', 0.10),
            arm_length=data.get('arm_length', 0.15),
            camera_offset=data.get('camera_offset', 0.05),
            yaw_min=data.get('yaw_limits', [-180, 180])[0],
            yaw_max=data.get('yaw_limits', [-180, 180])[1],
            pitch_min=data.get('pitch_limits', [-90, 90])[0],
            pitch_max=data.get('pitch_limits', [-90, 90])[1],
            roll_min=data.get('roll_limits', [-180, 180])[0],
            roll_max=data.get('roll_limits', [-180, 180])[1],
            yaw_direction=data.get('directions', [1, 1, 1])[0],
            pitch_direction=data.get('directions', [1, 1, 1])[1],
            roll_direction=data.get('directions', [1, 1, 1])[2],
            yaw_zero=data.get('zero_positions', [0, 0, 0])[0],
            pitch_zero=data.get('zero_positions', [0, 0, 0])[1],
            roll_zero=data.get('zero_positions', [0, 0, 0])[2]
        )


@dataclass
class CameraPose:
    """
    Complete camera pose in 3D space

    Attributes:
        position: (x, y, z) camera position in world frame
        forward: unit vector pointing where camera looks
        up: unit vector pointing camera "up" direction
        right: unit vector pointing camera "right" (computed from forward × up)
    """
    position: np.ndarray        # shape (3,)
    forward: np.ndarray         # shape (3,) - unit vector
    up: np.ndarray             # shape (3,) - unit vector

    def __post_init__(self):
        """Ensure vectors are unit vectors"""
        self.forward = self.forward / np.linalg.norm(self.forward)
        self.up = self.up / np.linalg.norm(self.up)

        # Ensure up is perpendicular to forward
        self.up = self.up - np.dot(self.up, self.forward) * self.forward
        self.up = self.up / np.linalg.norm(self.up)

    @property
    def right(self) -> np.ndarray:
        """Compute right vector from forward and up"""
        return np.cross(self.forward, self.up)

    def rotation_matrix(self) -> np.ndarray:
        """
        Get 3x3 rotation matrix representing camera orientation
        Columns are: [right, up, -forward] (OpenCV convention)
        """
        return np.column_stack([self.right, self.up, -self.forward])

    def look_at(self, target: np.ndarray) -> 'CameraPose':
        """
        Create new pose looking at target point while maintaining roll

        Args:
            target: (x, y, z) point to look at

        Returns:
            New CameraPose with forward vector pointing at target
        """
        new_forward = target - self.position
        new_forward = new_forward / np.linalg.norm(new_forward)

        # Maintain roll by keeping up vector in same plane
        # Project current up onto plane perpendicular to new forward
        new_up = self.up - np.dot(self.up, new_forward) * new_forward

        # Handle degenerate case (looking straight up/down)
        if np.linalg.norm(new_up) < 0.01:
            # Use right vector to reconstruct up
            temp_right = self.right - np.dot(self.right, new_forward) * new_forward
            if np.linalg.norm(temp_right) > 0.01:
                temp_right = temp_right / np.linalg.norm(temp_right)
                new_up = np.cross(new_forward, temp_right)
            else:
                # Default up vector
                new_up = np.array([0, 0, 1])

        new_up = new_up / np.linalg.norm(new_up)

        return CameraPose(
            position=self.position.copy(),
            forward=new_forward,
            up=new_up
        )


@dataclass
class JointAngles:
    """
    Joint angles for the 3DOF gimbal (in degrees)
    """
    yaw: float      # Servo 1 - base rotation
    pitch: float    # Servo 2 - shoulder rotation
    roll: float     # Servo 3 - camera roll

    def to_radians(self) -> Tuple[float, float, float]:
        """Convert to radians"""
        return (
            np.deg2rad(self.yaw),
            np.deg2rad(self.pitch),
            np.deg2rad(self.roll)
        )

    def to_array(self) -> np.ndarray:
        """Convert to numpy array"""
        return np.array([self.yaw, self.pitch, self.roll])

    def __str__(self) -> str:
        return f"Yaw: {self.yaw:.1f}°, Pitch: {self.pitch:.1f}°, Roll: {self.roll:.1f}°"


class OrbitAxis(Enum):
    """Predefined orbital planes"""
    HORIZONTAL = "horizontal"      # XY plane (equatorial)
    VERTICAL_X = "vertical_x"      # XZ plane (meridian along X)
    VERTICAL_Y = "vertical_y"      # YZ plane (meridian along Y)
    CUSTOM = "custom"              # Custom axis vector


# ============================================================================
# Kinematic Solver
# ============================================================================

class Kinematics3DOF:
    """
    Forward and inverse kinematics solver for 3DOF gimbal

    This solver handles the transformation between:
    - Joint space: (yaw, pitch, roll) servo angles
    - Task space: Camera pose (position + orientation) in 3D

    Key constraints:
    - Camera position is determined by yaw and pitch
    - Camera orientation is constrained to look at target
    - Roll determines camera rotation around optical axis
    """

    def __init__(self, params: KinematicParameters):
        """
        Initialize kinematics solver

        Args:
            params: Physical parameters of the gimbal
        """
        self.params = params
        self.logger = logging.getLogger(f"{__name__}.Kinematics3DOF")

    def forward_kinematics(self, angles: JointAngles) -> CameraPose:
        """
        Compute camera pose from joint angles

        Args:
            angles: Joint angles (yaw, pitch, roll)

        Returns:
            CameraPose: Camera position and orientation in world frame

        Process:
            1. Apply yaw rotation to base frame
            2. Translate up to shoulder (Servo 2)
            3. Apply pitch rotation
            4. Translate along arm to wrist (Servo 3)
            5. Apply roll rotation
            6. Translate to camera optical center
        """
        yaw_rad, pitch_rad, roll_rad = angles.to_radians()

        # Start at origin (Servo 1 position)
        position = np.array([0.0, 0.0, 0.0])

        # Rotation matrix for yaw (around Z-axis)
        cos_yaw, sin_yaw = np.cos(yaw_rad), np.sin(yaw_rad)
        R_yaw = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw,  cos_yaw, 0],
            [0,        0,       1]
        ])

        # Move up to Servo 2 position
        position += np.array([0, 0, self.params.base_height])

        # Rotation matrix for pitch (around Y-axis in yawed frame)
        cos_pitch, sin_pitch = np.cos(pitch_rad), np.sin(pitch_rad)
        R_pitch_local = np.array([
            [cos_pitch,  0, sin_pitch],
            [0,          1, 0],
            [-sin_pitch, 0, cos_pitch]
        ])

        # Combine yaw and pitch
        R_yaw_pitch = R_yaw @ R_pitch_local

        # Arm extends along X-axis in pitched frame
        arm_vector = R_yaw_pitch @ np.array([self.params.arm_length, 0, 0])
        position += arm_vector

        # Camera offset along arm direction (after pitch)
        camera_vector = R_yaw_pitch @ np.array([self.params.camera_offset, 0, 0])
        position += camera_vector

        # Roll rotation (around X-axis in pitched frame)
        cos_roll, sin_roll = np.cos(roll_rad), np.sin(roll_rad)
        R_roll_local = np.array([
            [1, 0,         0],
            [0, cos_roll, -sin_roll],
            [0, sin_roll,  cos_roll]
        ])

        # Complete rotation matrix
        R_total = R_yaw_pitch @ R_roll_local

        # Camera forward is along +X in local frame (after all rotations)
        forward = R_total @ np.array([1, 0, 0])

        # Camera up is along +Z in local frame (after all rotations)
        up = R_total @ np.array([0, 0, 1])

        return CameraPose(
            position=position,
            forward=forward,
            up=up
        )

    def inverse_kinematics(
        self,
        target_position: np.ndarray,
        camera_position: np.ndarray,
        camera_roll: float = 0.0,
        initial_guess: Optional[JointAngles] = None
    ) -> Optional[JointAngles]:
        """
        Compute joint angles to achieve desired camera pose

        Args:
            target_position: (x, y, z) point camera should look at
            camera_position: (x, y, z) desired camera location
            camera_roll: desired roll angle (degrees) around optical axis
            initial_guess: starting point for iterative solver (optional)

        Returns:
            JointAngles if solution found, None if unreachable

        Algorithm:
            1. Compute required yaw from XY projection
            2. Compute required pitch from geometry
            3. Verify reachability
            4. Apply roll directly

        Constraints:
            - Camera must be within workspace (reachable by arm lengths)
            - Joint angles must be within limits
            - Camera forward vector must point at target
        """
        # Vector from origin to desired camera position
        cam_pos = camera_position

        # Check if position is reachable
        distance_xy = np.sqrt(cam_pos[0]**2 + cam_pos[1]**2)
        distance_z = cam_pos[2]

        # Compute required arm extension
        height_diff = distance_z - self.params.base_height
        total_arm = self.params.arm_length + self.params.camera_offset

        # Check reachability
        required_reach = np.sqrt(distance_xy**2 + height_diff**2)
        if required_reach > total_arm * 1.001:  # Small tolerance
            self.logger.warning(
                f"Position unreachable: needs {required_reach:.3f}m, "
                f"have {total_arm:.3f}m"
            )
            return None

        # 1. SOLVE YAW
        # Yaw is determined by XY position of camera
        yaw_rad = np.arctan2(cam_pos[1], cam_pos[0])
        yaw_deg = np.rad2deg(yaw_rad)

        # Apply direction and zero offset
        yaw_deg = yaw_deg * self.params.yaw_direction + self.params.yaw_zero

        # Normalize to [-180, 180]
        yaw_deg = self._normalize_angle(yaw_deg)

        # Check yaw limits
        if not (self.params.yaw_min <= yaw_deg <= self.params.yaw_max):
            self.logger.warning(f"Yaw {yaw_deg:.1f}° outside limits")
            return None

        # 2. SOLVE PITCH
        # After yaw rotation, arm extends in X direction
        # Need to reach (distance_xy, 0, height_diff) in yawed frame

        # Horizontal distance from Servo 2 to camera
        horizontal_reach = distance_xy

        # Vertical distance from Servo 2 to camera
        vertical_reach = height_diff

        # Use law of cosines
        # We have a right triangle in the YZ plane (after yaw rotation)
        reach_distance = np.sqrt(horizontal_reach**2 + vertical_reach**2)

        # Pitch angle to achieve this reach
        if reach_distance > 0:
            pitch_rad = np.arctan2(vertical_reach, horizontal_reach)
        else:
            pitch_rad = 0

        pitch_deg = np.rad2deg(pitch_rad)

        # Apply direction and zero offset
        pitch_deg = pitch_deg * self.params.pitch_direction + self.params.pitch_zero

        # Normalize
        pitch_deg = self._normalize_angle(pitch_deg)

        # Check pitch limits
        if not (self.params.pitch_min <= pitch_deg <= self.params.pitch_max):
            self.logger.warning(f"Pitch {pitch_deg:.1f}° outside limits")
            return None

        # 3. APPLY ROLL
        # Roll is independent and directly specified
        roll_deg = camera_roll * self.params.roll_direction + self.params.roll_zero
        roll_deg = self._normalize_angle(roll_deg)

        # Check roll limits
        if not (self.params.roll_min <= roll_deg <= self.params.roll_max):
            self.logger.warning(f"Roll {roll_deg:.1f}° outside limits")
            return None

        # 4. VERIFY SOLUTION
        # Compute forward kinematics and check if we hit target
        solution = JointAngles(yaw=yaw_deg, pitch=pitch_deg, roll=roll_deg)
        computed_pose = self.forward_kinematics(solution)

        # Check position error
        position_error = np.linalg.norm(computed_pose.position - camera_position)
        if position_error > 0.01:  # 1cm tolerance
            self.logger.warning(
                f"Position error too large: {position_error*1000:.1f}mm"
            )
            # Still return solution, might be close enough

        # Check if camera is looking at target
        to_target = target_position - computed_pose.position
        to_target_normalized = to_target / np.linalg.norm(to_target)

        dot_product = np.dot(computed_pose.forward, to_target_normalized)
        angle_error = np.rad2deg(np.arccos(np.clip(dot_product, -1, 1)))

        if angle_error > 5.0:  # 5 degree tolerance
            self.logger.warning(
                f"Camera not pointing at target: {angle_error:.1f}° off"
            )

        return solution

    def _normalize_angle(self, angle_deg: float) -> float:
        """Normalize angle to [-180, 180] range"""
        while angle_deg > 180:
            angle_deg -= 360
        while angle_deg < -180:
            angle_deg += 360
        return angle_deg

    def check_reachability(self, point: np.ndarray) -> bool:
        """
        Check if a point in 3D space is reachable by the camera

        Args:
            point: (x, y, z) position to check

        Returns:
            True if reachable, False otherwise
        """
        distance_xy = np.sqrt(point[0]**2 + point[1]**2)
        height_diff = point[2] - self.params.base_height
        required_reach = np.sqrt(distance_xy**2 + height_diff**2)
        total_arm = self.params.arm_length + self.params.camera_offset

        return required_reach <= total_arm


# ============================================================================
# Orbital Path Generator
# ============================================================================

class OrbitalPathGenerator:
    """
    Generates circular and spherical paths for 3D scanning

    Creates trajectories where camera orbits around object with
    inward-facing orientation for complete 360° scanning
    """

    def __init__(self, kinematics: Kinematics3DOF):
        """
        Initialize path generator

        Args:
            kinematics: Kinematics solver for IK
        """
        self.kinematics = kinematics
        self.logger = logging.getLogger(f"{__name__}.OrbitalPathGenerator")

    def generate_circular_orbit(
        self,
        center: np.ndarray,
        radius: float,
        axis: np.ndarray,
        num_points: int,
        start_angle: float = 0.0,
        end_angle: float = 360.0,
        camera_roll: float = 0.0
    ) -> List[JointAngles]:
        """
        Generate circular orbital path around a center point

        Args:
            center: (x, y, z) center point to orbit around
            radius: orbital radius (meters)
            axis: (x, y, z) unit vector defining orbital plane normal
            num_points: number of waypoints in orbit
            start_angle: starting angle (degrees)
            end_angle: ending angle (degrees)
            camera_roll: camera roll angle (degrees)

        Returns:
            List of JointAngles for each waypoint

        The orbit is a circle in a plane perpendicular to axis vector
        Camera always faces inward toward center point
        """
        axis = np.array(axis)
        axis = axis / np.linalg.norm(axis)  # Normalize

        # Create two perpendicular vectors in the orbital plane
        # Choose an arbitrary vector not parallel to axis
        if abs(axis[2]) < 0.9:
            arbitrary = np.array([0, 0, 1])
        else:
            arbitrary = np.array([1, 0, 0])

        # First basis vector in plane
        v1 = np.cross(axis, arbitrary)
        v1 = v1 / np.linalg.norm(v1)

        # Second basis vector in plane (perpendicular to v1 and axis)
        v2 = np.cross(axis, v1)
        v2 = v2 / np.linalg.norm(v2)

        # Generate orbit points
        angles = np.linspace(
            np.deg2rad(start_angle),
            np.deg2rad(end_angle),
            num_points,
            endpoint=(end_angle - start_angle >= 360)
        )

        joint_angles_list = []

        for angle in angles:
            # Position on circle
            position = (
                center +
                radius * np.cos(angle) * v1 +
                radius * np.sin(angle) * v2
            )

            # Solve IK to point camera at center from this position
            joint_angles = self.kinematics.inverse_kinematics(
                target_position=center,
                camera_position=position,
                camera_roll=camera_roll
            )

            if joint_angles is None:
                self.logger.warning(
                    f"Point unreachable at angle {np.rad2deg(angle):.1f}°"
                )
                continue

            joint_angles_list.append(joint_angles)

        return joint_angles_list

    def generate_horizontal_orbit(
        self,
        center: np.ndarray,
        radius: float,
        num_points: int = 36,
        camera_roll: float = 0.0
    ) -> List[JointAngles]:
        """
        Generate horizontal orbit (XY plane / equatorial)

        Args:
            center: (x, y, z) center point
            radius: orbital radius
            num_points: number of waypoints (default 36 = 10° steps)
            camera_roll: camera roll angle

        Returns:
            List of joint angles for orbit
        """
        return self.generate_circular_orbit(
            center=center,
            radius=radius,
            axis=np.array([0, 0, 1]),  # Z-axis (vertical)
            num_points=num_points,
            camera_roll=camera_roll
        )

    def generate_vertical_orbit(
        self,
        center: np.ndarray,
        radius: float,
        axis_direction: str = 'x',
        num_points: int = 36,
        camera_roll: float = 0.0
    ) -> List[JointAngles]:
        """
        Generate vertical orbit (meridian)

        Args:
            center: (x, y, z) center point
            radius: orbital radius
            axis_direction: 'x' or 'y' - which horizontal axis to use
            num_points: number of waypoints
            camera_roll: camera roll angle

        Returns:
            List of joint angles for orbit
        """
        if axis_direction.lower() == 'x':
            axis = np.array([1, 0, 0])  # Orbit in YZ plane
        elif axis_direction.lower() == 'y':
            axis = np.array([0, 1, 0])  # Orbit in XZ plane
        else:
            raise ValueError(f"Invalid axis_direction: {axis_direction}")

        return self.generate_circular_orbit(
            center=center,
            radius=radius,
            axis=axis,
            num_points=num_points,
            camera_roll=camera_roll
        )

    def generate_spherical_scan(
        self,
        center: np.ndarray,
        radius: float,
        num_rings: int = 5,
        points_per_ring: int = 36,
        camera_roll: float = 0.0
    ) -> List[JointAngles]:
        """
        Generate spherical scan pattern with multiple orbital rings

        Args:
            center: (x, y, z) center point
            radius: orbital radius
            num_rings: number of latitude rings
            points_per_ring: points per ring
            camera_roll: camera roll angle

        Returns:
            List of joint angles covering sphere

        Creates orbits at different elevation angles to cover sphere
        """
        all_waypoints = []

        # Generate rings at different elevations
        # From bottom to top (or vice versa)
        for ring_idx in range(num_rings):
            # Elevation angle from -90° to +90°
            elevation = -90 + (180.0 * (ring_idx + 1) / (num_rings + 1))
            elevation_rad = np.deg2rad(elevation)

            # Compute ring radius at this elevation
            ring_radius = radius * np.cos(elevation_rad)

            # Height of this ring
            ring_height = radius * np.sin(elevation_rad)

            # Center of this ring
            ring_center = center + np.array([0, 0, ring_height])

            # Generate horizontal orbit at this elevation
            if ring_radius > 0.01:  # Skip if too close to poles
                ring_waypoints = self.generate_horizontal_orbit(
                    center=ring_center,
                    radius=ring_radius,
                    num_points=points_per_ring,
                    camera_roll=camera_roll
                )
                all_waypoints.extend(ring_waypoints)

        return all_waypoints

    def generate_spiral_scan(
        self,
        center: np.ndarray,
        radius: float,
        height_range: Tuple[float, float],
        num_revolutions: int = 3,
        points_per_revolution: int = 36,
        camera_roll: float = 0.0
    ) -> List[JointAngles]:
        """
        Generate spiral scan pattern

        Args:
            center: (x, y, z) center point
            radius: orbital radius
            height_range: (min_height, max_height) relative to center
            num_revolutions: number of times to circle while changing height
            points_per_revolution: density of waypoints
            camera_roll: camera roll angle

        Returns:
            List of joint angles for spiral path
        """
        total_points = num_revolutions * points_per_revolution
        waypoints = []

        for i in range(total_points):
            # Angle around circle
            angle = 2 * np.pi * (i / points_per_revolution)

            # Height varies linearly
            t = i / (total_points - 1) if total_points > 1 else 0
            height = height_range[0] + t * (height_range[1] - height_range[0])

            # Position on spiral
            position = center + np.array([
                radius * np.cos(angle),
                radius * np.sin(angle),
                height
            ])

            # Solve IK
            joint_angles = self.kinematics.inverse_kinematics(
                target_position=center,
                camera_position=position,
                camera_roll=camera_roll
            )

            if joint_angles is not None:
                waypoints.append(joint_angles)

        return waypoints


# ============================================================================
# Trajectory Interpolator
# ============================================================================

class TrajectoryInterpolator:
    """
    Smooth trajectory generation between waypoints

    Ensures all three servos move synchronously and smoothly
    """

    @staticmethod
    def interpolate_linear(
        start: JointAngles,
        end: JointAngles,
        num_steps: int
    ) -> List[JointAngles]:
        """
        Linear interpolation between two joint configurations

        Args:
            start: Starting joint angles
            end: Ending joint angles
            num_steps: Number of intermediate steps

        Returns:
            List of interpolated joint angles
        """
        if num_steps < 2:
            return [start, end]

        start_arr = start.to_array()
        end_arr = end.to_array()

        interpolated = []
        for i in range(num_steps):
            t = i / (num_steps - 1)
            angles = start_arr + t * (end_arr - start_arr)
            interpolated.append(JointAngles(*angles))

        return interpolated

    @staticmethod
    def interpolate_path(
        waypoints: List[JointAngles],
        steps_per_segment: int = 10
    ) -> List[JointAngles]:
        """
        Interpolate smooth path through multiple waypoints

        Args:
            waypoints: List of waypoint joint angles
            steps_per_segment: Interpolation density between waypoints

        Returns:
            Smooth trajectory through all waypoints
        """
        if len(waypoints) < 2:
            return waypoints

        smooth_path = []
        for i in range(len(waypoints) - 1):
            segment = TrajectoryInterpolator.interpolate_linear(
                waypoints[i],
                waypoints[i + 1],
                steps_per_segment
            )
            # Avoid duplicating waypoints
            if i < len(waypoints) - 2:
                smooth_path.extend(segment[:-1])
            else:
                smooth_path.extend(segment)

        return smooth_path

    @staticmethod
    def compute_trajectory_duration(
        waypoints: List[JointAngles],
        max_velocities: Tuple[float, float, float] = (60, 60, 120)  # deg/s
    ) -> float:
        """
        Compute total time needed for trajectory

        Args:
            waypoints: Path to execute
            max_velocities: (yaw, pitch, roll) max speeds in deg/s

        Returns:
            Total duration in seconds
        """
        if len(waypoints) < 2:
            return 0.0

        total_time = 0.0

        for i in range(len(waypoints) - 1):
            delta = waypoints[i + 1].to_array() - waypoints[i].to_array()

            # Time needed for each joint
            times = np.abs(delta) / np.array(max_velocities)

            # Slowest joint determines segment time
            segment_time = np.max(times)
            total_time += segment_time

        return total_time


# ============================================================================
# Configuration Management
# ============================================================================

class Gimbal3DOFConfig:
    """Configuration file management for 3DOF gimbal"""

    DEFAULT_CONFIG = {
        'kinematics': {
            'base_height': 0.10,
            'arm_length': 0.15,
            'camera_offset': 0.05,
            'yaw_limits': [-180, 180],
            'pitch_limits': [-90, 90],
            'roll_limits': [-180, 180],
            'directions': [1, 1, 1],
            'zero_positions': [0, 0, 0]
        },
        'servo_ids': {
            'yaw': 1,
            'pitch': 2,
            'roll': 3
        },
        'serial': {
            'port': '/dev/ttyUSB0',
            'baudrate': 1000000
        },
        'motion': {
            'max_velocities': [60, 60, 120],  # deg/s
            'acceleration': 100,
            'interpolation_steps': 10
        },
        'scanning': {
            'default_radius': 0.3,
            'default_num_points': 36,
            'default_roll_speed': 30  # deg/s for continuous roll scanning
        }
    }

    @classmethod
    def load(cls, config_path: Path) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        if not config_path.exists():
            logger.warning(f"Config not found: {config_path}, using defaults")
            return cls.DEFAULT_CONFIG.copy()

        with open(config_path, 'r') as f:
            config = json.load(f)

        # Merge with defaults
        merged = cls.DEFAULT_CONFIG.copy()
        merged.update(config)
        return merged

    @classmethod
    def save(cls, config: Dict[str, Any], config_path: Path):
        """Save configuration to JSON file"""
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved config to {config_path}")


# ============================================================================
# Main Module Interface
# ============================================================================

__all__ = [
    'KinematicParameters',
    'CameraPose',
    'JointAngles',
    'OrbitAxis',
    'Kinematics3DOF',
    'OrbitalPathGenerator',
    'TrajectoryInterpolator',
    'Gimbal3DOFConfig'
]
