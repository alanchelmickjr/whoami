#!/usr/bin/env python3
"""
Orbital 3D Scanning Example

Demonstrates using the 3DOF gimbal to perform complete 360° 3D scans
of objects by orbiting the camera around them while capturing depth data.

This example shows:
1. Setting up the 3DOF gimbal
2. Defining scan target (object position and size)
3. Performing horizontal orbital scan
4. Performing spherical multi-ring scan
5. Continuous roll scanning at each position
6. Integrating point clouds into complete 3D model
"""

import numpy as np
import logging
import time
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from whoami.gimbal_3dof_controller import Gimbal3DOFController
from whoami.scanner_3d import Scanner3D, ScanMode, Scanner3DConfig
from whoami.gimbal_3dof import JointAngles

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Orbital Scanning System
# ============================================================================

class OrbitalScanner:
    """
    Complete orbital 3D scanning system

    Coordinates gimbal motion with 3D scanner to capture complete
    object geometry from multiple angles
    """

    def __init__(
        self,
        gimbal_config_path: Optional[Path] = None,
        scanner_config: Optional[Scanner3DConfig] = None
    ):
        """
        Initialize orbital scanner

        Args:
            gimbal_config_path: Path to gimbal configuration
            scanner_config: 3D scanner configuration
        """
        # Initialize 3DOF gimbal
        self.gimbal = Gimbal3DOFController(
            config_path=gimbal_config_path
        )

        # Initialize 3D scanner
        if scanner_config is None:
            scanner_config = Scanner3DConfig(
                resolution=(1280, 720),
                fps=30,
                scan_mode=ScanMode.SINGLE_SHOT
            )
        self.scanner = Scanner3D(scanner_config)

        # Scanning state
        self.point_clouds = []
        self.current_scan_index = 0

    def connect(self) -> bool:
        """Connect to gimbal and camera"""
        logger.info("Connecting to orbital scanner...")

        # Connect gimbal
        if not self.gimbal.connect():
            logger.error("Failed to connect to gimbal")
            return False

        # Connect scanner
        if not self.scanner.initialize():
            logger.error("Failed to initialize scanner")
            return False

        logger.info("Orbital scanner ready!")
        return True

    def disconnect(self):
        """Disconnect all hardware"""
        self.gimbal.disconnect()
        self.scanner.shutdown()

    def capture_callback(self, waypoint_index: int, angles: JointAngles):
        """
        Callback function called at each scan position

        Args:
            waypoint_index: Index of current waypoint
            angles: Current joint angles
        """
        logger.info(f"Capturing at waypoint {waypoint_index}: {angles}")

        try:
            # Give gimbal time to settle
            time.sleep(0.2)

            # Capture point cloud
            point_cloud = self.scanner.capture_point_cloud()

            if point_cloud is not None:
                self.point_clouds.append(point_cloud)
                logger.info(f"  Captured {len(point_cloud.points)} points")
            else:
                logger.warning(f"  Capture failed at waypoint {waypoint_index}")

        except Exception as e:
            logger.error(f"Capture error: {e}")

    def scan_object_horizontal(
        self,
        object_center: np.ndarray,
        scan_radius: float = 0.3,
        num_positions: int = 36
    ) -> List:
        """
        Scan object with horizontal orbit (equatorial)

        Args:
            object_center: (x, y, z) center of object to scan
            scan_radius: orbital radius in meters
            num_positions: number of scan positions around orbit

        Returns:
            List of captured point clouds
        """
        logger.info("="*60)
        logger.info("HORIZONTAL ORBITAL SCAN")
        logger.info(f"Center: {object_center}")
        logger.info(f"Radius: {scan_radius}m")
        logger.info(f"Positions: {num_positions}")
        logger.info("="*60)

        self.point_clouds = []

        success = self.gimbal.scan_horizontal_orbit(
            center=object_center,
            radius=scan_radius,
            num_points=num_positions,
            camera_roll=0.0,
            scan_callback=self.capture_callback
        )

        if success:
            logger.info(f"Scan complete! Captured {len(self.point_clouds)} point clouds")
        else:
            logger.error("Scan failed")

        return self.point_clouds

    def scan_object_spherical(
        self,
        object_center: np.ndarray,
        scan_radius: float = 0.3,
        num_rings: int = 5,
        points_per_ring: int = 36
    ) -> List:
        """
        Scan object with spherical pattern (multiple orbital rings)

        Args:
            object_center: (x, y, z) center of object
            scan_radius: orbital radius
            num_rings: number of latitude rings
            points_per_ring: scan positions per ring

        Returns:
            List of captured point clouds
        """
        logger.info("="*60)
        logger.info("SPHERICAL ORBITAL SCAN")
        logger.info(f"Center: {object_center}")
        logger.info(f"Radius: {scan_radius}m")
        logger.info(f"Rings: {num_rings}, Points/ring: {points_per_ring}")
        logger.info("="*60)

        self.point_clouds = []

        success = self.gimbal.scan_spherical(
            center=object_center,
            radius=scan_radius,
            num_rings=num_rings,
            points_per_ring=points_per_ring,
            camera_roll=0.0,
            scan_callback=self.capture_callback
        )

        if success:
            logger.info(f"Scan complete! Captured {len(self.point_clouds)} point clouds")
        else:
            logger.error("Scan failed")

        return self.point_clouds

    def scan_with_rotating_camera(
        self,
        object_center: np.ndarray,
        scan_radius: float = 0.3,
        orbital_positions: int = 12,
        roll_angles: List[float] = [0, 90, 180, 270]
    ) -> List:
        """
        Scan with camera roll at each orbital position

        Combines orbital motion with camera rotation to capture
        more geometric detail from each angle

        Args:
            object_center: (x, y, z) center of object
            scan_radius: orbital radius
            orbital_positions: number of positions around orbit
            roll_angles: list of roll angles to capture at each position

        Returns:
            List of all captured point clouds
        """
        logger.info("="*60)
        logger.info("ORBITAL SCAN WITH CAMERA ROTATION")
        logger.info(f"Center: {object_center}")
        logger.info(f"Orbital positions: {orbital_positions}")
        logger.info(f"Roll angles: {roll_angles}")
        logger.info("="*60)

        self.point_clouds = []

        # Generate orbital positions
        angles = np.linspace(0, 360, orbital_positions, endpoint=False)

        for i, angle in enumerate(angles):
            # Calculate camera position on orbit
            angle_rad = np.deg2rad(angle)
            camera_pos = object_center + np.array([
                scan_radius * np.cos(angle_rad),
                scan_radius * np.sin(angle_rad),
                0
            ])

            logger.info(f"\nOrbital position {i+1}/{orbital_positions} (angle={angle:.1f}°)")

            # Capture at multiple roll angles
            for roll_angle in roll_angles:
                logger.info(f"  Roll angle: {roll_angle}°")

                # Move to position with this roll
                success = self.gimbal.move_to_pose(
                    target_point=object_center,
                    camera_position=camera_pos,
                    camera_roll=roll_angle,
                    blocking=True
                )

                if not success:
                    logger.warning(f"  Position unreachable")
                    continue

                # Capture
                time.sleep(0.2)
                point_cloud = self.scanner.capture_point_cloud()

                if point_cloud is not None:
                    self.point_clouds.append(point_cloud)
                    logger.info(f"    Captured {len(point_cloud.points)} points")

        logger.info(f"\nScan complete! Captured {len(self.point_clouds)} point clouds")
        return self.point_clouds

    def merge_point_clouds(self) -> Optional:
        """
        Merge all captured point clouds into single model

        Returns:
            Merged point cloud or None if merge failed
        """
        if len(self.point_clouds) == 0:
            logger.error("No point clouds to merge")
            return None

        logger.info(f"Merging {len(self.point_clouds)} point clouds...")

        try:
            # Use scanner's merge functionality
            merged = self.scanner.merge_point_clouds(self.point_clouds)
            logger.info(f"Merged model has {len(merged.points)} points")
            return merged

        except Exception as e:
            logger.error(f"Merge failed: {e}")
            return None

    def save_scan(self, output_path: Path, generate_mesh: bool = True):
        """
        Save captured scan data

        Args:
            output_path: Directory to save files
            generate_mesh: Generate mesh from point cloud
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving scan to {output_path}")

        # Merge point clouds
        merged = self.merge_point_clouds()

        if merged is None:
            return

        # Save point cloud
        cloud_file = output_path / "scan_pointcloud.ply"
        self.scanner.save_point_cloud(merged, cloud_file)
        logger.info(f"Saved point cloud: {cloud_file}")

        # Generate and save mesh
        if generate_mesh:
            logger.info("Generating mesh...")
            mesh = self.scanner.generate_mesh(merged)

            if mesh is not None:
                mesh_file = output_path / "scan_mesh.ply"
                self.scanner.save_mesh(mesh, mesh_file)
                logger.info(f"Saved mesh: {mesh_file}")

                # Also save as STL for 3D printing
                stl_file = output_path / "scan_mesh.stl"
                mesh.export(str(stl_file))
                logger.info(f"Saved STL: {stl_file}")


# ============================================================================
# Example Usage
# ============================================================================

def example_desktop_object_scan():
    """
    Example: Scan object on desktop

    Assumes object is on desk in front of robot at approximately
    30cm distance and at robot's natural neck height
    """
    logger.info("="*70)
    logger.info("EXAMPLE: Desktop Object Scanning")
    logger.info("="*70)

    # Initialize scanner
    scanner = OrbitalScanner()

    if not scanner.connect():
        logger.error("Failed to connect")
        return

    try:
        # Define object position (adjust for your setup)
        # Assuming robot is at origin, object is 30cm forward, at desk height
        object_center = np.array([0.3, 0.0, 0.0])  # 30cm forward, centered, desk height

        # Scan radius - orbit 20cm around object
        scan_radius = 0.20

        # === QUICK SCAN (12 positions, horizontal only) ===
        logger.info("\n" + "="*70)
        logger.input("Performing quick horizontal scan...")
        logger.info("Press Enter to start quick scan or Ctrl+C to skip")
        input()

        point_clouds = scanner.scan_object_horizontal(
            object_center=object_center,
            scan_radius=scan_radius,
            num_positions=12  # Every 30 degrees
        )

        # Save quick scan
        scanner.save_scan(Path("data/scans/quick_scan"))

        # === DETAILED SCAN (spherical coverage) ===
        logger.info("\n" + "="*70)
        logger.info("Performing detailed spherical scan...")
        logger.info("Press Enter to start detailed scan or Ctrl+C to skip")
        input()

        point_clouds = scanner.scan_object_spherical(
            object_center=object_center,
            scan_radius=scan_radius,
            num_rings=3,  # 3 elevation rings
            points_per_ring=24  # 24 points per ring
        )

        # Save detailed scan
        scanner.save_scan(Path("data/scans/detailed_scan"))

        # === ULTRA DETAILED (with camera rotation) ===
        logger.info("\n" + "="*70)
        logger.info("Performing ultra-detailed scan with camera rotation...")
        logger.info("Press Enter to start ultra scan or Ctrl+C to skip")
        input()

        point_clouds = scanner.scan_with_rotating_camera(
            object_center=object_center,
            scan_radius=scan_radius,
            orbital_positions=12,
            roll_angles=[0, 45, 90, 135]  # 4 roll angles at each position
        )

        # Save ultra scan
        scanner.save_scan(Path("data/scans/ultra_scan"))

        logger.info("\n" + "="*70)
        logger.info("All scans complete!")
        logger.info("="*70)

    except KeyboardInterrupt:
        logger.info("\nScan interrupted by user")

    except Exception as e:
        logger.error(f"Error during scan: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean up
        scanner.disconnect()


def example_calibration_scan():
    """
    Example: Calibration and test scan

    Moves gimbal through test positions to verify kinematics
    """
    logger.info("="*70)
    logger.info("CALIBRATION AND TEST")
    logger.info("="*70)

    scanner = OrbitalScanner()

    if not scanner.connect():
        return

    try:
        # Run calibration
        logger.info("\nRunning gimbal calibration...")
        scanner.gimbal.calibrate()

        # Test forward kinematics
        logger.info("\nTesting forward kinematics...")
        test_angles = JointAngles(yaw=45, pitch=30, roll=0)
        pose = scanner.gimbal.kinematics.forward_kinematics(test_angles)

        logger.info(f"Angles: {test_angles}")
        logger.info(f"Camera position: {pose.position}")
        logger.info(f"Camera forward: {pose.forward}")
        logger.info(f"Camera up: {pose.up}")

        # Test inverse kinematics
        logger.info("\nTesting inverse kinematics...")
        target = np.array([0.3, 0.0, 0.0])
        camera_pos = np.array([0.2, 0.2, 0.1])

        result_angles = scanner.gimbal.kinematics.inverse_kinematics(
            target_position=target,
            camera_position=camera_pos
        )

        if result_angles:
            logger.info(f"Target: {target}")
            logger.info(f"Camera: {camera_pos}")
            logger.info(f"Solved angles: {result_angles}")

            # Move to computed position
            logger.info("\nMoving to computed position...")
            scanner.gimbal.move_to_angles(result_angles)
        else:
            logger.error("Inverse kinematics failed")

        # Return home
        logger.info("\nReturning home...")
        scanner.gimbal.home()

    finally:
        scanner.disconnect()


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Orbital 3D Scanning Examples")
    parser.add_argument(
        'mode',
        choices=['scan', 'calibrate', 'test'],
        help='Operation mode'
    )

    args = parser.parse_args()

    if args.mode == 'scan':
        example_desktop_object_scan()
    elif args.mode == 'calibrate':
        example_calibration_scan()
    elif args.mode == 'test':
        example_calibration_scan()


if __name__ == "__main__":
    main()
