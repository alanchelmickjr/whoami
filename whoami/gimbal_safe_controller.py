"""
Safe Gimbal Controller with Fault Tolerance

CRITICAL: Gimbal failures won't crash the robot

Wraps gimbal_3dof_controller with safety monitoring:
- Real-time health monitoring
- Automatic failure detection
- Safe fallback behavior
- Graceful degradation
- Error recovery

Philosophy: "If the neck/head gimbal fails, the robot does not fail"
"""

import numpy as np
import logging
import time
from typing import Optional, Callable, Dict, Any
from pathlib import Path

from whoami.servo_safety import (
    ServoSafetyMonitor,
    ServoHealth,
    FailureMode,
    SafetyLimits
)

logger = logging.getLogger(__name__)

# Try to import gimbal controller
try:
    from whoami.gimbal_3dof_controller import Gimbal3DOFController
    HAS_GIMBAL = True
except ImportError:
    HAS_GIMBAL = False
    logger.warning("Gimbal controller not available")


# ============================================================================
# Safe Gimbal Controller
# ============================================================================

class SafeGimbalController:
    """
    Fault-tolerant gimbal controller

    Wraps Gimbal3DOFController with comprehensive safety monitoring
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        safety_limits: Optional[SafetyLimits] = None,
        enable_recovery: bool = True
    ):
        """
        Initialize safe gimbal controller

        Args:
            config_path: Path to gimbal config
            safety_limits: Safety limits
            enable_recovery: Enable automatic recovery from failures
        """
        self.config_path = config_path
        self.enable_recovery = enable_recovery

        # Initialize underlying gimbal controller
        self.gimbal: Optional[Gimbal3DOFController] = None
        self.gimbal_available = False

        if HAS_GIMBAL:
            try:
                self.gimbal = Gimbal3DOFController(config_path=config_path)
                self.gimbal_available = True
                logger.info("Gimbal controller initialized")
            except Exception as e:
                logger.error(f"Failed to initialize gimbal: {e}")
                logger.warning("Running in degraded mode (no gimbal)")
        else:
            logger.warning("Gimbal controller not available - degraded mode")

        # Initialize safety monitor
        servo_ids = [1, 2, 3]  # Yaw, Pitch, Roll
        self.safety = ServoSafetyMonitor(
            servo_ids=servo_ids,
            safety_limits=safety_limits or SafetyLimits()
        )

        # Register safety callbacks
        self.safety.on_failure_detected = self._on_failure
        self.safety.on_emergency_stop = self._on_emergency_stop

        # Operation mode
        self.operational = self.gimbal_available
        self.degraded_mode = not self.gimbal_available

        # Statistics
        self.operation_count = 0
        self.failure_count = 0
        self.recovery_count = 0

        # Start safety monitoring
        self.safety.start()

        logger.info(
            f"Safe gimbal controller ready "
            f"({'operational' if self.operational else 'degraded'})"
        )

    # ========================================================================
    # Safety Checks
    # ========================================================================

    def _check_operational(self) -> bool:
        """
        Check if gimbal is operational

        Returns:
            True if safe to operate
        """
        if not self.gimbal_available:
            return False

        if self.safety.emergency_stopped:
            logger.warning("Emergency stop active - gimbal disabled")
            return False

        # Check system health
        system_health = self.safety.get_system_health()

        if system_health == ServoHealth.FAILED:
            logger.error("System health FAILED - gimbal disabled")
            self.operational = False
            return False

        if system_health == ServoHealth.ERROR:
            logger.warning("System health ERROR - attempting recovery")

            if self.enable_recovery:
                if self._attempt_recovery():
                    logger.info("Recovery successful - resuming operation")
                    return True

            logger.error("Recovery failed - gimbal disabled")
            self.operational = False
            return False

        return True

    def _on_failure(self, servo_id: int, failures: list):
        """Callback when failure detected"""
        logger.warning(f"Servo {servo_id} failure: {[f.value for f in failures]}")

        self.failure_count += 1

        # Check if critical failure
        critical_failures = [
            FailureMode.COMMUNICATION_LOST,
            FailureMode.MECHANICAL_JAM,
            FailureMode.OVERHEATING
        ]

        if any(f in critical_failures for f in failures):
            logger.error(f"Critical failure on servo {servo_id}")

            # Move to safe position
            self._move_to_safe_position()

            # Disable gimbal
            self.operational = False

    def _on_emergency_stop(self):
        """Callback when emergency stop triggered"""
        logger.critical("Emergency stop - disabling gimbal")
        self.operational = False
        self.degraded_mode = True

    def _attempt_recovery(self) -> bool:
        """
        Attempt to recover from failures

        Returns:
            True if recovery successful
        """
        logger.info("Attempting failure recovery...")

        failed_servos = self.safety.get_failed_servos()

        for servo_id in failed_servos:
            success = self.safety.attempt_recovery(servo_id)

            if success:
                logger.info(f"Servo {servo_id} recovered")
                self.recovery_count += 1
            else:
                logger.error(f"Servo {servo_id} recovery failed")
                return False

        # Test gimbal operation
        try:
            # Move to home position
            self._move_to_safe_position()
            time.sleep(1.0)

            # Check if successful
            if self._check_operational():
                logger.info("✓ Recovery successful")
                return True

        except Exception as e:
            logger.error(f"Recovery test failed: {e}")

        return False

    def _move_to_safe_position(self):
        """Move gimbal to safe position"""
        if self.gimbal:
            try:
                logger.info("Moving to safe position...")
                self.safety.move_to_safe_position("home", self.gimbal.servo_controller)
            except Exception as e:
                logger.error(f"Failed to move to safe position: {e}")

    # ========================================================================
    # Wrapped Gimbal Operations (Fault-Tolerant)
    # ========================================================================

    def move_to_angles(
        self,
        yaw: float,
        pitch: float,
        roll: float,
        speed: float = 100.0
    ) -> bool:
        """
        Move gimbal to specific angles (SAFE)

        Args:
            yaw: Yaw angle (degrees)
            pitch: Pitch angle (degrees)
            roll: Roll angle (degrees)
            speed: Movement speed

        Returns:
            True if successful
        """
        # Check if operational
        if not self._check_operational():
            logger.warning("Gimbal not operational - skipping move")
            return False

        try:
            # Update safety targets
            self.safety.set_target_position(1, yaw)
            self.safety.set_target_position(2, pitch)
            self.safety.set_target_position(3, roll)

            # Execute move
            self.gimbal.move_to_angles(yaw, pitch, roll, speed)

            # Wait for completion
            time.sleep(0.5)

            # Update operation count
            self.operation_count += 1

            # Verify position
            time.sleep(0.5)
            if not self._verify_position_reached():
                logger.warning("Position verification failed")

            return True

        except Exception as e:
            logger.error(f"Move failed: {e}")
            self.failure_count += 1
            return False

    def scan_horizontal_orbit(
        self,
        center: np.ndarray,
        radius: float,
        num_points: int = 36,
        camera_roll: float = 0.0,
        scan_callback: Optional[Callable] = None
    ) -> bool:
        """
        Perform horizontal orbital scan (SAFE)

        Args:
            center: Center point
            radius: Orbital radius
            num_points: Number of scan points
            camera_roll: Camera roll angle
            scan_callback: Callback at each position

        Returns:
            True if successful
        """
        # Check operational
        if not self._check_operational():
            logger.warning("Gimbal not operational - scan aborted")
            return False

        try:
            # Wrap callback with safety monitoring
            def safe_callback(idx, pose):
                # Check health during scan
                if not self._check_operational():
                    logger.error("Health check failed during scan - aborting")
                    raise RuntimeError("Scan aborted - safety issue")

                # Call user callback
                if scan_callback:
                    scan_callback(idx, pose)

            # Execute scan
            success = self.gimbal.scan_horizontal_orbit(
                center=center,
                radius=radius,
                num_points=num_points,
                camera_roll=camera_roll,
                scan_callback=safe_callback
            )

            if success:
                self.operation_count += 1

            return success

        except Exception as e:
            logger.error(f"Orbital scan failed: {e}")
            self.failure_count += 1

            # Move to safe position
            self._move_to_safe_position()

            return False

    def scan_spherical(
        self,
        center: np.ndarray,
        radius: float,
        num_rings: int = 5,
        points_per_ring: int = 36
    ) -> bool:
        """
        Perform spherical scan (SAFE)

        Args:
            center: Center point
            radius: Orbital radius
            num_rings: Number of elevation rings
            points_per_ring: Points per ring

        Returns:
            True if successful
        """
        # Check operational
        if not self._check_operational():
            logger.warning("Gimbal not operational - scan aborted")
            return False

        try:
            success = self.gimbal.scan_spherical(
                center=center,
                radius=radius,
                num_rings=num_rings,
                points_per_ring=points_per_ring
            )

            if success:
                self.operation_count += 1
            else:
                self.failure_count += 1

            return success

        except Exception as e:
            logger.error(f"Spherical scan failed: {e}")
            self.failure_count += 1
            self._move_to_safe_position()
            return False

    def continuous_roll_scan(
        self,
        target_point: np.ndarray,
        camera_position: np.ndarray,
        roll_speed: float = 30.0,
        duration: float = 12.0
    ) -> bool:
        """
        Continuous roll scan (SAFE)

        Args:
            target_point: Point to look at
            camera_position: Camera position
            roll_speed: Roll speed (deg/s)
            duration: Scan duration

        Returns:
            True if successful
        """
        # Check operational
        if not self._check_operational():
            logger.warning("Gimbal not operational - scan aborted")
            return False

        try:
            # Monitor health during scan
            start_time = time.time()

            success = self.gimbal.continuous_roll_scan(
                target_point=target_point,
                camera_position=camera_position,
                roll_speed=roll_speed,
                duration=duration
            )

            # Check health after scan
            if not self._check_operational():
                logger.warning("Health degraded after roll scan")

            if success:
                self.operation_count += 1

            return success

        except Exception as e:
            logger.error(f"Continuous roll scan failed: {e}")
            self.failure_count += 1
            self._move_to_safe_position()
            return False

    def _verify_position_reached(self) -> bool:
        """Verify servos reached target positions"""
        for servo_id in [1, 2, 3]:
            status = self.safety.servo_status[servo_id]

            if status.position_error > 10.0:
                logger.warning(
                    f"Servo {servo_id} position error: {status.position_error:.1f}°"
                )
                return False

        return True

    # ========================================================================
    # Graceful Degradation
    # ========================================================================

    def get_capabilities(self) -> Dict[str, bool]:
        """
        Get current capabilities

        Returns:
            Dictionary of capability flags
        """
        return {
            'gimbal_operational': self.operational,
            'can_move': self.operational,
            'can_scan': self.operational,
            'degraded_mode': self.degraded_mode,
            'emergency_stopped': self.safety.emergency_stopped
        }

    def get_degradation_level(self) -> str:
        """
        Get degradation level

        Returns:
            'fully_operational', 'degraded', or 'failed'
        """
        if self.operational and not self.degraded_mode:
            return 'fully_operational'
        elif self.degraded_mode and not self.safety.emergency_stopped:
            return 'degraded'
        else:
            return 'failed'

    # ========================================================================
    # Diagnostics
    # ========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status"""
        return {
            'operational': self.operational,
            'degraded_mode': self.degraded_mode,
            'degradation_level': self.get_degradation_level(),
            'capabilities': self.get_capabilities(),
            'statistics': {
                'operations': self.operation_count,
                'failures': self.failure_count,
                'recoveries': self.recovery_count
            },
            'safety': self.safety.get_diagnostics()
        }

    def print_status(self):
        """Print human-readable status"""
        print("\n" + "="*70)
        print("SAFE GIMBAL CONTROLLER STATUS")
        print("="*70)

        status = self.get_status()

        # Operational status
        if status['operational']:
            print("\nStatus: ✓ OPERATIONAL")
        elif status['degraded_mode']:
            print("\nStatus: ⚠ DEGRADED MODE")
        else:
            print("\nStatus: ✗ FAILED")

        # Capabilities
        print("\nCapabilities:")
        caps = status['capabilities']
        for cap, available in caps.items():
            icon = "✓" if available else "✗"
            print(f"  {icon} {cap}")

        # Statistics
        stats = status['statistics']
        print(f"\nStatistics:")
        print(f"  Operations: {stats['operations']}")
        print(f"  Failures: {stats['failures']}")
        print(f"  Recoveries: {stats['recoveries']}")

        if stats['operations'] > 0:
            success_rate = (1 - stats['failures'] / stats['operations']) * 100
            print(f"  Success Rate: {success_rate:.1f}%")

        # Safety health
        print(f"\nSystem Health: {status['safety']['system_health'].upper()}")

        print("="*70 + "\n")

    # ========================================================================
    # Emergency Controls
    # ========================================================================

    def emergency_stop(self):
        """Emergency stop gimbal"""
        logger.critical("🛑 EMERGENCY STOP")
        self.safety.emergency_stop()
        self.operational = False

    def reset(self) -> bool:
        """
        Reset after emergency stop

        Returns:
            True if reset successful
        """
        logger.info("Resetting gimbal controller...")

        # Clear emergency stop
        self.safety.emergency_stopped = False

        # Attempt recovery
        if self._attempt_recovery():
            self.operational = True
            self.degraded_mode = False
            logger.info("✓ Reset successful")
            return True
        else:
            logger.error("✗ Reset failed")
            return False

    # ========================================================================
    # Lifecycle
    # ========================================================================

    def shutdown(self):
        """Shutdown controller safely"""
        logger.info("Shutting down safe gimbal controller...")

        # Stop safety monitoring
        self.safety.stop()

        # Move to safe position
        self._move_to_safe_position()

        # Shutdown gimbal
        if self.gimbal:
            self.gimbal.shutdown()

        logger.info("✓ Shutdown complete")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()


# ============================================================================
# Convenience Function
# ============================================================================

def create_safe_gimbal(
    config_path: Optional[Path] = None,
    **kwargs
) -> SafeGimbalController:
    """
    Create safe gimbal controller with default settings

    Args:
        config_path: Config path
        **kwargs: Additional arguments

    Returns:
        SafeGimbalController instance
    """
    return SafeGimbalController(config_path=config_path, **kwargs)


# ============================================================================
# Module Interface
# ============================================================================

__all__ = [
    'SafeGimbalController',
    'create_safe_gimbal'
]
