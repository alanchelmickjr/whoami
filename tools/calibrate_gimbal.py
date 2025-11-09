#!/usr/bin/env python3
"""
Safe Gimbal Calibration Utility

CRITICAL: Safe calibration that won't crash the robot if servos fail

Features:
- Step-by-step calibration with safety checks
- Automatic range of motion testing
- Center point calibration
- Angle validation
- Emergency stop capability
- Comprehensive diagnostics

Usage:
    python tools/calibrate_gimbal.py
    python tools/calibrate_gimbal.py --servo 1
    python tools/calibrate_gimbal.py --test-only
"""

import sys
import time
import numpy as np
from pathlib import Path
import argparse
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from whoami.servo_safety import (
    ServoSafetyMonitor,
    ServoHealth,
    SafetyLimits,
    SafePosition
)

# Try to import servo controller
try:
    from whoami.feetech_servo import FeetechServoController
    HAS_HARDWARE = True
except ImportError:
    HAS_HARDWARE = False
    print("⚠ Hardware not available - running in simulation mode")


# ============================================================================
# Safe Calibration System
# ============================================================================

class SafeGimbalCalibrator:
    """
    Safe gimbal calibration system

    Ensures calibration process won't damage servos or robot
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        simulate: bool = False
    ):
        """
        Initialize calibrator

        Args:
            config_path: Path to gimbal config
            simulate: Run in simulation mode
        """
        self.simulate = simulate or not HAS_HARDWARE

        # Load config
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config/gimbal_3dof_config.json"

        self.config_path = config_path
        self.config = self._load_config()

        # Servo IDs
        self.servo_ids = [
            self.config['servo_ids']['yaw'],
            self.config['servo_ids']['pitch'],
            self.config['servo_ids']['roll']
        ]

        # Initialize safety monitor
        self.safety = ServoSafetyMonitor(
            servo_ids=self.servo_ids,
            safety_limits=SafetyLimits(
                max_temperature=70.0,  # Conservative for calibration
                max_current=900.0,
                max_position_error=15.0,  # Larger tolerance for initial calibration
                communication_timeout=3.0
            )
        )

        # Initialize servo controller (if hardware available)
        self.servo_controller = None
        if not self.simulate:
            try:
                self.servo_controller = FeetechServoController(
                    port=self.config.get('port', '/dev/ttyUSB0'),
                    baudrate=self.config.get('baudrate', 1000000)
                )
                print("✓ Connected to servo controller")
            except Exception as e:
                print(f"✗ Failed to connect to servos: {e}")
                print("Running in simulation mode")
                self.simulate = True

        # Calibration results
        self.calibration_data = {
            'yaw': {},
            'pitch': {},
            'roll': {}
        }

        # Emergency stop flag
        self.stopped = False

        print(f"Gimbal calibrator initialized ({'simulation' if self.simulate else 'hardware'} mode)")

    def _load_config(self) -> dict:
        """Load gimbal configuration"""
        if not self.config_path.exists():
            print(f"⚠ Config not found: {self.config_path}")
            print("Using default configuration")
            return self._default_config()

        with open(self.config_path, 'r') as f:
            return json.load(f)

    def _default_config(self) -> dict:
        """Default configuration"""
        return {
            'servo_ids': {'yaw': 1, 'pitch': 2, 'roll': 3},
            'port': '/dev/ttyUSB0',
            'baudrate': 1000000,
            'kinematics': {
                'yaw_limits': [-180, 180],
                'pitch_limits': [-90, 90],
                'roll_limits': [-180, 180]
            }
        }

    # ========================================================================
    # Safety Checks
    # ========================================================================

    def check_safety(self) -> bool:
        """
        Comprehensive safety check

        Returns:
            True if safe to proceed
        """
        print("\nRunning safety checks...")

        # Check system health
        system_health = self.safety.get_system_health()

        if system_health in [ServoHealth.ERROR, ServoHealth.FAILED]:
            print(f"✗ System health: {system_health.value}")
            print("  Cannot proceed with calibration")
            return False

        # Check individual servos
        for servo_id in self.servo_ids:
            status = self.safety.servo_status[servo_id]

            # Check responsiveness (if hardware connected)
            if not self.simulate:
                if not status.is_responsive(timeout=3.0):
                    print(f"✗ Servo {servo_id} not responding")
                    return False

            # Check temperature
            if status.temperature > 60.0:
                print(f"✗ Servo {servo_id} too hot: {status.temperature:.1f}°C")
                print("  Let servos cool down before calibration")
                return False

        print("✓ All safety checks passed")
        return True

    def wait_for_safety(self, timeout: float = 5.0) -> bool:
        """
        Wait for safe conditions

        Returns:
            True if safe within timeout
        """
        start_time = time.time()

        while (time.time() - start_time) < timeout:
            if self.check_safety():
                return True

            time.sleep(0.5)

        return False

    # ========================================================================
    # Calibration Steps
    # ========================================================================

    def calibrate_center_positions(self):
        """
        Step 1: Calibrate center positions

        Ensures 0° position is mechanically centered
        """
        print("\n" + "="*70)
        print("STEP 1: Center Position Calibration")
        print("="*70)

        servo_names = {
            self.config['servo_ids']['yaw']: 'YAW (base rotation)',
            self.config['servo_ids']['pitch']: 'PITCH (arm up/down)',
            self.config['servo_ids']['roll']: 'ROLL (camera spin)'
        }

        for servo_id in self.servo_ids:
            print(f"\nCalibrating Servo {servo_id}: {servo_names[servo_id]}")

            # Safety check
            if not self.check_safety():
                print("✗ Safety check failed - aborting")
                return False

            # Move to center
            print(f"Moving to center position (0°)...")

            if not self.simulate:
                try:
                    # Move slowly to center
                    self.servo_controller.set_position(servo_id, 0.0, speed=50)
                    time.sleep(2.0)

                    # Read actual position
                    actual_pos = self.servo_controller.get_position(servo_id)
                    print(f"Actual position: {actual_pos:.2f}°")

                    # Update safety monitor
                    self.safety.update_servo_status(
                        servo_id,
                        position=actual_pos,
                        communication_success=True
                    )

                except Exception as e:
                    print(f"✗ Error: {e}")
                    return False
            else:
                print("[SIMULATION] Moved to 0°")
                actual_pos = 0.0

            # Verify position
            print(f"\nPlease verify servo {servo_id} is mechanically centered:")
            print(f"  - {servo_names[servo_id]}")
            print(f"  - Should be in neutral, centered position")

            response = input("Is servo centered? (yes/no): ").strip().lower()

            if response != 'yes':
                print("⚠ Center calibration incomplete")
                print("  Please adjust mechanical assembly or servo offset")
                return False

            # Store calibration
            axis = [k for k, v in self.config['servo_ids'].items() if v == servo_id][0]
            self.calibration_data[axis]['center_position'] = actual_pos
            self.calibration_data[axis]['center_verified'] = True

            print(f"✓ Servo {servo_id} center calibration complete")

        print("\n✓ All center positions calibrated")
        return True

    def calibrate_range_of_motion(self):
        """
        Step 2: Calibrate range of motion

        Tests limits and ensures safe operation
        """
        print("\n" + "="*70)
        print("STEP 2: Range of Motion Calibration")
        print("="*70)

        axes = ['yaw', 'pitch', 'roll']

        for axis in axes:
            servo_id = self.config['servo_ids'][axis]

            print(f"\nCalibrating {axis.upper()} range of motion...")

            # Get configured limits
            limits = self.config['kinematics'][f'{axis}_limits']
            min_angle, max_angle = limits

            # Test minimum position
            print(f"\nTesting minimum position: {min_angle}°")

            if not self._test_position(servo_id, min_angle):
                print(f"✗ Failed to reach {min_angle}°")
                print("  Adjusting limit...")
                min_angle = self._find_safe_limit(servo_id, min_angle, direction='min')

            self.calibration_data[axis]['min_angle'] = min_angle

            # Return to center
            self._move_to_center(servo_id)
            time.sleep(1.0)

            # Test maximum position
            print(f"\nTesting maximum position: {max_angle}°")

            if not self._test_position(servo_id, max_angle):
                print(f"✗ Failed to reach {max_angle}°")
                print("  Adjusting limit...")
                max_angle = self._find_safe_limit(servo_id, max_angle, direction='max')

            self.calibration_data[axis]['max_angle'] = max_angle

            # Return to center
            self._move_to_center(servo_id)

            print(f"✓ {axis.upper()} range: [{min_angle:.1f}°, {max_angle:.1f}°]")

        print("\n✓ Range of motion calibration complete")
        return True

    def _test_position(self, servo_id: int, angle: float) -> bool:
        """
        Test moving to specific position

        Returns:
            True if position reached safely
        """
        # Safety check before move
        if not self.check_safety():
            return False

        # Set target for error tracking
        self.safety.set_target_position(servo_id, angle)

        if not self.simulate:
            try:
                # Move slowly
                self.servo_controller.set_position(servo_id, angle, speed=30)

                # Wait for movement
                time.sleep(2.0)

                # Read actual position
                actual_pos = self.servo_controller.get_position(servo_id)

                # Update safety
                self.safety.update_servo_status(
                    servo_id,
                    position=actual_pos,
                    communication_success=True
                )

                # Check position error
                error = abs(actual_pos - angle)

                if error > 15.0:
                    print(f"⚠ Large position error: {error:.1f}°")
                    return False

                print(f"  Reached {actual_pos:.1f}° (target: {angle:.1f}°)")
                return True

            except Exception as e:
                print(f"✗ Error moving servo: {e}")
                self.safety.update_servo_status(
                    servo_id,
                    communication_success=False
                )
                return False
        else:
            print(f"  [SIMULATION] Reached {angle:.1f}°")
            return True

    def _move_to_center(self, servo_id: int):
        """Move servo to center position"""
        if not self.simulate:
            self.servo_controller.set_position(servo_id, 0.0, speed=50)
        time.sleep(1.0)

    def _find_safe_limit(
        self,
        servo_id: int,
        target_angle: float,
        direction: str
    ) -> float:
        """
        Find safe limit by incremental testing

        Args:
            servo_id: Servo to test
            target_angle: Target limit
            direction: 'min' or 'max'

        Returns:
            Safe limit angle
        """
        print(f"  Finding safe {direction} limit...")

        # Start from center
        current = 0.0
        step = 10.0 if direction == 'max' else -10.0
        last_safe = current

        while abs(current) < abs(target_angle):
            current += step

            if self._test_position(servo_id, current):
                last_safe = current
            else:
                print(f"  Limit reached at {last_safe:.1f}°")
                break

            time.sleep(0.5)

        return last_safe

    def calibrate_angle_accuracy(self):
        """
        Step 3: Calibrate angle accuracy

        Tests accuracy at known positions
        """
        print("\n" + "="*70)
        print("STEP 3: Angle Accuracy Calibration")
        print("="*70)

        test_angles = [-45, 0, 45]

        for axis in ['yaw', 'pitch', 'roll']:
            servo_id = self.config['servo_ids'][axis]

            print(f"\nTesting {axis.upper()} accuracy...")

            errors = []

            for target_angle in test_angles:
                # Check if in range
                if 'min_angle' in self.calibration_data[axis]:
                    min_a = self.calibration_data[axis]['min_angle']
                    max_a = self.calibration_data[axis]['max_angle']

                    if target_angle < min_a or target_angle > max_a:
                        continue

                if not self.simulate:
                    # Move to position
                    self.servo_controller.set_position(servo_id, target_angle, speed=50)
                    time.sleep(1.5)

                    # Read actual
                    actual = self.servo_controller.get_position(servo_id)
                    error = actual - target_angle

                    errors.append(error)

                    print(f"  {target_angle:+6.1f}° → {actual:+6.1f}° (error: {error:+5.2f}°)")
                else:
                    print(f"  {target_angle:+6.1f}° → {target_angle:+6.1f}° (error: 0.00°)")
                    errors.append(0.0)

            # Compute statistics
            if errors:
                mean_error = np.mean(errors)
                std_error = np.std(errors)

                self.calibration_data[axis]['mean_error'] = mean_error
                self.calibration_data[axis]['std_error'] = std_error

                print(f"  Mean error: {mean_error:+.2f}° ± {std_error:.2f}°")

            # Return to center
            self._move_to_center(servo_id)

        print("\n✓ Angle accuracy calibration complete")
        return True

    # ========================================================================
    # Full Calibration
    # ========================================================================

    def run_full_calibration(self):
        """Run complete calibration sequence"""
        print("\n" + "="*70)
        print("SAFE GIMBAL CALIBRATION")
        print("="*70)
        print("\nThis will calibrate all gimbal servos safely")
        print("You can press Ctrl+C at any time to emergency stop")
        print("="*70)

        # Start safety monitoring
        self.safety.start()

        try:
            # Initial safety check
            print("\nPerforming initial safety check...")
            if not self.wait_for_safety(timeout=10.0):
                print("\n✗ Initial safety check failed")
                print("Check servo connections and try again")
                return False

            # Step 1: Center positions
            if not self.calibrate_center_positions():
                print("\n✗ Center calibration failed")
                return False

            input("\nPress Enter to continue to range of motion calibration...")

            # Step 2: Range of motion
            if not self.calibrate_range_of_motion():
                print("\n✗ Range calibration failed")
                return False

            input("\nPress Enter to continue to accuracy calibration...")

            # Step 3: Accuracy
            if not self.calibrate_angle_accuracy():
                print("\n✗ Accuracy calibration failed")
                return False

            # Success!
            print("\n" + "="*70)
            print("✓ CALIBRATION COMPLETE")
            print("="*70)

            self.print_calibration_results()

            # Save results
            self.save_calibration()

            return True

        except KeyboardInterrupt:
            print("\n\n🛑 CALIBRATION INTERRUPTED")
            self.emergency_stop()
            return False

        except Exception as e:
            print(f"\n✗ Calibration error: {e}")
            self.emergency_stop()
            return False

        finally:
            self.safety.stop()

    # ========================================================================
    # Results and Persistence
    # ========================================================================

    def print_calibration_results(self):
        """Print calibration results"""
        print("\nCalibration Results:")
        print("-" * 70)

        for axis in ['yaw', 'pitch', 'roll']:
            print(f"\n{axis.upper()}:")

            data = self.calibration_data[axis]

            if 'center_verified' in data:
                print(f"  Center: {data.get('center_position', 0.0):.1f}° ✓")

            if 'min_angle' in data:
                print(f"  Range: [{data['min_angle']:.1f}°, {data['max_angle']:.1f}°]")

            if 'mean_error' in data:
                print(f"  Accuracy: {data['mean_error']:+.2f}° ± {data['std_error']:.2f}°")

    def save_calibration(self):
        """Save calibration results"""
        output_path = Path(__file__).parent.parent / "config/gimbal_calibration.json"

        calibration_output = {
            'timestamp': time.time(),
            'servo_ids': self.config['servo_ids'],
            'calibration': self.calibration_data
        }

        with open(output_path, 'w') as f:
            json.dump(calibration_output, f, indent=2)

        print(f"\n✓ Calibration saved to: {output_path}")

    # ========================================================================
    # Emergency Stop
    # ========================================================================

    def emergency_stop(self):
        """Emergency stop all servos"""
        print("\n🛑 EMERGENCY STOP")

        self.stopped = True

        # Trigger safety monitor emergency stop
        self.safety.emergency_stop()

        # Move to safe position
        if self.servo_controller:
            print("Moving to safe position...")
            self.safety.move_to_safe_position("home", self.servo_controller)

        print("✓ Robot in safe state")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Safe gimbal calibration")
    parser.add_argument('--config', type=Path, help='Gimbal config path')
    parser.add_argument('--servo', type=int, help='Calibrate specific servo only')
    parser.add_argument('--test-only', action='store_true', help='Test without calibration')
    parser.add_argument('--simulate', action='store_true', help='Run in simulation mode')

    args = parser.parse_args()

    # Create calibrator
    calibrator = SafeGimbalCalibrator(
        config_path=args.config,
        simulate=args.simulate
    )

    if args.test_only:
        # Just test safety
        print("Running safety test...")
        calibrator.safety.start()
        time.sleep(2.0)
        calibrator.safety.print_health_report()
        calibrator.safety.stop()
        return

    # Run full calibration
    success = calibrator.run_full_calibration()

    if success:
        print("\n✓ Calibration successful!")
        print("Gimbal is ready for use")
    else:
        print("\n✗ Calibration incomplete")
        print("Please review errors and try again")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
