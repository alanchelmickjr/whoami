#!/usr/bin/env python3
"""
Safe Gimbal Operation Demo

Demonstrates fault-tolerant gimbal control:
- Real-time safety monitoring
- Automatic failure detection
- Graceful degradation
- Error recovery
- Emergency stop

Philosophy: "If the neck/head gimbal fails, the robot does not fail"
"""

import numpy as np
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from whoami.gimbal_safe_controller import SafeGimbalController


# ============================================================================
# Demo Scenarios
# ============================================================================

def demo_basic_safe_operation():
    """
    Demo 1: Basic safe gimbal operation
    """
    print("\n" + "="*70)
    print("DEMO 1: Basic Safe Gimbal Operation")
    print("="*70)

    # Create safe gimbal controller
    with SafeGimbalController() as gimbal:
        # Check status
        gimbal.print_status()

        # Check capabilities
        caps = gimbal.get_capabilities()

        if not caps['can_move']:
            print("\n⚠ Gimbal not available - running in degraded mode")
            print("Robot continues to function without gimbal")
            return

        print("\n✓ Gimbal operational - performing safe movements")

        # Safe movement
        print("\nMoving to look forward...")
        success = gimbal.move_to_angles(yaw=0.0, pitch=0.0, roll=0.0)

        if success:
            print("✓ Movement successful")
        else:
            print("✗ Movement failed - but robot is still operational")

        time.sleep(1.0)

        # Safe pan left
        print("\nPanning left...")
        success = gimbal.move_to_angles(yaw=45.0, pitch=0.0, roll=0.0)

        if success:
            print("✓ Pan successful")

        time.sleep(1.0)

        # Return to center
        gimbal.move_to_angles(yaw=0.0, pitch=0.0, roll=0.0)

    print("\n✓ Demo 1 complete - gimbal shutdown safely")


def demo_failure_handling():
    """
    Demo 2: Handling servo failures gracefully
    """
    print("\n" + "="*70)
    print("DEMO 2: Failure Handling")
    print("="*70)

    gimbal = SafeGimbalController()

    print("\nGimbal initialized with safety monitoring")

    # Simulate checking health
    health = gimbal.safety.get_system_health()
    print(f"Initial health: {health.value}")

    # Show what happens if gimbal fails
    print("\nDemonstrating graceful degradation...")
    print("If gimbal fails:")
    print("  1. Failure is detected immediately")
    print("  2. Gimbal moves to safe position")
    print("  3. Gimbal operations are disabled")
    print("  4. Robot continues to operate in degraded mode")
    print("  5. No crash, no damage")

    # Check capabilities
    caps = gimbal.get_capabilities()
    print(f"\nCurrent capabilities:")
    for cap, available in caps.items():
        icon = "✓" if available else "✗"
        print(f"  {icon} {cap}")

    gimbal.shutdown()
    print("\n✓ Demo 2 complete")


def demo_safe_scanning():
    """
    Demo 3: Safe orbital scanning with health monitoring
    """
    print("\n" + "="*70)
    print("DEMO 3: Safe Orbital Scanning")
    print("="*70)

    gimbal = SafeGimbalController()

    # Check if scanning available
    caps = gimbal.get_capabilities()

    if not caps['can_scan']:
        print("\n⚠ Scanning not available (degraded mode)")
        print("Robot continues operation without scanning")
        gimbal.shutdown()
        return

    print("\n✓ Scanning available - performing safe orbital scan")

    # Orbital scan with health monitoring
    target_center = np.array([0.25, 0.0, 0.0])

    print(f"\nScanning object at {target_center}...")

    def scan_callback(idx, pose):
        """Called at each scan position"""
        print(f"  Scan position {idx}...")

        # Health is automatically monitored
        # If failure occurs, scan will abort safely

    success = gimbal.scan_horizontal_orbit(
        center=target_center,
        radius=0.15,
        num_points=24,
        scan_callback=scan_callback
    )

    if success:
        print("\n✓ Scan completed successfully")
    else:
        print("\n⚠ Scan failed or aborted - robot still operational")

    gimbal.shutdown()
    print("\n✓ Demo 3 complete")


def demo_emergency_stop():
    """
    Demo 4: Emergency stop functionality
    """
    print("\n" + "="*70)
    print("DEMO 4: Emergency Stop")
    print("="*70)

    gimbal = SafeGimbalController()

    print("\nGimbal operational")

    # Simulate emergency condition
    print("\nSimulating emergency condition...")
    print("Press Ctrl+C to trigger emergency stop")

    try:
        # Simulate some operations
        for i in range(5):
            print(f"Operation {i+1}...")
            time.sleep(1.0)

            # Could be interrupted by Ctrl+C

    except KeyboardInterrupt:
        print("\n\n🛑 EMERGENCY STOP TRIGGERED")

        # Emergency stop
        gimbal.emergency_stop()

        print("✓ Robot in safe state")
        print("  - All servos halted")
        print("  - Moved to safe position")
        print("  - Robot still operational (degraded mode)")

    gimbal.print_status()
    gimbal.shutdown()
    print("\n✓ Demo 4 complete")


def demo_automatic_recovery():
    """
    Demo 5: Automatic error recovery
    """
    print("\n" + "="*70)
    print("DEMO 5: Automatic Recovery")
    print("="*70)

    gimbal = SafeGimbalController(enable_recovery=True)

    print("\nGimbal with automatic recovery enabled")

    print("\nIf transient errors occur:")
    print("  1. Error is detected")
    print("  2. Recovery is attempted automatically")
    print("  3. If successful, operation continues")
    print("  4. If failed, degraded mode activated")

    # Show statistics
    status = gimbal.get_status()
    stats = status['statistics']

    print(f"\nCurrent statistics:")
    print(f"  Operations: {stats['operations']}")
    print(f"  Failures: {stats['failures']}")
    print(f"  Recoveries: {stats['recoveries']}")

    gimbal.shutdown()
    print("\n✓ Demo 5 complete")


def demo_degraded_mode():
    """
    Demo 6: Robot operation in degraded mode
    """
    print("\n" + "="*70)
    print("DEMO 6: Degraded Mode Operation")
    print("="*70)

    gimbal = SafeGimbalController()

    print("\nDemonstrating robot operation when gimbal unavailable")

    degradation = gimbal.get_degradation_level()
    print(f"Degradation level: {degradation}")

    if degradation == 'fully_operational':
        print("\n✓ Gimbal fully operational")
        print("\nCapabilities available:")
        print("  ✓ Camera positioning")
        print("  ✓ Orbital scanning")
        print("  ✓ Object tracking")
        print("  ✓ 3D reconstruction")

    elif degradation == 'degraded':
        print("\n⚠ Gimbal in degraded mode")
        print("\nRobot still functions:")
        print("  ✓ Face recognition (fixed camera)")
        print("  ✓ Object detection (limited FOV)")
        print("  ✓ Spatial awareness (reduced)")
        print("  ✓ Brain/reasoning (unaffected)")
        print("\nReduced capabilities:")
        print("  ✗ Camera positioning")
        print("  ✗ Orbital scanning")

    else:
        print("\n✗ Gimbal failed")
        print("\nRobot still functions:")
        print("  ✓ Core systems operational")
        print("  ✓ Can request manual camera adjustment")
        print("\nDisabled:")
        print("  ✗ All gimbal operations")

    gimbal.shutdown()
    print("\n✓ Demo 6 complete")


def demo_complete_pipeline():
    """
    Demo 7: Complete safe pipeline
    """
    print("\n" + "="*70)
    print("DEMO 7: Complete Safe Pipeline")
    print("="*70)

    print("\nDemonstrating complete safe operation pipeline:")

    # 1. Initialize with safety
    print("\n1. Initialize with safety monitoring...")
    gimbal = SafeGimbalController()

    # 2. Check health
    print("\n2. Check system health...")
    gimbal.print_status()

    # 3. Verify capabilities
    print("\n3. Verify capabilities...")
    caps = gimbal.get_capabilities()

    can_proceed = caps['gimbal_operational']

    if can_proceed:
        print("✓ All systems go")

        # 4. Perform operation
        print("\n4. Perform safe operation...")
        success = gimbal.move_to_angles(0, 0, 0)

        if success:
            print("✓ Operation successful")
        else:
            print("⚠ Operation failed - falling back")

    else:
        print("⚠ Gimbal not available - using fallback strategy")
        print("  → Robot continues with fixed camera")

    # 5. Monitor health
    print("\n5. Continuous health monitoring...")
    print("   (Running in background)")

    # 6. Shutdown safely
    print("\n6. Safe shutdown...")
    gimbal.shutdown()

    print("\n✓ Demo 7 complete - Full safe pipeline demonstrated")


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all demos"""
    print("\n" + "="*70)
    print("SAFE GIMBAL OPERATION DEMOS")
    print("="*70)
    print("\nPhilosophy: 'If the neck/head gimbal fails, the robot does not fail'")
    print("\nThese demos show fault-tolerant gimbal control:")
    print("  • Real-time safety monitoring")
    print("  • Automatic failure detection")
    print("  • Graceful degradation")
    print("  • Error recovery")
    print("  • Emergency stop")
    print("="*70)

    demos = [
        ("Basic Safe Operation", demo_basic_safe_operation),
        ("Failure Handling", demo_failure_handling),
        ("Safe Scanning", demo_safe_scanning),
        ("Emergency Stop", demo_emergency_stop),
        ("Automatic Recovery", demo_automatic_recovery),
        ("Degraded Mode", demo_degraded_mode),
        ("Complete Pipeline", demo_complete_pipeline),
    ]

    try:
        for i, (name, demo_func) in enumerate(demos, 1):
            print(f"\n\n{'='*70}")
            print(f"Running Demo {i}/{len(demos)}: {name}")
            print(f"{'='*70}")

            demo_func()

            if i < len(demos):
                input("\nPress Enter for next demo (or Ctrl+C to exit)...")

    except KeyboardInterrupt:
        print("\n\nDemos interrupted by user")

    print("\n" + "="*70)
    print("ALL DEMOS COMPLETE")
    print("="*70)
    print("\nKey Takeaways:")
    print("  ✓ Gimbal failures are detected immediately")
    print("  ✓ Robot never crashes from gimbal issues")
    print("  ✓ Graceful degradation maintains core functionality")
    print("  ✓ Automatic recovery from transient errors")
    print("  ✓ Emergency stop works instantly")
    print("  ✓ Robot remains operational in all scenarios")
    print("\n🛡 SAFETY FIRST: Gimbal failures won't crash the robot!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
