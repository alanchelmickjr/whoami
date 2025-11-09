#!/usr/bin/env python3
"""
Test script for 3DOF gimbal kinematics

Tests forward and inverse kinematics to verify mathematical model
Run this BEFORE connecting to physical hardware
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from whoami.gimbal_3dof import (
    KinematicParameters,
    Kinematics3DOF,
    JointAngles,
    OrbitalPathGenerator
)

print("="*70)
print("3DOF GIMBAL KINEMATICS TEST")
print("="*70)

# Create kinematic parameters (adjust for your build)
params = KinematicParameters(
    base_height=0.10,      # 10cm
    arm_length=0.15,       # 15cm
    camera_offset=0.05     # 5cm
)

print("\nKinematic Parameters:")
print(f"  Base height: {params.base_height*100:.1f}cm")
print(f"  Arm length: {params.arm_length*100:.1f}cm")
print(f"  Camera offset: {params.camera_offset*100:.1f}cm")
print(f"  Max reach: {(params.base_height + params.arm_length + params.camera_offset)*100:.1f}cm")

# Create kinematics solver
kinematics = Kinematics3DOF(params)
path_gen = OrbitalPathGenerator(kinematics)

print("\n" + "="*70)
print("TEST 1: Forward Kinematics")
print("="*70)

test_angles = [
    JointAngles(0, 0, 0),
    JointAngles(45, 0, 0),
    JointAngles(0, 45, 0),
    JointAngles(0, 0, 90),
    JointAngles(45, 30, 0),
]

for angles in test_angles:
    pose = kinematics.forward_kinematics(angles)
    print(f"\nAngles: {angles}")
    print(f"  Position: [{pose.position[0]:.3f}, {pose.position[1]:.3f}, {pose.position[2]:.3f}]")
    print(f"  Forward:  [{pose.forward[0]:.3f}, {pose.forward[1]:.3f}, {pose.forward[2]:.3f}]")
    print(f"  Up:       [{pose.up[0]:.3f}, {pose.up[1]:.3f}, {pose.up[2]:.3f}]")

print("\n" + "="*70)
print("TEST 2: Inverse Kinematics")
print("="*70)

# Test IK for orbiting around a target
target = np.array([0.25, 0.0, 0.0])  # 25cm forward
print(f"\nTarget point: {target}")

# Generate positions on a circle around target
radius = 0.10  # 10cm orbital radius
num_points = 8

print(f"Orbital radius: {radius*100:.1f}cm")
print(f"Testing {num_points} positions around orbit...")

success_count = 0
for i in range(num_points):
    angle = 2 * np.pi * i / num_points

    # Position on circle
    camera_pos = target + np.array([
        radius * np.cos(angle),
        radius * np.sin(angle),
        0
    ])

    # Solve IK
    result = kinematics.inverse_kinematics(
        target_position=target,
        camera_position=camera_pos,
        camera_roll=0.0
    )

    if result:
        success_count += 1
        print(f"  Position {i+1}/8: ✓ {result}")

        # Verify by forward kinematics
        computed_pose = kinematics.forward_kinematics(result)
        pos_error = np.linalg.norm(computed_pose.position - camera_pos)

        if pos_error > 0.01:
            print(f"    WARNING: Position error {pos_error*1000:.1f}mm")
    else:
        print(f"  Position {i+1}/8: ✗ UNREACHABLE")

print(f"\nSuccess rate: {success_count}/{num_points} ({100*success_count/num_points:.0f}%)")

print("\n" + "="*70)
print("TEST 3: Orbital Path Generation")
print("="*70)

# Generate horizontal orbit
print("\nGenerating horizontal orbit...")
waypoints = path_gen.generate_horizontal_orbit(
    center=target,
    radius=radius,
    num_points=12
)

print(f"Generated {len(waypoints)} waypoints:")
for i, wp in enumerate(waypoints[:4]):  # Show first 4
    print(f"  {i+1}. {wp}")
if len(waypoints) > 4:
    print(f"  ... and {len(waypoints)-4} more")

# Generate vertical orbit
print("\nGenerating vertical orbit...")
waypoints = path_gen.generate_vertical_orbit(
    center=target,
    radius=radius,
    axis_direction='x',
    num_points=12
)

print(f"Generated {len(waypoints)} waypoints")

# Generate spherical scan
print("\nGenerating spherical scan pattern...")
waypoints = path_gen.generate_spherical_scan(
    center=target,
    radius=radius,
    num_rings=3,
    points_per_ring=12
)

print(f"Generated {len(waypoints)} waypoints across {3} rings")

print("\n" + "="*70)
print("TEST 4: Workspace Analysis")
print("="*70)

# Test reachability at different distances
print("\nTesting reachability at various distances...")

test_distances = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]

for dist in test_distances:
    point = np.array([dist, 0, 0])
    reachable = kinematics.check_reachability(point)
    status = "✓ REACHABLE" if reachable else "✗ OUT OF REACH"
    print(f"  {dist*100:5.1f}cm: {status}")

print("\n" + "="*70)
print("TEST 5: Round-Trip Accuracy")
print("="*70)

# Test FK → IK → FK round trip
print("\nTesting forward → inverse → forward kinematics...")

original_angles = JointAngles(30, 20, 0)
print(f"Original angles: {original_angles}")

# Forward kinematics
pose1 = kinematics.forward_kinematics(original_angles)
print(f"FK result: position={pose1.position}")

# Inverse kinematics (use same target and position)
recovered_angles = kinematics.inverse_kinematics(
    target_position=pose1.position + pose1.forward * 0.1,  # 10cm in front
    camera_position=pose1.position,
    camera_roll=original_angles.roll
)

if recovered_angles:
    print(f"IK result: {recovered_angles}")

    # Forward again
    pose2 = kinematics.forward_kinematics(recovered_angles)
    print(f"FK again: position={pose2.position}")

    # Compare
    angle_diff = np.abs(recovered_angles.to_array() - original_angles.to_array())
    pos_diff = np.linalg.norm(pose2.position - pose1.position)

    print(f"\nAccuracy:")
    print(f"  Angle differences: {angle_diff}")
    print(f"  Position error: {pos_diff*1000:.2f}mm")

    if np.all(angle_diff < 5) and pos_diff < 0.01:
        print("  ✓ PASSED: Round-trip accurate within tolerances")
    else:
        print("  ⚠ WARNING: Round-trip errors exceed tolerances")
else:
    print("  ✗ FAILED: IK could not solve")

print("\n" + "="*70)
print("ALL TESTS COMPLETE")
print("="*70)
print("\nIf all tests passed, kinematics are working correctly!")
print("You can now test with physical hardware.\n")
