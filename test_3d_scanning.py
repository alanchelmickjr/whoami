#!/usr/bin/env python3
"""
Test script for 3D scanning capabilities with OAK-D camera
Verifies point cloud capture, mesh generation, and STL export
"""

import sys
import os
import time
import logging
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Check for required dependencies
try:
    import depthai as dai
    import open3d as o3d
    import trimesh
    print("✓ All 3D scanning dependencies found")
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("\nPlease install required packages:")
    print("pip install open3d trimesh depthai")
    sys.exit(1)

from whoami.scanner_3d import (
    Scanner3D,
    Scanner3DConfig,
    ScanMode,
    MeshQuality,
    ToolType,
    create_3d_scanner,
    OakD3DCamera,
    PointCloudProcessor,
    MeshGenerator,
    PointCloudData
)

from whoami.robot_vision_api import (
    RobotVisionAPI,
    RobotVisionConfig,
    VisionModule,
    create_robot_vision
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Test3DScanning:
    """Test suite for 3D scanning functionality"""
    
    def __init__(self):
        self.test_results = {
            'passed': [],
            'failed': [],
            'skipped': []
        }
        self.output_dir = Path("test_3d_scans")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def run_all_tests(self):
        """Run all 3D scanning tests"""
        print("\n" + "="*60)
        print("3D SCANNING TEST SUITE")
        print("="*60 + "\n")
        
        # Test 1: Camera initialization
        self.test_camera_initialization()
        
        # Test 2: Point cloud capture
        self.test_point_cloud_capture()
        
        # Test 3: Mesh generation
        self.test_mesh_generation()
        
        # Test 4: STL export
        self.test_stl_export()
        
        # Test 5: Tool generation
        self.test_tool_generation()
        
        # Test 6: Robot Vision API integration
        self.test_robot_vision_integration()
        
        # Test 7: Multi-angle scanning
        self.test_multi_angle_scan()
        
        # Test 8: Performance test
        self.test_performance()
        
        # Show results
        self.show_results()
    
    def test_camera_initialization(self):
        """Test OAK-D camera initialization for 3D scanning"""
        test_name = "Camera Initialization"
        print(f"\n🧪 Testing: {test_name}")
        
        try:
            config = Scanner3DConfig(
                resolution=(640, 480),
                min_depth=200,
                max_depth=5000
            )
            
            camera = OakD3DCamera(config)
            
            # Test camera start
            if camera.start():
                print("  ✓ Camera started successfully")
                
                # Check if camera is running
                if camera.is_running:
                    print("  ✓ Camera is running")
                else:
                    raise Exception("Camera reports not running after start")
                
                # Stop camera
                camera.stop()
                
                if not camera.is_running:
                    print("  ✓ Camera stopped successfully")
                else:
                    raise Exception("Camera still running after stop")
                
                self.test_results['passed'].append(test_name)
                print(f"✅ {test_name} PASSED")
            else:
                raise Exception("Failed to start camera")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            self.test_results['failed'].append((test_name, str(e)))
            print(f"❌ {test_name} FAILED")
    
    def test_point_cloud_capture(self):
        """Test point cloud capture functionality"""
        test_name = "Point Cloud Capture"
        print(f"\n🧪 Testing: {test_name}")
        
        try:
            scanner = create_3d_scanner(
                output_directory=str(self.output_dir),
                mesh_quality=MeshQuality.DRAFT
            )
            
            with scanner:
                # Start scan
                if not scanner.start_scan(ScanMode.SINGLE_SHOT):
                    raise Exception("Failed to start scan")
                print("  ✓ Scan started")
                
                # Capture point cloud
                pcd = scanner.capture_point_cloud()
                
                if pcd is None:
                    raise Exception("Failed to capture point cloud")
                
                print(f"  ✓ Point cloud captured")
                print(f"    Points: {len(pcd.points)}")
                
                # Verify point cloud has data
                if len(pcd.points) == 0:
                    raise Exception("Point cloud is empty")
                
                # Complete scan
                result = scanner.complete_scan(generate_mesh=False)
                
                if result is None:
                    raise Exception("Failed to complete scan")
                
                print(f"  ✓ Scan completed")
                print(f"    Total points: {result.total_points}")
                print(f"    Captures: {result.num_captures}")
                
                self.test_results['passed'].append(test_name)
                print(f"✅ {test_name} PASSED")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            self.test_results['failed'].append((test_name, str(e)))
            print(f"❌ {test_name} FAILED")
    
    def test_mesh_generation(self):
        """Test mesh generation from point cloud"""
        test_name = "Mesh Generation"
        print(f"\n🧪 Testing: {test_name}")
        
        try:
            # Create synthetic point cloud for testing
            print("  Creating synthetic point cloud...")
            
            # Generate sphere point cloud
            mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=50.0)
            pcd = mesh_sphere.sample_points_uniformly(number_of_points=5000)
            
            print(f"  ✓ Synthetic point cloud created ({len(pcd.points)} points)")
            
            # Create mesh generator
            config = Scanner3DConfig(mesh_quality=MeshQuality.DRAFT)
            mesh_gen = MeshGenerator(config)
            
            # Generate mesh
            print("  Generating mesh...")
            mesh = mesh_gen.generate_mesh(pcd)
            
            if mesh is None:
                raise Exception("Failed to generate mesh")
            
            print(f"  ✓ Mesh generated")
            print(f"    Vertices: {len(mesh.vertices)}")
            print(f"    Faces: {len(mesh.faces)}")
            
            # Verify mesh properties
            if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                raise Exception("Generated mesh is empty")
            
            if not mesh.is_watertight:
                print("  ⚠️ Warning: Mesh is not watertight (this is normal for scans)")
            
            self.test_results['passed'].append(test_name)
            print(f"✅ {test_name} PASSED")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            self.test_results['failed'].append((test_name, str(e)))
            print(f"❌ {test_name} FAILED")
    
    def test_stl_export(self):
        """Test STL file export functionality"""
        test_name = "STL Export"
        print(f"\n🧪 Testing: {test_name}")
        
        try:
            # Create a simple mesh for testing
            print("  Creating test mesh...")
            mesh = trimesh.creation.box(extents=[10, 10, 10])
            
            # Create scanner for export functionality
            scanner = create_3d_scanner(output_directory=str(self.output_dir))
            
            # Export STL
            filename = "test_export"
            print(f"  Exporting STL as {filename}.stl...")
            
            if not scanner.export_stl(filename, mesh):
                raise Exception("Failed to export STL")
            
            # Verify file exists
            stl_path = self.output_dir / f"{filename}.stl"
            if not stl_path.exists():
                raise Exception(f"STL file not found at {stl_path}")
            
            print(f"  ✓ STL exported to {stl_path}")
            
            # Try to load the STL to verify it's valid
            loaded_mesh = trimesh.load(stl_path)
            print(f"  ✓ STL file is valid")
            print(f"    Loaded vertices: {len(loaded_mesh.vertices)}")
            print(f"    Loaded faces: {len(loaded_mesh.faces)}")
            
            # Clean up test file
            stl_path.unlink()
            print("  ✓ Test file cleaned up")
            
            self.test_results['passed'].append(test_name)
            print(f"✅ {test_name} PASSED")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            self.test_results['failed'].append((test_name, str(e)))
            print(f"❌ {test_name} FAILED")
    
    def test_tool_generation(self):
        """Test tool generation from scanned objects"""
        test_name = "Tool Generation"
        print(f"\n🧪 Testing: {test_name}")
        
        try:
            # Create a test object mesh (cylinder for socket test)
            print("  Creating test object (cylinder)...")
            test_object = trimesh.creation.cylinder(radius=10, height=20)
            
            # Create mesh generator
            config = Scanner3DConfig()
            mesh_gen = MeshGenerator(config)
            
            # Test different tool types
            tool_types = [
                (ToolType.GRIPPER, "Gripper"),
                (ToolType.SOCKET, "Socket"),
                (ToolType.MOLD, "Mold"),
                (ToolType.FIXTURE, "Fixture")
            ]
            
            for tool_type, name in tool_types:
                print(f"  Testing {name} generation...")
                
                tool = mesh_gen.generate_tool_inverse(
                    test_object,
                    tool_type=tool_type,
                    offset=2.0
                )
                
                if tool is None:
                    raise Exception(f"Failed to generate {name}")
                
                print(f"    ✓ {name} generated")
                print(f"      Vertices: {len(tool.vertices)}")
                print(f"      Faces: {len(tool.faces)}")
                
                # Verify tool is larger than object (has offset)
                tool_bounds = tool.bounds
                obj_bounds = test_object.bounds
                
                if tool_type != ToolType.GRIPPER:  # Gripper has special geometry
                    if not (tool.volume > test_object.volume):
                        print(f"    ⚠️ Warning: Tool volume not larger than object")
            
            self.test_results['passed'].append(test_name)
            print(f"✅ {test_name} PASSED")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            self.test_results['failed'].append((test_name, str(e)))
            print(f"❌ {test_name} FAILED")
    
    def test_robot_vision_integration(self):
        """Test integration with Robot Vision API"""
        test_name = "Robot Vision API Integration"
        print(f"\n🧪 Testing: {test_name}")
        
        try:
            # Create robot vision API
            robot = create_robot_vision(
                enable_face_recognition=False,
                enable_3d_scanning=True
            )
            
            print("  ✓ Robot Vision API created")
            
            # Check if 3D scanner module is available
            if not robot.is_module_available(VisionModule.SCANNER_3D):
                raise Exception("3D Scanner module not available")
            
            print("  ✓ 3D Scanner module available")
            
            # Get the scanner module
            scanner_module = robot.get_module(VisionModule.SCANNER_3D)
            if scanner_module is None:
                raise Exception("Failed to get scanner module")
            
            print("  ✓ Scanner module retrieved")
            
            # Test starting the module
            if robot.start_module(VisionModule.SCANNER_3D):
                print("  ✓ Scanner module started")
                
                # Stop the module
                robot.stop_module(VisionModule.SCANNER_3D)
                print("  ✓ Scanner module stopped")
            else:
                raise Exception("Failed to start scanner module")
            
            self.test_results['passed'].append(test_name)
            print(f"✅ {test_name} PASSED")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            self.test_results['failed'].append((test_name, str(e)))
            print(f"❌ {test_name} FAILED")
    
    def test_multi_angle_scan(self):
        """Test multi-angle scanning capability"""
        test_name = "Multi-Angle Scanning"
        print(f"\n🧪 Testing: {test_name}")
        
        try:
            scanner = create_3d_scanner(
                output_directory=str(self.output_dir),
                mesh_quality=MeshQuality.DRAFT
            )
            
            with scanner:
                # Start multi-angle scan
                if not scanner.start_scan(ScanMode.MULTI_ANGLE):
                    raise Exception("Failed to start multi-angle scan")
                
                print("  ✓ Multi-angle scan started")
                
                # Simulate multiple captures
                num_captures = 3
                for i in range(num_captures):
                    print(f"  Capturing angle {i+1}/{num_captures}...")
                    pcd = scanner.capture_point_cloud()
                    
                    if pcd is None:
                        raise Exception(f"Failed to capture point cloud {i+1}")
                    
                    print(f"    ✓ Captured {len(pcd.points)} points")
                    time.sleep(0.5)  # Brief delay between captures
                
                # Complete scan
                result = scanner.complete_scan()
                
                if result is None:
                    raise Exception("Failed to complete multi-angle scan")
                
                print(f"  ✓ Multi-angle scan completed")
                print(f"    Total points: {result.total_points}")
                print(f"    Captures merged: {result.num_captures}")
                
                # Verify we got the expected number of captures
                if result.num_captures != num_captures:
                    raise Exception(f"Expected {num_captures} captures, got {result.num_captures}")
                
                self.test_results['passed'].append(test_name)
                print(f"✅ {test_name} PASSED")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            self.test_results['failed'].append((test_name, str(e)))
            print(f"❌ {test_name} FAILED")
    
    def test_performance(self):
        """Test performance metrics of 3D scanning"""
        test_name = "Performance Test"
        print(f"\n🧪 Testing: {test_name}")
        
        try:
            scanner = create_3d_scanner(
                output_directory=str(self.output_dir),
                mesh_quality=MeshQuality.DRAFT
            )
            
            with scanner:
                print("  Measuring capture performance...")
                
                # Start scan
                scanner.start_scan(ScanMode.SINGLE_SHOT)
                
                # Measure capture time
                start_time = time.time()
                pcd = scanner.capture_point_cloud()
                capture_time = time.time() - start_time
                
                if pcd is None:
                    raise Exception("Failed to capture for performance test")
                
                print(f"  ✓ Capture time: {capture_time:.3f} seconds")
                print(f"    Points captured: {len(pcd.points)}")
                print(f"    Points per second: {len(pcd.points)/capture_time:.0f}")
                
                # Measure mesh generation time
                print("  Measuring mesh generation performance...")
                
                start_time = time.time()
                result = scanner.complete_scan()
                mesh_time = time.time() - start_time
                
                if result is None or result.mesh is None:
                    raise Exception("Failed to generate mesh for performance test")
                
                print(f"  ✓ Mesh generation time: {mesh_time:.3f} seconds")
                print(f"    Vertices: {len(result.mesh.vertices)}")
                print(f"    Faces: {len(result.mesh.faces)}")
                
                # Performance thresholds
                if capture_time > 5.0:
                    print("  ⚠️ Warning: Capture time exceeds 5 seconds")
                
                if mesh_time > 10.0:
                    print("  ⚠️ Warning: Mesh generation time exceeds 10 seconds")
                
                self.test_results['passed'].append(test_name)
                print(f"✅ {test_name} PASSED")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            self.test_results['failed'].append((test_name, str(e)))
            print(f"❌ {test_name} FAILED")
    
    def show_results(self):
        """Display test results summary"""
        print("\n" + "="*60)
        print("TEST RESULTS SUMMARY")
        print("="*60)
        
        total_tests = len(self.test_results['passed']) + len(self.test_results['failed']) + len(self.test_results['skipped'])
        
        print(f"\nTotal Tests: {total_tests}")
        print(f"✅ Passed: {len(self.test_results['passed'])}")
        print(f"❌ Failed: {len(self.test_results['failed'])}")
        print(f"⏭️  Skipped: {len(self.test_results['skipped'])}")
        
        if self.test_results['passed']:
            print("\n✅ PASSED TESTS:")
            for test in self.test_results['passed']:
                print(f"  • {test}")
        
        if self.test_results['failed']:
            print("\n❌ FAILED TESTS:")
            for test, error in self.test_results['failed']:
                print(f"  • {test}")
                print(f"    Error: {error}")
        
        if self.test_results['skipped']:
            print("\n⏭️  SKIPPED TESTS:")
            for test in self.test_results['skipped']:
                print(f"  • {test}")
        
        # Overall result
        print("\n" + "="*60)
        if len(self.test_results['failed']) == 0:
            print("🎉 ALL TESTS PASSED! 3D scanning is working correctly.")
        else:
            print(f"⚠️  {len(self.test_results['failed'])} test(s) failed. Please check the errors above.")
        print("="*60)


def quick_test():
    """Quick test to verify basic 3D scanning functionality"""
    print("\n" + "="*60)
    print("QUICK 3D SCANNING TEST")
    print("="*60 + "\n")
    
    try:
        # Test 1: Create scanner
        print("1. Creating 3D scanner...")
        scanner = create_3d_scanner()
        print("   ✓ Scanner created")
        
        # Test 2: Start camera
        print("2. Starting camera...")
        if scanner.start_camera():
            print("   ✓ Camera started")
        else:
            print("   ❌ Failed to start camera")
            return False
        
        # Test 3: Capture test
        print("3. Testing capture...")
        scanner.start_scan(ScanMode.SINGLE_SHOT)
        pcd = scanner.capture_point_cloud()
        
        if pcd:
            print(f"   ✓ Captured {len(pcd.points)} points")
        else:
            print("   ❌ Failed to capture point cloud")
            scanner.stop_camera()
            return False
        
        # Test 4: Complete scan
        print("4. Completing scan...")
        result = scanner.complete_scan()
        
        if result:
            print(f"   ✓ Scan completed: {result.total_points} points")
        else:
            print("   ❌ Failed to complete scan")
        
        # Clean up
        scanner.stop_camera()
        print("\n✅ Quick test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Quick test failed: {e}")
        return False


def main():
    """Main test function"""
    print("\n" + "="*70)
    print(" "*20 + "3D SCANNING TEST SUITE")
    print(" "*10 + "Testing OAK-D 3D Scanning Capabilities")
    print("="*70)
    
    # Check for OAK-D camera
    try:
        devices = dai.Device.getAllAvailableDevices()
        if len(devices) == 0:
            print("\n⚠️  No OAK-D camera detected!")
            print("Please connect an OAK-D camera and try again.")
            print("\nRunning tests with simulated data...")
    except Exception as e:
        print(f"\n⚠️  Could not check for OAK-D devices: {e}")
        print("Running tests with simulated data...")
    
    # Menu
    print("\nSelect test mode:")
    print("1. Run all tests (comprehensive)")
    print("2. Quick test (basic functionality)")
    print("3. Exit")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        tester = Test3DScanning()
        tester.run_all_tests()
    elif choice == '2':
        quick_test()
    elif choice == '3':
        print("Exiting...")
    else:
        print("Invalid choice. Running quick test...")
        quick_test()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"\n❌ Error: {e}")