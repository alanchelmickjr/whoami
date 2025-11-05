#!/usr/bin/env python3
"""
Robot Tool Maker - Example workflow for autonomous tool creation
Demonstrates how a robot can scan objects and create custom tools for manipulation
"""

import sys
import os
import time
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whoami.robot_vision_api import (
    RobotVisionAPI,
    RobotVisionConfig,
    VisionModule,
    create_robot_vision
)
from whoami.scanner_3d import (
    ScanMode,
    MeshQuality,
    ToolType,
    Scanner3DConfig
)
from whoami.face_recognition_api import CameraType


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RobotToolMaker:
    """
    Robot Tool Maker - Autonomous tool creation system
    
    This robot can:
    1. Scan objects in 3D
    2. Analyze their shape
    3. Create custom tools (grippers, sockets, molds, fixtures)
    4. Export STL files for 3D printing
    5. Learn from successful tool creations
    """
    
    def __init__(self, output_dir: str = "robot_tools"):
        """Initialize the Robot Tool Maker"""
        
        # Create output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure vision system
        config = RobotVisionConfig(
            camera_type=CameraType.OAK_D,
            camera_resolution=(1280, 720),
            enable_face_recognition=True,  # To identify who needs the tool
            enable_3d_scanning=True,
            scan_output_directory=str(self.output_dir),
            scanner_3d_config=Scanner3DConfig(
                mesh_quality=MeshQuality.HIGH,
                auto_save=True,
                output_directory=str(self.output_dir)
            )
        )
        
        # Initialize robot vision
        self.vision = RobotVisionAPI(config)
        
        # Tool creation history
        self.tool_history: List[Dict[str, Any]] = []
        
        # Known object-tool mappings
        self.object_tool_map = {
            'bolt': ToolType.SOCKET,
            'nut': ToolType.SOCKET,
            'handle': ToolType.GRIPPER,
            'bottle': ToolType.GRIPPER,
            'figurine': ToolType.MOLD,
            'part': ToolType.FIXTURE,
            'default': ToolType.GRIPPER
        }
        
        logger.info("Robot Tool Maker initialized")
    
    def demonstrate_tool_creation_workflow(self):
        """
        Complete demonstration of tool creation workflow
        """
        print("\n" + "="*60)
        print("ROBOT TOOL MAKER - Complete Workflow Demo")
        print("="*60 + "\n")
        
        with self.vision:
            # Step 1: Identify the operator (optional)
            print("Step 1: Identifying operator...")
            self._identify_operator()
            
            # Step 2: Scan a bolt to create a socket
            print("\nStep 2: Creating socket wrench from bolt scan...")
            self._create_socket_from_bolt()
            
            # Step 3: Scan an object to create a gripper
            print("\nStep 3: Creating custom gripper from object scan...")
            self._create_gripper_from_object()
            
            # Step 4: Create a mold for replication
            print("\nStep 4: Creating mold for object replication...")
            self._create_mold_for_replication()
            
            # Step 5: Create a fixture for holding
            print("\nStep 5: Creating fixture for precision work...")
            self._create_fixture_for_holding()
            
            # Step 6: Show tool creation summary
            print("\nStep 6: Tool Creation Summary...")
            self._show_summary()
    
    def _identify_operator(self) -> Optional[str]:
        """Identify who is operating the robot"""
        try:
            print("Looking for operator...")
            
            # Try to identify person for a few seconds
            for i in range(10):
                results = self.vision.identify_person()
                if results:
                    for result in results:
                        if result.name != "Unknown":
                            print(f"✓ Operator identified: {result.name}")
                            print(f"  Confidence: {result.confidence:.2%}")
                            return result.name
                time.sleep(0.5)
            
            print("No known operator identified - proceeding as guest")
            return "Guest"
            
        except Exception as e:
            logger.error(f"Failed to identify operator: {e}")
            return None
    
    def _create_socket_from_bolt(self) -> bool:
        """Scan a bolt and create a matching socket wrench"""
        try:
            print("\n📹 Scanning bolt head...")
            print("Place the bolt with head facing the camera")
            input("Press Enter when ready to scan...")
            
            # Perform single-shot scan for bolt head
            scan_result = self.vision.scan_object(
                scan_mode=ScanMode.SINGLE_SHOT,
                num_captures=1
            )
            
            if not scan_result:
                print("❌ Failed to scan bolt")
                return False
            
            print(f"✓ Scan complete: {scan_result.total_points:,} points captured")
            print(f"  Bounding box: {scan_result.bounding_box['extents']}")
            
            # Create socket tool
            print("\n🔧 Generating socket wrench...")
            
            # Calculate offset based on tolerance (typically 0.5-1mm for sockets)
            offset = 0.5  # mm clearance
            
            success = self.vision.create_tool(
                tool_type=ToolType.SOCKET,
                offset=offset,
                export_stl=True,
                filename=f"socket_bolt_{int(time.time())}"
            )
            
            if success:
                print("✓ Socket wrench STL file created successfully!")
                print("  Ready for 3D printing")
                
                # Log the creation
                self.tool_history.append({
                    'type': 'socket',
                    'object': 'bolt',
                    'timestamp': time.time(),
                    'points': scan_result.total_points,
                    'offset': offset
                })
                return True
            else:
                print("❌ Failed to create socket wrench")
                return False
                
        except Exception as e:
            logger.error(f"Socket creation failed: {e}")
            return False
    
    def _create_gripper_from_object(self) -> bool:
        """Scan an object and create custom gripper fingers"""
        try:
            print("\n📹 Scanning object for gripper creation...")
            print("Place the object to be gripped")
            input("Press Enter when ready to scan...")
            
            # Perform multi-angle scan for better gripper fit
            print("Performing multi-angle scan (3 captures)...")
            
            scan_result = self.vision.scan_object(
                scan_mode=ScanMode.MULTI_ANGLE,
                num_captures=3
            )
            
            if not scan_result:
                print("❌ Failed to scan object")
                return False
            
            print(f"✓ Scan complete: {scan_result.total_points:,} points captured")
            
            # Analyze object dimensions
            extents = scan_result.bounding_box['extents']
            print(f"  Object dimensions: {extents[0]:.1f} x {extents[1]:.1f} x {extents[2]:.1f} mm")
            
            # Create gripper with appropriate offset
            print("\n🤖 Generating custom gripper...")
            
            # Calculate gripper offset based on object size
            # Larger objects need more clearance
            max_dimension = max(extents)
            offset = 2.0 if max_dimension < 50 else 3.0  # mm
            
            success = self.vision.create_tool(
                tool_type=ToolType.GRIPPER,
                offset=offset,
                export_stl=True,
                filename=f"gripper_custom_{int(time.time())}"
            )
            
            if success:
                print("✓ Custom gripper STL file created successfully!")
                print(f"  Gripper offset: {offset}mm")
                print("  Ready for 3D printing")
                
                # Log the creation
                self.tool_history.append({
                    'type': 'gripper',
                    'object': 'custom',
                    'timestamp': time.time(),
                    'points': scan_result.total_points,
                    'offset': offset,
                    'dimensions': extents
                })
                return True
            else:
                print("❌ Failed to create gripper")
                return False
                
        except Exception as e:
            logger.error(f"Gripper creation failed: {e}")
            return False
    
    def _create_mold_for_replication(self) -> bool:
        """Create a two-part mold for object replication"""
        try:
            print("\n📹 Scanning object for mold creation...")
            print("Place the object to be replicated")
            input("Press Enter when ready to scan...")
            
            # Perform high-quality scan for mold
            scan_result = self.vision.scan_object(
                scan_mode=ScanMode.MULTI_ANGLE,
                num_captures=4  # More captures for better mold quality
            )
            
            if not scan_result:
                print("❌ Failed to scan object")
                return False
            
            print(f"✓ Scan complete: {scan_result.total_points:,} points captured")
            
            # Create mold
            print("\n🏭 Generating two-part casting mold...")
            
            # Molds need more offset for material thickness
            offset = 5.0  # mm for mold walls
            
            success = self.vision.create_tool(
                tool_type=ToolType.MOLD,
                offset=offset,
                export_stl=True,
                filename=f"mold_twopart_{int(time.time())}"
            )
            
            if success:
                print("✓ Two-part mold STL files created successfully!")
                print("  Top and bottom mold halves ready")
                print("  Includes alignment features")
                print("  Ready for 3D printing and casting")
                
                # Log the creation
                self.tool_history.append({
                    'type': 'mold',
                    'object': 'replication',
                    'timestamp': time.time(),
                    'points': scan_result.total_points,
                    'offset': offset
                })
                return True
            else:
                print("❌ Failed to create mold")
                return False
                
        except Exception as e:
            logger.error(f"Mold creation failed: {e}")
            return False
    
    def _create_fixture_for_holding(self) -> bool:
        """Create a holding fixture for precision work"""
        try:
            print("\n📹 Scanning object for fixture creation...")
            print("Place the object that needs to be held")
            input("Press Enter when ready to scan...")
            
            # Single scan is usually enough for fixtures
            scan_result = self.vision.scan_object(
                scan_mode=ScanMode.SINGLE_SHOT,
                num_captures=1
            )
            
            if not scan_result:
                print("❌ Failed to scan object")
                return False
            
            print(f"✓ Scan complete: {scan_result.total_points:,} points captured")
            
            # Create fixture
            print("\n🔩 Generating holding fixture...")
            
            # Fixtures need minimal offset for tight fit
            offset = 1.0  # mm for snug fit
            
            success = self.vision.create_tool(
                tool_type=ToolType.FIXTURE,
                offset=offset,
                export_stl=True,
                filename=f"fixture_holder_{int(time.time())}"
            )
            
            if success:
                print("✓ Holding fixture STL file created successfully!")
                print("  V-shaped cradle design")
                print("  Stable base plate included")
                print("  Ready for 3D printing")
                
                # Log the creation
                self.tool_history.append({
                    'type': 'fixture',
                    'object': 'holder',
                    'timestamp': time.time(),
                    'points': scan_result.total_points,
                    'offset': offset
                })
                return True
            else:
                print("❌ Failed to create fixture")
                return False
                
        except Exception as e:
            logger.error(f"Fixture creation failed: {e}")
            return False
    
    def _show_summary(self):
        """Show summary of all tools created"""
        print("\n" + "="*60)
        print("TOOL CREATION SUMMARY")
        print("="*60)
        
        if not self.tool_history:
            print("No tools created in this session")
            return
        
        print(f"\nTotal tools created: {len(self.tool_history)}")
        print("\nTools created:")
        
        for i, tool in enumerate(self.tool_history, 1):
            print(f"\n{i}. {tool['type'].upper()} Tool")
            print(f"   Object: {tool['object']}")
            print(f"   Points scanned: {tool['points']:,}")
            print(f"   Offset used: {tool['offset']}mm")
            if 'dimensions' in tool:
                dims = tool['dimensions']
                print(f"   Object size: {dims[0]:.1f} x {dims[1]:.1f} x {dims[2]:.1f} mm")
            print(f"   Created: {time.strftime('%H:%M:%S', time.localtime(tool['timestamp']))}")
        
        print(f"\n✓ All STL files saved to: {self.output_dir}")
        print("✓ Files are ready for 3D printing")
    
    def autonomous_tool_creation(self, object_type: str = "unknown") -> bool:
        """
        Autonomous tool creation based on object type
        
        Args:
            object_type: Type of object (bolt, nut, handle, etc.)
        
        Returns:
            True if tool created successfully
        """
        print(f"\n🤖 Autonomous Tool Creation for: {object_type}")
        
        # Determine appropriate tool type
        tool_type = self.object_tool_map.get(
            object_type.lower(), 
            self.object_tool_map['default']
        )
        
        print(f"Selected tool type: {tool_type.value}")
        
        with self.vision:
            # Scan the object
            print("Scanning object...")
            scan_result = self.vision.scan_object(
                scan_mode=ScanMode.SINGLE_SHOT if tool_type == ToolType.SOCKET 
                else ScanMode.MULTI_ANGLE,
                num_captures=1 if tool_type == ToolType.SOCKET else 3
            )
            
            if not scan_result:
                print("❌ Scan failed")
                return False
            
            print(f"✓ Scanned {scan_result.total_points:,} points")
            
            # Create appropriate tool
            print(f"Creating {tool_type.value}...")
            
            # Dynamic offset based on tool type
            offset_map = {
                ToolType.SOCKET: 0.5,
                ToolType.GRIPPER: 2.0,
                ToolType.MOLD: 5.0,
                ToolType.FIXTURE: 1.0
            }
            offset = offset_map.get(tool_type, 2.0)
            
            success = self.vision.create_tool(
                tool_type=tool_type,
                offset=offset,
                export_stl=True,
                filename=f"{tool_type.value}_{object_type}_{int(time.time())}"
            )
            
            if success:
                print(f"✓ {tool_type.value} created successfully!")
                return True
            else:
                print(f"❌ Failed to create {tool_type.value}")
                return False
    
    def batch_tool_creation(self, objects: List[str]):
        """
        Create tools for multiple objects in sequence
        
        Args:
            objects: List of object types to create tools for
        """
        print(f"\n📦 Batch Tool Creation - {len(objects)} objects")
        
        successful = 0
        failed = 0
        
        for obj in objects:
            print(f"\nProcessing: {obj}")
            if self.autonomous_tool_creation(obj):
                successful += 1
            else:
                failed += 1
            
            # Brief pause between scans
            if obj != objects[-1]:
                print("Preparing for next object...")
                time.sleep(2)
        
        print(f"\n📊 Batch Complete:")
        print(f"   Successful: {successful}")
        print(f"   Failed: {failed}")
        print(f"   Success rate: {successful/len(objects)*100:.1f}%")


def main():
    """Main function to demonstrate robot tool making capabilities"""
    
    print("\n" + "="*70)
    print(" "*20 + "ROBOT TOOL MAKER v1.0")
    print(" "*15 + "Autonomous Tool Creation System")
    print("="*70)
    
    # Create robot tool maker
    robot = RobotToolMaker(output_dir="robot_tools_demo")
    
    # Menu options
    while True:
        print("\n" + "-"*50)
        print("Select an option:")
        print("1. Run complete workflow demonstration")
        print("2. Create socket from bolt")
        print("3. Create custom gripper")
        print("4. Create casting mold")
        print("5. Create holding fixture")
        print("6. Autonomous tool creation (specify object)")
        print("7. Batch tool creation")
        print("0. Exit")
        print("-"*50)
        
        choice = input("\nEnter your choice (0-7): ").strip()
        
        if choice == '0':
            print("\n👋 Goodbye! Happy tool making!")
            break
            
        elif choice == '1':
            robot.demonstrate_tool_creation_workflow()
            
        elif choice == '2':
            with robot.vision:
                robot._create_socket_from_bolt()
                
        elif choice == '3':
            with robot.vision:
                robot._create_gripper_from_object()
                
        elif choice == '4':
            with robot.vision:
                robot._create_mold_for_replication()
                
        elif choice == '5':
            with robot.vision:
                robot._create_fixture_for_holding()
                
        elif choice == '6':
            object_type = input("Enter object type (bolt/nut/handle/bottle/part): ").strip()
            if object_type:
                robot.autonomous_tool_creation(object_type)
                
        elif choice == '7':
            print("\nEnter object types (comma-separated):")
            objects_input = input("Example: bolt,nut,handle: ").strip()
            if objects_input:
                objects = [obj.strip() for obj in objects_input.split(',')]
                robot.batch_tool_creation(objects)
        
        else:
            print("❌ Invalid choice. Please try again.")
    
    # Show final summary
    if robot.tool_history:
        robot._show_summary()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Tool creation interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"\n❌ Error: {e}")