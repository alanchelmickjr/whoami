#!/usr/bin/env python3
"""
Robot Eyes Movement Demonstration
Shows how to use gimbal control and vision behaviors for natural robot eye movements

This example demonstrates:
- Face centering and tracking
- Environmental scanning patterns
- Object inspection routines
- Idle curiosity behaviors
- Search patterns for lost targets
- Expressive gestures
"""

import sys
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from whoami.robot_vision_api import (
    RobotVisionAPI,
    RobotVisionConfig,
    CameraType,
    create_robot_vision
)


class RobotEyesDemo:
    """
    Demonstration of robot eye movements and vision behaviors
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize robot eyes demo
        
        Args:
            config_file: Optional path to gimbal config file
        """
        # Load configuration
        self.gimbal_config = self._load_config(config_file)
        
        # Create robot vision with gimbal control
        self.robot = self._create_robot_vision()
        
        # Track demo state
        self.current_person = None
        self.demo_running = False
        
        print("=" * 60)
        print("Robot Eyes Movement Demonstration")
        print("=" * 60)
    
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load gimbal configuration from file"""
        if config_file and Path(config_file).exists():
            with open(config_file, 'r') as f:
                config_data = json.load(f)
                print(f"Loaded configuration from {config_file}")
                return config_data
        else:
            # Default configuration
            return {
                "gimbal": {
                    "servos": {
                        "pan": {"id": 1, "min_angle": -90, "max_angle": 90},
                        "tilt": {"id": 2, "min_angle": -45, "max_angle": 45}
                    },
                    "communication": {
                        "serial_port": "/dev/ttyUSB0",
                        "baudrate": 1000000
                    }
                }
            }
    
    def _create_robot_vision(self) -> RobotVisionAPI:
        """Create robot vision API with gimbal control"""
        # Extract gimbal settings
        gimbal_settings = self.gimbal_config.get("gimbal", {})
        
        config = RobotVisionConfig(
            camera_type=CameraType.OAK_D,
            enable_face_recognition=True,
            enable_3d_scanning=True,
            enable_gimbal_control=True,
            
            # Gimbal configuration
            gimbal_serial_port=gimbal_settings.get("communication", {}).get("serial_port", "/dev/ttyUSB0"),
            gimbal_baudrate=gimbal_settings.get("communication", {}).get("baudrate", 1000000),
            enable_vision_tracking=True,
            enable_curiosity_mode=True
        )
        
        # Create robot vision
        robot = RobotVisionAPI(config)
        
        # Register callbacks
        robot.register_callback('on_person_identified', self._on_person_identified)
        robot.register_callback('on_target_lost', self._on_target_lost)
        robot.register_callback('on_behavior_change', self._on_behavior_change)
        robot.register_callback('on_gimbal_movement', self._on_gimbal_movement)
        
        return robot
    
    # ========================================================================
    # Demonstration Scenarios
    # ========================================================================
    
    def demo_greeting(self):
        """Demonstrate greeting behavior"""
        print("\n🤝 Greeting Demonstration")
        print("-" * 40)
        
        if not self.robot.start_gimbal_control():
            print("❌ Failed to start gimbal control")
            return
        
        print("Robot is greeting...")
        
        # Get gimbal module
        gimbal = self.robot.get_module(self.robot.capabilities.keys().__iter__().__next__())
        if gimbal and hasattr(gimbal, 'system'):
            gimbal.system.behaviors.greeting_gesture()
            time.sleep(2)
        
        print("✓ Greeting complete!")
    
    def demo_face_tracking(self, duration: float = 30.0):
        """
        Demonstrate face tracking behavior
        
        Args:
            duration: How long to track faces
        """
        print("\n👤 Face Tracking Demonstration")
        print("-" * 40)
        print("Looking for faces to track...")
        
        if not self.robot.enable_face_tracking():
            print("❌ Failed to enable face tracking")
            return
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # Get frame and process for faces
            results = self.robot.identify_person()
            
            if results:
                print(f"Tracking {len(results)} face(s)")
                for result in results:
                    if result.name != "Unknown":
                        print(f"  - {result.name} (confidence: {result.confidence:.2f})")
            
            time.sleep(0.1)
        
        self.robot.disable_face_tracking()
        print("✓ Face tracking demonstration complete!")
    
    def demo_object_scanning(self):
        """Demonstrate object scanning pattern"""
        print("\n📦 Object Scanning Demonstration")
        print("-" * 40)
        print("Performing systematic object scan...")
        
        if not self.robot.scan_with_gimbal(num_positions=8):
            print("❌ Failed to start scanning")
            return
        
        # Wait for scan to complete
        time.sleep(10)
        
        print("✓ Object scanning complete!")
    
    def demo_environmental_exploration(self):
        """Demonstrate environmental exploration"""
        print("\n🌍 Environmental Exploration Demonstration")
        print("-" * 40)
        print("Exploring the environment...")
        
        if not self.robot.explore_with_gimbal():
            print("❌ Failed to start exploration")
            return
        
        # Let it explore for a while
        time.sleep(15)
        
        print("✓ Environmental exploration complete!")
    
    def demo_curiosity_mode(self, duration: float = 20.0):
        """
        Demonstrate curiosity mode
        
        Args:
            duration: How long to run curiosity mode
        """
        print("\n🤔 Curiosity Mode Demonstration")
        print("-" * 40)
        print("Enabling curiosity mode - watch random explorations...")
        
        if not self.robot.enable_curiosity_mode():
            print("❌ Failed to enable curiosity mode")
            return
        
        # Let curiosity run
        time.sleep(duration)
        
        print("✓ Curiosity mode demonstration complete!")
    
    def demo_search_pattern(self):
        """Demonstrate search pattern for lost target"""
        print("\n🔍 Search Pattern Demonstration")
        print("-" * 40)
        print("Searching for lost target...")
        
        if not self.robot.search_for_person():
            print("❌ Failed to start search")
            return
        
        # Let search run for a bit
        time.sleep(10)
        
        print("✓ Search pattern demonstration complete!")
    
    def demo_expressive_gestures(self):
        """Demonstrate expressive gestures"""
        print("\n😊 Expressive Gestures Demonstration")
        print("-" * 40)
        
        if not self.robot.start_gimbal_control():
            print("❌ Failed to start gimbal control")
            return
        
        # Get gimbal module for direct access to behaviors
        gimbal = None
        for module in self.robot.capabilities.values():
            if hasattr(module.instance, 'system'):
                gimbal = module.instance.system
                break
        
        if not gimbal:
            print("❌ Gimbal system not available")
            return
        
        gestures = [
            ("Alert", gimbal.perform_alert),
            ("Greeting", gimbal.perform_greeting),
            ("Looking around", lambda: gimbal.behaviors.scan_environment())
        ]
        
        for name, gesture_func in gestures:
            print(f"Performing: {name}")
            gesture_func()
            time.sleep(3)
        
        print("✓ Expressive gestures complete!")
    
    def demo_3d_scanning_with_movement(self):
        """Demonstrate 3D scanning with gimbal movements"""
        print("\n🎯 3D Scanning with Movement Demonstration")
        print("-" * 40)
        print("Combining 3D scanning with gimbal movements...")
        
        # Start 3D scanner
        if self.robot.start_module(self.robot.VisionModule.SCANNER_3D):
            print("3D scanner started")
            
            # Use gimbal to scan around object
            if self.robot.scan_with_gimbal(num_positions=6):
                print("Capturing from multiple angles...")
                time.sleep(15)
                
                # Complete the scan
                result = self.robot.scan_object()
                if result:
                    print(f"✓ Captured {result.total_points} points")
                else:
                    print("❌ Scan failed")
            else:
                print("❌ Failed to start gimbal scanning")
            
            self.robot.stop_module(self.robot.VisionModule.SCANNER_3D)
        else:
            print("❌ Failed to start 3D scanner")
    
    def demo_person_following(self, target_name: str = None, duration: float = 30.0):
        """
        Demonstrate following a specific person
        
        Args:
            target_name: Name of person to follow
            duration: How long to follow
        """
        print("\n🚶 Person Following Demonstration")
        print("-" * 40)
        
        if target_name:
            print(f"Looking for {target_name} to follow...")
        else:
            print("Following the closest person...")
        
        if not self.robot.enable_face_tracking(target_name):
            print("❌ Failed to enable tracking")
            return
        
        start_time = time.time()
        tracking_stats = {
            'frames_tracked': 0,
            'frames_lost': 0,
            'total_frames': 0
        }
        
        while time.time() - start_time < duration:
            results = self.robot.identify_person()
            tracking_stats['total_frames'] += 1
            
            if results:
                found_target = False
                for result in results:
                    if target_name is None or result.name == target_name:
                        found_target = True
                        tracking_stats['frames_tracked'] += 1
                        
                        # Center on the face
                        self.robot.center_on_face(result.bbox)
                        break
                
                if not found_target:
                    tracking_stats['frames_lost'] += 1
            else:
                tracking_stats['frames_lost'] += 1
            
            time.sleep(0.05)
        
        self.robot.disable_face_tracking()
        
        # Print statistics
        print("\nTracking Statistics:")
        print(f"  Total frames: {tracking_stats['total_frames']}")
        print(f"  Frames tracked: {tracking_stats['frames_tracked']}")
        print(f"  Frames lost: {tracking_stats['frames_lost']}")
        print(f"  Tracking rate: {tracking_stats['frames_tracked']/tracking_stats['total_frames']*100:.1f}%")
        
        print("✓ Person following demonstration complete!")
    
    # ========================================================================
    # Callbacks
    # ========================================================================
    
    def _on_person_identified(self, result):
        """Callback when person is identified"""
        if result.name != "Unknown":
            if result.name != self.current_person:
                print(f"👋 Hello, {result.name}!")
                self.current_person = result.name
    
    def _on_target_lost(self, target_id):
        """Callback when tracking target is lost"""
        print(f"👁️ Lost track of {target_id}")
    
    def _on_behavior_change(self, behavior):
        """Callback when behavior changes"""
        print(f"🎭 Behavior changed to: {behavior}")
    
    def _on_gimbal_movement(self, position):
        """Callback for gimbal movements"""
        if isinstance(position, tuple):
            print(f"📐 Gimbal position: pan={position[0]:.1f}°, tilt={position[1]:.1f}°")
    
    # ========================================================================
    # Main Demo Runner
    # ========================================================================
    
    def run_full_demo(self):
        """Run complete demonstration of all features"""
        print("\n" + "=" * 60)
        print("Starting Full Robot Eyes Movement Demonstration")
        print("=" * 60)
        
        try:
            # Start robot vision
            with self.robot:
                # Run through all demonstrations
                demos = [
                    ("Greeting", self.demo_greeting),
                    ("Face Tracking", lambda: self.demo_face_tracking(20)),
                    ("Object Scanning", self.demo_object_scanning),
                    ("Environmental Exploration", self.demo_environmental_exploration),
                    ("Search Pattern", self.demo_search_pattern),
                    ("Curiosity Mode", lambda: self.demo_curiosity_mode(15)),
                    ("Expressive Gestures", self.demo_expressive_gestures),
                    ("Person Following", lambda: self.demo_person_following(duration=20))
                ]
                
                for i, (name, demo_func) in enumerate(demos, 1):
                    print(f"\n[{i}/{len(demos)}] {name}")
                    print("=" * 60)
                    
                    try:
                        demo_func()
                    except Exception as e:
                        print(f"❌ Error in {name}: {e}")
                    
                    if i < len(demos):
                        print("\nMoving to next demo in 3 seconds...")
                        time.sleep(3)
                
                # Return to home
                print("\n🏠 Returning to home position...")
                self.robot.gimbal_home()
                
        except KeyboardInterrupt:
            print("\n\n⚠️ Demo interrupted by user")
        except Exception as e:
            print(f"\n❌ Demo error: {e}")
        finally:
            print("\n" + "=" * 60)
            print("Robot Eyes Movement Demonstration Complete!")
            print("=" * 60)
    
    def run_interactive_demo(self):
        """Run interactive demonstration with menu"""
        print("\n" + "=" * 60)
        print("Interactive Robot Eyes Movement Demo")
        print("=" * 60)
        
        with self.robot:
            while True:
                print("\nSelect a demonstration:")
                print("1. Greeting")
                print("2. Face Tracking")
                print("3. Object Scanning")
                print("4. Environmental Exploration")
                print("5. Search Pattern")
                print("6. Curiosity Mode")
                print("7. Expressive Gestures")
                print("8. Person Following")
                print("9. 3D Scanning with Movement")
                print("0. Exit")
                
                try:
                    choice = input("\nEnter choice (0-9): ").strip()
                    
                    if choice == '0':
                        break
                    elif choice == '1':
                        self.demo_greeting()
                    elif choice == '2':
                        duration = float(input("Duration (seconds, default 30): ") or "30")
                        self.demo_face_tracking(duration)
                    elif choice == '3':
                        self.demo_object_scanning()
                    elif choice == '4':
                        self.demo_environmental_exploration()
                    elif choice == '5':
                        self.demo_search_pattern()
                    elif choice == '6':
                        duration = float(input("Duration (seconds, default 20): ") or "20")
                        self.demo_curiosity_mode(duration)
                    elif choice == '7':
                        self.demo_expressive_gestures()
                    elif choice == '8':
                        name = input("Person name to follow (or press Enter for any): ").strip() or None
                        duration = float(input("Duration (seconds, default 30): ") or "30")
                        self.demo_person_following(name, duration)
                    elif choice == '9':
                        self.demo_3d_scanning_with_movement()
                    else:
                        print("Invalid choice!")
                    
                except KeyboardInterrupt:
                    print("\nReturning to menu...")
                except Exception as e:
                    print(f"Error: {e}")
            
            # Return home before exit
            print("\nReturning to home position...")
            self.robot.gimbal_home()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Robot Eyes Movement Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full demo with default config
  python robot_eyes_movement.py
  
  # Run with custom config file
  python robot_eyes_movement.py --config ../config/gimbal_config.json
  
  # Run interactive mode
  python robot_eyes_movement.py --interactive
  
  # Run specific demo
  python robot_eyes_movement.py --demo face_tracking --duration 60
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        help='Path to gimbal configuration file',
        default='../config/gimbal_config.json'
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive mode'
    )
    
    parser.add_argument(
        '--demo', '-d',
        choices=['greeting', 'face_tracking', 'object_scanning', 
                'exploration', 'search', 'curiosity', 'gestures', 
                'following', '3d_scanning', 'full'],
        help='Run specific demonstration',
        default='full'
    )
    
    parser.add_argument(
        '--duration', '-t',
        type=float,
        default=30.0,
        help='Duration for timed demonstrations (seconds)'
    )
    
    parser.add_argument(
        '--target', 
        help='Name of person to track/follow'
    )
    
    args = parser.parse_args()
    
    # Create demo instance
    demo = RobotEyesDemo(args.config)
    
    try:
        if args.interactive:
            demo.run_interactive_demo()
        elif args.demo == 'full':
            demo.run_full_demo()
        else:
            # Run specific demo
            with demo.robot:
                if args.demo == 'greeting':
                    demo.demo_greeting()
                elif args.demo == 'face_tracking':
                    demo.demo_face_tracking(args.duration)
                elif args.demo == 'object_scanning':
                    demo.demo_object_scanning()
                elif args.demo == 'exploration':
                    demo.demo_environmental_exploration()
                elif args.demo == 'search':
                    demo.demo_search_pattern()
                elif args.demo == 'curiosity':
                    demo.demo_curiosity_mode(args.duration)
                elif args.demo == 'gestures':
                    demo.demo_expressive_gestures()
                elif args.demo == 'following':
                    demo.demo_person_following(args.target, args.duration)
                elif args.demo == '3d_scanning':
                    demo.demo_3d_scanning_with_movement()
                
                # Return home
                demo.robot.gimbal_home()
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()