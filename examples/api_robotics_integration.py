#!/usr/bin/env python3
"""
Face Recognition API - Robotics Integration Example

This script demonstrates how to integrate the Face Recognition API into
robotics applications with:
- Real-time person tracking
- Event-driven robot behaviors
- State management for robot interactions
- Person-specific actions and responses
- Safety zones and access control
- Data logging for robot-human interactions

Requirements:
- OAK-D camera or webcam
- Python 3.7+
- Required packages: numpy, opencv-python, face-recognition, depthai
"""

import sys
import os
import time
import threading
import queue
import json
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import List, Dict, Any, Optional, Callable
from enum import Enum

# Add parent directory to path to import whoami module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whoami.face_recognition_api import (
    FaceRecognitionAPI,
    RecognitionConfig,
    CameraType,
    RecognitionModel,
    RecognitionResult,
    create_face_recognition_api
)


class RobotState(Enum):
    """Robot operational states"""
    IDLE = "idle"
    GREETING = "greeting"
    TRACKING = "tracking"
    INTERACTING = "interacting"
    ALERT = "alert"
    EMERGENCY = "emergency"


class PersonProfile:
    """Profile for known persons with robot interaction preferences"""
    
    def __init__(self, name: str, role: str = "guest", preferences: Dict = None):
        self.name = name
        self.role = role  # guest, operator, admin, restricted
        self.preferences = preferences or {}
        self.last_seen = None
        self.interaction_count = 0
        self.total_time_visible = 0
        self.first_seen = None
    
    def update_seen(self):
        """Update last seen timestamp"""
        now = datetime.now()
        if self.first_seen is None:
            self.first_seen = now
        self.last_seen = now
        self.interaction_count += 1


class RobotVisionSystem:
    """
    Advanced robot vision system using the Face Recognition API
    Provides real-time person tracking and robot behavior management
    """
    
    def __init__(self, config: Optional[RecognitionConfig] = None):
        # Initialize API
        self.config = config or RecognitionConfig(
            camera_type=CameraType.OAK_D,
            tolerance=0.5,
            process_every_n_frames=2,
            face_detection_scale=0.5
        )
        self.api = FaceRecognitionAPI(self.config)
        
        # State management
        self.state = RobotState.IDLE
        self.running = False
        self.lock = threading.RLock()
        
        # Person tracking
        self.current_people: Dict[str, PersonProfile] = {}
        self.person_profiles: Dict[str, PersonProfile] = {}
        self.tracking_history = deque(maxlen=100)
        
        # Robot behavior callbacks
        self.behavior_callbacks: Dict[str, List[Callable]] = {
            'on_person_detected': [],
            'on_person_lost': [],
            'on_greeting_needed': [],
            'on_security_alert': [],
            'on_state_change': []
        }
        
        # Interaction zones (in pixels from camera center)
        self.zones = {
            'close': 150,    # Very close to robot
            'near': 300,     # Interaction distance
            'far': 500       # Detection distance
        }
        
        # Safety and security
        self.restricted_persons = set()
        self.authorized_operators = set()
        
        # Statistics
        self.stats = {
            'total_people_seen': 0,
            'total_interactions': 0,
            'total_greetings': 0,
            'security_alerts': 0
        }
        
        # Register API callbacks
        self._register_api_callbacks()
        
        # Threading
        self.vision_thread = None
        self.behavior_thread = None
        self.command_queue = queue.Queue()
    
    def _register_api_callbacks(self):
        """Register callbacks with the Face Recognition API"""
        self.api.register_callback('on_face_recognized', self._on_face_recognized)
        self.api.register_callback('on_error', self._on_error)
    
    def _on_face_recognized(self, results: List[RecognitionResult]):
        """Handle face recognition events from API"""
        with self.lock:
            recognized_names = set()
            
            for result in results:
                if result.name != "Unknown" and result.confidence > 0.7:
                    recognized_names.add(result.name)
                    
                    # Update or create person profile
                    if result.name not in self.person_profiles:
                        self.person_profiles[result.name] = PersonProfile(result.name)
                    
                    profile = self.person_profiles[result.name]
                    profile.update_seen()
                    
                    # Calculate distance from center (simplified)
                    top, right, bottom, left = result.location
                    center_x = (left + right) / 2
                    center_y = (top + bottom) / 2
                    frame_center_x = self.config.camera_resolution[0] / 2
                    frame_center_y = self.config.camera_resolution[1] / 2
                    distance = ((center_x - frame_center_x)**2 + (center_y - frame_center_y)**2)**0.5
                    
                    # Update current people
                    if result.name not in self.current_people:
                        self.current_people[result.name] = profile
                        self._trigger_callback('on_person_detected', profile, distance)
                        
                        # Check if greeting is needed
                        if self._should_greet(profile):
                            self._trigger_callback('on_greeting_needed', profile)
                    
                    # Security check
                    if result.name in self.restricted_persons:
                        self._trigger_callback('on_security_alert', profile, "Restricted person detected")
            
            # Check for people who left
            lost_people = set(self.current_people.keys()) - recognized_names
            for name in lost_people:
                profile = self.current_people.pop(name)
                self._trigger_callback('on_person_lost', profile)
            
            # Update tracking history
            self.tracking_history.append({
                'timestamp': datetime.now(),
                'people': list(recognized_names)
            })
    
    def _on_error(self, error):
        """Handle errors from API"""
        print(f"[ROBOT-ERROR] Vision system error: {error}")
        if "camera" in str(error).lower():
            self.change_state(RobotState.ALERT)
    
    def _should_greet(self, profile: PersonProfile) -> bool:
        """Determine if robot should greet a person"""
        if profile.last_seen is None:
            return True
        
        # Greet if not seen for more than 1 hour
        time_since_seen = datetime.now() - profile.last_seen
        return time_since_seen > timedelta(hours=1)
    
    def _trigger_callback(self, event: str, *args, **kwargs):
        """Trigger behavior callbacks"""
        for callback in self.behavior_callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                print(f"[ROBOT-ERROR] Callback error for {event}: {e}")
    
    def register_behavior(self, event: str, callback: Callable):
        """Register a robot behavior callback"""
        if event in self.behavior_callbacks:
            self.behavior_callbacks[event].append(callback)
    
    def change_state(self, new_state: RobotState):
        """Change robot state"""
        with self.lock:
            if self.state != new_state:
                old_state = self.state
                self.state = new_state
                print(f"[ROBOT-STATE] {old_state.value} -> {new_state.value}")
                self._trigger_callback('on_state_change', old_state, new_state)
    
    def start(self) -> bool:
        """Start the robot vision system"""
        print("[ROBOT] Starting vision system...")
        
        # Start camera
        if not self.api.start_camera():
            print("[ROBOT-ERROR] Failed to start camera")
            return False
        
        self.running = True
        
        # Start vision thread
        self.vision_thread = threading.Thread(target=self._vision_loop, daemon=True)
        self.vision_thread.start()
        
        # Start behavior thread
        self.behavior_thread = threading.Thread(target=self._behavior_loop, daemon=True)
        self.behavior_thread.start()
        
        self.change_state(RobotState.IDLE)
        print("[ROBOT] Vision system started successfully")
        return True
    
    def stop(self):
        """Stop the robot vision system"""
        print("[ROBOT] Stopping vision system...")
        self.running = False
        
        if self.vision_thread:
            self.vision_thread.join(timeout=2.0)
        if self.behavior_thread:
            self.behavior_thread.join(timeout=2.0)
        
        self.api.stop_camera()
        self.api.save_database()
        
        print("[ROBOT] Vision system stopped")
    
    def _vision_loop(self):
        """Main vision processing loop"""
        while self.running:
            try:
                frame = self.api.get_frame()
                if frame is not None:
                    # Process frame for recognition
                    self.api.process_frame(frame)
                
                time.sleep(0.05)  # ~20 FPS
                
            except Exception as e:
                print(f"[ROBOT-ERROR] Vision loop error: {e}")
    
    def _behavior_loop(self):
        """Robot behavior management loop"""
        while self.running:
            try:
                # Process commands
                try:
                    command = self.command_queue.get(timeout=0.1)
                    self._execute_command(command)
                except queue.Empty:
                    pass
                
                # Update robot state based on current situation
                with self.lock:
                    if self.current_people:
                        if self.state == RobotState.IDLE:
                            self.change_state(RobotState.TRACKING)
                    else:
                        if self.state == RobotState.TRACKING:
                            self.change_state(RobotState.IDLE)
                
            except Exception as e:
                print(f"[ROBOT-ERROR] Behavior loop error: {e}")
    
    def _execute_command(self, command: Dict[str, Any]):
        """Execute robot command"""
        cmd_type = command.get('type')
        
        if cmd_type == 'greet':
            person = command.get('person')
            self._greet_person(person)
        elif cmd_type == 'follow':
            person = command.get('person')
            self._follow_person(person)
        # Add more commands as needed
    
    def _greet_person(self, person: PersonProfile):
        """Greet a person"""
        self.change_state(RobotState.GREETING)
        print(f"[ROBOT-ACTION] Greeting {person.name}")
        # In real robot: move to person, play greeting sound, etc.
        time.sleep(2)  # Simulate greeting action
        self.stats['total_greetings'] += 1
        self.change_state(RobotState.IDLE)
    
    def _follow_person(self, person: PersonProfile):
        """Follow a person"""
        print(f"[ROBOT-ACTION] Following {person.name}")
        # In real robot: activate tracking motors, maintain distance, etc.
    
    # Public methods for robot control
    
    def get_current_people(self) -> List[str]:
        """Get list of currently visible people"""
        with self.lock:
            return list(self.current_people.keys())
    
    def is_person_visible(self, name: str) -> bool:
        """Check if a specific person is currently visible"""
        with self.lock:
            return name in self.current_people
    
    def get_person_profile(self, name: str) -> Optional[PersonProfile]:
        """Get profile for a person"""
        with self.lock:
            return self.person_profiles.get(name)
    
    def add_authorized_operator(self, name: str):
        """Add an authorized operator"""
        self.authorized_operators.add(name)
        print(f"[ROBOT-SECURITY] Added authorized operator: {name}")
    
    def add_restricted_person(self, name: str):
        """Add a restricted person"""
        self.restricted_persons.add(name)
        print(f"[ROBOT-SECURITY] Added restricted person: {name}")
    
    def send_command(self, command: Dict[str, Any]):
        """Send command to robot"""
        self.command_queue.put(command)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get robot vision statistics"""
        with self.lock:
            api_stats = self.api.get_statistics()
            return {
                **self.stats,
                **api_stats,
                'current_state': self.state.value,
                'people_visible': len(self.current_people),
                'total_profiles': len(self.person_profiles)
            }


class InteractiveRobot:
    """
    Example interactive robot with personality and behaviors
    """
    
    def __init__(self, vision_system: RobotVisionSystem):
        self.vision = vision_system
        self.personality = {
            'friendly': True,
            'formal': False,
            'verbose': True
        }
        
        # Register robot behaviors
        self.vision.register_behavior('on_person_detected', self.on_person_detected)
        self.vision.register_behavior('on_person_lost', self.on_person_lost)
        self.vision.register_behavior('on_greeting_needed', self.on_greeting_needed)
        self.vision.register_behavior('on_security_alert', self.on_security_alert)
        self.vision.register_behavior('on_state_change', self.on_state_change)
        
        # Conversation memory
        self.conversations = defaultdict(list)
    
    def on_person_detected(self, profile: PersonProfile, distance: float):
        """React to person detection"""
        zone = self._get_zone(distance)
        
        if self.personality['verbose']:
            print(f"[ROBOT] I see {profile.name} in the {zone} zone")
        
        if profile.role == "operator":
            print(f"[ROBOT] Ready for your commands, {profile.name}")
        elif profile.role == "admin":
            print(f"[ROBOT] Admin access granted. Welcome, {profile.name}")
        
        # Log interaction
        self.conversations[profile.name].append({
            'time': datetime.now(),
            'event': 'detected',
            'zone': zone
        })
    
    def on_person_lost(self, profile: PersonProfile):
        """React to person leaving"""
        if self.personality['friendly']:
            print(f"[ROBOT] Goodbye, {profile.name}! See you next time!")
        else:
            print(f"[ROBOT] {profile.name} has left the area")
        
        # Log interaction
        self.conversations[profile.name].append({
            'time': datetime.now(),
            'event': 'departed'
        })
    
    def on_greeting_needed(self, profile: PersonProfile):
        """Greet a person"""
        greeting = self._generate_greeting(profile)
        print(f"[ROBOT] {greeting}")
        
        # Send greeting command
        self.vision.send_command({
            'type': 'greet',
            'person': profile
        })
    
    def on_security_alert(self, profile: PersonProfile, reason: str):
        """Handle security alerts"""
        print(f"[ROBOT-ALERT] Security issue: {reason}")
        print(f"[ROBOT] Attention! {profile.name} - {reason}")
        
        # In real robot: sound alarm, notify security, etc.
        self.vision.change_state(RobotState.ALERT)
    
    def on_state_change(self, old_state: RobotState, new_state: RobotState):
        """React to state changes"""
        if new_state == RobotState.EMERGENCY:
            print("[ROBOT] EMERGENCY MODE ACTIVATED!")
        elif new_state == RobotState.IDLE and old_state == RobotState.TRACKING:
            print("[ROBOT] Area clear, returning to standby")
    
    def _get_zone(self, distance: float) -> str:
        """Determine which zone a person is in"""
        if distance < self.vision.zones['close']:
            return 'close'
        elif distance < self.vision.zones['near']:
            return 'near'
        elif distance < self.vision.zones['far']:
            return 'far'
        else:
            return 'distant'
    
    def _generate_greeting(self, profile: PersonProfile) -> str:
        """Generate personalized greeting"""
        if profile.interaction_count == 0:
            return f"Hello! I don't believe we've met. I'm your robot assistant. Nice to meet you, {profile.name}!"
        elif profile.interaction_count < 5:
            return f"Hello again, {profile.name}! Good to see you!"
        else:
            return f"Welcome back, {profile.name}! Always a pleasure!"


class SecurityRobot:
    """
    Security-focused robot implementation
    """
    
    def __init__(self, vision_system: RobotVisionSystem):
        self.vision = vision_system
        self.alert_active = False
        self.security_log = []
        
        # Security configuration
        self.max_unknown_tolerance = 3  # Max unknown faces before alert
        self.restricted_areas = {}
        
        # Register behaviors
        self.vision.register_behavior('on_person_detected', self.check_authorization)
        self.vision.register_behavior('on_security_alert', self.handle_security_alert)
    
    def check_authorization(self, profile: PersonProfile, distance: float):
        """Check if person is authorized"""
        if profile.name == "Unknown":
            self.handle_unknown_person()
        elif profile.name in self.vision.restricted_persons:
            self.handle_restricted_person(profile)
        elif profile.role == "admin" or profile.role == "operator":
            print(f"[SECURITY] Authorized: {profile.name} ({profile.role})")
        else:
            print(f"[SECURITY] Guest detected: {profile.name}")
    
    def handle_unknown_person(self):
        """Handle detection of unknown person"""
        print("[SECURITY] Unknown person detected - monitoring...")
        
        # Count unknown faces in recent history
        recent_unknown = sum(
            1 for entry in list(self.vision.tracking_history)[-10:]
            if "Unknown" in entry.get('people', [])
        )
        
        if recent_unknown >= self.max_unknown_tolerance:
            self.raise_security_alert("Multiple unknown persons detected")
    
    def handle_restricted_person(self, profile: PersonProfile):
        """Handle detection of restricted person"""
        self.raise_security_alert(f"Restricted person detected: {profile.name}")
    
    def handle_security_alert(self, profile: PersonProfile, reason: str):
        """Handle security alert"""
        self.alert_active = True
        
        # Log security event
        event = {
            'timestamp': datetime.now().isoformat(),
            'person': profile.name,
            'reason': reason,
            'action': 'alert_raised'
        }
        self.security_log.append(event)
        
        print(f"[SECURITY-ALERT] {reason}")
        print("[SECURITY] Initiating security protocol...")
        
        # In real robot: activate alarms, lock doors, notify security, etc.
    
    def raise_security_alert(self, reason: str):
        """Raise a security alert"""
        self.vision.change_state(RobotState.ALERT)
        print(f"[SECURITY-ALERT] {reason}")
    
    def clear_alert(self):
        """Clear security alert"""
        self.alert_active = False
        self.vision.change_state(RobotState.IDLE)
        print("[SECURITY] Alert cleared")


def demo_basic_robot():
    """
    Demonstrate basic robot vision integration
    """
    print("=" * 60)
    print("Basic Robot Vision Demo")
    print("=" * 60)
    
    # Create vision system
    vision = RobotVisionSystem()
    
    # Add some known people to database
    vision.api.load_database("robot_faces.pkl")
    
    # Start vision system
    if not vision.start():
        print("Failed to start vision system")
        return
    
    try:
        print("\nRobot vision active. Monitoring for people...")
        print("Press Ctrl+C to stop\n")
        
        last_people = []
        
        while True:
            current_people = vision.get_current_people()
            
            # Report changes
            if current_people != last_people:
                if current_people:
                    print(f"[ROBOT] People in view: {', '.join(current_people)}")
                else:
                    print("[ROBOT] No people currently visible")
                
                last_people = current_people
            
            # Display statistics periodically
            time.sleep(5)
            stats = vision.get_statistics()
            print(f"[STATS] State: {stats['current_state']}, "
                  f"Greetings: {stats['total_greetings']}, "
                  f"People visible: {stats['people_visible']}")
    
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    
    finally:
        vision.stop()
        
        # Display final statistics
        stats = vision.get_statistics()
        print("\nFinal Statistics:")
        print(f"  Total people seen: {stats['total_profiles']}")
        print(f"  Total greetings: {stats['total_greetings']}")
        print(f"  Security alerts: {stats['security_alerts']}")


def demo_interactive_robot():
    """
    Demonstrate interactive robot with personality
    """
    print("=" * 60)
    print("Interactive Robot Demo")
    print("=" * 60)
    
    # Create vision system
    vision = RobotVisionSystem()
    
    # Create interactive robot
    robot = InteractiveRobot(vision)
    
    # Configure robot personality
    robot.personality['friendly'] = True
    robot.personality['verbose'] = True
    
    # Start system
    if not vision.start():
        print("Failed to start vision system")
        return
    
    try:
        print("\nInteractive robot active...")
        print("The robot will greet and interact with people")
        print("Press Ctrl+C to stop\n")
        
        # Run for 60 seconds
        for i in range(60):
            time.sleep(1)
            
            # Periodically check for specific people
            if i % 10 == 0:
                if vision.is_person_visible("Admin"):
                    print("[ROBOT] Admin present - ready for configuration")
    
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    
    finally:
        vision.stop()
        
        # Display conversation logs
        print("\nConversation Summary:")
        for person, events in robot.conversations.items():
            print(f"  {person}: {len(events)} events")


def demo_security_robot():
    """
    Demonstrate security-focused robot
    """
    print("=" * 60)
    print("Security Robot Demo")
    print("=" * 60)
    
    # Create vision system
    vision = RobotVisionSystem()
    
    # Create security robot
    security = SecurityRobot(vision)
    
    # Configure security settings
    vision.add_authorized_operator("Operator1")
    vision.add_restricted_person("Restricted1")
    
    # Start system
    if not vision.start():
        print("Failed to start vision system")
        return
    
    try:
        print("\nSecurity robot active...")
        print("Monitoring for unauthorized access")
        print("Press Ctrl+C to stop\n")
        
        # Run security monitoring
        for i in range(30):
            time.sleep(1)
            
            # Check security status
            if security.alert_active:
                print("[SECURITY] ALERT ACTIVE - Monitoring situation...")
    
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    
    finally:
        vision.stop()
        
        # Display security log
        if security.security_log:
            print("\nSecurity Log:")
            for event in security.security_log:
                print(f"  {event['timestamp']}: {event['reason']}")


def main():
    """
    Main function to run robotics demos
    """
    print("\n" + "=" * 60)
    print("Face Recognition API - Robotics Integration")
    print("=" * 60)
    print("\nSelect a demo to run:")
    print("1. Basic Robot Vision")
    print("2. Interactive Robot")
    print("3. Security Robot")
    print("4. Run All Demos")
    print("0. Exit")
    
    try:
        choice = input("\nEnter your choice (0-4): ").strip()
        
        if choice == '1':
            demo_basic_robot()
        elif choice == '2':
            demo_interactive_robot()
        elif choice == '3':
            demo_security_robot()
        elif choice == '4':
            # Run all demos
            demo_basic_robot()
            print("\n" + "=" * 60 + "\n")
            demo_interactive_robot()
            print("\n" + "=" * 60 + "\n")
            demo_security_robot()
        elif choice == '0':
            print("Exiting...")
        else:
            print("Invalid choice. Please run the script again.")
    
    except Exception as e:
        print(f"\n✗ Error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Robotics integration demo completed")
    print("=" * 60)


if __name__ == "__main__":
    main()