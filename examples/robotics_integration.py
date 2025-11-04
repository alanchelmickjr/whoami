"""
Example: Robotics Integration
Example of integrating WhoAmI into a robotics framework
"""

from whoami.face_recognizer import FaceRecognizer
import threading
import time


class RobotVision:
    """Example robot vision system using WhoAmI"""
    
    def __init__(self):
        self.recognizer = FaceRecognizer()
        self.running = False
        self.current_people = []
        self.lock = threading.Lock()
    
    def start(self):
        """Start the vision system"""
        print("Starting robot vision system...")
        
        if not self.recognizer.start_camera():
            print("Error: Could not start camera")
            return False
        
        self.running = True
        self.vision_thread = threading.Thread(target=self._vision_loop, daemon=True)
        self.vision_thread.start()
        
        print("Vision system started!")
        return True
    
    def stop(self):
        """Stop the vision system"""
        print("Stopping vision system...")
        self.running = False
        if hasattr(self, 'vision_thread'):
            self.vision_thread.join(timeout=2.0)
        self.recognizer.stop_camera()
        print("Vision system stopped")
    
    def _vision_loop(self):
        """Main vision processing loop"""
        while self.running:
            frame = self.recognizer.get_frame()
            
            if frame is not None:
                # Detect and recognize faces
                _, face_encodings = self.recognizer.detect_faces(frame)
                
                if face_encodings:
                    results = self.recognizer.recognize_faces(face_encodings)
                    
                    # Update current people list
                    with self.lock:
                        self.current_people = [name for name, conf in results if name != "Unknown"]
                
                else:
                    with self.lock:
                        self.current_people = []
            
            time.sleep(0.2)  # Process at ~5 FPS
    
    def get_current_people(self):
        """Get list of currently visible people"""
        with self.lock:
            return self.current_people.copy()
    
    def is_person_visible(self, name):
        """Check if a specific person is currently visible"""
        with self.lock:
            return name in self.current_people


def main():
    """Example usage in a robot application"""
    
    # Initialize robot vision
    robot_vision = RobotVision()
    
    if not robot_vision.start():
        return
    
    try:
        print("\nRobot vision active. Monitoring for known faces...")
        print("Press Ctrl+C to stop\n")
        
        last_people = []
        
        while True:
            # Get currently visible people
            current_people = robot_vision.get_current_people()
            
            # Check for changes
            if current_people != last_people:
                if current_people:
                    print(f"People detected: {', '.join(current_people)}")
                    
                    # Example: Robot could greet people
                    for person in current_people:
                        if person not in last_people:
                            print(f"  -> Hello, {person}!")
                else:
                    if last_people:
                        print("No known people visible")
                
                last_people = current_people
            
            # Example: Check for specific person
            if robot_vision.is_person_visible("Alice"):
                # Robot could perform Alice-specific actions
                pass
            
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    
    finally:
        robot_vision.stop()


if __name__ == "__main__":
    main()
