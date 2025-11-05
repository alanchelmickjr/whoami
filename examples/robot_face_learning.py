#!/usr/bin/env python3
"""
Example: Robot Face Learning System
Shows how a robot would programmatically learn and remember faces
"""

import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from whoami.face_recognition_api import (
    create_face_recognition_api, 
    CameraType, 
    RecognitionResult
)

class RobotFaceLearningSystem:
    """
    Robot system for learning and remembering faces with associated data
    This would integrate with the robot's VLA/VLM for intelligent interactions
    """
    
    def __init__(self):
        """Initialize the robot's face learning system"""
        # Initialize face recognition API
        self.face_api = create_face_recognition_api(
            database_path="robot_memory.pkl",
            camera_type=CameraType.OAK_D,
            tolerance=0.6
        )
        
        # Robot's memory of people (would be stored in persistent database)
        self.person_memory: Dict[str, Dict[str, Any]] = {}
        
        # Interaction state
        self.current_interaction = None
        self.unknown_face_buffer = []
        
    def process_visual_frame(self, frame: np.ndarray) -> List[RecognitionResult]:
        """
        Process a frame from robot's vision system
        This would be called continuously during robot operation
        """
        # Detect and recognize faces
        results = self.face_api.process_frame(frame)
        
        # Process each detected face
        for result in results:
            if result.name == "Unknown" or result.name.startswith("unknown_"):
                # Unknown person detected - robot should interact
                self._handle_unknown_person(result)
            else:
                # Known person - retrieve their information
                self._handle_known_person(result)
        
        return results
    
    def _handle_unknown_person(self, result: RecognitionResult):
        """
        Handle detection of unknown person
        Robot would initiate interaction to learn about them
        """
        # In a real robot, this would trigger:
        # 1. Robot approaches person
        # 2. Robot asks for their name
        # 3. Robot asks permission to remember them
        print(f"[ROBOT] Unknown person detected at location {result.location}")
        
        # Simulate robot asking for name (would use speech recognition)
        # In real implementation, this would be async with speech/NLP
        self.current_interaction = {
            'face_location': result.location,
            'encoding': result.encoding,
            'timestamp': self._get_timestamp()
        }
        
        # Robot would say: "Hello! I don't think we've met. May I ask your name?"
        
    def _handle_known_person(self, result: RecognitionResult):
        """
        Handle detection of known person
        Robot retrieves their information and personalizes interaction
        """
        person_data = self.person_memory.get(result.name, {})
        
        # Robot would use this data to personalize interaction
        print(f"[ROBOT] Recognized {result.name} (confidence: {result.confidence:.2f})")
        
        if person_data:
            # Use remembered information
            preferences = person_data.get('preferences', {})
            last_interaction = person_data.get('last_interaction')
            
            # Robot could say things like:
            # "Hello [name]! Good to see you again."
            # "Would you like your usual coffee order?"
            # "How did that project you mentioned go?"
    
    def learn_new_person(self, 
                        name: str,
                        frame: np.ndarray,
                        face_location: Tuple[int, int, int, int],
                        additional_info: Optional[Dict[str, Any]] = None):
        """
        Learn a new person after robot interaction
        This would be called after robot gets person's name via speech recognition
        
        Args:
            name: Person's name obtained via speech recognition
            frame: Current camera frame
            face_location: Location of the person's face
            additional_info: Any additional information to remember
        """
        # Add face to recognition database using specific location
        success = self.face_api.add_face_at_location(
            name=name,
            frame=frame,
            face_location=face_location
        )
        
        if success:
            # Store additional information about the person
            self.person_memory[name] = {
                'first_met': self._get_timestamp(),
                'last_interaction': self._get_timestamp(),
                'interaction_count': 1,
                'preferences': additional_info or {},
                'notes': []
            }
            
            print(f"[ROBOT] Learned new person: {name}")
            # Robot would say: "Nice to meet you, [name]! I'll remember you."
            return True
        else:
            print(f"[ROBOT] Failed to learn face for {name}")
            return False
    
    def update_person_memory(self, 
                            name: str, 
                            new_info: Dict[str, Any]):
        """
        Update remembered information about a person
        Robot learns more about people over time
        """
        if name in self.person_memory:
            # Update interaction timestamp
            self.person_memory[name]['last_interaction'] = self._get_timestamp()
            self.person_memory[name]['interaction_count'] += 1
            
            # Merge new information
            if 'preferences' in new_info:
                self.person_memory[name]['preferences'].update(new_info['preferences'])
            
            if 'notes' in new_info:
                self.person_memory[name]['notes'].extend(new_info['notes'])
            
            print(f"[ROBOT] Updated memory for {name}")
    
    def handle_multiple_unknown_faces(self, 
                                     frame: np.ndarray,
                                     face_locations: List[Tuple[int, int, int, int]]):
        """
        Handle situation with multiple unknown people
        Robot would interact with them one by one
        """
        print(f"[ROBOT] Detected {len(face_locations)} unknown people")
        
        # Robot strategy for multiple unknown faces:
        # 1. Approach closest person first
        # 2. Or prioritize based on who is looking at robot (using face landmarks)
        # 3. Or interact with group collectively
        
        for idx, location in enumerate(face_locations):
            # Temporarily assign unknown_N naming
            temp_name = None  # Will auto-generate unknown_N
            
            # Add face with automatic numbering
            success = self.face_api.add_face_at_location(
                name=temp_name,
                frame=frame,
                face_location=location
            )
            
            if success:
                print(f"[ROBOT] Temporarily cataloged unknown person {idx+1}")
                # Robot would later update this when learning their actual name
    
    def batch_learn_from_group(self, 
                              frame: np.ndarray,
                              names_and_locations: List[Tuple[str, Tuple[int, int, int, int]]]):
        """
        Learn multiple people at once (e.g., during introductions)
        Robot could be introduced to a group of people
        """
        for name, location in names_and_locations:
            success = self.face_api.add_face_at_location(
                name=name,
                frame=frame,
                face_location=location
            )
            
            if success:
                self.person_memory[name] = {
                    'first_met': self._get_timestamp(),
                    'last_interaction': self._get_timestamp(),
                    'interaction_count': 1,
                    'met_in_group': True,
                    'preferences': {},
                    'notes': ['Met as part of group introduction']
                }
        
        print(f"[ROBOT] Learned {len(names_and_locations)} people from group")
    
    def get_person_history(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve all remembered information about a person
        Robot uses this to personalize interactions
        """
        return self.person_memory.get(name)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def demonstrate_robot_workflow(self):
        """
        Demonstrate typical robot workflow for face learning
        """
        print("\n" + "="*60)
        print("ROBOT FACE LEARNING WORKFLOW DEMONSTRATION")
        print("="*60)
        
        print("\n1. INITIAL ENCOUNTER - Unknown Person")
        print("-" * 40)
        print("[ROBOT VISION] Detects unknown face")
        print("[ROBOT SPEECH] 'Hello! I don't think we've met. May I ask your name?'")
        print("[HUMAN] 'I'm Alice'")
        print("[ROBOT] Adds face as 'Alice' to database")
        print("[ROBOT SPEECH] 'Nice to meet you, Alice! I'll remember you.'")
        
        print("\n2. LEARNING PREFERENCES")
        print("-" * 40)
        print("[ROBOT] 'Alice, what's your favorite beverage?'")
        print("[HUMAN] 'I like green tea'")
        print("[ROBOT] Updates Alice's preferences: {'favorite_beverage': 'green tea'}")
        
        print("\n3. SUBSEQUENT ENCOUNTER")
        print("-" * 40)
        print("[ROBOT VISION] Detects face, recognizes as 'Alice' (confidence: 0.89)")
        print("[ROBOT] Retrieves Alice's information from memory")
        print("[ROBOT SPEECH] 'Hello Alice! Would you like some green tea?'")
        
        print("\n4. GROUP SCENARIO")
        print("-" * 40)
        print("[ROBOT VISION] Detects 3 unknown faces")
        print("[ROBOT] Temporarily catalogs as unknown_1, unknown_2, unknown_3")
        print("[ROBOT SPEECH] 'Hello everyone! I'm the assistant robot. May I learn your names?'")
        print("[HUMANS] Introduce themselves")
        print("[ROBOT] Updates unknown_1 → 'Bob', unknown_2 → 'Carol', unknown_3 → 'Dave'")
        
        print("\n5. CONTINUOUS LEARNING")
        print("-" * 40)
        print("[ROBOT] Over time, learns:")
        print("  - Bob prefers coffee, arrives at 9 AM daily")
        print("  - Carol is allergic to peanuts")
        print("  - Dave works in engineering department")
        print("[ROBOT] Uses this information to provide personalized assistance")
        
        print("\n" + "="*60)
        print("END OF DEMONSTRATION")
        print("="*60)


def main():
    """Main demonstration"""
    # Create robot face learning system
    robot = RobotFaceLearningSystem()
    
    # Demonstrate the workflow
    robot.demonstrate_robot_workflow()
    
    print("\n" + "="*60)
    print("KEY FEATURES FOR ROBOTIC USE:")
    print("="*60)
    print("✓ Automatic numbering for unknown faces (unknown_1, unknown_2, etc.)")
    print("✓ Programmatic face selection by location or index")
    print("✓ Integration with robot's memory system")
    print("✓ Continuous learning and preference tracking")
    print("✓ Group handling capabilities")
    print("✓ Personalized interactions based on remembered data")
    print("\nThe system is ready for integration with VLA/VLM models!")


if __name__ == "__main__":
    main()