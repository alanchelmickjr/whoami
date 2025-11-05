"""
Face Recognition Core Module
Handles face detection, encoding, and recognition using Oak D Series 3
"""

import cv2
import depthai as dai
import numpy as np
import face_recognition
from typing import List, Tuple, Optional, Dict
import pickle
import os


class FaceRecognizer:
    """Core face recognition class using Oak D camera and face_recognition library"""
    
    def __init__(self, database_path: str = "face_database.pkl"):
        """
        Initialize the face recognizer
        
        Args:
            database_path: Path to save/load face encodings database
        """
        self.database_path = database_path
        self.known_face_encodings = []
        self.known_face_names = []
        self.pipeline = None
        self.device = None
        self._unknown_counter = 0  # Counter for automatic unknown face numbering
        
        # Load existing database if available
        self.load_database()
        self._initialize_unknown_counter()
    
    def create_pipeline(self) -> dai.Pipeline:
        """
        Create DepthAI pipeline for Oak D camera (DepthAI V3 API)
        
        Returns:
            Configured DepthAI pipeline with output queue
        """
        pipeline = dai.Pipeline()
        
        # Use the new Camera node with V3 API - don't forget .build()
        cam = pipeline.create(dai.node.Camera).build()
        
        # Request output with specific resolution and format
        # This replaces the old setPreviewSize and preview output
        self.camera_output = cam.requestOutput((640, 480), type=dai.ImgFrame.Type.RGB888p)
        
        # Create output queue directly from the camera output (no more XLink nodes needed)
        self.output_queue = self.camera_output.createOutputQueue()
        
        return pipeline
    
    def start_camera(self) -> bool:
        """
        Start the Oak D camera with V3 API
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.pipeline = self.create_pipeline()
            # V3 API: Use pipeline.start() instead of dai.Device(pipeline)
            self.pipeline.start()
            return True
        except Exception as e:
            print(f"Error starting camera: {e}")
            return False
    
    def stop_camera(self):
        """Stop the Oak D camera"""
        if self.pipeline:
            # V3 API: Stop the pipeline instead of closing device
            self.pipeline.stop()
            self.pipeline = None
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get current frame from camera using V3 API
        
        Returns:
            Frame as numpy array or None if unavailable
        """
        if not self.pipeline or not self.pipeline.isRunning():
            return None
        
        try:
            # V3 API: Use the output queue created during pipeline setup
            if self.output_queue.has():
                in_rgb = self.output_queue.get()
                frame = in_rgb.getCvFrame()
                return frame
            return None
        except Exception as e:
            print(f"Error getting frame: {e}")
            return None
    
    def detect_faces(self, frame: np.ndarray) -> Tuple[List, List]:
        """
        Detect faces in the frame
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (face_locations, face_encodings)
        """
        # Convert BGR to RGB if needed
        rgb_frame = frame
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            # Check if it's BGR (OpenCV format)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find face locations and encodings
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        return face_locations, face_encodings
    
    def recognize_faces(self, face_encodings: List) -> List[Tuple[str, float]]:
        """
        Recognize faces from encodings
        
        Args:
            face_encodings: List of face encodings to recognize
            
        Returns:
            List of tuples (name, confidence) for each face
        """
        results = []
        
        for face_encoding in face_encodings:
            name = "Unknown"
            confidence = 0.0
            
            if len(self.known_face_encodings) > 0:
                # Compare with known faces
                matches = face_recognition.compare_faces(
                    self.known_face_encodings, face_encoding, tolerance=0.6
                )
                face_distances = face_recognition.face_distance(
                    self.known_face_encodings, face_encoding
                )
                
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = 1.0 - face_distances[best_match_index]
            
            results.append((name, confidence))
        
        return results
    
    def add_face(self, name: Optional[str] = None, frame: Optional[np.ndarray] = None,
                 face_index: int = 0) -> bool:
        """
        Add a new face to the database
        
        Args:
            name: Name of the person (if None, auto-generates unknown_N)
            frame: Frame containing the face
            face_index: Index of face to add when multiple faces detected (default: 0)
            
        Returns:
            True if successful, False otherwise
        """
        # Auto-generate name if not provided
        if name is None or name.strip() == "":
            name = self._get_next_unknown_name()
            print(f"Auto-generating name: {name}")
        
        # Detect faces in the frame
        face_locations, face_encodings = self.detect_faces(frame)
        
        if len(face_encodings) == 0:
            print("No face detected in the frame")
            return False
        
        if len(face_encodings) > 1:
            print(f"Multiple faces detected ({len(face_encodings)}). Using face at index {face_index}.")
            if face_index >= len(face_encodings):
                print(f"Face index {face_index} out of range (detected {len(face_encodings)} faces)")
                return False
        
        # Add the face encoding and name
        self.known_face_encodings.append(face_encodings[face_index])
        self.known_face_names.append(name)
        
        # Save database
        self.save_database()
        
        return True
    
    def add_face_at_location(self, name: Optional[str] = None,
                             frame: np.ndarray = None,
                             face_location: Tuple[int, int, int, int] = None) -> bool:
        """
        Add a specific face at a given location to the database
        
        Args:
            name: Name of the person (if None, auto-generates unknown_N)
            frame: Frame containing the face
            face_location: Location of face to add (top, right, bottom, left)
            
        Returns:
            True if successful, False otherwise
        """
        # Auto-generate name if not provided
        if name is None or name.strip() == "":
            name = self._get_next_unknown_name()
            print(f"Auto-generating name: {name}")
        
        if frame is None or face_location is None:
            print("Frame and face location are required")
            return False
        
        # Detect all faces to find the matching one
        face_locations, face_encodings = self.detect_faces(frame)
        
        if len(face_encodings) == 0:
            print("No faces detected in frame")
            return False
        
        # Find the face closest to the specified location
        best_match_idx = None
        min_distance = float('inf')
        
        for idx, location in enumerate(face_locations):
            # Calculate distance between centers of bounding boxes
            det_top, det_right, det_bottom, det_left = location
            spec_top, spec_right, spec_bottom, spec_left = face_location
            
            det_center_x = (det_left + det_right) / 2
            det_center_y = (det_top + det_bottom) / 2
            spec_center_x = (spec_left + spec_right) / 2
            spec_center_y = (spec_top + spec_bottom) / 2
            
            distance = ((det_center_x - spec_center_x) ** 2 +
                       (det_center_y - spec_center_y) ** 2) ** 0.5
            
            if distance < min_distance:
                min_distance = distance
                best_match_idx = idx
        
        if best_match_idx is None:
            print("Could not match face location")
            return False
        
        # Use a reasonable threshold for matching (e.g., 50 pixels)
        if min_distance > 50:
            print(f"Face location match distance ({min_distance:.1f}) may be too far")
        
        # Add the matched face
        self.known_face_encodings.append(face_encodings[best_match_idx])
        self.known_face_names.append(name)
        
        # Save database
        self.save_database()
        print(f"Added face '{name}' at location {face_location}")
        
        return True
    
    def _get_next_unknown_name(self) -> str:
        """Generate the next available unknown_N name"""
        self._unknown_counter += 1
        # Check if this number is already used
        while f"unknown_{self._unknown_counter}" in self.known_face_names:
            self._unknown_counter += 1
        return f"unknown_{self._unknown_counter}"
    
    def _initialize_unknown_counter(self) -> None:
        """Initialize the unknown counter based on existing unknown faces"""
        # Find the highest unknown number in the database
        max_num = 0
        for name in self.known_face_names:
            if name.startswith("unknown_"):
                try:
                    num = int(name.split("_")[1])
                    max_num = max(max_num, num)
                except (IndexError, ValueError):
                    pass
        self._unknown_counter = max_num
    
    def remove_face(self, name: str) -> bool:
        """
        Remove a face from the database
        
        Args:
            name: Name of the person to remove
            
        Returns:
            True if successful, False otherwise
        """
        if name not in self.known_face_names:
            return False
        
        # Find all indices with this name
        indices = [i for i, n in enumerate(self.known_face_names) if n == name]
        
        # Remove in reverse order to maintain indices
        for index in sorted(indices, reverse=True):
            del self.known_face_encodings[index]
            del self.known_face_names[index]
        
        # Save database
        self.save_database()
        
        return True
    
    def get_all_names(self) -> List[str]:
        """
        Get list of all known names
        
        Returns:
            List of unique names in the database
        """
        return list(set(self.known_face_names))
    
    def save_database(self):
        """Save face encodings database to file"""
        data = {
            'encodings': self.known_face_encodings,
            'names': self.known_face_names
        }
        
        with open(self.database_path, 'wb') as f:
            pickle.dump(data, f)
    
    def load_database(self):
        """Load face encodings database from file"""
        if os.path.exists(self.database_path):
            try:
                with open(self.database_path, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data.get('encodings', [])
                    self.known_face_names = data.get('names', [])
                print(f"Loaded {len(self.known_face_names)} faces from database")
            except Exception as e:
                print(f"Error loading database: {e}")
                self.known_face_encodings = []
                self.known_face_names = []
    
    def clear_database(self):
        """Clear all faces from the database"""
        self.known_face_encodings = []
        self.known_face_names = []
        self.save_database()
