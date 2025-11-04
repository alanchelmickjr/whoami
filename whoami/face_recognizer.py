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
        
        # Load existing database if available
        self.load_database()
    
    def create_pipeline(self) -> dai.Pipeline:
        """
        Create DepthAI pipeline for Oak D camera
        
        Returns:
            Configured DepthAI pipeline
        """
        pipeline = dai.Pipeline()
        
        # Create color camera node
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setPreviewSize(640, 480)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        
        # Create output node
        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        
        # Link camera to output
        cam_rgb.preview.link(xout_rgb.input)
        
        return pipeline
    
    def start_camera(self) -> bool:
        """
        Start the Oak D camera
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.pipeline = self.create_pipeline()
            self.device = dai.Device(self.pipeline)
            return True
        except Exception as e:
            print(f"Error starting camera: {e}")
            return False
    
    def stop_camera(self):
        """Stop the Oak D camera"""
        if self.device:
            self.device.close()
            self.device = None
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get current frame from camera
        
        Returns:
            Frame as numpy array or None if unavailable
        """
        if not self.device:
            return None
        
        try:
            queue = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            in_rgb = queue.get()
            frame = in_rgb.getCvFrame()
            return frame
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
    
    def add_face(self, name: str, frame: np.ndarray) -> bool:
        """
        Add a new face to the database
        
        Args:
            name: Name of the person
            frame: Frame containing the face
            
        Returns:
            True if successful, False otherwise
        """
        # Detect faces in the frame
        face_locations, face_encodings = self.detect_faces(frame)
        
        if len(face_encodings) == 0:
            print("No face detected in the frame")
            return False
        
        if len(face_encodings) > 1:
            print("Multiple faces detected. Please ensure only one face is in the frame")
            return False
        
        # Add the face encoding and name
        self.known_face_encodings.append(face_encodings[0])
        self.known_face_names.append(name)
        
        # Save database
        self.save_database()
        
        return True
    
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
