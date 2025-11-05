"""
Face Recognition API - Refactored Class-Based Library
A clean, modular, and thread-safe face recognition library with separated concerns
"""

import cv2
import depthai as dai
import numpy as np
import face_recognition
import pickle
import os
import logging
import threading
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any, Union, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
from enum import Enum
import time


# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes and Enums
# ============================================================================

class RecognitionModel(Enum):
    """Supported face recognition models"""
    HOG = "hog"
    CNN = "cnn"


class CameraType(Enum):
    """Supported camera types"""
    OAK_D = "oak_d"
    WEBCAM = "webcam"
    VIDEO_FILE = "video_file"


@dataclass
class FaceDetection:
    """Face detection result"""
    location: Tuple[int, int, int, int]  # (top, right, bottom, left)
    encoding: Optional[np.ndarray] = None
    landmarks: Optional[Dict[str, List[Tuple[int, int]]]] = None


@dataclass
class RecognitionResult:
    """Face recognition result"""
    name: str
    confidence: float
    location: Tuple[int, int, int, int]
    encoding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecognitionConfig:
    """Configuration for face recognition"""
    # Recognition parameters
    tolerance: float = 0.6
    model: RecognitionModel = RecognitionModel.HOG
    num_jitters: int = 1
    
    # Camera parameters
    camera_type: CameraType = CameraType.OAK_D
    camera_resolution: Tuple[int, int] = (640, 480)
    camera_fps: int = 30
    
    # Database parameters
    database_path: str = "face_database.pkl"
    auto_save: bool = True
    
    # Processing parameters
    process_every_n_frames: int = 1
    face_detection_scale: float = 1.0
    min_face_size: int = 20
    
    # Thread safety
    enable_threading: bool = True
    
    # Logging
    log_level: str = "INFO"


# ============================================================================
# Abstract Base Classes
# ============================================================================

class CameraInterface(ABC):
    """Abstract base class for camera interfaces"""
    
    @abstractmethod
    def start(self) -> bool:
        """Start the camera"""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop the camera"""
        pass
    
    @abstractmethod
    def get_frame(self) -> Optional[np.ndarray]:
        """Get current frame from camera"""
        pass
    
    @abstractmethod
    def is_running(self) -> bool:
        """Check if camera is running"""
        pass
    
    @property
    @abstractmethod
    def resolution(self) -> Tuple[int, int]:
        """Get camera resolution"""
        pass


# ============================================================================
# Camera Implementations
# ============================================================================

class OakDCamera(CameraInterface):
    """Oak D Series 3 camera implementation using DepthAI V3 API"""
    
    def __init__(self, resolution: Tuple[int, int] = (640, 480)):
        self._resolution = resolution
        self.pipeline = None
        self.camera_output = None
        self.output_queue = None
        self._lock = threading.Lock()
        logger.debug(f"Initialized OakDCamera with resolution {resolution}")
    
    def start(self) -> bool:
        """Start the Oak D camera"""
        try:
            with self._lock:
                self.pipeline = self._create_pipeline()
                self.pipeline.start()
                logger.info("Oak D camera started successfully")
                return True
        except Exception as e:
            logger.error(f"Failed to start Oak D camera: {e}")
            return False
    
    def stop(self) -> None:
        """Stop the Oak D camera"""
        with self._lock:
            if self.pipeline:
                self.pipeline.stop()
                self.pipeline = None
                logger.info("Oak D camera stopped")
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get current frame from camera"""
        with self._lock:
            if not self.is_running():
                return None
            
            try:
                if self.output_queue and self.output_queue.has():
                    in_rgb = self.output_queue.get()
                    frame = in_rgb.getCvFrame()
                    return frame
                return None
            except Exception as e:
                logger.error(f"Error getting frame: {e}")
                return None
    
    def is_running(self) -> bool:
        """Check if camera is running"""
        with self._lock:
            return self.pipeline is not None and self.pipeline.isRunning()
    
    @property
    def resolution(self) -> Tuple[int, int]:
        """Get camera resolution"""
        return self._resolution
    
    def _create_pipeline(self) -> dai.Pipeline:
        """Create DepthAI pipeline for Oak D camera"""
        pipeline = dai.Pipeline()
        
        # Create camera node with V3 API
        cam = pipeline.create(dai.node.Camera).build()
        
        # Request output with specific resolution and format
        self.camera_output = cam.requestOutput(
            self._resolution, 
            type=dai.ImgFrame.Type.RGB888p
        )
        
        # Create output queue
        self.output_queue = self.camera_output.createOutputQueue()
        
        return pipeline


class WebcamCamera(CameraInterface):
    """Standard webcam implementation using OpenCV"""
    
    def __init__(self, camera_index: int = 0, resolution: Tuple[int, int] = (640, 480)):
        self.camera_index = camera_index
        self._resolution = resolution
        self.cap = None
        self._lock = threading.Lock()
        logger.debug(f"Initialized WebcamCamera with index {camera_index}")
    
    def start(self) -> bool:
        """Start the webcam"""
        try:
            with self._lock:
                self.cap = cv2.VideoCapture(self.camera_index)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._resolution[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._resolution[1])
                
                if self.cap.isOpened():
                    logger.info("Webcam started successfully")
                    return True
                else:
                    logger.error("Failed to open webcam")
                    return False
        except Exception as e:
            logger.error(f"Failed to start webcam: {e}")
            return False
    
    def stop(self) -> None:
        """Stop the webcam"""
        with self._lock:
            if self.cap:
                self.cap.release()
                self.cap = None
                logger.info("Webcam stopped")
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get current frame from webcam"""
        with self._lock:
            if not self.is_running():
                return None
            
            try:
                ret, frame = self.cap.read()
                if ret:
                    # Convert BGR to RGB
                    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return None
            except Exception as e:
                logger.error(f"Error getting frame: {e}")
                return None
    
    def is_running(self) -> bool:
        """Check if webcam is running"""
        with self._lock:
            return self.cap is not None and self.cap.isOpened()
    
    @property
    def resolution(self) -> Tuple[int, int]:
        """Get camera resolution"""
        return self._resolution


# ============================================================================
# Face Database Manager
# ============================================================================

class FaceDatabaseManager:
    """Manages face encodings database with thread-safe operations"""
    
    def __init__(self, database_path: str = "face_database.pkl", auto_save: bool = True):
        self.database_path = database_path
        self.auto_save = auto_save
        self.encodings: List[np.ndarray] = []
        self.names: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        
        # Load existing database
        self.load()
    
    def add_face(self, name: str, encoding: np.ndarray, 
                 metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add a face to the database"""
        with self._lock:
            try:
                self.encodings.append(encoding)
                self.names.append(name)
                self.metadata.append(metadata or {})
                
                if self.auto_save:
                    self.save()
                
                logger.info(f"Added face for {name} to database")
                return True
            except Exception as e:
                logger.error(f"Failed to add face: {e}")
                return False
    
    def remove_face(self, name: str, remove_all: bool = True) -> bool:
        """Remove face(s) from database"""
        with self._lock:
            try:
                if name not in self.names:
                    logger.warning(f"Name {name} not found in database")
                    return False
                
                if remove_all:
                    # Remove all instances of the name
                    indices = [i for i, n in enumerate(self.names) if n == name]
                else:
                    # Remove only first instance
                    indices = [self.names.index(name)]
                
                # Remove in reverse order to maintain indices
                for index in sorted(indices, reverse=True):
                    del self.encodings[index]
                    del self.names[index]
                    del self.metadata[index]
                
                if self.auto_save:
                    self.save()
                
                logger.info(f"Removed {len(indices)} face(s) for {name}")
                return True
            except Exception as e:
                logger.error(f"Failed to remove face: {e}")
                return False
    
    def get_all_names(self) -> List[str]:
        """Get list of all unique names in database"""
        with self._lock:
            return list(set(self.names))
    
    def get_face_count(self, name: Optional[str] = None) -> int:
        """Get count of faces in database"""
        with self._lock:
            if name:
                return self.names.count(name)
            return len(self.names)
    
    def clear(self) -> None:
        """Clear all faces from database"""
        with self._lock:
            self.encodings.clear()
            self.names.clear()
            self.metadata.clear()
            
            if self.auto_save:
                self.save()
            
            logger.info("Cleared all faces from database")
    
    def save(self, path: Optional[str] = None) -> bool:
        """Save database to file"""
        save_path = path or self.database_path
        
        with self._lock:
            try:
                data = {
                    'encodings': self.encodings,
                    'names': self.names,
                    'metadata': self.metadata,
                    'version': '2.0'  # Version for compatibility checking
                }
                
                with open(save_path, 'wb') as f:
                    pickle.dump(data, f)
                
                logger.info(f"Saved database with {len(self.names)} faces to {save_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to save database: {e}")
                return False
    
    def load(self, path: Optional[str] = None) -> bool:
        """Load database from file"""
        load_path = path or self.database_path
        
        if not os.path.exists(load_path):
            logger.info("No existing database found, starting fresh")
            return True
        
        with self._lock:
            try:
                with open(load_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Handle both old and new database formats
                if 'version' in data:
                    # New format
                    self.encodings = data.get('encodings', [])
                    self.names = data.get('names', [])
                    self.metadata = data.get('metadata', [])
                else:
                    # Old format compatibility
                    self.encodings = data.get('encodings', [])
                    self.names = data.get('names', [])
                    self.metadata = [{}] * len(self.names)
                
                logger.info(f"Loaded {len(self.names)} faces from {load_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to load database: {e}")
                return False
    
    def get_encodings_and_names(self) -> Tuple[List[np.ndarray], List[str]]:
        """Get encodings and names for recognition"""
        with self._lock:
            return self.encodings.copy(), self.names.copy()


# ============================================================================
# Face Detector
# ============================================================================

class FaceDetector:
    """Handles face detection operations"""
    
    def __init__(self, model: RecognitionModel = RecognitionModel.HOG,
                 scale: float = 1.0, min_face_size: int = 20):
        self.model = model
        self.scale = scale
        self.min_face_size = min_face_size
        logger.debug(f"Initialized FaceDetector with model {model.value}")
    
    def detect_faces(self, frame: np.ndarray, 
                    return_encodings: bool = True,
                    num_jitters: int = 1) -> List[FaceDetection]:
        """Detect faces in frame and optionally compute encodings"""
        try:
            # Ensure frame is in RGB format
            rgb_frame = self._ensure_rgb(frame)
            
            # Scale frame if needed
            if self.scale != 1.0:
                scaled_frame = cv2.resize(
                    rgb_frame, 
                    None, 
                    fx=self.scale, 
                    fy=self.scale
                )
            else:
                scaled_frame = rgb_frame
            
            # Detect face locations
            face_locations = face_recognition.face_locations(
                scaled_frame, 
                model=self.model.value
            )
            
            # Scale locations back to original size
            if self.scale != 1.0:
                face_locations = [
                    tuple(int(coord / self.scale) for coord in loc)
                    for loc in face_locations
                ]
            
            # Filter out small faces
            face_locations = [
                loc for loc in face_locations
                if self._get_face_size(loc) >= self.min_face_size
            ]
            
            # Compute encodings if requested
            if return_encodings and face_locations:
                face_encodings = face_recognition.face_encodings(
                    rgb_frame, 
                    face_locations,
                    num_jitters=num_jitters
                )
            else:
                face_encodings = [None] * len(face_locations)
            
            # Create FaceDetection objects
            detections = [
                FaceDetection(location=loc, encoding=enc)
                for loc, enc in zip(face_locations, face_encodings)
            ]
            
            logger.debug(f"Detected {len(detections)} faces")
            return detections
            
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return []
    
    def detect_face_landmarks(self, frame: np.ndarray, 
                             face_location: Optional[Tuple[int, int, int, int]] = None) -> Optional[Dict]:
        """Detect facial landmarks"""
        try:
            rgb_frame = self._ensure_rgb(frame)
            
            if face_location:
                landmarks = face_recognition.face_landmarks(
                    rgb_frame, 
                    [face_location]
                )
            else:
                landmarks = face_recognition.face_landmarks(rgb_frame)
            
            return landmarks[0] if landmarks else None
            
        except Exception as e:
            logger.error(f"Landmark detection failed: {e}")
            return None
    
    @staticmethod
    def _ensure_rgb(frame: np.ndarray) -> np.ndarray:
        """Ensure frame is in RGB format"""
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            # Assume BGR if it's a 3-channel image from OpenCV
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    
    @staticmethod
    def _get_face_size(location: Tuple[int, int, int, int]) -> int:
        """Calculate face size from location"""
        top, right, bottom, left = location
        return min(bottom - top, right - left)


# ============================================================================
# Face Recognizer
# ============================================================================

class FaceRecognizerEngine:
    """Handles face recognition/matching operations"""
    
    def __init__(self, tolerance: float = 0.6):
        self.tolerance = tolerance
        self._cache = {}  # Cache for recent recognitions
        self._cache_ttl = 1.0  # Cache TTL in seconds
        logger.debug(f"Initialized FaceRecognizerEngine with tolerance {tolerance}")
    
    def recognize_faces(self, 
                        face_encodings: List[np.ndarray],
                        known_encodings: List[np.ndarray],
                        known_names: List[str],
                        use_cache: bool = True) -> List[RecognitionResult]:
        """Recognize faces by comparing with known faces"""
        results = []
        
        for face_encoding in face_encodings:
            # Check cache first
            if use_cache:
                cached_result = self._check_cache(face_encoding)
                if cached_result:
                    results.append(cached_result)
                    continue
            
            # Perform recognition
            result = self._recognize_single_face(
                face_encoding, 
                known_encodings, 
                known_names
            )
            
            # Cache result
            if use_cache:
                self._update_cache(face_encoding, result)
            
            results.append(result)
        
        return results
    
    def _recognize_single_face(self, 
                               face_encoding: np.ndarray,
                               known_encodings: List[np.ndarray],
                               known_names: List[str]) -> RecognitionResult:
        """Recognize a single face"""
        if not known_encodings:
            return RecognitionResult(
                name="Unknown",
                confidence=0.0,
                location=(0, 0, 0, 0),
                encoding=face_encoding
            )
        
        # Compare with known faces
        matches = face_recognition.compare_faces(
            known_encodings, 
            face_encoding, 
            tolerance=self.tolerance
        )
        
        face_distances = face_recognition.face_distance(
            known_encodings, 
            face_encoding
        )
        
        # Find best match
        best_match_index = np.argmin(face_distances)
        
        if matches[best_match_index]:
            name = known_names[best_match_index]
            confidence = 1.0 - face_distances[best_match_index]
        else:
            name = "Unknown"
            confidence = 0.0
        
        return RecognitionResult(
            name=name,
            confidence=confidence,
            location=(0, 0, 0, 0),  # Will be updated by caller
            encoding=face_encoding,
            metadata={
                'distance': float(face_distances[best_match_index]) if matches[best_match_index] else None,
                'match_index': int(best_match_index) if matches[best_match_index] else None
            }
        )
    
    def _check_cache(self, encoding: np.ndarray) -> Optional[RecognitionResult]:
        """Check if encoding is in cache"""
        # Simple cache implementation - can be improved with better hashing
        current_time = time.time()
        
        # Clean old cache entries
        self._cache = {
            k: v for k, v in self._cache.items()
            if current_time - v[1] < self._cache_ttl
        }
        
        # Check for match (simplified - in production use better comparison)
        for cached_encoding, (result, timestamp) in self._cache.items():
            if np.array_equal(cached_encoding, encoding):
                return result
        
        return None
    
    def _update_cache(self, encoding: np.ndarray, result: RecognitionResult) -> None:
        """Update cache with new result"""
        # Simple cache - store with timestamp
        # In production, use better key generation
        key = id(encoding)  # Simple key for demo
        self._cache[key] = (result, time.time())
    
    def clear_cache(self) -> None:
        """Clear recognition cache"""
        self._cache.clear()


# ============================================================================
# Main Face Recognition API
# ============================================================================

class FaceRecognitionAPI:
    """
    Main Face Recognition API - Clean interface for face recognition operations
    
    This is the main entry point for using the face recognition system.
    It provides a high-level, thread-safe API for all face recognition operations.
    """
    
    def __init__(self, config: Optional[RecognitionConfig] = None):
        """
        Initialize Face Recognition API
        
        Args:
            config: Configuration object. If None, uses default configuration.
        """
        self.config = config or RecognitionConfig()
        
        # Configure logging
        logging.basicConfig(level=getattr(logging, self.config.log_level))
        
        # Initialize components
        self.camera: Optional[CameraInterface] = None
        self.database = FaceDatabaseManager(
            self.config.database_path,
            self.config.auto_save
        )
        self.detector = FaceDetector(
            self.config.model,
            self.config.face_detection_scale,
            self.config.min_face_size
        )
        self.recognizer = FaceRecognizerEngine(self.config.tolerance)
        
        # Thread safety
        self._lock = threading.RLock() if self.config.enable_threading else None
        
        # Processing state
        self._frame_counter = 0
        self._is_processing = False
        
        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            'on_face_detected': [],
            'on_face_recognized': [],
            'on_face_added': [],
            'on_face_removed': [],
            'on_error': []
        }
        
        logger.info("Face Recognition API initialized")
    
    # ========================================================================
    # Camera Management
    # ========================================================================
    
    def start_camera(self, camera_type: Optional[CameraType] = None) -> bool:
        """
        Start the camera
        
        Args:
            camera_type: Type of camera to use. If None, uses config default.
        
        Returns:
            True if camera started successfully
        """
        with self._get_lock():
            try:
                # Stop existing camera if running
                if self.camera and self.camera.is_running():
                    self.stop_camera()
                
                # Create camera based on type
                cam_type = camera_type or self.config.camera_type
                
                if cam_type == CameraType.OAK_D:
                    self.camera = OakDCamera(self.config.camera_resolution)
                elif cam_type == CameraType.WEBCAM:
                    self.camera = WebcamCamera(
                        camera_index=0,
                        resolution=self.config.camera_resolution
                    )
                else:
                    raise ValueError(f"Unsupported camera type: {cam_type}")
                
                # Start camera
                success = self.camera.start()
                if success:
                    logger.info(f"Started {cam_type.value} camera")
                return success
                
            except Exception as e:
                logger.error(f"Failed to start camera: {e}")
                self._trigger_callback('on_error', e)
                return False
    
    def stop_camera(self) -> None:
        """Stop the camera"""
        with self._get_lock():
            if self.camera:
                self.camera.stop()
                self.camera = None
                logger.info("Camera stopped")
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get current frame from camera
        
        Returns:
            Frame as numpy array or None if unavailable
        """
        with self._get_lock():
            if not self.camera or not self.camera.is_running():
                return None
            return self.camera.get_frame()
    
    def is_camera_running(self) -> bool:
        """Check if camera is running"""
        with self._get_lock():
            return self.camera is not None and self.camera.is_running()
    
    # ========================================================================
    # Face Detection and Recognition
    # ========================================================================
    
    def detect_faces(self, frame: np.ndarray, 
                    compute_encodings: bool = True) -> List[FaceDetection]:
        """
        Detect faces in a frame
        
        Args:
            frame: Input frame
            compute_encodings: Whether to compute face encodings
        
        Returns:
            List of face detections
        """
        with self._get_lock():
            self._frame_counter += 1
            
            # Skip frames based on configuration
            if self._frame_counter % self.config.process_every_n_frames != 0:
                return []
            
            detections = self.detector.detect_faces(
                frame,
                return_encodings=compute_encodings,
                num_jitters=self.config.num_jitters
            )
            
            # Trigger callback
            if detections:
                self._trigger_callback('on_face_detected', detections)
            
            return detections
    
    def recognize_faces(self, 
                       face_detections: List[FaceDetection],
                       use_cache: bool = True) -> List[RecognitionResult]:
        """
        Recognize faces from detections
        
        Args:
            face_detections: List of face detections with encodings
            use_cache: Whether to use recognition cache
        
        Returns:
            List of recognition results
        """
        with self._get_lock():
            # Get known faces from database
            known_encodings, known_names = self.database.get_encodings_and_names()
            
            # Extract encodings from detections
            face_encodings = [d.encoding for d in face_detections if d.encoding is not None]
            
            if not face_encodings:
                return []
            
            # Perform recognition
            results = self.recognizer.recognize_faces(
                face_encodings,
                known_encodings,
                known_names,
                use_cache=use_cache
            )
            
            # Update results with locations
            for result, detection in zip(results, face_detections):
                result.location = detection.location
            
            # Trigger callback
            if results:
                self._trigger_callback('on_face_recognized', results)
            
            return results
    
    def process_frame(self, frame: Optional[np.ndarray] = None) -> List[RecognitionResult]:
        """
        Process a frame for face recognition (combined detect + recognize)
        
        Args:
            frame: Frame to process. If None, gets frame from camera.
        
        Returns:
            List of recognition results
        """
        with self._get_lock():
            # Get frame if not provided
            if frame is None:
                frame = self.get_frame()
                if frame is None:
                    return []
            
            # Detect faces
            detections = self.detect_faces(frame, compute_encodings=True)
            
            if not detections:
                return []
            
            # Recognize faces
            return self.recognize_faces(detections)
    
    # ========================================================================
    # Database Management
    # ========================================================================
    
    def add_face(self, name: str, frame: Optional[np.ndarray] = None,
                encoding: Optional[np.ndarray] = None,
                metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a face to the database
        
        Args:
            name: Name of the person
            frame: Frame containing the face (provide either frame or encoding)
            encoding: Pre-computed face encoding (provide either frame or encoding)
            metadata: Optional metadata to store with the face
        
        Returns:
            True if successful
        """
        with self._get_lock():
            try:
                # Get encoding from frame if not provided
                if encoding is None:
                    if frame is None:
                        # Try to get frame from camera
                        frame = self.get_frame()
                        if frame is None:
                            logger.error("No frame or encoding provided")
                            return False
                    
                    # Detect faces in frame
                    detections = self.detector.detect_faces(
                        frame,
                        return_encodings=True,
                        num_jitters=self.config.num_jitters
                    )
                    
                    if not detections:
                        logger.warning("No face detected in frame")
                        return False
                    
                    if len(detections) > 1:
                        logger.warning("Multiple faces detected. Using first face.")
                    
                    encoding = detections[0].encoding
                
                # Add to database
                success = self.database.add_face(name, encoding, metadata)
                
                if success:
                    self._trigger_callback('on_face_added', name)
                
                return success
                
            except Exception as e:
                logger.error(f"Failed to add face: {e}")
                self._trigger_callback('on_error', e)
                return False
    
    def remove_face(self, name: str, remove_all: bool = True) -> bool:
        """
        Remove face(s) from database
        
        Args:
            name: Name of the person to remove
            remove_all: If True, removes all instances. If False, removes only first.
        
        Returns:
            True if successful
        """
        with self._get_lock():
            success = self.database.remove_face(name, remove_all)
            if success:
                self._trigger_callback('on_face_removed', name)
            return success
    
    def get_all_names(self) -> List[str]:
        """Get list of all unique names in database"""
        with self._get_lock():
            return self.database.get_all_names()
    
    def get_face_count(self, name: Optional[str] = None) -> int:
        """Get count of faces in database"""
        with self._get_lock():
            return self.database.get_face_count(name)
    
    def clear_database(self) -> None:
        """Clear all faces from database"""
        with self._get_lock():
            self.database.clear()
    
    def save_database(self, path: Optional[str] = None) -> bool:
        """Save database to file"""
        with self._get_lock():
            return self.database.save(path)
    
    def load_database(self, path: Optional[str] = None) -> bool:
        """Load database from file"""
        with self._get_lock():
            return self.database.load(path)
    
    # ========================================================================
    # Batch Operations
    # ========================================================================
    
    def add_faces_from_directory(self, directory: str, 
                                 pattern: str = "*.jpg") -> Dict[str, int]:
        """
        Add faces from a directory of images
        
        Args:
            directory: Directory containing images
            pattern: File pattern to match
        
        Returns:
            Dictionary mapping names to number of faces added
        """
        import glob
        
        results = {}
        
        with self._get_lock():
            for image_path in glob.glob(os.path.join(directory, pattern)):
                # Extract name from filename
                name = os.path.splitext(os.path.basename(image_path))[0]
                
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    logger.warning(f"Could not load image: {image_path}")
                    continue
                
                # Add face
                if self.add_face(name, image):
                    results[name] = results.get(name, 0) + 1
                    logger.info(f"Added face for {name} from {image_path}")
        
        return results
    
    def recognize_faces_in_image(self, image_path: str) -> List[RecognitionResult]:
        """
        Recognize faces in a single image file
        
        Args:
            image_path: Path to image file
        
        Returns:
            List of recognition results
        """
        with self._get_lock():
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return []
            
            # Process frame
            return self.process_frame(image)
    
    # ========================================================================
    # Callbacks and Events
    # ========================================================================
    
    def register_callback(self, event: str, callback: Callable) -> None:
        """
        Register a callback for an event
        
        Args:
            event: Event name ('on_face_detected', 'on_face_recognized', etc.)
            callback: Callback function
        """
        if event in self._callbacks:
            self._callbacks[event].append(callback)
            logger.debug(f"Registered callback for {event}")
        else:
            logger.warning(f"Unknown event: {event}")
    
    def unregister_callback(self, event: str, callback: Callable) -> None:
        """Unregister a callback"""
        if event in self._callbacks and callback in self._callbacks[event]:
            self._callbacks[event].remove(callback)
            logger.debug(f"Unregistered callback for {event}")
    
    def _trigger_callback(self, event: str, *args, **kwargs) -> None:
        """Trigger callbacks for an event"""
        for callback in self._callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Callback error for {event}: {e}")
    
    # ========================================================================
    # Context Manager Support
    # ========================================================================
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_camera()
        if self.config.auto_save:
            self.save_database()
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def _get_lock(self):
        """Get thread lock if threading is enabled"""
        if self._lock:
            return self._lock
        else:
            # Return a dummy context manager if threading is disabled
            return contextmanager(lambda: iter([None]))()
    
    def get_config(self) -> RecognitionConfig:
        """Get current configuration"""
        return self.config
    
    def update_config(self, **kwargs) -> None:
        """Update configuration parameters"""
        with self._get_lock():
            for key, value in kwargs.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                    logger.debug(f"Updated config: {key} = {value}")
                else:
                    logger.warning(f"Unknown config parameter: {key}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get API statistics"""
        with self._get_lock():
            return {
                'total_faces': self.database.get_face_count(),
                'unique_people': len(self.database.get_all_names()),
                'frames_processed': self._frame_counter,
                'camera_running': self.is_camera_running(),
                'config': {
                    'tolerance': self.config.tolerance,
                    'model': self.config.model.value,
                    'camera_type': self.config.camera_type.value
                }
            }
    
    # ========================================================================
    # Backwards Compatibility Methods
    # ========================================================================
    
    def start(self) -> bool:
        """Backwards compatible method for starting the system"""
        return self.start_camera()
    
    def stop(self) -> None:
        """Backwards compatible method for stopping the system"""
        self.stop_camera()
    
    def recognize(self, face_encodings: List[np.ndarray]) -> List[Tuple[str, float]]:
        """
        Backwards compatible recognition method
        
        Args:
            face_encodings: List of face encodings
        
        Returns:
            List of (name, confidence) tuples
        """
        # Create fake detections for compatibility
        detections = [
            FaceDetection(location=(0, 0, 0, 0), encoding=enc)
            for enc in face_encodings
        ]
        
        results = self.recognize_faces(detections)
        return [(r.name, r.confidence) for r in results]


# ============================================================================
# Factory Functions
# ============================================================================

def create_face_recognition_api(
    database_path: str = "face_database.pkl",
    camera_type: CameraType = CameraType.OAK_D,
    tolerance: float = 0.6,
    **kwargs
) -> FaceRecognitionAPI:
    """
    Factory function to create Face Recognition API with common settings
    
    Args:
        database_path: Path to face database
        camera_type: Type of camera to use
        tolerance: Recognition tolerance (0.0 to 1.0)
        **kwargs: Additional configuration parameters
    
    Returns:
        Configured FaceRecognitionAPI instance
    """
    config = RecognitionConfig(
        database_path=database_path,
        camera_type=camera_type,
        tolerance=tolerance,
        **kwargs
    )
    
    return FaceRecognitionAPI(config)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example: Basic usage
    api = create_face_recognition_api()
    
    # Start camera
    if api.start_camera():
        print("Camera started successfully")
        
        # Add a face
        print("Position your face and press Enter...")
        input()
        if api.add_face("Test User"):
            print("Face added successfully")
        
        # Recognize faces
        print("Starting recognition...")
        for _ in range(100):
            results = api.process_frame()
            for result in results:
                if result.name != "Unknown":
                    print(f"Recognized: {result.name} ({result.confidence:.2f})")
        
        # Stop camera
        api.stop_camera()
    
    # Example: Using context manager
    with create_face_recognition_api() as api:
        api.start_camera()
        # ... do recognition ...
        pass  # Camera automatically stopped on exit