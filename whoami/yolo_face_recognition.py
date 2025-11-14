"""
YOLO-based Face Recognition for WhoAmI K-1 System

Integrates YOLOv8 face detection with deep learning face recognition
and voice interaction for the K-1 Booster robot.

Features:
- YOLOv8 for fast face detection
- DeepFace for face recognition/embeddings
- Voice interaction for name collection
- Hardware-optimized for Jetson Orin NX
"""

import cv2
import numpy as np
import logging
import time
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import pickle
import threading

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("ultralytics not available. Install with: pip install ultralytics")

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    logging.warning("deepface not available. Install with: pip install deepface")

try:
    import depthai as dai
    DEPTHAI_AVAILABLE = True
except ImportError:
    DEPTHAI_AVAILABLE = False
    logging.warning("depthai not available for Oak D camera")

# Import voice interaction
try:
    from whoami.voice_interaction import VoiceInteraction
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False
    logging.warning("Voice interaction not available")

logger = logging.getLogger(__name__)


def format_time_delta(seconds: float) -> str:
    """
    Format time difference in human-friendly format

    Args:
        seconds: Time difference in seconds

    Returns:
        Formatted string like "1 day 3 hours and 23 seconds"
    """
    if seconds < 0:
        return "just now"

    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    parts = []
    if days > 0:
        parts.append(f"{days} day{'s' if days != 1 else ''}")
    if hours > 0:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes > 0 and days == 0:  # Skip minutes if we have days
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if secs > 0 and days == 0 and hours == 0:  # Only show seconds if < 1 hour
        parts.append(f"{secs} second{'s' if secs != 1 else ''}")

    if not parts:
        return "just now"

    if len(parts) == 1:
        return parts[0]
    elif len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    else:
        return f"{', '.join(parts[:-1])}, and {parts[-1]}"


@dataclass
class YOLOFaceDetection:
    """Face detection result from YOLO"""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    embedding: Optional[np.ndarray] = None


@dataclass
class FaceRecognitionResult:
    """Face recognition result"""
    name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    embedding: Optional[np.ndarray] = None


class YOLOFaceDetector:
    """
    YOLO-based face detector optimized for Jetson

    Uses YOLOv8n (nano) for real-time performance on K-1 Booster
    """

    def __init__(
        self,
        model_path: str = 'yolov8n-face.pt',
        confidence_threshold: float = 0.5,
        device: str = 'cuda:0'
    ):
        """
        Initialize YOLO face detector

        Args:
            model_path: Path to YOLO model weights
            confidence_threshold: Minimum detection confidence
            device: Device to run on ('cuda:0' for Jetson GPU)
        """
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics not installed. Run: pip install ultralytics")

        self.confidence_threshold = confidence_threshold
        self.device = device

        # Load YOLO model
        try:
            self.model = YOLO(model_path)
            self.model.to(device)
            logger.info(f"YOLO model loaded: {model_path} on {device}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            # Try default face detection model
            try:
                self.model = YOLO('yolov8n.pt')  # General object detection
                logger.info("Using YOLOv8n general model (will detect 'person' class)")
            except:
                raise RuntimeError("Could not load any YOLO model")

        # Warm up model
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(dummy, verbose=False)
        logger.info("YOLO model warmed up")

    def detect_faces(self, frame: np.ndarray) -> List[YOLOFaceDetection]:
        """
        Detect faces in frame using YOLO

        Args:
            frame: Input frame (BGR or RGB)

        Returns:
            List of face detections
        """
        try:
            # Run inference
            results = self.model(frame, verbose=False, conf=self.confidence_threshold)

            detections = []
            for result in results:
                boxes = result.boxes

                for box in boxes:
                    # Get bbox coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])

                    # Filter by confidence
                    if conf >= self.confidence_threshold:
                        detections.append(YOLOFaceDetection(
                            bbox=(int(x1), int(y1), int(x2), int(y2)),
                            confidence=conf
                        ))

            logger.debug(f"Detected {len(detections)} faces")
            return detections

        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return []


class DeepFaceRecognizer:
    """
    Face recognition using DeepFace embeddings

    Uses FaceNet or ArcFace for face embeddings and matching
    """

    def __init__(
        self,
        model_name: str = 'Facenet',  # or 'ArcFace', 'VGG-Face'
        database_path: str = 'face_database_yolo.pkl',
        distance_metric: str = 'cosine',
        threshold: float = 0.4
    ):
        """
        Initialize DeepFace recognizer

        Args:
            model_name: Model for embeddings ('Facenet', 'ArcFace', 'VGG-Face')
            database_path: Path to face database
            distance_metric: Distance metric ('cosine', 'euclidean')
            threshold: Recognition threshold
        """
        if not DEEPFACE_AVAILABLE:
            raise ImportError("deepface not installed. Run: pip install deepface")

        self.model_name = model_name
        self.distance_metric = distance_metric
        self.threshold = threshold
        self.database_path = database_path

        # Face database
        self.face_db: Dict[str, List[np.ndarray]] = {}
        self.metadata_db: Dict[str, Dict[str, Any]] = {}  # Store metadata like last_seen
        self._lock = threading.RLock()

        # Load existing database
        self.load_database()

        logger.info(f"DeepFace recognizer initialized with {model_name}")

    def extract_embedding(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding using DeepFace

        Args:
            face_img: Cropped face image (RGB)

        Returns:
            Face embedding vector or None if failed
        """
        try:
            # Ensure minimum size
            if face_img.shape[0] < 80 or face_img.shape[1] < 80:
                face_img = cv2.resize(face_img, (160, 160))

            # Extract embedding
            embedding_obj = DeepFace.represent(
                img_path=face_img,
                model_name=self.model_name,
                enforce_detection=False,
                detector_backend='skip'  # We already detected face with YOLO
            )

            # Get embedding vector
            if isinstance(embedding_obj, list) and len(embedding_obj) > 0:
                embedding = np.array(embedding_obj[0]['embedding'])
                return embedding

            return None

        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}")
            return None

    def add_face(self, name: str, embedding: np.ndarray) -> bool:
        """
        Add face embedding to database

        Args:
            name: Person's name
            embedding: Face embedding vector

        Returns:
            True if successful
        """
        with self._lock:
            try:
                if name not in self.face_db:
                    self.face_db[name] = []
                    # Initialize metadata for new person
                    self.metadata_db[name] = {
                        'first_seen': time.time(),
                        'last_seen': time.time(),
                        'encounter_count': 0
                    }

                self.face_db[name].append(embedding)
                self.save_database()

                logger.info(f"Added face for {name} (total: {len(self.face_db[name])} embeddings)")
                return True

            except Exception as e:
                logger.error(f"Failed to add face: {e}")
                return False

    def recognize_face(self, embedding: np.ndarray) -> Tuple[str, float]:
        """
        Recognize face from embedding

        Args:
            embedding: Face embedding to match

        Returns:
            (name, confidence) tuple
        """
        with self._lock:
            if not self.face_db:
                return ("Unknown", 0.0)

            best_match_name = "Unknown"
            best_distance = float('inf')

            # Compare with all known faces
            for name, embeddings in self.face_db.items():
                for known_embedding in embeddings:
                    # Calculate distance
                    if self.distance_metric == 'cosine':
                        distance = self._cosine_distance(embedding, known_embedding)
                    else:
                        distance = self._euclidean_distance(embedding, known_embedding)

                    if distance < best_distance:
                        best_distance = distance
                        best_match_name = name

            # Check if distance is within threshold
            if best_distance > self.threshold:
                return ("Unknown", 0.0)

            # Convert distance to confidence
            confidence = 1.0 - (best_distance / self.threshold)
            confidence = max(0.0, min(1.0, confidence))

            return (best_match_name, confidence)

    @staticmethod
    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine distance"""
        return 1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    @staticmethod
    def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate euclidean distance"""
        return np.linalg.norm(a - b)

    def remove_face(self, name: str) -> bool:
        """Remove all faces for a person"""
        with self._lock:
            if name in self.face_db:
                del self.face_db[name]
                if name in self.metadata_db:
                    del self.metadata_db[name]
                self.save_database()
                logger.info(f"Removed all faces for {name}")
                return True
            return False

    def update_last_seen(self, name: str) -> None:
        """Update the last_seen timestamp for a person"""
        with self._lock:
            if name not in self.metadata_db:
                self.metadata_db[name] = {
                    'first_seen': time.time(),
                    'last_seen': time.time(),
                    'encounter_count': 0
                }

            self.metadata_db[name]['last_seen'] = time.time()
            self.metadata_db[name]['encounter_count'] = self.metadata_db[name].get('encounter_count', 0) + 1
            self.save_database()

    def get_time_since_last_seen(self, name: str) -> Optional[float]:
        """
        Get time in seconds since person was last seen

        Args:
            name: Person's name

        Returns:
            Seconds since last seen, or None if never seen
        """
        with self._lock:
            if name in self.metadata_db and 'last_seen' in self.metadata_db[name]:
                return time.time() - self.metadata_db[name]['last_seen']
            return None

    def get_all_names(self) -> List[str]:
        """Get all known names"""
        with self._lock:
            return list(self.face_db.keys())

    def save_database(self) -> bool:
        """Save face database to disk"""
        with self._lock:
            try:
                data = {
                    'face_db': self.face_db,
                    'metadata_db': self.metadata_db,
                    'version': '2.0'
                }
                with open(self.database_path, 'wb') as f:
                    pickle.dump(data, f)
                logger.debug(f"Saved database with {len(self.face_db)} people")
                return True
            except Exception as e:
                logger.error(f"Failed to save database: {e}")
                return False

    def load_database(self) -> bool:
        """Load face database from disk"""
        if not Path(self.database_path).exists():
            logger.info("No existing database found")
            return True

        with self._lock:
            try:
                with open(self.database_path, 'rb') as f:
                    data = pickle.load(f)

                # Handle both old and new database formats
                if isinstance(data, dict) and 'version' in data:
                    # New format with metadata
                    self.face_db = data.get('face_db', {})
                    self.metadata_db = data.get('metadata_db', {})
                else:
                    # Old format - just face embeddings
                    self.face_db = data
                    self.metadata_db = {}

                logger.info(f"Loaded database with {len(self.face_db)} people")
                return True
            except Exception as e:
                logger.error(f"Failed to load database: {e}")
                return False


class OakDCameraYOLO:
    """Oak D camera interface for YOLO face recognition"""

    def __init__(self, resolution: Tuple[int, int] = (1280, 720), fps: int = 30):
        if not DEPTHAI_AVAILABLE:
            raise ImportError("depthai not installed")

        self.resolution = resolution
        self.fps = fps
        self.pipeline = None
        self.output_queue = None
        self._running = False

    def start(self) -> bool:
        """Start Oak D camera"""
        try:
            self.pipeline = dai.Pipeline()

            # Create camera node
            cam = self.pipeline.create(dai.node.Camera).build()

            # Request RGB output
            self.camera_output = cam.requestOutput(
                self.resolution,
                type=dai.ImgFrame.Type.RGB888p
            )

            # Create output queue
            self.output_queue = self.camera_output.createOutputQueue()

            # Start pipeline
            self.pipeline.start()
            self._running = True

            logger.info(f"Oak D camera started: {self.resolution}@{self.fps}fps")
            return True

        except Exception as e:
            logger.error(f"Failed to start Oak D camera: {e}")
            return False

    def get_frame(self) -> Optional[np.ndarray]:
        """Get frame from camera"""
        if not self._running or not self.output_queue:
            return None

        try:
            if self.output_queue.has():
                in_rgb = self.output_queue.get()
                frame = in_rgb.getCvFrame()
                return frame
            return None
        except Exception as e:
            logger.error(f"Error getting frame: {e}")
            return None

    def stop(self) -> None:
        """Stop camera"""
        if self.pipeline:
            self.pipeline.stop()
            self._running = False
            logger.info("Oak D camera stopped")

    def is_running(self) -> bool:
        """Check if camera is running"""
        return self._running


class K1FaceRecognitionSystem:
    """
    Complete face recognition system for K-1 Booster

    Integrates:
    - YOLO face detection
    - DeepFace recognition
    - Voice interaction
    - Oak D camera
    """

    def __init__(
        self,
        yolo_model: str = 'yolov8n.pt',
        deepface_model: str = 'Facenet',
        enable_voice: bool = True,
        camera_resolution: Tuple[int, int] = (1280, 720)
    ):
        """
        Initialize K-1 face recognition system

        Args:
            yolo_model: YOLO model path
            deepface_model: DeepFace model name
            enable_voice: Enable voice interaction
            camera_resolution: Camera resolution
        """
        # Initialize components
        self.detector = YOLOFaceDetector(model_path=yolo_model)
        self.recognizer = DeepFaceRecognizer(model_name=deepface_model)

        # Voice interaction
        self.voice = None
        if enable_voice and VOICE_AVAILABLE:
            self.voice = VoiceInteraction()
            logger.info("Voice interaction enabled")

        # Camera
        self.camera = OakDCameraYOLO(resolution=camera_resolution)

        # Tracking
        self.recently_greeted = {}  # {name: timestamp}
        self.greet_cooldown = 60.0  # seconds

        logger.info("K-1 Face Recognition System initialized")

    def start(self) -> bool:
        """Start the system"""
        return self.camera.start()

    def stop(self) -> None:
        """Stop the system"""
        self.camera.stop()
        self.recognizer.save_database()

    def process_frame(
        self,
        frame: Optional[np.ndarray] = None,
        ask_unknown: bool = True,
        greet_known: bool = True,
        wave_on_greet: bool = False
    ) -> List[FaceRecognitionResult]:
        """
        Process a frame for face recognition with voice interaction

        Args:
            frame: Input frame (if None, gets from camera)
            ask_unknown: Ask unknown people for their names
            greet_known: Greet known people
            wave_on_greet: Wave after greeting known people

        Returns:
            List of recognition results
        """
        # Get frame from camera if not provided
        if frame is None:
            frame = self.camera.get_frame()
            if frame is None:
                return []

        # Detect faces
        detections = self.detector.detect_faces(frame)

        if not detections:
            return []

        results = []
        current_time = time.time()

        for detection in detections:
            # Extract face region
            x1, y1, x2, y2 = detection.bbox
            face_img = frame[y1:y2, x1:x2]

            if face_img.size == 0:
                continue

            # Get embedding
            embedding = self.recognizer.extract_embedding(face_img)

            if embedding is None:
                continue

            # Recognize face
            name, confidence = self.recognizer.recognize_face(embedding)

            # Handle unknown person
            if name == "Unknown" and ask_unknown and self.voice:
                # Only ask once per detection session
                if not any(r.name == "Unknown" for r in results):
                    learned_name = self._ask_for_name(frame, face_img, embedding)
                    if learned_name:
                        name = learned_name
                        confidence = 1.0

            # Greet known person
            elif name != "Unknown" and greet_known and self.voice:
                last_greet = self.recently_greeted.get(name, 0)
                if current_time - last_greet > self.greet_cooldown:
                    # Get time since last seen
                    time_since = self.recognizer.get_time_since_last_seen(name)

                    if time_since is not None and time_since > 60:  # More than 1 minute
                        # Create personalized greeting with time
                        time_str = format_time_delta(time_since)
                        greeting = f"Hi {name}, it's been {time_str} since we last talked! How are you?"
                        self.voice.say(greeting)
                    else:
                        # First time seeing them or saw them very recently
                        self.voice.welcome_back(name)

                    # Wave if requested
                    if wave_on_greet:
                        self.wave()

                    self.recently_greeted[name] = current_time

                # Update last_seen timestamp
                self.recognizer.update_last_seen(name)

            # Create result
            result = FaceRecognitionResult(
                name=name,
                confidence=confidence,
                bbox=detection.bbox,
                embedding=embedding
            )
            results.append(result)

        return results

    def _ask_for_name(
        self,
        frame: np.ndarray,
        face_img: np.ndarray,
        embedding: np.ndarray
    ) -> Optional[str]:
        """
        Ask unknown person for their name via voice

        Args:
            frame: Full frame (for display/context)
            face_img: Cropped face image
            embedding: Face embedding

        Returns:
            Name if learned, None otherwise
        """
        if not self.voice:
            return None

        try:
            # Announce unknown person
            self.voice.say("Hello! I don't think we've met before.")
            time.sleep(0.5)

            # Ask for name
            name = self.voice.ask_name(
                prompt="What's your name?",
                max_attempts=3
            )

            if name:
                # Add to database
                self.recognizer.add_face(name, embedding)

                # Greet the person
                self.voice.greet_person(name)
                self.recently_greeted[name] = time.time()

                logger.info(f"Learned new person: {name}")
                return name

        except Exception as e:
            logger.error(f"Error asking for name: {e}")

        return None

    def add_face_from_frame(
        self,
        name: str,
        frame: Optional[np.ndarray] = None
    ) -> bool:
        """
        Manually add a face from current frame

        Args:
            name: Person's name
            frame: Frame to use (if None, gets from camera)

        Returns:
            True if successful
        """
        if frame is None:
            frame = self.camera.get_frame()
            if frame is None:
                logger.error("No frame available")
                return False

        # Detect faces
        detections = self.detector.detect_faces(frame)

        if not detections:
            logger.warning("No face detected")
            return False

        if len(detections) > 1:
            logger.warning(f"Multiple faces detected ({len(detections)}), using first")

        # Use first detection
        detection = detections[0]
        x1, y1, x2, y2 = detection.bbox
        face_img = frame[y1:y2, x1:x2]

        # Extract embedding
        embedding = self.recognizer.extract_embedding(face_img)

        if embedding is None:
            logger.error("Failed to extract embedding")
            return False

        # Add to database
        return self.recognizer.add_face(name, embedding)

    def wave(self) -> None:
        """
        Perform a friendly wave gesture using the robot's arm

        Currently a placeholder - will control arm when implemented
        """
        logger.info("Waving!")
        # TODO: Implement arm wave motion
        # The K-1 robots have arms, so this should control the arm to wave
        # Example wave sequence:
        #   - Raise arm to shoulder height
        #   - Rotate wrist left-right-left-right (3-4 times)
        #   - Lower arm back to rest position
        # May also combine with head nod for more personality
        pass

    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        names = self.recognizer.get_all_names()
        stats = {
            'known_people': len(names),
            'people': names,
            'voice_enabled': self.voice is not None,
            'camera_running': self.camera.is_running()
        }

        # Add metadata for each person
        for name in names:
            if name in self.recognizer.metadata_db:
                metadata = self.recognizer.metadata_db[name]
                time_since = self.recognizer.get_time_since_last_seen(name)
                if time_since is not None:
                    stats[f'{name}_last_seen'] = format_time_delta(time_since)
                    stats[f'{name}_encounters'] = metadata.get('encounter_count', 0)

        return stats


def main():
    """
    Main function for testing K-1 face recognition
    """
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(description='K-1 Face Recognition System')
    parser.add_argument('--no-voice', action='store_true', help='Disable voice interaction')
    parser.add_argument('--model', default='yolov8n.pt', help='YOLO model path')
    args = parser.parse_args()

    # Create system
    system = K1FaceRecognitionSystem(
        yolo_model=args.model,
        enable_voice=not args.no_voice
    )

    # Start system
    if not system.start():
        logger.error("Failed to start system")
        return

    logger.info("K-1 Face Recognition System started")
    logger.info("Press Ctrl+C to exit")

    try:
        frame_count = 0
        while True:
            # Process frame
            results = system.process_frame(
                ask_unknown=True,
                greet_known=True
            )

            # Log results
            if results:
                for result in results:
                    logger.info(
                        f"Frame {frame_count}: {result.name} "
                        f"(confidence: {result.confidence:.2f}) "
                        f"at {result.bbox}"
                    )

            frame_count += 1
            time.sleep(0.1)  # ~10 FPS

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        system.stop()

        # Print statistics
        stats = system.get_statistics()
        logger.info(f"Final statistics: {stats}")


if __name__ == '__main__':
    main()
