# Face Recognition API Reference

## Table of Contents
- [Overview](#overview)
- [Core Classes](#core-classes)
  - [FaceRecognitionAPI](#facerecognitionapi)
  - [RecognitionConfig](#recognitionconfig)
  - [Data Classes](#data-classes)
  - [Camera Interfaces](#camera-interfaces)
  - [Database Manager](#database-manager)
  - [Face Detector](#face-detector)
  - [Recognition Engine](#recognition-engine)
- [Enumerations](#enumerations)
- [Factory Functions](#factory-functions)
- [Error Handling](#error-handling)
- [Thread Safety](#thread-safety)

## Overview

The Face Recognition API is a comprehensive, thread-safe library for face detection and recognition. It provides a clean, modular architecture with separated concerns for camera management, face detection, recognition, and database operations.

### Key Features
- Multiple camera support (OAK-D, Webcam)
- Thread-safe operations
- Configurable recognition models (HOG, CNN)
- Event callbacks
- Batch processing capabilities
- Context manager support
- Recognition caching for performance

## Core Classes

### FaceRecognitionAPI

The main entry point for all face recognition operations.

```python
class FaceRecognitionAPI(config: Optional[RecognitionConfig] = None)
```

#### Constructor Parameters
- `config` (Optional[RecognitionConfig]): Configuration object. If None, uses default configuration.

#### Camera Management Methods

##### `start_camera(camera_type: Optional[CameraType] = None) -> bool`
Start the camera system.

**Parameters:**
- `camera_type` (Optional[CameraType]): Type of camera to use. If None, uses config default.

**Returns:**
- `bool`: True if camera started successfully

**Example:**
```python
api = FaceRecognitionAPI()
if api.start_camera(CameraType.OAK_D):
    print("Camera started")
```

##### `stop_camera() -> None`
Stop the camera system.

**Example:**
```python
api.stop_camera()
```

##### `get_frame() -> Optional[np.ndarray]`
Get current frame from camera.

**Returns:**
- `Optional[np.ndarray]`: Frame as numpy array or None if unavailable

**Thread Safety:** Thread-safe with internal locking

##### `is_camera_running() -> bool`
Check if camera is currently running.

**Returns:**
- `bool`: True if camera is active

#### Face Detection and Recognition Methods

##### `detect_faces(frame: np.ndarray, compute_encodings: bool = True) -> List[FaceDetection]`
Detect faces in a frame.

**Parameters:**
- `frame` (np.ndarray): Input frame in RGB or BGR format
- `compute_encodings` (bool): Whether to compute face encodings for recognition

**Returns:**
- `List[FaceDetection]`: List of detected faces with locations and optionally encodings

**Performance Considerations:**
- Processing is skipped based on `process_every_n_frames` configuration
- Face detection can be scaled using `face_detection_scale` for speed

**Example:**
```python
frame = api.get_frame()
detections = api.detect_faces(frame, compute_encodings=True)
for detection in detections:
    print(f"Face at {detection.location}")
```

##### `recognize_faces(face_detections: List[FaceDetection], use_cache: bool = True) -> List[RecognitionResult]`
Recognize faces from detections.

**Parameters:**
- `face_detections` (List[FaceDetection]): List of face detections with encodings
- `use_cache` (bool): Whether to use recognition cache for performance

**Returns:**
- `List[RecognitionResult]`: List of recognition results with names and confidence scores

**Performance Notes:**
- Recognition cache TTL is 1.0 second by default
- Cache significantly improves performance for consecutive frames

##### `process_frame(frame: Optional[np.ndarray] = None) -> List[RecognitionResult]`
Combined detection and recognition in a single call.

**Parameters:**
- `frame` (Optional[np.ndarray]): Frame to process. If None, gets frame from camera.

**Returns:**
- `List[RecognitionResult]`: List of recognition results

**Example:**
```python
# Process current camera frame
results = api.process_frame()

# Process specific frame
image = cv2.imread("photo.jpg")
results = api.process_frame(image)
```

#### Database Management Methods

##### `add_face(name: str, frame: Optional[np.ndarray] = None, encoding: Optional[np.ndarray] = None, metadata: Optional[Dict[str, Any]] = None) -> bool`
Add a face to the recognition database.

**Parameters:**
- `name` (str): Name of the person
- `frame` (Optional[np.ndarray]): Frame containing the face (provide either frame or encoding)
- `encoding` (Optional[np.ndarray]): Pre-computed face encoding
- `metadata` (Optional[Dict[str, Any]]): Additional metadata to store

**Returns:**
- `bool`: True if successful

**Notes:**
- If multiple faces are detected in frame, uses the first one
- Automatically saves database if `auto_save` is enabled

**Example:**
```python
# Add from current camera frame
api.add_face("John Doe")

# Add from specific image
image = cv2.imread("john.jpg")
api.add_face("John Doe", frame=image, metadata={"employee_id": "12345"})
```

##### `remove_face(name: str, remove_all: bool = True) -> bool`
Remove face(s) from database.

**Parameters:**
- `name` (str): Name of the person to remove
- `remove_all` (bool): If True, removes all instances. If False, removes only first.

**Returns:**
- `bool`: True if successful

##### `get_all_names() -> List[str]`
Get list of all unique names in database.

**Returns:**
- `List[str]`: List of unique names

##### `get_face_count(name: Optional[str] = None) -> int`
Get count of faces in database.

**Parameters:**
- `name` (Optional[str]): Specific name to count. If None, returns total count.

**Returns:**
- `int`: Number of faces

##### `clear_database() -> None`
Clear all faces from database.

##### `save_database(path: Optional[str] = None) -> bool`
Save database to file.

**Parameters:**
- `path` (Optional[str]): Save path. If None, uses config default.

**Returns:**
- `bool`: True if successful

##### `load_database(path: Optional[str] = None) -> bool`
Load database from file.

**Parameters:**
- `path` (Optional[str]): Load path. If None, uses config default.

**Returns:**
- `bool`: True if successful

#### Batch Operations

##### `add_faces_from_directory(directory: str, pattern: str = "*.jpg") -> Dict[str, int]`
Add faces from a directory of images.

**Parameters:**
- `directory` (str): Directory containing images
- `pattern` (str): File pattern to match

**Returns:**
- `Dict[str, int]`: Dictionary mapping names to number of faces added

**Notes:**
- Extracts person name from filename (without extension)
- Skips files that cannot be loaded

**Example:**
```python
# Directory structure:
# faces/
#   john_doe.jpg
#   jane_smith.jpg
results = api.add_faces_from_directory("faces/")
# Results: {"john_doe": 1, "jane_smith": 1}
```

##### `recognize_faces_in_image(image_path: str) -> List[RecognitionResult]`
Recognize faces in a single image file.

**Parameters:**
- `image_path` (str): Path to image file

**Returns:**
- `List[RecognitionResult]`: List of recognition results

#### Event Callbacks

##### `register_callback(event: str, callback: Callable) -> None`
Register a callback for an event.

**Parameters:**
- `event` (str): Event name
- `callback` (Callable): Callback function

**Available Events:**
- `on_face_detected`: Triggered when faces are detected
- `on_face_recognized`: Triggered when faces are recognized
- `on_face_added`: Triggered when face is added to database
- `on_face_removed`: Triggered when face is removed from database
- `on_error`: Triggered on errors

**Example:**
```python
def on_recognized(results):
    for result in results:
        print(f"Recognized: {result.name}")

api.register_callback('on_face_recognized', on_recognized)
```

##### `unregister_callback(event: str, callback: Callable) -> None`
Unregister a callback.

#### Utility Methods

##### `get_config() -> RecognitionConfig`
Get current configuration object.

##### `update_config(**kwargs) -> None`
Update configuration parameters.

**Example:**
```python
api.update_config(tolerance=0.5, process_every_n_frames=2)
```

##### `get_statistics() -> Dict[str, Any]`
Get API statistics.

**Returns:**
```python
{
    'total_faces': int,           # Total faces in database
    'unique_people': int,         # Number of unique people
    'frames_processed': int,      # Total frames processed
    'camera_running': bool,       # Camera status
    'config': {                   # Current configuration
        'tolerance': float,
        'model': str,
        'camera_type': str
    }
}
```

#### Context Manager Support

The API supports context manager protocol for automatic cleanup:

```python
with FaceRecognitionAPI() as api:
    api.start_camera()
    # ... do recognition ...
    # Camera automatically stopped and database saved on exit
```

### RecognitionConfig

Configuration dataclass for the Face Recognition API.

```python
@dataclass
class RecognitionConfig:
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
```

#### Configuration Parameters

##### Recognition Parameters
- `tolerance` (float): Recognition tolerance (0.0-1.0). Lower values are more strict. Default: 0.6
- `model` (RecognitionModel): Face detection model (HOG or CNN). HOG is faster, CNN is more accurate. Default: HOG
- `num_jitters` (int): Number of times to re-sample face when calculating encoding. Higher is more accurate but slower. Default: 1

##### Camera Parameters
- `camera_type` (CameraType): Type of camera to use. Default: OAK_D
- `camera_resolution` (Tuple[int, int]): Camera resolution. Default: (640, 480)
- `camera_fps` (int): Camera frames per second. Default: 30

##### Database Parameters
- `database_path` (str): Path to face database file. Default: "face_database.pkl"
- `auto_save` (bool): Automatically save database after modifications. Default: True

##### Processing Parameters
- `process_every_n_frames` (int): Process every Nth frame for performance. Default: 1
- `face_detection_scale` (float): Scale factor for face detection (smaller is faster). Default: 1.0
- `min_face_size` (int): Minimum face size in pixels. Default: 20

##### Thread Safety
- `enable_threading` (bool): Enable thread-safe operations. Default: True

##### Logging
- `log_level` (str): Logging level (DEBUG, INFO, WARNING, ERROR). Default: "INFO"

### Data Classes

#### FaceDetection
Represents a detected face.

```python
@dataclass
class FaceDetection:
    location: Tuple[int, int, int, int]  # (top, right, bottom, left)
    encoding: Optional[np.ndarray] = None
    landmarks: Optional[Dict[str, List[Tuple[int, int]]]] = None
```

#### RecognitionResult
Represents a face recognition result.

```python
@dataclass
class RecognitionResult:
    name: str
    confidence: float
    location: Tuple[int, int, int, int]
    encoding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Metadata Fields:**
- `distance`: Face distance from best match
- `match_index`: Index of best match in database

### Camera Interfaces

#### CameraInterface (Abstract Base Class)
Abstract base class for camera implementations.

```python
class CameraInterface(ABC):
    @abstractmethod
    def start(self) -> bool: ...
    
    @abstractmethod
    def stop(self) -> None: ...
    
    @abstractmethod
    def get_frame(self) -> Optional[np.ndarray]: ...
    
    @abstractmethod
    def is_running(self) -> bool: ...
    
    @property
    @abstractmethod
    def resolution(self) -> Tuple[int, int]: ...
```

#### OakDCamera
Implementation for OAK-D Series 3 cameras using DepthAI V3 API.

```python
class OakDCamera(CameraInterface):
    def __init__(self, resolution: Tuple[int, int] = (640, 480))
```

**Thread Safety:** All methods are thread-safe with internal locking.

#### WebcamCamera
Implementation for standard webcams using OpenCV.

```python
class WebcamCamera(CameraInterface):
    def __init__(self, camera_index: int = 0, resolution: Tuple[int, int] = (640, 480))
```

**Parameters:**
- `camera_index` (int): Camera index for OpenCV. Default: 0
- `resolution` (Tuple[int, int]): Desired resolution. Default: (640, 480)

### Database Manager

#### FaceDatabaseManager
Manages face encodings database with thread-safe operations.

```python
class FaceDatabaseManager:
    def __init__(self, database_path: str = "face_database.pkl", auto_save: bool = True)
```

**Features:**
- Thread-safe operations with RLock
- Automatic saving on modifications (configurable)
- Backward compatibility with older database formats
- Metadata support for each face

**Database Format:**
```python
{
    'encodings': List[np.ndarray],
    'names': List[str],
    'metadata': List[Dict[str, Any]],
    'version': '2.0'
}
```

### Face Detector

#### FaceDetector
Handles face detection operations.

```python
class FaceDetector:
    def __init__(self, model: RecognitionModel = RecognitionModel.HOG,
                 scale: float = 1.0, min_face_size: int = 20)
```

**Methods:**

##### `detect_faces(frame: np.ndarray, return_encodings: bool = True, num_jitters: int = 1) -> List[FaceDetection]`
Detect faces and optionally compute encodings.

##### `detect_face_landmarks(frame: np.ndarray, face_location: Optional[Tuple[int, int, int, int]] = None) -> Optional[Dict]`
Detect facial landmarks (eyes, nose, mouth, etc.).

**Performance Optimization:**
- Frame scaling for faster detection
- Minimum face size filtering
- Automatic RGB conversion

### Recognition Engine

#### FaceRecognizerEngine
Handles face recognition/matching operations.

```python
class FaceRecognizerEngine:
    def __init__(self, tolerance: float = 0.6)
```

**Features:**
- Recognition caching for performance
- Configurable tolerance
- Distance-based confidence scoring

**Cache Details:**
- TTL: 1.0 second (configurable)
- Automatic cleanup of expired entries
- Significant performance improvement for real-time recognition

## Enumerations

### RecognitionModel
Supported face recognition models.

```python
class RecognitionModel(Enum):
    HOG = "hog"    # Histogram of Oriented Gradients (faster)
    CNN = "cnn"    # Convolutional Neural Network (more accurate)
```

### CameraType
Supported camera types.

```python
class CameraType(Enum):
    OAK_D = "oak_d"
    WEBCAM = "webcam"
    VIDEO_FILE = "video_file"  # Future implementation
```

## Factory Functions

### create_face_recognition_api
Factory function to create Face Recognition API with common settings.

```python
def create_face_recognition_api(
    database_path: str = "face_database.pkl",
    camera_type: CameraType = CameraType.OAK_D,
    tolerance: float = 0.6,
    **kwargs
) -> FaceRecognitionAPI
```

**Parameters:**
- `database_path` (str): Path to face database
- `camera_type` (CameraType): Type of camera to use
- `tolerance` (float): Recognition tolerance (0.0 to 1.0)
- `**kwargs`: Additional configuration parameters

**Example:**
```python
api = create_face_recognition_api(
    database_path="my_faces.pkl",
    camera_type=CameraType.WEBCAM,
    tolerance=0.5,
    process_every_n_frames=2,
    log_level="DEBUG"
)
```

## Error Handling

The API uses Python's logging module for error reporting and debugging.

### Logging Levels
- `DEBUG`: Detailed information for debugging
- `INFO`: General informational messages
- `WARNING`: Warning messages for potential issues
- `ERROR`: Error messages for failures

### Error Callbacks
Register an error callback to handle errors programmatically:

```python
def handle_error(error):
    print(f"Error occurred: {error}")
    # Log to file, send alert, etc.

api.register_callback('on_error', handle_error)
```

### Common Errors
1. **Camera Initialization Failure**: Camera not connected or in use
2. **Face Detection Failure**: Invalid frame format or corrupted image
3. **Database Load/Save Failure**: File permissions or corruption
4. **Recognition Failure**: No faces in database or invalid encodings

## Thread Safety

### Thread-Safe Operations
All public methods of FaceRecognitionAPI are thread-safe when `enable_threading` is True in configuration.

### Locking Strategy
- Uses `threading.RLock` for recursive locking
- Database operations are atomic
- Camera operations are synchronized
- Recognition cache is thread-safe

### Multi-threaded Usage Example
```python
import threading

api = FaceRecognitionAPI()

def recognition_thread():
    while True:
        results = api.process_frame()
        # Process results

def database_thread():
    # Periodically update database
    api.add_face("New Person")

# Start threads
threading.Thread(target=recognition_thread).start()
threading.Thread(target=database_thread).start()
```

### Performance Considerations

#### Frame Processing
- Use `process_every_n_frames` to reduce CPU usage
- Scale down frames with `face_detection_scale` for faster detection
- Enable recognition cache for real-time applications

#### Model Selection
- **HOG Model**: 
  - Faster (5-10ms per frame)
  - Lower accuracy
  - Good for real-time applications
  - Works on CPU

- **CNN Model**:
  - Slower (50-200ms per frame)
  - Higher accuracy
  - Better for batch processing
  - Benefits from GPU acceleration

#### Database Size
- Performance degrades linearly with database size
- Consider using multiple smaller databases for different contexts
- Regularly clean up duplicate encodings

#### Memory Management
- Frame buffers are automatically managed
- Recognition cache has automatic cleanup
- Database is loaded into memory for fast access

### Best Practices

1. **Resource Management**
   - Always use context managers or ensure proper cleanup
   - Stop camera when not in use
   - Save database regularly

2. **Configuration Tuning**
   - Adjust tolerance based on use case (0.4-0.6 typical)
   - Use frame skipping for non-critical applications
   - Scale detection for performance vs accuracy trade-off

3. **Error Handling**
   - Always check return values
   - Register error callbacks for production
   - Implement retry logic for camera operations

4. **Database Management**
   - Keep backups of face database
   - Implement versioning for database updates
   - Clean up old/duplicate entries periodically

5. **Security Considerations**
   - Encrypt database files in production
   - Implement access controls
   - Log all recognition events for audit
   - Consider privacy regulations (GDPR, etc.)