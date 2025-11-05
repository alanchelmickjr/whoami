# Face Recognition API Usage Guide

## Table of Contents
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Common Use Cases](#common-use-cases)
  - [Basic Face Recognition Setup](#basic-face-recognition-setup)
  - [Adding Faces to the Database](#adding-faces-to-the-database)
  - [Real-Time Face Recognition](#real-time-face-recognition)
  - [Batch Processing Images](#batch-processing-images)
  - [Using Different Camera Types](#using-different-camera-types)
  - [Event Callback Usage](#event-callback-usage)
  - [Multi-Threaded Applications](#multi-threaded-applications)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Performance Tuning](#performance-tuning)

## Quick Start

The Face Recognition API provides a simple, intuitive interface for face recognition tasks. Here's how to get started in just a few lines of code:

```python
from whoami.face_recognition_api import create_face_recognition_api

# Create API instance
api = create_face_recognition_api()

# Start camera and recognize faces
with api:
    api.start_camera()
    
    # Add a face to database
    print("Position your face and press Enter...")
    input()
    api.add_face("Your Name")
    
    # Start recognition
    while True:
        results = api.process_frame()
        for result in results:
            print(f"Recognized: {result.name} (Confidence: {result.confidence:.2f})")
```

## Installation

### Prerequisites

1. **Python 3.7+**
2. **Required packages:**
   ```bash
   pip install numpy opencv-python face-recognition depthai
   ```

3. **For OAK-D Camera:**
   - Install DepthAI: `pip install depthai`
   - Ensure OAK-D camera is connected via USB 3.0

4. **For Webcam:**
   - OpenCV with camera support
   - Camera permissions enabled

### Import the API

```python
from whoami.face_recognition_api import (
    FaceRecognitionAPI,
    RecognitionConfig,
    CameraType,
    RecognitionModel,
    create_face_recognition_api
)
```

## Common Use Cases

### Basic Face Recognition Setup

#### Simple Setup with Defaults

```python
from whoami.face_recognition_api import create_face_recognition_api

# Create API with default settings
api = create_face_recognition_api()

# Start using the API
api.start_camera()
# ... perform recognition tasks
api.stop_camera()
```

#### Custom Configuration

```python
from whoami.face_recognition_api import FaceRecognitionAPI, RecognitionConfig, CameraType, RecognitionModel

# Create custom configuration
config = RecognitionConfig(
    tolerance=0.5,                         # Stricter matching
    model=RecognitionModel.CNN,            # More accurate model
    camera_type=CameraType.WEBCAM,         # Use webcam
    camera_resolution=(1280, 720),         # HD resolution
    process_every_n_frames=2,               # Process every 2nd frame
    face_detection_scale=0.5,               # Scale down for faster detection
    database_path="my_faces.pkl",          # Custom database path
    auto_save=True,                         # Auto-save changes
    log_level="DEBUG"                       # Detailed logging
)

# Create API with custom config
api = FaceRecognitionAPI(config)
```

### Adding Faces to the Database

#### Add Face from Camera

```python
# Method 1: Add from current camera frame
api.start_camera()
print("Position your face in front of the camera and press Enter...")
input()

if api.add_face("John Doe"):
    print("Face added successfully!")
else:
    print("Failed to add face - ensure a face is visible")
```

#### Add Face from Image File

```python
import cv2

# Method 2: Add from image file
image = cv2.imread("photos/john_doe.jpg")
if api.add_face("John Doe", frame=image):
    print("Face added from image")

# With metadata
api.add_face(
    name="Jane Smith",
    frame=image,
    metadata={
        "employee_id": "EMP001",
        "department": "Engineering",
        "date_added": "2024-01-15"
    }
)
```

#### Batch Add from Directory

```python
# Add all faces from a directory
# Files should be named as: person_name.jpg
results = api.add_faces_from_directory(
    directory="employee_photos/",
    pattern="*.jpg"
)

print(f"Added faces: {results}")
# Output: {'john_doe': 1, 'jane_smith': 1, ...}
```

#### Add Multiple Photos of Same Person

```python
# Add multiple angles/photos for better recognition
import os

person_name = "John Doe"
photos_dir = "john_photos/"

for photo_file in os.listdir(photos_dir):
    if photo_file.endswith(".jpg"):
        image = cv2.imread(os.path.join(photos_dir, photo_file))
        api.add_face(person_name, frame=image)
        print(f"Added {photo_file} for {person_name}")
```

### Real-Time Face Recognition

#### Basic Real-Time Recognition

```python
import cv2
from whoami.face_recognition_api import create_face_recognition_api

api = create_face_recognition_api()
api.start_camera()

# Load some faces first
api.load_database("known_faces.pkl")

print("Starting real-time recognition. Press 'q' to quit.")

while True:
    # Get frame from camera
    frame = api.get_frame()
    if frame is None:
        continue
    
    # Process frame for recognition
    results = api.process_frame(frame)
    
    # Draw results on frame
    for result in results:
        # Get face location
        top, right, bottom, left = result.location
        
        # Draw rectangle around face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Add name label
        label = f"{result.name} ({result.confidence:.2f})"
        cv2.putText(frame, label, (left, top - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display frame
    cv2.imshow('Face Recognition', frame)
    
    # Check for quit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

api.stop_camera()
cv2.destroyAllWindows()
```

#### Optimized Real-Time Recognition

```python
# Optimize for performance
config = RecognitionConfig(
    process_every_n_frames=3,      # Process every 3rd frame
    face_detection_scale=0.5,       # Scale down for detection
    model=RecognitionModel.HOG,     # Faster model
    tolerance=0.6                   # Standard tolerance
)

api = FaceRecognitionAPI(config)
api.start_camera()

# Track recognized faces to reduce redundant processing
last_recognized = {}
recognition_timeout = 2.0  # seconds

import time

while True:
    frame = api.get_frame()
    if frame is None:
        continue
    
    current_time = time.time()
    results = api.process_frame(frame)
    
    for result in results:
        # Check if recently recognized
        if result.name in last_recognized:
            if current_time - last_recognized[result.name] < recognition_timeout:
                continue  # Skip if recently recognized
        
        # New or timeout recognition
        last_recognized[result.name] = current_time
        print(f"Recognized: {result.name} at {current_time}")
        
        # Trigger action (e.g., log entry, door unlock, etc.)
        if result.name != "Unknown" and result.confidence > 0.8:
            print(f"High confidence match for {result.name}")
```

### Batch Processing Images

#### Process Single Image

```python
# Recognize faces in a single image
results = api.recognize_faces_in_image("group_photo.jpg")

for result in results:
    print(f"Found: {result.name} at position {result.location}")
```

#### Process Multiple Images

```python
import glob
import cv2

# Process all images in a directory
image_files = glob.glob("test_images/*.jpg")

for image_file in image_files:
    print(f"\nProcessing: {image_file}")
    
    results = api.recognize_faces_in_image(image_file)
    
    if results:
        # Load image for annotation
        image = cv2.imread(image_file)
        
        for result in results:
            top, right, bottom, left = result.location
            
            # Draw rectangle
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Add label
            label = f"{result.name}"
            cv2.putText(image, label, (left, top - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Save annotated image
        output_file = image_file.replace(".jpg", "_annotated.jpg")
        cv2.imwrite(output_file, image)
        print(f"Saved annotated image: {output_file}")
    else:
        print(f"No faces found in {image_file}")
```

#### Batch Analysis with Statistics

```python
import json
from collections import defaultdict

def batch_analyze_images(api, image_directory):
    """Analyze all images and generate statistics"""
    
    stats = {
        'total_images': 0,
        'total_faces': 0,
        'people_count': defaultdict(int),
        'unknown_faces': 0,
        'processing_times': []
    }
    
    import time
    image_files = glob.glob(f"{image_directory}/*.jpg")
    
    for image_file in image_files:
        start_time = time.time()
        
        results = api.recognize_faces_in_image(image_file)
        
        processing_time = time.time() - start_time
        stats['processing_times'].append(processing_time)
        stats['total_images'] += 1
        stats['total_faces'] += len(results)
        
        for result in results:
            if result.name == "Unknown":
                stats['unknown_faces'] += 1
            else:
                stats['people_count'][result.name] += 1
    
    # Calculate averages
    if stats['processing_times']:
        stats['avg_processing_time'] = sum(stats['processing_times']) / len(stats['processing_times'])
        stats['avg_faces_per_image'] = stats['total_faces'] / stats['total_images']
    
    return stats

# Run analysis
stats = batch_analyze_images(api, "event_photos/")

# Save statistics
with open('analysis_results.json', 'w') as f:
    json.dump(stats, f, indent=2)

print(f"Analyzed {stats['total_images']} images")
print(f"Found {stats['total_faces']} faces")
print(f"Identified people: {dict(stats['people_count'])}")
```

### Using Different Camera Types

#### OAK-D Camera Setup

```python
from whoami.face_recognition_api import create_face_recognition_api, CameraType

# Create API for OAK-D camera
api = create_face_recognition_api(
    camera_type=CameraType.OAK_D,
    camera_resolution=(1920, 1080)  # Full HD
)

# Start OAK-D camera
if api.start_camera():
    print("OAK-D camera started successfully")
    
    # OAK-D specific features can be accessed through the camera object
    # The API handles all the complexity internally
    
    while True:
        frame = api.get_frame()
        if frame is not None:
            results = api.process_frame(frame)
            # Process results...
else:
    print("Failed to start OAK-D camera - check connection")
```

#### Webcam Setup

```python
# Create API for webcam
api = create_face_recognition_api(
    camera_type=CameraType.WEBCAM,
    camera_resolution=(640, 480)
)

# Start webcam
if api.start_camera():
    print("Webcam started successfully")
    # Use API normally...
```

#### Switching Between Cameras

```python
# Start with webcam
api = create_face_recognition_api()
api.start_camera(CameraType.WEBCAM)

# ... do some processing ...

# Switch to OAK-D
api.stop_camera()
api.start_camera(CameraType.OAK_D)

# ... continue processing ...
```

### Event Callback Usage

#### Basic Callbacks

```python
# Define callback functions
def on_face_detected(detections):
    """Called when faces are detected"""
    print(f"Detected {len(detections)} face(s)")
    for detection in detections:
        print(f"  Face at {detection.location}")

def on_face_recognized(results):
    """Called when faces are recognized"""
    for result in results:
        if result.name != "Unknown":
            print(f"✓ Recognized: {result.name} (Confidence: {result.confidence:.2f})")
        else:
            print(f"✗ Unknown face detected")

def on_face_added(name):
    """Called when a face is added to database"""
    print(f"📝 Added {name} to database")
    # Could trigger additional actions like sending notification

def on_error(error):
    """Called when an error occurs"""
    print(f"❌ Error: {error}")
    # Log to file, send alert, etc.

# Register callbacks
api.register_callback('on_face_detected', on_face_detected)
api.register_callback('on_face_recognized', on_face_recognized)
api.register_callback('on_face_added', on_face_added)
api.register_callback('on_error', on_error)

# Use API normally - callbacks will be triggered automatically
api.start_camera()
api.add_face("Test User")
results = api.process_frame()
```

#### Advanced Callback Usage

```python
import datetime
import json

class RecognitionLogger:
    """Custom logger using callbacks"""
    
    def __init__(self, log_file="recognition_log.json"):
        self.log_file = log_file
        self.log_data = []
    
    def on_recognized(self, results):
        """Log recognition events"""
        for result in results:
            if result.name != "Unknown":
                entry = {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'name': result.name,
                    'confidence': result.confidence,
                    'location': result.location
                }
                self.log_data.append(entry)
                
                # Save to file
                with open(self.log_file, 'w') as f:
                    json.dump(self.log_data, f, indent=2)
    
    def on_error(self, error):
        """Log errors"""
        entry = {
            'timestamp': datetime.datetime.now().isoformat(),
            'type': 'error',
            'message': str(error)
        }
        self.log_data.append(entry)

# Create logger and register callbacks
logger = RecognitionLogger()
api.register_callback('on_face_recognized', logger.on_recognized)
api.register_callback('on_error', logger.on_error)
```

#### Event-Driven Actions

```python
class AccessControlSystem:
    """Example access control using callbacks"""
    
    def __init__(self, api):
        self.api = api
        self.authorized_users = ["John Doe", "Jane Smith"]
        self.access_log = []
        
        # Register callback
        api.register_callback('on_face_recognized', self.check_access)
    
    def check_access(self, results):
        """Check if recognized person has access"""
        for result in results:
            if result.name in self.authorized_users and result.confidence > 0.8:
                self.grant_access(result.name)
            elif result.name != "Unknown":
                self.deny_access(result.name)
    
    def grant_access(self, name):
        """Grant access to authorized person"""
        print(f"✅ ACCESS GRANTED: {name}")
        self.access_log.append({
            'time': datetime.datetime.now(),
            'name': name,
            'status': 'granted'
        })
        # Trigger door unlock, etc.
    
    def deny_access(self, name):
        """Deny access to unauthorized person"""
        print(f"❌ ACCESS DENIED: {name}")
        self.access_log.append({
            'time': datetime.datetime.now(),
            'name': name,
            'status': 'denied'
        })
        # Trigger alert, etc.

# Set up access control
access_system = AccessControlSystem(api)
api.start_camera()

# System will automatically handle access control
while True:
    api.process_frame()
    time.sleep(0.1)
```

### Multi-Threaded Applications

#### Basic Threading

```python
import threading
import queue
import time

class ThreadedFaceRecognition:
    def __init__(self, api):
        self.api = api
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue()
        self.running = False
    
    def capture_thread(self):
        """Thread for capturing frames"""
        while self.running:
            frame = self.api.get_frame()
            if frame is not None:
                try:
                    self.frame_queue.put(frame, timeout=0.1)
                except queue.Full:
                    pass  # Drop frame if queue is full
    
    def recognition_thread(self):
        """Thread for processing frames"""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                results = self.api.process_frame(frame)
                if results:
                    self.result_queue.put(results)
            except queue.Empty:
                pass
    
    def display_thread(self):
        """Thread for displaying results"""
        while self.running:
            try:
                results = self.result_queue.get(timeout=0.1)
                for result in results:
                    print(f"Recognized: {result.name} ({result.confidence:.2f})")
            except queue.Empty:
                pass
    
    def start(self):
        """Start all threads"""
        self.running = True
        self.api.start_camera()
        
        # Start threads
        threading.Thread(target=self.capture_thread, daemon=True).start()
        threading.Thread(target=self.recognition_thread, daemon=True).start()
        threading.Thread(target=self.display_thread, daemon=True).start()
    
    def stop(self):
        """Stop all threads"""
        self.running = False
        self.api.stop_camera()

# Use threaded recognition
api = create_face_recognition_api()
threaded_system = ThreadedFaceRecognition(api)

threaded_system.start()
time.sleep(30)  # Run for 30 seconds
threaded_system.stop()
```

#### Producer-Consumer Pattern

```python
import concurrent.futures
import threading

class FaceRecognitionPipeline:
    def __init__(self, api, num_workers=2):
        self.api = api
        self.num_workers = num_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
        self.results_lock = threading.Lock()
        self.all_results = []
    
    def process_frame_async(self, frame):
        """Process frame asynchronously"""
        future = self.executor.submit(self.api.process_frame, frame)
        future.add_done_callback(self.handle_result)
        return future
    
    def handle_result(self, future):
        """Handle recognition result"""
        try:
            results = future.result()
            with self.results_lock:
                self.all_results.extend(results)
                # Process results
                for result in results:
                    if result.name != "Unknown":
                        print(f"Found: {result.name}")
        except Exception as e:
            print(f"Error processing frame: {e}")
    
    def process_video(self, video_path):
        """Process video file with multiple workers"""
        cap = cv2.VideoCapture(video_path)
        futures = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Submit frame for processing
            future = self.process_frame_async(frame)
            futures.append(future)
            
            # Limit number of pending futures
            if len(futures) > self.num_workers * 2:
                concurrent.futures.wait(futures[:self.num_workers])
                futures = futures[self.num_workers:]
        
        # Wait for all remaining futures
        concurrent.futures.wait(futures)
        cap.release()
        
        return self.all_results

# Use pipeline
api = create_face_recognition_api()
pipeline = FaceRecognitionPipeline(api, num_workers=4)

results = pipeline.process_video("surveillance_video.mp4")
print(f"Processed video, found {len(results)} faces")
```

## Best Practices

### 1. Resource Management

Always properly manage resources using context managers or explicit cleanup:

```python
# Good: Using context manager
with create_face_recognition_api() as api:
    api.start_camera()
    # ... do work ...
    # Camera automatically stopped and database saved

# Good: Explicit cleanup
api = create_face_recognition_api()
try:
    api.start_camera()
    # ... do work ...
finally:
    api.stop_camera()
    api.save_database()
```

### 2. Database Management

```python
# Regular backups
import shutil
import datetime

def backup_database(api):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"backups/face_db_{timestamp}.pkl"
    api.save_database(backup_path)
    print(f"Database backed up to {backup_path}")

# Clean duplicate entries
def clean_database(api):
    """Remove duplicate encodings for same person"""
    names = api.get_all_names()
    for name in names:
        count = api.get_face_count(name)
        if count > 5:  # Keep max 5 encodings per person
            # Remove and re-add limited number
            # (Implementation depends on your needs)
            pass
```

### 3. Performance Optimization

```python
# For real-time applications
config = RecognitionConfig(
    model=RecognitionModel.HOG,        # Faster model
    process_every_n_frames=3,           # Skip frames
    face_detection_scale=0.5,           # Scale down detection
    tolerance=0.6                       # Standard tolerance
)

# For accuracy-critical applications
config = RecognitionConfig(
    model=RecognitionModel.CNN,         # More accurate
    num_jitters=2,                      # Better encoding
    process_every_n_frames=1,           # Process all frames
    face_detection_scale=1.0,           # Full resolution
    tolerance=0.4                       # Stricter matching
)
```

### 4. Error Handling

```python
def safe_recognition(api, max_retries=3):
    """Robust recognition with retry logic"""
    for attempt in range(max_retries):
        try:
            frame = api.get_frame()
            if frame is None:
                time.sleep(0.1)
                continue
            
            results = api.process_frame(frame)
            return results
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(0.5)
    
    return []
```

### 5. Security Considerations

```python
import hashlib
import json

class SecureFaceDatabase:
    """Example of adding security layers"""
    
    def __init__(self, api):
        self.api = api
        self.audit_log = []
    
    def add_face_with_audit(self, name, frame, added_by):
        """Add face with audit trail"""
        # Generate hash of encoding for verification
        success = self.api.add_face(name, frame)
        
        if success:
            self.audit_log.append({
                'action': 'add_face',
                'name': name,
                'added_by': added_by,
                'timestamp': datetime.datetime.now().isoformat()
            })
            
            # Save audit log
            with open('audit_log.json', 'w') as f:
                json.dump(self.audit_log, f)
        
        return success
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Camera Not Starting

```python
# Check camera availability
import platform

def diagnose_camera(api):
    """Diagnose camera issues"""
    
    # Try different camera types
    camera_types = [CameraType.OAK_D, CameraType.WEBCAM]
    
    for cam_type in camera_types:
        print(f"Testing {cam_type.value}...")
        if api.start_camera(cam_type):
            print(f"✓ {cam_type.value} works!")
            api.stop_camera()
        else:
            print(f"✗ {cam_type.value} failed")
    
    # Check system
    print(f"System: {platform.system()}")
    print(f"Python: {platform.python_version()}")

# Run diagnosis
api = create_face_recognition_api()
diagnose_camera(api)
```

#### 2. Poor Recognition Accuracy

```python
def test_recognition_settings(api, test_image):
    """Test different tolerance settings"""
    
    tolerances = [0.4, 0.5, 0.6, 0.7]
    
    for tolerance in tolerances:
        api.update_config(tolerance=tolerance)
        results = api.recognize_faces_in_image(test_image)
        
        print(f"Tolerance {tolerance}:")
        for result in results:
            print(f"  {result.name}: {result.confidence:.2f}")
```

#### 3. Performance Issues

```python
def measure_performance(api):
    """Measure processing performance"""
    import time
    
    # Test different configurations
    configs = [
        {"process_every_n_frames": 1, "face_detection_scale": 1.0},
        {"process_every_n_frames": 2, "face_detection_scale": 0.5},
        {"process_every_n_frames": 3, "face_detection_scale": 0.25},
    ]
    
    for config in configs:
        api.update_config(**config)
        
        start_time = time.time()
        frames_processed = 0
        
        while time.time() - start_time < 10:  # Test for 10 seconds
            frame = api.get_frame()
            if frame is not None:
                api.process_frame(frame)
                frames_processed += 1
        
        fps = frames_processed / 10
        print(f"Config {config}: {fps:.2f} FPS")
```

#### 4. Database Issues

```python
def verify_database(api):
    """Verify database integrity"""
    
    # Check database
    print(f"Total faces: {api.get_face_count()}")
    print(f"Unique people: {len(api.get_all_names())}")
    
    # Test save/load
    test_path = "test_db.pkl"
    if api.save_database(test_path):
        print("✓ Database save works")
        
        # Clear and reload
        api.clear_database()
        if api.load_database(test_path):
            print("✓ Database load works")
            print(f"Reloaded {api.get_face_count()} faces")
    
    # Cleanup
    import os
    if os.path.exists(test_path):
        os.remove(test_path)
```

## Performance Tuning

### Configuration Guidelines

| Use Case | Model | Tolerance | Scale | Frame Skip |
|----------|-------|-----------|-------|------------|
| Real-time monitoring | HOG | 0.6 | 0.5 | 2-3 |
| High accuracy | CNN | 0.4 | 1.0 | 1 |
| Battery-powered | HOG | 0.6 | 0.25 | 5 |
| Large database | HOG | 0.5 | 0.5 | 2 |

### Memory Optimization

```python
# For systems with limited memory
config = RecognitionConfig(
    face_detection_scale=0.25,          # Reduce memory usage
    process_every_n_frames=5,            # Process fewer frames
    camera_resolution=(320, 240)         # Lower resolution
)

# Clear cache periodically
def periodic_cleanup(api, interval=60):
    """Clear cache periodically to free memory"""
    import threading
    
    def cleanup():
        while True:
            time.sleep(interval)
            api.recognizer.clear_cache()
            print("Cache cleared")
    
    threading.Thread(target=cleanup, daemon=True).start()
```

### CPU Optimization

```python
# Optimize for multi-core systems
import multiprocessing

def optimal_config():
    """Get optimal configuration for system"""
    cores = multiprocessing.cpu_count()
    
    if cores >= 8:
        # High-end system
        return RecognitionConfig(
            model=RecognitionModel.CNN,
            process_every_n_frames=1,
            face_detection_scale=1.0
        )
    elif cores >= 4:
        # Mid-range system
        return RecognitionConfig(
            model=RecognitionModel.HOG,
            process_every_n_frames=2,
            face_detection_scale=0.5
        )
    else:
        # Low-end system
        return RecognitionConfig(
            model=RecognitionModel.HOG,
            process_every_n_frames=3,
            face_detection_scale=0.25
        )

api = FaceRecognitionAPI(optimal_config())
```

### Network Optimization (for remote cameras)

```python
class RemoteCameraOptimizer:
    """Optimize for network cameras"""
    
    def __init__(self, api):
        self.api = api
        self.frame_buffer = []
        self.buffer_size = 5
    
    def buffered_recognition(self):
        """Buffer frames for batch processing"""
        while True:
            frame = self.api.get_frame()
            if frame is not None:
                self.frame_buffer.append(frame)
                
                if len(self.frame_buffer) >= self.buffer_size:
                    # Process batch
                    for buffered_frame in self.frame_buffer:
                        self.api.process_frame(buffered_frame)
                    
                    self.frame_buffer.clear()
```

## Advanced Tips

### 1. Custom Face Encodings

```python
# Generate more robust encodings
def create_robust_encoding(api, name, images):
    """Create encoding from multiple images"""
    encodings = []
    
    for image_path in images:
        image = cv2.imread(image_path)
        detections = api.detect_faces(image, compute_encodings=True)
        
        if detections and detections[0].encoding is not None:
            encodings.append(detections[0].encoding)
    
    if encodings:
        # Average encoding for robustness
        average_encoding = np.mean(encodings, axis=0)
        api.add_face(name, encoding=average_encoding)
        return True
    
    return False
```

### 2. Dynamic Tolerance Adjustment

```python
class AdaptiveRecognition:
    """Dynamically adjust tolerance based on conditions"""
    
    def __init__(self, api):
        self.api = api
        self.base_tolerance = 0.6
        self.lighting_factor = 1.0
    
    def adjust_for_lighting(self, frame):
        """Adjust tolerance based on lighting conditions"""
        # Calculate average brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        brightness = np.mean(gray)
        
        if brightness < 50:  # Dark
            self.lighting_factor = 1.2
        elif brightness > 200:  # Bright
            self.lighting_factor = 1.1
        else:  # Normal
            self.lighting_factor = 1.0
        
        new_tolerance = self.base_tolerance * self.lighting_factor
        self.api.update_config(tolerance=min(new_tolerance, 0.8))
```

### 3. Integration with Other Systems

```python
# Example: Integration with notification system
class NotificationIntegration:
    def __init__(self, api):
        self.api = api
        api.register_callback('on_face_recognized', self.send_notification)
    
    def send_notification(self, results):
        for result in results:
            if result.name != "Unknown" and result.confidence > 0.9:
                # Send notification (email, SMS, webhook, etc.)
                self.notify_arrival(result.name)
    
    def notify_arrival(self, name):
        # Implement notification logic
        print(f"NOTIFICATION: {name} has arrived")
        # Could integrate with:
        # - Email services
        # - SMS gateways
        # - Slack/Discord webhooks
        # - Home automation systems
```

This comprehensive usage guide covers all major use cases and provides practical examples for implementing face recognition in various scenarios. The guide includes best practices, troubleshooting tips, and performance optimization strategies to help developers make the most of the Face Recognition API.