# WhoAmI - Facial Recognition System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Oak D](https://img.shields.io/badge/Oak%20D-Series%203-green.svg)](https://docs.luxonis.com/)

A simple, portable, and modular facial recognition framework designed for Oak D Series 3 cameras and Jetson Orin Nano. Built with security and ease of use in mind.

## Features

- 🎥 **Oak D Series 3 Support**: Optimized for Oak D cameras using DepthAI SDK
- 🖥️ **Simple GUI**: Easy-to-use graphical interface built with tkinter
- 💻 **CLI Interface**: Command-line interface for headless operation
- 👤 **Face Management**: Add, remove, and manage known faces
- 🔒 **Secure**: Local processing, no cloud dependencies
- 🚀 **Portable**: Lightweight and modular design for easy integration
- 🤖 **Robot-Ready**: Perfect for robotics applications on Jetson platforms
- 🔌 **New Class-Based API**: Clean, modular, and thread-safe face recognition library
- 📊 **Batch Processing**: Process multiple images and videos efficiently
- 🔔 **Event Callbacks**: React to face detection and recognition events
- 🧵 **Thread-Safe**: Designed for multi-threaded applications

## Hardware Requirements

- Oak D Series 3 camera (or compatible DepthAI device)
- Jetson Orin Nano (or any system with USB 3.0)
- Python 3.8 or higher

## Installation

### Quick Install

```bash
# Clone the repository
git clone https://github.com/alanchelmickjr/whoami.git
cd whoami

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

### System Dependencies

For Jetson Orin Nano:
```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-tk cmake libopencv-dev
```

For face_recognition library dependencies:
```bash
sudo apt-get install -y build-essential cmake libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev
```

## Usage

### GUI Application

Start the graphical interface:

```bash
python run_gui.py
# Or if installed:
whoami-gui
```

**GUI Features:**
1. **Start Camera**: Initialize the Oak D camera
2. **Add Face**: Capture and store a new face
3. **Remove Face**: Delete a face from the database
4. **Clear All Faces**: Remove all stored faces
5. **Real-time Recognition**: Automatically recognize faces in the camera feed

### CLI Application

Use the command-line interface:

```bash
# List all known faces
python run_cli.py list

# Add a new face
python run_cli.py add "John Doe"

# Remove a face
python run_cli.py remove "John Doe"

# Run real-time recognition
python run_cli.py recognize

# Clear all faces
python run_cli.py clear
```

### 🆕 Face Recognition API (Refactored)

The new Face Recognition API provides a clean, class-based interface with improved modularity and features:

```python
from whoami.face_recognition_api import create_face_recognition_api

# Quick start with the new API
api = create_face_recognition_api()

# Start camera and process frames
with api:
    api.start_camera()
    
    # Add a face to database
    api.add_face("John Doe")
    
    # Real-time recognition
    while True:
        results = api.process_frame()
        for result in results:
            print(f"Recognized: {result.name} ({result.confidence:.2f})")
```

**Key Features of the New API:**
- **Separated Concerns**: Camera, detection, recognition, and database are separate components
- **Event-Driven**: Register callbacks for face detection, recognition, and other events
- **Thread-Safe**: Built-in support for multi-threaded applications
- **Flexible Configuration**: Extensive configuration options via `RecognitionConfig`
- **Multiple Camera Types**: Support for OAK-D and webcam with easy extensibility
- **Batch Processing**: Efficiently process directories of images or video files
- **Context Manager Support**: Automatic resource cleanup

See the comprehensive [API Reference](docs/API_REFERENCE.md) and [Usage Guide](docs/USAGE_GUIDE.md) for detailed documentation.

## Architecture

The system is designed with modularity in mind:

```
whoami/
├── whoami/
│   ├── __init__.py                  # Package initialization
│   ├── face_recognizer.py           # Original core face recognition logic
│   ├── face_recognition_api.py      # 🆕 New refactored API
│   ├── gui.py                       # GUI application
│   ├── cli.py                       # CLI application
│   └── config.py                    # Configuration management
├── docs/
│   ├── API_REFERENCE.md             # 🆕 Complete API documentation
│   └── USAGE_GUIDE.md               # 🆕 Usage guide with examples
├── examples/
│   ├── api_basic_usage.py          # 🆕 Basic API usage examples
│   ├── api_advanced_features.py    # 🆕 Advanced features demo
│   ├── api_robotics_integration.py # 🆕 Robotics integration examples
│   └── api_batch_processing.py     # 🆕 Batch processing examples
├── run_gui.py                      # GUI entry point
├── run_cli.py                      # CLI entry point
├── requirements.txt                 # Python dependencies
└── setup.py                        # Installation script
```

### Core Components

1. **FaceRecognizer** (`face_recognizer.py`): Core facial recognition engine
   - Camera interface via DepthAI
   - Face detection and encoding
   - Face database management
   - Recognition with confidence scores

2. **GUI** (`gui.py`): Tkinter-based graphical interface
   - Live camera feed display
   - Face management controls
   - Visual feedback for recognition

3. **CLI** (`cli.py`): Command-line interface
   - Headless operation support
   - Scriptable face management
   - Real-time recognition mode

4. **Config** (`config.py`): Configuration system
   - Camera settings
   - Recognition parameters
   - GUI preferences

## Integration

This framework is designed to be easily integrated into larger robotics systems. You can use either the original API or the new refactored API:

### Using the New Face Recognition API (Recommended)

```python
from whoami.face_recognition_api import (
    FaceRecognitionAPI,
    RecognitionConfig,
    CameraType,
    RecognitionModel
)

# Configure the API
config = RecognitionConfig(
    camera_type=CameraType.OAK_D,
    tolerance=0.5,
    model=RecognitionModel.HOG,  # or CNN for higher accuracy
    process_every_n_frames=2,     # Skip frames for performance
    database_path="my_faces.pkl"
)

# Initialize API
api = FaceRecognitionAPI(config)

# Use as context manager for automatic cleanup
with api:
    api.start_camera()
    
    # Add faces to database
    api.add_face("John Doe")
    
    # Process frames
    while True:
        results = api.process_frame()
        for result in results:
            if result.name != "Unknown":
                print(f"Recognized: {result.name} ({result.confidence:.2f})")
                print(f"Location: {result.location}")

# Or use event callbacks
def on_face_recognized(results):
    for result in results:
        print(f"Event: Recognized {result.name}")

api.register_callback('on_face_recognized', on_face_recognized)
```

### Using the Original API

```python
from whoami.face_recognizer import FaceRecognizer

# Initialize recognizer
recognizer = FaceRecognizer(database_path="my_faces.pkl")

# Start camera
recognizer.start_camera()

# Get frame and recognize
frame = recognizer.get_frame()
face_locations, face_encodings = recognizer.detect_faces(frame)
results = recognizer.recognize_faces(face_encodings)

# results contains [(name, confidence), ...]
for (name, confidence) in results:
    print(f"Detected: {name} (confidence: {confidence:.2f})")

# Stop camera
recognizer.stop_camera()
```

## Configuration

Create a `config.json` file to customize settings:

```json
{
    "camera": {
        "preview_width": 640,
        "preview_height": 480,
        "fps": 30
    },
    "recognition": {
        "tolerance": 0.6,
        "database_path": "face_database.pkl"
    },
    "gui": {
        "window_width": 1000,
        "window_height": 700,
        "video_width": 640,
        "video_height": 480
    }
}
```

## Security Features

- **Local Processing**: All face recognition happens on-device
- **No Cloud**: No internet connection required
- **Encrypted Storage**: Face encodings stored in binary format
- **Privacy First**: No images stored, only mathematical encodings

## Performance Tips

For Jetson Orin Nano:
- Enable maximum performance mode: `sudo nvpmodel -m 0`
- Set CPU to max frequency: `sudo jetson_clocks`
- Consider using CUDA acceleration for face_recognition library

## Troubleshooting

### Camera Not Detected
```bash
# Check if Oak D is connected
lsusb | grep Movidius

# Verify DepthAI installation
python -c "import depthai; print(depthai.__version__)"
```

### Performance Issues
- Reduce camera resolution in config
- Lower FPS setting
- Ensure adequate lighting for better detection

### Import Errors
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

## Development

To contribute or extend the framework:

```bash
# Install in development mode
pip install -e .

# Run tests (if available)
python -m pytest tests/

# Run example scripts
python examples/api_basic_usage.py
python examples/api_advanced_features.py
python examples/api_robotics_integration.py
python examples/api_batch_processing.py
```

## 📚 Documentation

- **[API Reference](docs/API_REFERENCE.md)**: Complete API documentation with all classes, methods, and parameters
- **[Usage Guide](docs/USAGE_GUIDE.md)**: Comprehensive guide with code examples and best practices
- **[Examples](examples/)**: Working example scripts demonstrating various features

### Example Scripts

1. **[api_basic_usage.py](examples/api_basic_usage.py)**: Basic face recognition operations
   - Starting/stopping camera
   - Adding faces to database
   - Real-time recognition
   - Database management

2. **[api_advanced_features.py](examples/api_advanced_features.py)**: Advanced capabilities
   - Event callbacks
   - Multi-threaded processing
   - Adaptive recognition
   - Facial landmarks analysis

3. **[api_robotics_integration.py](examples/api_robotics_integration.py)**: Robotics applications
   - Person tracking
   - State management
   - Security robot implementation
   - Interactive robot behaviors

4. **[api_batch_processing.py](examples/api_batch_processing.py)**: Batch operations
   - Process directories of images
   - Video file processing
   - Export results to CSV/JSON
   - Dataset analysis

## License

See LICENSE file for details.

## Support

For issues and questions, please open an issue on GitHub.

## Roadmap

- [x] ~~Class-based refactored API~~ ✅ Complete!
- [x] ~~Event callbacks system~~ ✅ Complete!
- [x] ~~Thread-safe operations~~ ✅ Complete!
- [x] ~~Batch processing support~~ ✅ Complete!
- [ ] Multi-camera support
- [ ] GPU acceleration on Jetson
- [ ] REST API for remote access
- [x] ~~Face detection confidence thresholds~~ ✅ Complete!
- [x] ~~Export/import face database~~ ✅ Complete!
- [ ] Integration with ROS (Robot Operating System)
- [ ] Real-time streaming API
- [ ] Face tracking across frames
- [ ] Age and gender estimation

## Credits

Built with:
- [DepthAI](https://github.com/luxonis/depthai-python) - Oak D camera interface
- [face_recognition](https://github.com/ageitgey/face_recognition) - Face recognition library
- [OpenCV](https://opencv.org/) - Computer vision operations
- [tkinter](https://docs.python.org/3/library/tkinter.html) - GUI framework
