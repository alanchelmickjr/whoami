# WhoAmI - Facial Recognition System

A simple, portable, and modular facial recognition framework designed for Oak D Series 3 cameras and Jetson Orin Nano. Built with security and ease of use in mind.

## Features

- 🎥 **Oak D Series 3 Support**: Optimized for Oak D cameras using DepthAI SDK
- 🖥️ **Simple GUI**: Easy-to-use graphical interface built with tkinter
- 💻 **CLI Interface**: Command-line interface for headless operation
- 👤 **Face Management**: Add, remove, and manage known faces
- 🔒 **Secure**: Local processing, no cloud dependencies
- 🚀 **Portable**: Lightweight and modular design for easy integration
- 🤖 **Robot-Ready**: Perfect for robotics applications on Jetson platforms

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

## Architecture

The system is designed with modularity in mind:

```
whoami/
├── whoami/
│   ├── __init__.py          # Package initialization
│   ├── face_recognizer.py   # Core face recognition logic
│   ├── gui.py               # GUI application
│   ├── cli.py               # CLI application
│   └── config.py            # Configuration management
├── run_gui.py               # GUI entry point
├── run_cli.py               # CLI entry point
├── requirements.txt         # Python dependencies
└── setup.py                 # Installation script
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

This framework is designed to be easily integrated into larger robotics systems:

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
```

## License

See LICENSE file for details.

## Support

For issues and questions, please open an issue on GitHub.

## Roadmap

- [ ] Multi-camera support
- [ ] GPU acceleration on Jetson
- [ ] REST API for remote access
- [ ] Face detection confidence thresholds
- [ ] Export/import face database
- [ ] Integration with ROS (Robot Operating System)

## Credits

Built with:
- [DepthAI](https://github.com/luxonis/depthai-python) - Oak D camera interface
- [face_recognition](https://github.com/ageitgey/face_recognition) - Face recognition library
- [OpenCV](https://opencv.org/) - Computer vision operations
- [tkinter](https://docs.python.org/3/library/tkinter.html) - GUI framework
