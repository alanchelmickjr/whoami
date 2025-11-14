# WhoAmI - Facial Recognition System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Node.js 18+](https://img.shields.io/badge/node.js-18+-green.svg)](https://nodejs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Oak D](https://img.shields.io/badge/Oak%20D-Series%203-green.svg)](https://docs.luxonis.com/)

A comprehensive, secure facial recognition framework designed for robotics platforms. Features both Python (Oak D cameras) and JavaScript (Gun.js secure database) implementations with hardware-backed encryption.

## Features

### Core Recognition
- 🎥 **Oak D Series 3 Support**: Optimized for Oak D cameras using DepthAI SDK
- 👤 **Face Management**: Add, remove, and manage known faces
- 🔌 **Class-Based API**: Clean, modular, and thread-safe face recognition library
- 📊 **Batch Processing**: Process multiple images and videos efficiently
- 🔔 **Event Callbacks**: React to face detection and recognition events
- 🧵 **Thread-Safe**: Designed for multi-threaded applications

### Interfaces
- 🖥️ **Simple GUI**: Easy-to-use graphical interface built with tkinter
- 💻 **CLI Interface**: Command-line interface for headless operation
- 🗣️ **Voice Interaction**: Ask names and greet people with audio feedback
- 🎤 **Speech Recognition**: Recognize voice input (online or offline)
- 🔊 **Text-to-Speech**: Provide audio feedback and status updates

### Hardware & Integration
- 🔒 **Secure**: Local processing, hardware-backed encryption, no cloud dependencies
- 🚀 **Portable**: Lightweight and modular design for easy integration
- 🤖 **Robot-Ready**: Perfect for robotics applications on Jetson platforms
- 🎛️ **Hardware Auto-Detection**: Automatic platform detection and configuration
- 🎮 **Gimbal Control**: 3-axis gimbal support for head/neck movement
- 🌐 **Remote Access**: VNC, SSH, and web interface support
- 🔌 **Multi-Platform**: Jetson (including K-1 Booster), Raspberry Pi, Mac, and Linux desktop

### Security (Gun.js Implementation)
- 🔐 **Hardware-Backed Encryption**: Keys derived from device-specific hardware identifiers
- 🔒 **Double-Layer Encryption**: AES-256-GCM + Gun.js SEA encryption
- 🛡️ **Zero Plaintext Storage**: No encryption keys or sensitive data in code
- 🎯 **Device-Specific**: Each robot has unique cryptographic identity
- ⚙️ **Reverse Engineering Resistant**: Data cannot be decrypted without specific hardware

## Hardware Requirements

### Supported Platforms

The WhoAmI system includes automatic hardware detection and configuration for multiple platforms:

- **NVIDIA Jetson**
  - Jetson Orin Nano DevKit
  - Jetson Orin NX DevKit
  - **Jetson Orin NX on K-1 Booster** (3-axis gimbal + audio + dual Ethernet)
  - Jetson AGX Orin DevKit
- **Raspberry Pi 4**
- **Apple Silicon Mac** (M1/M2/M3/M4)
- **Generic Linux Desktop** (x86_64)

### Core Requirements

**Python Implementation:**
- Oak D Series 3 camera (or compatible DepthAI device)
- USB 3.0 port
- Python 3.8 or higher

**JavaScript Implementation:**
- Node.js >= 18.0.0
- USB Camera or CSI Camera
- OpenCV (for opencv4nodejs)
- CUDA support (for GPU acceleration)

### K-1 Booster Configuration

The K-1 booster carrier board adds advanced capabilities:
- **3-Axis Gimbal System**: 2-axis head (pan/tilt) + 1-axis neck (tilt)
- **Audio I/O**: Voice reporting, audio tracking, speech recognition
- **Dual Ethernet**: Primary and backup network with automatic failover
- **Remote Access**: VNC, SSH, and web interface support
- **Operational Modes**: Remote VNC, direct access, or autonomous

See [K-1 Booster Setup Guide](docs/K1_BOOSTER_SETUP.md) for detailed configuration.

## Installation

### Python Implementation (Quick Install)

```bash
# Clone the repository
git clone https://github.com/alanchelmickjr/whoami.git
cd whoami

# Install Python dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

### JavaScript Implementation

```bash
# Install Node.js dependencies
npm install

# Download face-api.js models
mkdir -p models
cd models
# Download models from https://github.com/vladmandic/face-api
# Required models:
# - ssdMobilenetv1
# - faceLandmark68Net
# - faceRecognitionNet
```

### System Dependencies

For Jetson Orin Nano:
```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-tk cmake libopencv-dev nodejs npm

# For voice interaction
sudo apt-get install -y portaudio19-dev espeak flac alsa-utils pulseaudio
```

For face_recognition library:
```bash
sudo apt-get install -y build-essential cmake libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev
```

## Hardware Configuration

### Automatic Hardware Detection

The system automatically detects your hardware platform and loads the appropriate configuration:

```python
from whoami.hardware_detector import detect_hardware, get_serial_port

# Detect hardware platform
platform = detect_hardware()
print(f"Running on: {platform}")
# Example output: "jetson_orin_nx_k1"

# Get serial port for this platform
port = get_serial_port()
print(f"Serial port: {port}")
# Example output: "/dev/ttyTHS1"
```

### Manual Hardware Override

Override auto-detection if needed:

```bash
# Set specific hardware profile
export WHOAMI_HARDWARE_PROFILE="jetson_orin_nx_k1"

# Override serial port
export WHOAMI_SERIAL_PORT="/dev/ttyTHS1"

# Run application
python -m whoami.gui
```

### K-1 Booster Testing

Multiple K-1 booster units are available for testing! The configuration includes:

**Serial Ports:**
- `/dev/ttyTHS1` - Head gimbal (pan/tilt)
- `/dev/ttyTHS2` - Neck gimbal (tilt)

**Audio:**
- Input: `hw:2,0` (USB microphone for voice input)
- Output: `hw:2,0` (USB speaker for voice reporting)

**Network:**
- `eth0` - Primary Ethernet
- `eth1` - Secondary Ethernet (automatic failover)

**Next Steps for K-1 Deployment:**
1. Test hardware detection on actual K-1 booster
2. Verify serial port assignments (ttyTHS1, ttyTHS2)
3. Configure audio devices (adjust hw:2,0 if needed)
4. Test GPIO probe pin for carrier detection (GPIO 194)
5. Integrate voice interaction with face recognition

## Usage

### Python GUI Application

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

### Python CLI Application

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

### Python Face Recognition API

The Face Recognition API provides a clean, class-based interface:

```python
from whoami.face_recognition_api import create_face_recognition_api

# Quick start with the API
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

### Voice Interaction

Ask people for their names and greet them:

```python
from whoami.voice_interaction import VoiceInteraction

# Initialize voice system
voice = VoiceInteraction()

# Ask for a name
name = voice.ask_name()
if name:
    voice.greet_person(name)
    print(f"Learned name: {name}")
```

### JavaScript Gun.js Implementation

Secure facial recognition with hardware-backed encryption:

```javascript
import { whoami } from './src/index.js';

// Initialize the system
await whoami.initialize('./config/config.json');

// Register a person
const image = await loadImage('path/to/person.jpg');
const faceId = await whoami.registerPerson(image, 'John Doe');

// Recognize a person
const result = await whoami.recognize(image);
if (result.recognized) {
  console.log(`Hello, ${result.personName}!`);
}

// List registered persons
const persons = await whoami.listRegistered();
console.log(persons);
```

## Architecture

### Python Implementation

```
whoami/
├── whoami/
│   ├── __init__.py                  # Package initialization
│   ├── face_recognizer.py           # Core face recognition logic
│   ├── face_recognition_api.py      # Class-based API
│   ├── voice_interaction.py         # Voice interaction system
│   ├── hardware_detector.py         # Hardware auto-detection
│   ├── gui.py                       # GUI application
│   ├── cli.py                       # CLI application
│   └── config.py                    # Configuration management
├── config/
│   ├── hardware/                    # Hardware profiles
│   │   ├── hardware_profiles.json  # Platform configurations
│   │   └── README.md
│   └── k1_booster_config.json      # K-1 specific config
├── docs/                            # Comprehensive documentation
├── examples/                        # Example scripts
└── requirements.txt                 # Python dependencies
```

### JavaScript Implementation with Gun.js

```
┌─────────────────────────────────────────────────────────┐
│                    WhoAmI System                        │
├─────────────────────────────────────────────────────────┤
│  Application Layer                                      │
│  - Facial Recognition (face-api.js)                    │
│  - Real-time Processing                                │
├─────────────────────────────────────────────────────────┤
│  Security Layer (Hardware-Backed)                       │
│  - AES-256-GCM Encryption                              │
│  - Hardware Key Derivation (CPU Serial + MAC)          │
│  - Scrypt Key Derivation Function                      │
├─────────────────────────────────────────────────────────┤
│  Database Layer (Gun.js)                                │
│  - SEA Encryption (Second Layer)                        │
│  - Decentralized P2P Database                          │
│  - Local-First Storage                                 │
├─────────────────────────────────────────────────────────┤
│  Hardware Layer (Jetson Nano)                           │
│  - GPU Acceleration for CV                              │
│  - Hardware Identifiers                                │
│  - Camera Interface                                     │
└─────────────────────────────────────────────────────────┘
```

**JavaScript Structure:**
```
src/
├── index.js              # Main application entry point
├── secureKeyManager.js   # Hardware-backed key management
├── secureDatabase.js     # Gun.js database with encryption
└── facialRecognition.js  # Face detection and recognition
```

## 📚 Documentation

### Core Documentation
- **[API Reference](docs/API_REFERENCE.md)**: Complete API documentation with all classes, methods, and parameters
- **[Usage Guide](docs/USAGE_GUIDE.md)**: Comprehensive guide with code examples and best practices
- **[Examples](examples/)**: Working example scripts demonstrating various features

### Hardware & Configuration
- **[Hardware Configuration Guide](docs/HARDWARE_CONFIG_GUIDE.md)**: Complete guide for hardware profiles and detection
- **[K-1 Booster Setup](docs/K1_BOOSTER_SETUP.md)**: Setup guide for Jetson Orin NX on K-1 booster
- **[Voice Interaction Guide](docs/VOICE_INTERACTION_GUIDE.md)**: Voice-based name asking and audio feedback
- **[Gimbal 3DOF Guide](docs/GIMBAL_3DOF_GUIDE.md)**: 3-axis gimbal system integration
- **[Genesis VLA Guide](docs/GENESIS_VLA_GUIDE.md)**: Vision-Language-Action model training
- **[Spatial Awareness Guide](docs/SPATIAL_AWARENESS_GUIDE.md)**: Environmental understanding
- **[Servo Safety Guide](docs/SERVO_SAFETY_GUIDE.md)**: Servo health monitoring and safety

### Security & Deployment
- **[Security Documentation](SECURITY.md)**: Hardware-backed encryption details
- **[Deployment Guide](DEPLOYMENT.md)**: Production deployment instructions
- **[API Documentation](API.md)**: JavaScript API reference

### Installation & Setup
- **[Installation Guide](INSTALLATION.md)**: Complete installation instructions
- **[Setup Quick Reference](SETUP_QUICK_REFERENCE.md)**: Quick setup commands
- **[Jetson & M4 Setup](SETUP_JETSON_M4.md)**: Platform-specific setup guide

## Configuration

### Python Configuration

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
        "window_height": 700
    }
}
```

### JavaScript Configuration

Edit `config/config.json`:

```json
{
  "modelsPath": "./models",
  "dataPath": "./data/gun",
  "minConfidence": 0.7,
  "descriptorThreshold": 0.6,
  "peers": [],
  "camera": {
    "deviceId": 0,
    "width": 640,
    "height": 480,
    "fps": 30
  }
}
```

## Security Features

### Hardware-Backed Encryption (Gun.js)

1. **Key Derivation**:
   - Reads CPU serial from `/proc/cpuinfo`
   - Reads MAC address from network interfaces
   - Combines and hashes to create unique hardware fingerprint
   - Uses scrypt (memory-hard KDF) to derive encryption keys

2. **Double Encryption**:
   ```
   Plain Data → Hardware Encryption → Gun.js SEA Encryption → Storage
   ```

3. **Tamper Detection**:
   - Uses GCM authentication tags
   - Any tampering causes decryption to fail
   - No silent data corruption possible

### Why This is Secure

- **No Key Extraction**: Keys derived from hardware on-the-fly, never stored
- **Device-Locked**: Data encrypted on one device cannot be decrypted on another
- **Memory-Hard KDF**: Resistant to brute force attacks using scrypt
- **Layered Defense**: Even if one encryption layer is broken, second layer protects data
- **P2P Security**: Gun.js provides additional SEA (Security, Encryption, Authorization)
- **Local Processing**: All face recognition happens on-device (Python implementation)
- **Privacy First**: No images stored, only mathematical encodings

## Roadmap

### Completed Features ✅
- [x] ~~Class-based refactored API~~
- [x] ~~Event callbacks system~~
- [x] ~~Thread-safe operations~~
- [x] ~~Batch processing support~~
- [x] ~~Face detection confidence thresholds~~
- [x] ~~Export/import face database~~
- [x] ~~Hardware configuration system~~
- [x] ~~K-1 Booster support with 3-axis gimbal~~
- [x] ~~Audio I/O configuration~~
- [x] ~~Multi-platform hardware detection~~
- [x] ~~Voice interaction system~~
- [x] ~~Gun.js secure database integration~~

### In Progress 🚧
- [ ] **Audio source tracking** - Orient gimbal toward speakers
- [ ] **Voice commands** - Control system via speech

### Planned Features 📋
- [ ] Multi-camera support
- [ ] GPU acceleration on Jetson
- [ ] REST API for remote access
- [ ] Integration with ROS (Robot Operating System)
- [ ] Real-time streaming API
- [ ] Face tracking across frames
- [ ] Age and gender estimation
- [ ] Emotion detection
- [ ] Multi-language voice support
- [ ] Voice biometrics for speaker identification
- [ ] Distributed Gun.js peer network

## Credits

Built with:
- [DepthAI](https://github.com/luxonis/depthai-python) - Oak D camera interface
- [face_recognition](https://github.com/ageitgey/face_recognition) - Face recognition library
- [Gun.js](https://gun.eco/) - Decentralized database
- [face-api.js](https://github.com/vladmandic/face-api) - JavaScript face recognition
- [OpenCV](https://opencv.org/) - Computer vision operations
- [tkinter](https://docs.python.org/3/library/tkinter.html) - GUI framework
- [pyttsx3](https://pyttsx3.readthedocs.io/) - Text-to-speech
- [Vosk](https://alphacephei.com/vosk/) - Offline speech recognition

## License

See LICENSE file for details.

## Support

For issues and questions, please open an issue on GitHub.

## ⚠️ Disclaimer

This system is designed for authorized use only. Ensure compliance with local privacy laws and regulations when deploying facial recognition technology.
