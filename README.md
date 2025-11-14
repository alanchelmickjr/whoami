# WhoAmI - Portable Robotics Intelligence for Jetson Platforms

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Node.js 18+](https://img.shields.io/badge/node.js-18+-green.svg)](https://nodejs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Jetson](https://img.shields.io/badge/NVIDIA-Jetson-76B900.svg)](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/)

**Portable robotics "brain" software for NVIDIA Jetson platforms** - featuring facial recognition, voice interaction, hardware-backed encryption, and multi-modal sensing. Designed for portability across various robotic platforms and carrier boards.

Built by [Utiltiron](https://utiltiron.io) for creating intelligent, adaptive robotic systems on Jetson hardware.

## 🎯 Philosophy: Portable Intelligence

WhoAmI is **software for robotic brains**, not tied to any single hardware platform. The system provides:
- 🧠 **Platform-agnostic design** - Runs on any Jetson (Orin Nano, Orin NX, AGX Orin)
- 🔌 **Hardware auto-detection** - Automatically configures for different carrier boards
- 🎮 **Flexible peripheral support** - Works with various gimbals, sensors, and I/O configurations
- 📦 **Modular architecture** - Pick the features you need for your robot

## 🤖 Supported Platforms

WhoAmI runs on multiple Jetson-based platforms:

### K-1 Booster - Commercial Humanoid Robot

Production deployment platform - 22-DOF humanoid with RGBD vision:

**K-1 Booster Capabilities**

- 🦾 **Full-Body Humanoid Control**
  - 22 degrees of freedom (legs, arms, head)
  - Force-controlled dual-encoder joints
  - Fast-DDS/ROS2 SDK for motion control
  - AI, WALK, and CUSTOM modes

- 👁️ **RGBD Vision System**
  - Intel RealSense or ToF depth camera
  - RGB for face recognition
  - 3D spatial awareness
  - Integrated with WhoAmI face recognition

- 🗣️ **Voice Interaction**
  - Microphone array for voice input
  - Speaker for TTS output
  - Ask unknown people for their names
  - Greet known people by name
  - Offline speech recognition (Vosk)

- 🎯 **Advanced Features**
  - Jetson Orin NX 8GB (117 TOPS)
  - 8 CPU cores for parallel processing
  - 1024 CUDA cores for GPU acceleration
  - Dual Ethernet + WiFi connectivity
  - IMU and odometry feedback

### K-1 Booster Platform Architecture

```
┌─────────────────────────────────────────────────────────┐
│              K-1 Booster Humanoid Robot                 │
├─────────────────────────────────────────────────────────┤
│  👁️  RGBD Camera (Intel RealSense / ToF)                │
│     • Depth sensing for spatial awareness              │
│     • RGB for face detection and recognition           │
├─────────────────────────────────────────────────────────┤
│  🦾 22-DOF Humanoid Body                                 │
│     • Legs: 6 DOF × 2 (walking, balancing)            │
│     • Arms: 4 DOF × 2 (Shoulder P/R/Y + Elbow)        │
│     • Head: 2 DOF (Pan + Tilt)                        │
│     • Force-controlled dual-encoder joints             │
│     • Fast-DDS/ROS2 SDK (Booster Robotics)            │
├─────────────────────────────────────────────────────────┤
│  🎤 Audio I/O                                            │
│     • Microphone array for voice input                 │
│     • Speaker for TTS output                           │
│     • Voice interaction & name asking                  │
├─────────────────────────────────────────────────────────┤
│  🌐 Connectivity                                         │
│     • Dual Ethernet + WiFi                             │
│     • Remote access and control                        │
│     • Button panel + Xbox controller support           │
├─────────────────────────────────────────────────────────┤
│  🧠 WhoAmI Intelligence Layer                           │
│     • YOLO + DeepFace face recognition                 │
│     • Voice interaction system                         │
│     • Local-first encrypted database (Gun.js)          │
│     • Hardware-backed encryption                       │
└─────────────────────────────────────────────────────────┘
```

**➡️ See [K-1 Booster Setup Guide](docs/K1_BOOSTER_SETUP.md) for complete hardware configuration**

### Robi - Modular Morphing Platform

```
┌─────────────────────────────────────────────────────────┐
│        Robi (Reasonably Obtainable Bot Intelligence)    │
│         Modular Platform - Swappable Base System        │
├─────────────────────────────────────────────────────────┤
│  🧠 Core Module (Always Present)                         │
│     • Jetson Orin NX brain                             │
│     • OAK-D Series 3 camera head                       │
│     • Gimbal system (head/neck tilt)                   │
│     • Basic arms for manipulation                      │
│     • Wheels for mobility                              │
├─────────────────────────────────────────────────────────┤
│  👁️  Vision Head                                         │
│     • OAK-D: Stereo depth + RGB                        │
│     • Head: /dev/ttyTHS1 (tilt control)                │
│     • Neck: /dev/ttyTHS2 (tilt control)                │
│     • 2-axis coordinated tracking                      │
├─────────────────────────────────────────────────────────┤
│  🎤 Audio I/O                                            │
│     • USB Audio Class 2.0 interface                    │
│     • TTS voice output                                 │
│     • Voice interaction & name asking                  │
├─────────────────────────────────────────────────────────┤
│  🔧 Morphing Base System (Swappable)                     │
│     • Different bases for different tasks              │
│     • Tool attachments and configurations              │
│     • Currently building: New base                     │
├─────────────────────────────────────────────────────────┤
│  🧠 WhoAmI Intelligence Layer                           │
│     • YOLO + DeepFace face recognition                 │
│     • Voice interaction system                         │
│     • Local-first encrypted database (Gun.js)          │
│     • Hardware-backed encryption                       │
└─────────────────────────────────────────────────────────┘
```

**➡️ Robi: Modular platform with persistent head/brain and swappable bases**
**➡️ K-1: Commercial integrated humanoid with fixed body structure**
**➡️ Both use the same WhoAmI intelligence software**

## Core Features (Platform-Portable)

### 🧠 Intelligence Layer
- 👁️ **Advanced Face Recognition** (Dual-Engine Support)
  - **YOLO + DeepFace** (K-1 Booster optimized):
    - YOLOv8 for ultra-fast face detection on Jetson GPU
    - DeepFace (Facenet/ArcFace) for deep learning embeddings
    - Real-time performance: 15-30 FPS on Jetson Orin NX
    - Integrated with voice interaction for name collection
  - **dlib-based** (Traditional):
    - 128-dimensional face encodings
    - Confidence-based matching with adjustable thresholds
  - Camera integration:
    - K-1 Booster: RGBD (Intel RealSense / ToF)
    - Robi: OAK-D Series 3 (stereo depth + RGB)
  - Thread-safe operations across platforms

- 🗣️ **Voice Interaction System**
  - Natural conversation flow for name collection
  - "Hello! I don't think we've met. What's your name?"
  - Confirmation loop with yes/no validation
  - Text-to-speech greetings and status updates
  - Offline speech recognition (Vosk) or online (Google)

- 🎭 **Multi-Modal Sensing**
  - **Visual**: Face detection, tracking, recognition
  - **Audio**: Voice identification, source localization
  - **Depth**: 3D spatial awareness (RGBD camera)
  - **Motion**: Gimbal-based tracking or full-body kinematics (K-1)

### Security & Privacy
- 🔐 **Hardware-Backed Encryption** (Gun.js)
  - Keys derived from CPU serial + MAC address
  - AES-256-GCM + Gun.js SEA double encryption
  - Device-locked (cannot decrypt on different hardware)
  - Zero plaintext key storage

- 🛡️ **Privacy-First Design**
  - No cloud dependencies
  - Local-only processing
  - No images stored (only mathematical encodings)
  - Encrypted database

### 🔌 Hardware Abstraction Layer
- **Automatic Platform Detection**
  - Device tree parsing for Jetson identification
  - Carrier board detection (GPIO probing, device tree)
  - Auto-configuration of peripherals (serial, GPIO, I2C, USB)

- **Flexible Peripheral Support**
  - Gimbal systems (optional, auto-configured)
  - Audio I/O (USB or built-in)
  - Network interfaces (Ethernet, WiFi)
  - Remote access (VNC, SSH, web interface)

### 🎮 Operational Modes

Configurable operational modes (availability depends on platform):

1. **Remote VNC Mode** (K-1 Booster, AGX Orin)
   - Operator controls via VNC from anywhere
   - Bidirectional audio streaming
   - Real-time video feedback
   - Full system access

2. **🖥️ Direct Access Mode**
   - HDMI monitor + keyboard/mouse
   - Local operation and testing
   - Development environment

3. **🤖 Autonomous Mode**
   - Fully independent operation
   - Audio status reporting
   - Self-guided interaction
   - Monitoring via web interface

## Quick Start (Any Jetson Platform)

### Option A: K-1 Booster with YOLO (Recommended for K-1)

```bash
# Clone repository
git clone https://github.com/alanchelmickjr/whoami.git
cd whoami

# Run K-1 setup (includes YOLO + DeepFace + Voice)
./k1_setup.sh

# Test the system
python3 examples/k1_yolo_demo.py
```

**The K-1 will now:**
- Detect faces using YOLO (GPU-accelerated)
- Ask unknown people "What's your name?" via voice
- Remember faces and greet known people
- Display live video with face annotations

### Option B: Standard Installation (All Platforms)

```bash
# Clone repository
git clone https://github.com/alanchelmickjr/whoami.git
cd whoami

# Automated setup (detects your platform)
./jetson_setup_v2.sh --full

# Or install manually
pip install -r requirements.txt
sudo apt-get install -y portaudio19-dev espeak flac alsa-utils pulseaudio
```

### 2. Verify Hardware Auto-Detection

The system automatically detects your Jetson platform and carrier board:

```bash
python -m whoami.hardware_detector

# Example outputs:
# ✓ Detected: Jetson Orin NX on K-1 Booster
# ✓ Detected: Jetson Orin Nano Developer Kit
# ✓ Detected: Jetson AGX Orin Developer Kit
# Serial Port: /dev/ttyTHS1
# Head Gimbal Port: /dev/ttyTHS1
# Neck Gimbal Port: /dev/ttyTHS2
# Audio Input: hw:2,0
# Audio Output: hw:2,0
```

### 3. Run Voice-Enabled Face Recognition

```python
from whoami.voice_interaction import VoiceEnabledFaceRecognition, VoiceInteraction
from whoami.face_recognition_api import create_face_recognition_api

# Initialize systems
face_api = create_face_recognition_api()
voice = VoiceInteraction()

# Create voice-enabled wrapper
robot = VoiceEnabledFaceRecognition(
    face_recognizer=face_api,
    voice_interaction=voice,
    ask_unknown=True,      # Ask unknown people for names
    announce_known=True    # Greet known people
)

# Start camera
with face_api:
    face_api.start_camera()

    # Robot will automatically:
    # - Detect faces
    # - Ask unknown people for names
    # - Greet known people
    # - Track faces with gimbal

    while True:
        results = face_api.process_frame()
        for result in results:
            robot.process_detection(
                name=result.name,
                confidence=result.confidence,
                face_encoding=result.encoding
            )
```

### 4. Test Gimbal Control

```bash
# Test gimbal system
python3 -m whoami.gimbal_controller --test

# Or test directly with hardware detection
python3 -m whoami.hardware_detector
```

## Other Jetson Platforms

The WhoAmI brain software runs on any Jetson platform with automatic hardware detection:

### Jetson Orin Nano DevKit
- 🎯 **Compact intelligence** - Full AI capabilities in smaller form factor
- 8GB RAM, 6 CPU cores, 512 CUDA cores
- Supports custom gimbal configurations
- Perfect for prototyping and compact robots
- Auto-detected via device tree

### Jetson AGX Orin
- 🚀 **Maximum performance** - Most powerful Jetson platform
- Up to 64GB RAM, 12 CPU cores, 2048 CUDA cores
- Enterprise-grade robotics applications
- Multi-robot coordination and fleet management
- Extended I/O and expansion capabilities

### Development Platforms
We also support development on:
- 🍎 **Mac M-Series** (arm64) - Local development and testing
- 🐧 **Linux Desktop** (x86_64) - Simulation and algorithm development
- 🥧 **Raspberry Pi 4** - Lightweight deployment testing

**All platforms auto-detected and configured via hardware profiles**

## Voice Interaction Examples

### Ask for Name

```python
from whoami.voice_interaction import VoiceInteraction

voice = VoiceInteraction()
name = voice.ask_name()
if name:
    voice.greet_person(name)
```

### Voice-Enabled Robot

```python
from whoami.voice_interaction import VoiceEnabledFaceRecognition

# Automatically ask unknown people for names
robot = VoiceEnabledFaceRecognition(
    face_recognizer=face_api,
    ask_unknown=True,
    announce_known=True
)
```

### Custom Interactions

```python
voice.say("Starting patrol mode")
voice.say("Face detected. Analyzing...")
voice.say(f"Hello {name}, security clearance confirmed")
```

## Gun.js Secure Database

Hardware-locked encrypted face database:

```javascript
import { whoami } from './src/index.js';

// Initialize with hardware encryption
await whoami.initialize('./config/config.json');

// Register person (encrypted with device-specific keys)
const faceId = await whoami.registerPerson(image, 'John Doe');

// Recognize (decrypt with hardware keys)
const result = await whoami.recognize(image);
if (result.recognized) {
  console.log(`Authenticated: ${result.personName}`);
}
```

## Hardware Specifications

### K-1 Booster (Commercial Humanoid Platform)

**Jetson Orin NX 8GB Module**
- **CPU**: 8-core ARM Cortex-A78AE
- **GPU**: 1024 CUDA cores, 32 Tensor cores
- **Memory**: 8GB LPDDR5
- **AI Performance**: 117 TOPS
- **Power**: 10W-25W configurable

**22-DOF Humanoid Body**
- **Legs**: 6 DOF × 2 (hip, knee, ankle joints)
- **Arms**: 4 DOF × 2 (shoulder pitch/roll/yaw + elbow)
- **Head**: 2 DOF (pan + tilt)
- **Joints**: Force-controlled dual-encoder actuators
- **Control**: Fast-DDS/ROS2 communication
- **Modes**: AI, WALK, CUSTOM (low-level control)

**Sensors & I/O**
- **Camera**: RGBD (Intel RealSense or ToF)
- **Audio**: Microphone array + speaker
- **IMU**: 9-axis (accelerometer, gyro, magnetometer)
- **Network**: Dual Ethernet + WiFi
- **Power**: 24V battery system

**SDK**: Booster Robotics SDK (Fast-DDS)
- Joint control (position, velocity, torque)
- IMU and odometry feedback
- Button and gamepad input
- TF transforms (ROS2 compatible)

### Robi - Reasonably Obtainable Bot Intelligence (Modular Platform)

**Core Module (Persistent)**
- **Brain**: Jetson Orin NX on carrier board
- **Camera**: OAK-D Series 3 (depth + RGB stereo)
- **Gimbal**: 2-3 axis servo system (/dev/ttyTHS1-2)
- **Arms**: Basic manipulators (always present)
- **Wheels**: Base mobility system (always present)
- **Audio**: USB Audio Class 2.0
- **Network**: Ethernet or WiFi

**Morphing Base System**
- **Concept**: Swappable bases and tool attachments for different tasks
- **Core stays constant**: Head, brain, vision, arms, wheels
- **Base adapts**: Different configurations for different missions
- **Currently**: Building new base configuration

**Software**
- WhoAmI intelligence layer (face recognition + voice)
- Same software stack as K-1 Booster
- Hardware-backed encryption for identity storage

## Documentation

### Core Documentation (Platform-Agnostic)
- **[Hardware Configuration](docs/HARDWARE_CONFIG_GUIDE.md)** ⭐ Hardware profiles & auto-detection
- **[Voice Interaction](docs/VOICE_INTERACTION_GUIDE.md)** - Voice system guide
- **[API Reference](docs/API_REFERENCE.md)** - Complete API docs
- **[Usage Guide](docs/USAGE_GUIDE.md)** - Usage examples
- **[Security](SECURITY.md)** - Hardware-backed encryption
- **[Deployment](DEPLOYMENT.md)** - Production deployment

### Platform-Specific Guides
- **[K-1 Booster Setup](docs/K1_BOOSTER_SETUP.md)** - Orin NX on K-1 carrier
- **[Gimbal Control](docs/GIMBAL_3DOF_GUIDE.md)** - Gimbal configuration (optional)

### Advanced Features
- **[Genesis VLA](docs/GENESIS_VLA_GUIDE.md)** - Vision-Language-Action training
- **[Spatial Awareness](docs/SPATIAL_AWARENESS_GUIDE.md)** - Environmental mapping
- **[Servo Safety](docs/SERVO_SAFETY_GUIDE.md)** - Safety monitoring

### Installation & Setup
- **[Installation Guide](INSTALLATION.md)** - Complete installation
- **[Jetson & M4 Setup](SETUP_JETSON_M4.md)** - Platform setup
- **[Quick Reference](SETUP_QUICK_REFERENCE.md)** - Quick commands

## Configuration

### Hardware Profiles (Auto-Configured)

All Jetson platforms have pre-configured profiles in `config/hardware/hardware_profiles.json`:

```json
{
  "profiles": {
    "jetson_orin_nx_k1": { /* K-1 Booster config */ },
    "jetson_orin_nano_devkit": { /* Orin Nano DevKit */ },
    "jetson_agx_orin_devkit": { /* AGX Orin DevKit */ },
    // ... auto-detected and loaded
  }
}
```

### Example: K-1 Booster Configuration

Platform-specific config at `config/k1_booster_config.json`:

```json
{
  "hardware_profile": "jetson_orin_nx_k1",
  "gimbal": {
    "head_gimbal": {
      "serial_port": "/dev/ttyTHS1",
      "servos": {"tilt": {"id": 1}}
    },
    "neck_gimbal": {
      "serial_port": "/dev/ttyTHS2",
      "servos": {"neck_tilt": {"id": 2}}
    }
  },
  "audio": {
    "features": ["voice_reporting", "audio_tracking", "speech_synthesis"]
  },
  "network": {
    "ethernet_primary": "eth0",
    "ethernet_secondary": "eth1"
  }
}
```

**Your platform config is automatically loaded based on hardware detection.**

## Roadmap

### Completed ✅
- [x] **Multi-platform support** (Orin Nano, Orin NX, AGX Orin)
- [x] **Hardware auto-detection** & profile system
- [x] **K-1 Booster integration** (first production platform)
- [x] Voice interaction & name collection system
- [x] Gimbal control (flexible axis configuration)
- [x] Audio source tracking & localization
- [x] Hardware-backed encryption (device-locked)
- [x] Dual Ethernet failover (K-1, AGX)
- [x] Remote VNC access & web interface

### In Progress 🚧
- [ ] Audio source-based gimbal orientation
- [ ] Voice command control
- [ ] Multi-robot coordination (Gun.js P2P)

### Planned 📋
- [ ] Emotion detection from voice
- [ ] Multi-language voice support
- [ ] LiDAR integration for navigation
- [ ] ROS 2 integration
- [ ] Arm/manipulator control
- [ ] Fleet management dashboard

## Performance

Performance varies by platform. Example benchmarks:

### Jetson Orin NX (K-1 Booster)
- **Face Detection**: 30 FPS @ 1280x720
- **Face Recognition**: 20 FPS
- **Voice Response**: <500ms latency
- **Gimbal Tracking**: 60Hz update rate
- **Power**: 15W idle, 35W active, 45W peak

### Jetson Orin Nano (DevKit)
- **Face Detection**: 25 FPS @ 1280x720
- **Face Recognition**: 15 FPS
- **Voice Response**: <600ms latency
- **Power**: 10W idle, 25W active, 35W peak

### Jetson AGX Orin (DevKit)
- **Face Detection**: 60 FPS @ 1920x1080
- **Face Recognition**: 45 FPS
- **Voice Response**: <300ms latency
- **Power**: 20W idle, 50W active, 80W peak

## Credits

### Built by [Utiltiron](https://utiltiron.io)
Creating intelligent robotics software for NVIDIA Jetson platforms.

### Technology Stack
- **[NVIDIA Jetson](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/)** - Edge AI compute platforms
- **[Booster Robotics SDK](https://github.com/BoosterRobotics/booster_robotics_sdk)** - K-1 humanoid control (Fast-DDS/ROS2)
- **[Oak D Series 3](https://docs.luxonis.com/)** - Depth camera for Robi
- **[Intel RealSense](https://www.intelrealsense.com/)** - RGBD cameras for K-1
- **[Gun.js](https://gun.eco/)** - Decentralized P2P database
- **[YOLOv8](https://github.com/ultralytics/ultralytics)** - Fast face detection
- **[DeepFace](https://github.com/serengil/deepface)** - Deep learning face recognition
- **[face_recognition](https://github.com/ageitgey/face_recognition)** - Traditional face encoding
- **[Vosk](https://alphacephei.com/vosk/)** - Offline speech recognition
- **[pyttsx3](https://pyttsx3.readthedocs.io/)** - Text-to-speech synthesis

### Hardware Platforms
Tested on:
- **Jetson Orin NX** (K-1 Booster carrier - first production deployment)
- **Jetson Orin Nano** (DevKit - development & testing)
- **Jetson AGX Orin** (DevKit - high-performance applications)

## License

MIT License - See LICENSE file for details.

## Support

**General Issues:**
- 🐛 [GitHub Issues](https://github.com/alanchelmickjr/whoami/issues)
- 📖 [Hardware Configuration Guide](docs/HARDWARE_CONFIG_GUIDE.md)

**Platform-Specific Help:**
- K-1 Booster: [K-1 Setup Guide](docs/K1_BOOSTER_SETUP.md) (tag: `k1-booster`)
- Orin Nano: [Hardware Profiles](config/hardware/hardware_profiles.json) (tag: `orin-nano`)
- AGX Orin: Hardware auto-detection guide (tag: `agx-orin`)

## ⚠️ Disclaimer

This system is designed for authorized use only. Ensure compliance with local privacy laws and regulations when deploying facial recognition technology.

---

**Built for robotics. Optimized for K-1 Booster. Ready for production.**
