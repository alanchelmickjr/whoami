# WhoAmI - Advanced Robotic Face Recognition

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Node.js 18+](https://img.shields.io/badge/node.js-18+-green.svg)](https://nodejs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Oak D](https://img.shields.io/badge/Oak%20D-Series%203-green.svg)](https://docs.luxonis.com/)
[![K-1 Booster](https://img.shields.io/badge/K--1-Booster-orange.svg)](https://www.aparobot.com/robots/booster-k1)

An advanced facial recognition and voice interaction system built for the **Jetson Orin NX K-1 Booster** robotics platform. Features 3-axis gimbal control, voice-based person identification, hardware-backed encryption, and multi-modal interaction capabilities.

## 🤖 Primary Platform: K-1 Booster Robot

**The WhoAmI system is optimized for the Jetson Orin NX on K-1 Booster carrier board** - a powerful robotics platform with advanced I/O capabilities:

### K-1 Booster Capabilities

- 🎮 **2-Axis Gimbal System**
  - 1-axis head gimbal (tilt) for camera/eye vertical movement
  - 1-axis neck gimbal (tilt) for head forward/back orientation
  - Coordinated movement for natural tracking and scanning
  - Feetech servo control via dual serial ports

- 🗣️ **Voice Interaction**
  - Ask unknown people for their names
  - Greet known people by name
  - Audio source tracking and localization
  - Text-to-speech status reporting
  - Offline speech recognition (Vosk)

- 🌐 **Dual Ethernet Connectivity**
  - Primary and backup network paths
  - Automatic failover for mission-critical operations
  - Remote VNC access for operators
  - Distributed Gun.js database sync

- 🎯 **Advanced Features**
  - 16GB RAM for complex AI workloads
  - 8 CPU cores for parallel processing
  - 1024 CUDA cores for GPU acceleration
  - Expanded I/O (6 USB ports, dual M.2, PCIe)
  - Hardware-backed encryption for secure data

### K-1 Robot Architecture

```
┌─────────────────────────────────────────────────────────┐
│         Jetson Orin NX on K-1 Booster Platform          │
├─────────────────────────────────────────────────────────┤
│  👁️  Oak D Series 3 Camera (Depth + Face Detection)     │
├─────────────────────────────────────────────────────────┤
│  🎮 Gimbal Control                                       │
│     • Head: /dev/ttyTHS1 (tilt)                        │
│     • Neck: /dev/ttyTHS2 (tilt)                        │
├─────────────────────────────────────────────────────────┤
│  🎤 Audio I/O (hw:2,0)                                   │
│     • Voice interaction & name asking                   │
│     • Audio source tracking                            │
│     • TTS status reporting                             │
├─────────────────────────────────────────────────────────┤
│  🌐 Dual Ethernet (eth0 + eth1)                         │
│     • VNC remote access (port 5900)                    │
│     • SSH control (port 22)                            │
│     • Web interface (port 8080)                        │
├─────────────────────────────────────────────────────────┤
│  🧠 Face Recognition + Gun.js Database                   │
│     • Local-first encrypted storage                    │
│     • Hardware-backed encryption                       │
│     • Distributed P2P sync                             │
└─────────────────────────────────────────────────────────┘
```

**➡️ See [K-1 Booster Setup Guide](docs/K1_BOOSTER_SETUP.md) for complete hardware configuration**

## Features

### Robot Interaction
- 🤝 **Natural Conversation Flow**
  - Robot: "Hello! I don't think we've met. What's your name?"
  - Person: "My name is John"
  - Robot: "Did you say John? Please say yes or no."
  - Person: "Yes"
  - Robot: "Nice to meet you, John!" (saves face + name)
  - Next time: "Welcome back, John!"

- 👁️ **Advanced Face Recognition**
  - Real-time face detection with Oak D Series 3
  - 128-dimensional face encoding
  - Confidence-based matching
  - Batch processing and event callbacks
  - Thread-safe operations

- 🎭 **Multi-Modal Sensing**
  - Visual: Face detection, tracking, recognition
  - Audio: Voice identification, source localization
  - Depth: 3D spatial awareness (Oak D stereo)
  - Motion: Gimbal-based tracking

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

### Operational Modes

The K-1 robot supports three operational modes:

1. **🎮 Remote VNC Mode**
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

## Quick Start (K-1 Booster)

### 1. Install on K-1

```bash
# Clone repository
git clone https://github.com/alanchelmickjr/whoami.git
cd whoami

# Run K-1 setup script
./jetson_setup_v2.sh --full

# Or install manually
pip install -r requirements.txt
sudo apt-get install -y portaudio19-dev espeak flac alsa-utils pulseaudio
```

### 2. Verify Hardware Detection

```bash
# Auto-detect K-1 booster
python -m whoami.hardware_detector

# Expected output:
# Detected Hardware: Jetson Orin NX on K-1 Booster
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
# Test head gimbal (tilt)
python3 -c "
from whoami.feetech_sdk import FeetchController
head = FeetchController('/dev/ttyTHS1', 1000000)
head.ping(1)  # Tilt
print('Head gimbal OK')
"

# Test neck gimbal (tilt)
python3 -c "
from whoami.feetech_sdk import FeetchController
neck = FeetchController('/dev/ttyTHS2', 1000000)
neck.ping(2)  # Neck tilt
print('Neck gimbal OK')
"
```

## Alternative Platforms

While optimized for K-1 Booster, the system also supports:

### Jetson Orin Nano (Lighter Alternative)
- Single-board DevKit configuration
- Custom gimbal configuration support
- 8GB RAM, 6 CPU cores
- Good for testing and development

### Other Platforms
- Jetson AGX Orin (64GB, more powerful)
- Raspberry Pi 4 (lightweight deployment)
- Mac M-Series (development/testing)
- Linux Desktop (simulation)

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

## K-1 Hardware Specs

### Jetson Orin NX Module
- **CPU**: 8-core ARM Cortex-A78AE
- **GPU**: 1024 CUDA cores, 32 Tensor cores
- **Memory**: 16GB LPDDR5
- **AI Performance**: 100 TOPS
- **Power**: 10W-25W configurable

### K-1 Carrier Board
- **Ethernet**: 2x Gigabit (failover capable)
- **USB**: 6 ports (4x USB 3.2)
- **Storage**: 2x M.2 slots (NVMe)
- **Expansion**: PCIe 4.0 x4
- **Serial**: 3x UART (servo control)
- **GPIO**: 40-pin header
- **Power**: 19V DC, 65W minimum

### Peripherals
- **Camera**: OAK-D Series 3 (depth + RGB)
- **Servos**: 3x Feetech STS/SCS (1Mbps)
- **Audio**: USB Audio Class 2.0
- **Network**: Dual Ethernet + optional WiFi

## Documentation

### K-1 Booster Guides
- **[K-1 Booster Setup](docs/K1_BOOSTER_SETUP.md)** ⭐ Complete setup guide
- **[Hardware Configuration](docs/HARDWARE_CONFIG_GUIDE.md)** - Hardware profiles
- **[Voice Interaction](docs/VOICE_INTERACTION_GUIDE.md)** - Voice system guide
- **[Gimbal 3DOF](docs/GIMBAL_3DOF_GUIDE.md)** - Gimbal control

### Core Documentation
- **[API Reference](docs/API_REFERENCE.md)** - Complete API docs
- **[Usage Guide](docs/USAGE_GUIDE.md)** - Usage examples
- **[Security](SECURITY.md)** - Hardware encryption
- **[Deployment](DEPLOYMENT.md)** - Production deployment

### Advanced Features
- **[Genesis VLA](docs/GENESIS_VLA_GUIDE.md)** - Vision-Language-Action training
- **[Spatial Awareness](docs/SPATIAL_AWARENESS_GUIDE.md)** - Environmental mapping
- **[Servo Safety](docs/SERVO_SAFETY_GUIDE.md)** - Safety monitoring

### Installation & Setup
- **[Installation Guide](INSTALLATION.md)** - Complete installation
- **[Jetson & M4 Setup](SETUP_JETSON_M4.md)** - Platform setup
- **[Quick Reference](SETUP_QUICK_REFERENCE.md)** - Quick commands

## Configuration

### K-1 Booster Config

Pre-configured profile at `config/k1_booster_config.json`:

```json
{
  "hardware_profile": "jetson_orin_nx_k1",
  "head_gimbal": {
    "serial_port": "/dev/ttyTHS1",
    "servos": {"pan": {"id": 1}, "tilt": {"id": 2}}
  },
  "neck_gimbal": {
    "serial_port": "/dev/ttyTHS2",
    "servos": {"neck_tilt": {"id": 3}}
  },
  "audio": {
    "input_device": "hw:2,0",
    "output_device": "hw:2,0",
    "features": ["voice_reporting", "audio_tracking", "speech_synthesis"]
  },
  "network": {
    "ethernet_primary": "eth0",
    "ethernet_secondary": "eth1",
    "vnc_port": 5900
  }
}
```

## Roadmap

### Completed ✅
- [x] K-1 Booster hardware profiles
- [x] 2-axis gimbal control
- [x] Voice interaction system
- [x] Audio source tracking
- [x] Hardware-backed encryption
- [x] Dual Ethernet failover
- [x] Remote VNC access
- [x] Multi-platform support

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

### K-1 Booster Performance
- **Face Detection**: 30 FPS (Oak D)
- **Face Recognition**: 20 FPS
- **Voice Response**: <500ms
- **Gimbal Tracking**: 60Hz
- **Network Latency**: <50ms (Gigabit)

### Power Consumption
- Idle: ~15W
- Active (all systems): ~35W
- Peak: ~45W

## Credits

Built with:
- [Oak D Series 3](https://docs.luxonis.com/) - Depth camera
- [Gun.js](https://gun.eco/) - Decentralized database
- [face_recognition](https://github.com/ageitgey/face_recognition) - Face detection
- [Vosk](https://alphacephei.com/vosk/) - Offline speech recognition
- [pyttsx3](https://pyttsx3.readthedocs.io/) - Text-to-speech
- [Feetech Servos](http://www.feetechrc.com/) - Gimbal control

**Powered by NVIDIA Jetson Orin NX on K-1 Booster Carrier Board**

## License

MIT License - See LICENSE file for details.

## Support

For K-1 Booster specific issues:
- 🐛 [GitHub Issues](https://github.com/alanchelmickjr/whoami/issues)
- 📖 [K-1 Setup Guide](docs/K1_BOOSTER_SETUP.md)
- 💬 Tag issues with `k1-booster`

## ⚠️ Disclaimer

This system is designed for authorized use only. Ensure compliance with local privacy laws and regulations when deploying facial recognition technology.

---

**Built for robotics. Optimized for K-1 Booster. Ready for production.**
