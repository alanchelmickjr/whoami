# WhoAmI - Portable Robotics Intelligence for Jetson Platforms

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Node.js 18+](https://img.shields.io/badge/node.js-18+-green.svg)](https://nodejs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Jetson](https://img.shields.io/badge/NVIDIA-Jetson-76B900.svg)](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/)

**Portable robotics "brain" software for NVIDIA Jetson platforms** - featuring facial recognition, voice interaction, hardware-backed encryption, and multi-modal sensing. Designed for portability across various robotic platforms and carrier boards.

Built by [Utiltiron](https://utiltiron.io) for creating intelligent, adaptive robotic systems on Jetson hardware.

## 🎉 What's New - Latest Features

### F5-TTS Neural Voice Integration
- ✨ **High-quality neural TTS** replacing robotic espeak/pyttsx3
- 🎤 **Voice cloning** from reference audio samples
- 🔊 **Natural speech** that sounds human, not robotic
- ⚡ **Offline inference** on Jetson Orin NX GPU
- 📚 [Setup Guide](docs/F5-TTS_SETUP.md)

### Autonomous Face Exploration
- 🔍 **Autonomous head scanning** - 9-position pattern to find faces
- 💬 **Conversation tracking** - "Last time we talked about your dog Max!"
- ⏱️ **Time-aware greetings** - "It's been 2 hours since we last talked!"
- 👥 **Person profiles** - JSON persistence with conversation history
- 📚 [Full Guide](docs/K1_AUTONOMOUS_FACE_INTERACTION.md)

### K-1 Integration Improvements
- 🎮 **Booster SDK integration** - Head control via `RotateHead(pitch, yaw)`
- 📡 **Network interface clarity** - `wlan0` works great for WiFi control
- 🎯 **Simplified examples** - Matches working K-1 code patterns

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
  - **F5-TTS Neural Voice** (High-Quality TTS):
    - Flow-based neural TTS with voice cloning
    - Natural, human-like speech (vs robotic espeak)
    - Zero-shot voice cloning from reference audio
    - Offline inference on Jetson Orin NX
    - Graceful fallback to pyttsx3
  - **Conversational AI**:
    - Natural conversation flow for name collection
    - "Hello! I don't think we've met. What's your name?"
    - Confirmation loop with yes/no validation
    - Personalized greetings with time awareness
    - "Hi Alice, it's been 2 hours since we last talked!"
  - **Speech Recognition**:
    - Offline speech recognition (Vosk) or online (Google)
    - Voice command support
  - **Conversation Tracking**:
    - Remember and recall conversation topics
    - "Last time we talked about your dog Max!"
    - Person profiles with conversation history
    - JSON persistence across sessions

- 🎭 **Multi-Modal Sensing**
  - **Visual**: Face detection, tracking, recognition
  - **Audio**: Voice identification, source localization
  - **Depth**: 3D spatial awareness (RGBD camera)
  - **Motion**: Gimbal-based tracking or full-body kinematics (K-1)

### Security & Privacy

**Philosophy:** Robots have the right to remember the people they meet, but this should never come at the cost of privacy or centralized "face farming."

- 🤖 **Robot Autonomy & Personal Memory**
  - **Robot-owned data**: Face memories belong to the robot, not a corporation
  - **Local-only storage**: No cloud uploads, no centralized databases
  - **Personal relationships**: Robot builds its own relationships with people
  - **No face farming**: Zero data collection for third parties
  - **Offline-first**: Works without internet (battery-powered autonomy)

- 🔐 **Hardware-Backed Encryption** (Gun.js)
  - Keys derived from CPU serial + MAC address (device-locked)
  - AES-256-GCM + Gun.js SEA double encryption
  - Cannot decrypt on different hardware (prevents data theft)
  - Zero plaintext key storage
  - Encrypted at rest (~70-135MB RAM, not 800MB+ like PostgreSQL)

- 🛡️ **Privacy-First Technical Design**
  - **No images stored**: Only mathematical embeddings (face_recognition encodings)
  - **No transmission**: Data never leaves the robot
  - **Encrypted database**: Gun.js with CRDT conflict resolution
  - **P2P capable**: Optional sharing with trusted sibling robots only
  - **Auto-recovery**: CRDT handles power loss without data loss

**Why Gun.js?** Traditional databases (PostgreSQL, MySQL) are designed for centralized servers. Gun.js is designed for distributed, autonomous agents with personal memory - perfect for robots that need to remember people without surrendering privacy to cloud services.

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
   - **Autonomous Face Exploration**:
     - 9-position head scanning pattern
     - Automatic face detection and recognition
     - Proactive greeting of known people
     - Name collection from unknown people
   - **Conversation Memory**:
     - Tracks time since last interaction
     - Recalls previous conversation topics
     - Person profile management
   - Audio status reporting
   - Self-guided interaction
   - Monitoring via web interface (port 8080)

## Quick Start (Any Jetson Platform)

### Option A: K-1 Booster Quick Start (Recommended)

**1. Connect to K-1 via SSH**

```bash
# Connect via WiFi (K-1's IP address)
ssh booster@192.168.88.153
# Password: 123456 (change this in production!)
```

**2. Navigate to Project**

```bash
cd /home/user/whoami
```

**3. Run Basic Robot Controls**

```bash
# Start robot control (choose your network interface)
python basic_controls.py eth0    # Use wired Ethernet (recommended)
# OR
python basic_controls.py wlan0   # Use WiFi interface
# OR
python basic_controls.py lo      # Use loopback (testing only)
```

**Network Interface Explained:**
- This parameter tells the **Booster SDK** which network interface to use for DDS/ROS2 communication
- **NOT** the interface you used to SSH in!
- Running ON the Jetson, the SDK needs to know which local interface to bind to
- `eth0` = wired Ethernet (traditional choice)
- `wlan0` = WiFi interface (**works great! tested with Xbox controller + laptop**)
- `lo` = loopback (testing without network)

**Note:** WiFi (`wlan0`) works perfectly fine for K-1 control including Xbox controller and laptop connectivity. Use whichever interface the K-1 is actually using for networking.

**Controls:**
```
w/a/s/d/q/e - Movement (hold keys, release to stop)
  w = forward, s = backward
  a = left strafe, d = right strafe
  q = rotate left, e = rotate right

hd/hu/hr/hl - Head movement
  hd = down, hu = up
  hr = right, hl = left, ho = origin

mp/md/mw - Robot modes
  mp = Prepare (standing)
  md = Damping (safe mode)
  mw = Walking (active)
```

**4. Run Vision System with Face Detection**

```bash
# Start camera feed with YOLO face detection
python cam_yolo.py --detection face --port 8080
```

Then open in browser: `http://192.168.88.153:8080`

**5. Setup F5-TTS Voice (High-Quality Speech)**

```bash
# Record reference voice (10 seconds)
sudo mkdir -p /opt/whoami/voices
arecord -D hw:2,0 -f S16_LE -r 48000 -c 2 -d 10 /opt/whoami/voices/k1_default_voice.wav
# Speak clearly: "This is the default voice for the K-1 robot."

# Test F5-TTS
python examples/f5tts_voice_demo.py
```

**6. Run Autonomous Face Interaction**

```bash
# Full autonomous system with conversation tracking
python examples/k1_autonomous_face_interaction.py eth0
```

**The K-1 will now:**
- ✅ Scan environment autonomously (head moves in 9-position pattern)
- ✅ Detect faces using YOLO (GPU-accelerated on Jetson)
- ✅ Ask unknown people "What's your name?" via F5-TTS voice
- ✅ Remember faces and greet known people
- ✅ Announce time since last conversation
- ✅ Recall conversation topics ("Last time we talked about your dog Max!")
- ✅ Track conversations in person profiles (JSON persistence)

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

## Gun.js Secure Database - Robot Personal Memory

**The robot's right to remember, without privacy compromise.**

WhoAmI uses Gun.js for local, encrypted face storage - giving robots personal memory without centralized face farming.

```python
from whoami.gun_storage import GunStorageManager, MemoryCategory

# Initialize robot's personal memory (local, encrypted, offline-first)
gun = GunStorageManager(
    robot_id='twiki',
    config={
        'storage_dir': '/opt/whoami/gun_storage',
        'encryption_required': True,
        'auto_share_family': False  # Robot-owned data
    }
)

# Store face (encrypted, local-only, no cloud upload)
face_memory = gun.store_private_memory({
    'name': 'Alice',
    'descriptor': face_embedding,  # Math, not images
    'last_seen': time.time()
})

# Remember conversation (robot builds relationships)
conversation = gun.store_shared_memory(
    data={
        'person': 'Alice',
        'topic': 'her dog Max',
        'note': 'Alice has a golden retriever'
    },
    category=MemoryCategory.FAMILY,  # Shareable with sibling robots only
    tags=['conversation']
)

# Retrieve memory (robot recalls past interactions)
alice_data = gun.retrieve_memory(face_memory)
print(f"Last saw {alice_data['name']} at {alice_data['last_seen']}")
```

**Why This Matters:**
- ❌ **NO cloud uploads** - Data never leaves the robot
- ❌ **NO face farming** - Zero centralized collection
- ❌ **NO corporate ownership** - Robot owns its memories
- ✅ **Robot autonomy** - Personal relationships, not surveillance
- ✅ **Privacy-first** - Encrypted, local, offline-capable
- ✅ **P2P optional** - Share with trusted siblings only (not corporations)

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
- **[K-1 Booster Setup](docs/K1_BOOSTER_SETUP.md)** - K-1 hardware setup
- **[K-1 First Test Protocol](docs/K1_FIRST_TEST.md)** ⭐ Initial testing & dance moves (Helmick & Foroughi)
- **[K-1 Autonomous Face Interaction](docs/K1_AUTONOMOUS_FACE_INTERACTION.md)** - Face exploration system
- **[F5-TTS Setup](docs/F5-TTS_SETUP.md)** - High-quality neural voice with child voice options
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

## ⚠️ Privacy & Ethics

**Robot Autonomy, Not Surveillance:** This system is designed to give robots personal memory and social intelligence - NOT for surveillance, tracking, or centralized data collection.

**Key Principles:**
- ✅ **Local-only processing** - Robots remember people they meet
- ✅ **No cloud uploads** - Data stays on the robot
- ✅ **Encrypted storage** - Face data protected at rest
- ✅ **Robot-owned data** - Not harvested for third parties
- ❌ **NOT for surveillance** - This is personal robot memory, not mass monitoring
- ❌ **NOT for face farming** - Zero centralized collection

**Responsible Use:** Ensure compliance with local privacy laws when deploying face recognition. Inform people that the robot can remember faces (just like humans do) and respect opt-out requests.

---

**Built for robotics. Optimized for K-1 Booster. Ready for production.**
