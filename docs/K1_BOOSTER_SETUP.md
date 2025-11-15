# K-1 Booster Robot Setup Guide

Complete setup guide for running WhoAmI on the K-1 humanoid robot from Booster Robotics.

## Overview

The K-1 is a 95cm tall commercial humanoid robot with:
- **Jetson Orin NX 16GB** (pre-installed by Booster)
- **Built-in motors** controlled via Booster SDK (DDS/ROS2)
- **2-DoF head control**: yaw ±60°, pitch -30° to 45°
- **18-DoF total**: Arms, legs, head movement
- **Zod camera** for vision (USB 3.0)
- **Audio I/O** for voice interaction

**Important**: K-1 uses the Booster SDK for all motor control. No external servos or serial connections needed - everything is controlled via DDS (Fast-DDS/ROS2).

## Prerequisites

Your K-1 should come with:
- JetPack 5.x pre-installed
- Booster SDK installed (check with `python3 -c "import booster_robotics_sdk_python"`)
- Network connectivity (WiFi or Ethernet)
- SSH access enabled

## Initial Connection

### 1. Find K-1 IP Address

Check your router or use network scan:
```bash
# From your computer
nmap -sn 192.168.x.0/24 | grep -i jetson
```

### 2. SSH to K-1

```bash
# Default credentials (change after first login!)
ssh booster@192.168.x.x
```

### 3. Verify Booster SDK

```bash
# Check SDK is installed
python3 -c "from booster_robotics_sdk_python import B1LocoClient, ChannelFactory; print('Booster SDK OK')"
```

If not installed, contact Booster Robotics support.

## WhoAmI Installation

### 1. Clone Repository

```bash
cd ~
git clone https://github.com/alanchelmickjr/whoami.git
cd whoami
```

### 2. Install System Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install essentials
sudo apt install -y \
    python3-pip \
    python3-venv \
    git \
    portaudio19-dev \
    libsndfile1 \
    ffmpeg \
    sox \
    espeak-ng \
    alsa-utils \
    pulseaudio
```

### 3. Install Python Dependencies

```bash
# Create virtual environment (optional but recommended)
python3 -m venv ~/whoami_env
source ~/whoami_env/bin/activate

# Install WhoAmI
cd ~/whoami
pip install -e .

# Install requirements
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
# Test imports
python3 -c "
from booster_robotics_sdk_python import B1LocoClient, ChannelFactory, RobotMode
from whoami.yolo_face_recognition import K1FaceRecognitionSystem
from whoami.voice_interaction import VoiceInteraction
print('All imports OK!')
"
```

### 5. Set Up Gun.js Storage (Local Face/Conversation Database)

```bash
# Create storage directory for face recognition data
sudo mkdir -p /opt/whoami/gun_storage
sudo chown -R booster:booster /opt/whoami/gun_storage

# Test Gun.js storage
python3 -c "
from whoami.gun_storage import GunStorageManager
gun = GunStorageManager(robot_id='twiki', config={'storage_dir': '/opt/whoami/gun_storage'})
print('Gun.js storage initialized!')
print(gun.get_storage_stats())
"
```

**Why Gun.js for K-1:**
- **Offline-first**: Works without internet (critical for battery-powered robot)
- **Low resource**: ~70-135MB RAM vs PostgreSQL's 800MB+
- **Embedded**: No database daemon to manage
- **P2P capable**: Future multi-robot sync when needed
- **Auto-recovery**: CRDT handles power loss gracefully

**Storage structure:**
```
/opt/whoami/gun_storage/
├── twiki/
│   ├── memories.db          # SQLite backend (face data, conversations)
│   ├── .master.key          # Encryption key (permissions: 600)
│   └── person_profiles.json # Face recognition profiles
```

## Configuration

### 1. Review K-1 Config

```bash
cat config/k1_booster_config.json
```

Key settings:
```json
{
  "voice_reporting": {
    "enabled": true,
    "engine": "f5-tts",
    "fallback_engine": "pyttsx3"
  },
  "face_recognition": {
    "enabled": true,
    "model": "yolov8n.pt",
    "face_db_path": "data/face_db.pkl"
  }
}
```

### 2. Network Interface Selection

The Booster SDK needs a network interface for DDS communication:

- **`127.0.0.1`** - Loopback (local testing only)
- **`eth0`** - Wired Ethernet (initial setup, tethered)
- **`wlan0`** - WiFi (wireless operation, tested with Xbox controller)

**Note**: This is for SDK's internal DDS communication, NOT for SSH!

## Basic Testing

### 1. Test Robot Connection

```python
# test_connection.py
from booster_robotics_sdk_python import B1LocoClient, ChannelFactory, RobotMode
import time

# Initialize SDK
# Use '127.0.0.1' for local testing, 'wlan0' for wireless, 'eth0' for wired
ChannelFactory.Instance().Init(0, '127.0.0.1')

booster = B1LocoClient()
booster.Init()

# Check robot state
print("Robot connected!")
print(f"Battery: {booster.GetBatteryPercentage()}%")

# Test mode change
print("Changing to PREP mode...")
booster.ChangeMode(RobotMode.kPrepare)
time.sleep(2)

print("Test complete!")
```

Run:
```bash
python3 test_connection.py
```

### 2. Test Head Movement

```python
# test_head.py
from booster_robotics_sdk_python import B1LocoClient, ChannelFactory, RobotMode
import time

ChannelFactory.Instance().Init(0, '127.0.0.1')
booster = B1LocoClient()
booster.Init()

# Must be in PREP or WALK mode for head control
booster.ChangeMode(RobotMode.kPrepare)
time.sleep(2)

print("Testing head movement...")

# Center
booster.RotateHead(0.0, 0.0)
time.sleep(1)

# Look left
booster.RotateHead(0.0, 0.785)  # 45° yaw
time.sleep(1)

# Look right
booster.RotateHead(0.0, -0.785)  # -45° yaw
time.sleep(1)

# Look up
booster.RotateHead(0.3, 0.0)  # 17° pitch
time.sleep(1)

# Center
booster.RotateHead(0.0, 0.0)
print("Head test complete!")
```

Run:
```bash
python3 test_head.py
```

### 3. Test Camera

```bash
# Simple camera test (if cam_yolo.py exists)
python3 cam_yolo.py
```

Or use OpenCV:
```python
import cv2

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if ret:
    print(f"Camera OK: {frame.shape}")
    cv2.imwrite('test_frame.jpg', frame)
else:
    print("Camera failed!")
cap.release()
```

### 4. Test Voice (pyttsx3)

```bash
# Test default TTS
python3 -c "
from whoami.voice_interaction import VoiceInteraction
voice = VoiceInteraction(tts_engine='pyttsx3')
voice.say('K-1 voice system operational')
"
```

## F5-TTS Setup (High-Quality Voice)

For natural-sounding neural TTS, install F5-TTS:

### 1. Install F5-TTS

```bash
# Install PyTorch (if not installed)
pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install F5-TTS
pip3 install f5-tts

# Verify
python3 -c "from f5_tts.api import F5TTS; print('F5-TTS OK')"
```

### 2. Prepare Voice Sample

F5-TTS uses voice cloning, so you need a reference audio file:

```bash
# Create voices directory
sudo mkdir -p /opt/whoami/voices

# Record 10-30 second sample (or use pre-recorded)
# Sample should be clear speech, no background noise
```

**Option A**: Record with arecord:
```bash
arecord -D hw:2,0 -f S16_LE -r 16000 -c 1 -d 15 /opt/whoami/voices/k1_voice.wav
# Speak clearly for 15 seconds
```

**Option B**: Use pitch-shifted sample:
```bash
# Start with any adult voice sample
sox adult_voice.wav /opt/whoami/voices/k1_voice.wav pitch 400
```

### 3. Configure F5-TTS

Edit `config/k1_booster_config.json`:
```json
{
  "voice_reporting": {
    "enabled": true,
    "engine": "f5-tts",
    "fallback_engine": "pyttsx3",
    "model_type": "F5-TTS",
    "ref_audio": "/opt/whoami/voices/k1_voice.wav",
    "ref_text": "This is the voice for the K-1 robot."
  }
}
```

### 4. Test F5-TTS

```bash
python3 examples/f5tts_voice_demo.py
```

See [F5-TTS Setup Guide](F5-TTS_SETUP.md) for details.

## Running WhoAmI Features

### 1. Face Recognition

```bash
# Run autonomous face exploration
python3 examples/k1_autonomous_face_interaction.py wlan0
```

See [K-1 Autonomous Face Interaction](K1_AUTONOMOUS_FACE_INTERACTION.md) for details.

### 2. Basic Controls

If you have working control scripts:
```bash
# Basic movement controls
python3 basic_controls.py 127.0.0.1

# Camera feed
python3 basic_cam.py

# YOLO detection
python3 cam_yolo.py
```

### 3. Voice Interaction

```python
from whoami.voice_interaction import VoiceInteraction

voice = VoiceInteraction(
    tts_engine='f5-tts',
    f5tts_ref_audio='/opt/whoami/voices/k1_voice.wav'
)

voice.say("Hello, I am K-1!")
```

### 4. Gun.js Storage with Face Recognition

```python
from whoami.gun_storage import GunStorageManager, MemoryCategory, TrustLevel
import time

# Initialize local storage (offline-first)
gun = GunStorageManager(
    robot_id='twiki',
    config={
        'storage_dir': '/opt/whoami/gun_storage',
        'encryption_required': True,
        'auto_share_family': False  # Keep private initially
    }
)

# Store face recognition data (local, encrypted)
face_memory = gun.store_private_memory({
    'name': 'Alice',
    'descriptor': face_embedding,  # From DeepFace
    'last_seen': time.time(),
    'encounter_count': 1
})

# Store conversation note (shareable with sibling robots)
conversation = gun.store_shared_memory(
    data={
        'person': 'Alice',
        'topic': 'her dog Max',
        'note': 'Alice has a golden retriever who loves swimming',
        'timestamp': time.time()
    },
    category=MemoryCategory.FAMILY,
    tags=['conversation', 'pets']
)

# Retrieve memory later
alice_data = gun.retrieve_memory(face_memory)
print(f"Last saw {alice_data['name']} at {alice_data['last_seen']}")

# Optional: Add sibling robot for P2P sync (when both on same network)
# gun.add_peer('robi', '192.168.88.154', 8765, TrustLevel.SIBLING)
# gun.share_with_sibling(conversation, 'robi')

# Check storage stats
stats = gun.get_storage_stats()
print(f"Private memories: {stats['private_memories']}")
print(f"Family memories: {stats['family_memories']}")
print(f"Trusted peers: {stats['trusted_peers']}")
```

**Benefits for Twiki:**
- **Works offline**: No internet needed for face recognition
- **Low overhead**: ~70-135MB RAM (vs PostgreSQL ~800MB)
- **Auto-encrypted**: Face data encrypted at rest
- **Future-proof**: Easy P2P sync when you add more robots
- **Battery-friendly**: Zero background workers draining power

## Robot Operational Modes

K-1 has four operational modes (from Booster SDK):

| Mode | Value | Description | Head Control | Use Case |
|------|-------|-------------|--------------|----------|
| DAMPING | `RobotMode.kDamping` | Motors relaxed, safe for handling | ❌ No | Safe shutdown, manual handling |
| PREPARE | `RobotMode.kPrepare` | Standing, ready for commands | ✅ Yes | Normal operation, head control |
| WALKING | `RobotMode.kWalking` | Walking/locomotion mode | ✅ Yes | Movement, navigation |
| CUSTOM | `RobotMode.kCustom` | Custom programmable mode | ⚙️ Varies | Advanced custom behaviors |

**Mode Transition Sequence**:
```python
# Safe startup sequence
booster.ChangeMode(RobotMode.kDamping)
time.sleep(1)

booster.ChangeMode(RobotMode.kPrepare)
time.sleep(2)  # Wait for robot to stand up

# Now you can control head, arms, etc.
booster.RotateHead(0.0, 0.0)
```

**Recommended Modes for Face Interaction Development**:

For head control and face interaction, use **PREPARE mode**:
```python
# Enable PREPARE mode - stable standing, head control active
booster.ChangeMode(RobotMode.kPrepare)
time.sleep(2.0)  # Wait for robot to stabilize

# Head control now active
booster.RotateHead(0.0, 0.785)  # Look left
time.sleep(1.0)

booster.RotateHead(0.0, -0.785)  # Look right
time.sleep(1.0)

booster.RotateHead(0.0, 0.0)  # Center
```

**Movement Control**:

For walking/navigation, use **WALKING mode**:
```python
# Enable walking mode
booster.ChangeMode(RobotMode.kWalking)
time.sleep(1.0)

# Locomotion commands (from basic_controls.py example)
booster.Move(0.2, 0.0, 0.0)  # Forward at 0.2 m/s
time.sleep(2.0)

booster.Move(0.0, 0.0, 0.5)  # Rotate at 0.5 rad/s
time.sleep(2.0)

booster.Move(0.0, 0.0, 0.0)  # Stop
```

**Safe Shutdown**:
```python
# Always return to DAMPING mode when done
booster.ChangeMode(RobotMode.kDamping)
```

## Troubleshooting

### Booster SDK Connection Failed

```bash
# Check network interface
ip addr show

# Test DDS communication
python3 -c "
from booster_robotics_sdk_python import ChannelFactory
ChannelFactory.Instance().Init(0, 'wlan0')
print('DDS OK')
"
```

### Head Not Moving

**Check mode:**
```python
# Head only works in PREP or WALK mode
booster.ChangeMode(RobotMode.kPrepare)
time.sleep(2)
booster.RotateHead(0.0, 0.0)
```

**Check ranges:**
```python
# K-1 head limits
# Yaw: ±60° (±1.047 rad)
# Pitch: -30° to 45° (-0.524 to 0.785 rad)

# This will fail (out of range):
booster.RotateHead(2.0, 2.0)  # ❌

# This will work:
booster.RotateHead(0.5, 0.5)  # ✅
```

### Camera Not Found

```bash
# List video devices (camera shows as "Zod")
v4l2-ctl --list-devices

# Should show something like:
# Zod (usb-...):
#   /dev/video0
#   /dev/video1

# Test OpenCV
python3 -c "import cv2; print(cv2.VideoCapture(0).read()[0])"
```

### F5-TTS Out of Memory

```bash
# Use smaller model
# In voice_interaction.py, use 'F5-TTS' (default) not 'E2-TTS'

# Or increase swap
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Audio Not Working

```bash
# List audio devices
aplay -l
arecord -l

# Test speaker
speaker-test -t wav -c 2

# Check PulseAudio
pulseaudio --check || pulseaudio --start
```

## Performance Optimization

### 1. Set Max Performance

```bash
# Jetson Orin NX max performance mode
sudo nvpmodel -m 0
sudo jetson_clocks

# Verify
sudo nvpmodel -q
```

### 2. Monitor Resources

```bash
# Real-time stats
tegrastats

# GPU usage
tegrastats | grep GR3D
```

### 3. YOLO Optimization

Use smaller model for faster detection:
```python
from ultralytics import YOLO

# Nano (fastest, 80 FPS on Orin NX)
model = YOLO('yolov8n.pt')

# Small (balanced)
model = YOLO('yolov8s.pt')

# Medium (slower but more accurate)
model = YOLO('yolov8m.pt')
```

## Safety Notes

### ⚠️ Important Safety Guidelines

1. **Always start in DAMP mode** - Motors are relaxed, safe to handle
2. **Clear space before WALK mode** - Robot needs room to move
3. **Emergency stop** - Change to DAMP mode immediately if needed
4. **Battery monitoring** - Check battery level regularly
5. **Head movement limits** - Stay within safe ranges to avoid damage

```python
# Emergency stop pattern
def emergency_stop(booster):
    booster.ChangeMode(RobotMode.kDamping)
    print("EMERGENCY STOP - Robot in DAMPING mode")
```

## Next Steps

1. **Test basic features** - Head movement, voice, camera
2. **Configure voice** - Set up F5-TTS with appropriate voice
3. **Face recognition** - Run autonomous face exploration
4. **Custom behaviors** - Develop your own interactions

## See Also

- [K-1 First Test Protocol](K1_FIRST_TEST.md) - Complete test procedures
- [K-1 Autonomous Face Interaction](K1_AUTONOMOUS_FACE_INTERACTION.md) - Face scanning and conversation tracking
- [F5-TTS Setup](F5-TTS_SETUP.md) - High-quality neural voice setup
- [Booster SDK Documentation](https://github.com/BoosterRobotics/booster_robotics_sdk) - Official SDK docs

## Support

For K-1 issues:
- **Booster Robotics Support** - Hardware and SDK issues
- **WhoAmI Issues** - [GitHub Issues](https://github.com/alanchelmickjr/whoami/issues)
- **Community** - Tag with `k1-robot`
