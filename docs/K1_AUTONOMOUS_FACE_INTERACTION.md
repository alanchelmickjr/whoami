# K-1 Autonomous Face Interaction

## Overview

The K-1 can autonomously explore its environment, find faces, and engage in personalized interactions with conversation recall.

## Features

### ✅ Already Working

1. **Face Detection & Recognition** (yolo_face_recognition.py)
   - YOLO v8 for fast face detection
   - DeepFace for face recognition
   - Real-time processing on Jetson Orin NX

2. **Personalized Greetings with Time Tracking**
   ```python
   "Hi Alice, it's been 2 hours and 15 minutes since we last talked!"
   ```
   - Tracks `first_seen`, `last_seen`, `encounter_count`
   - Human-friendly time formatting

3. **Voice Interaction**
   - Ask unknown people for their names
   - TTS greetings (pyttsx3 or F5-TTS)
   - Speech recognition for responses

4. **Head Movement** (K-1 Booster SDK)
   - 2 DoF head (yaw, pitch)
   - `RotateHead(pitch, yaw)` API
   - Range: yaw ±60°, pitch -30° to 45°

### ✨ New: Autonomous Exploration

**k1_face_explorer.py** adds:

1. **Autonomous Head Scanning**
   ```python
   # Predefined scan positions
   scan_positions = [
       (0.0, 0.0),      # Center
       (-0.785, 0.0),   # Left (-45°)
       (0.785, 0.0),    # Right (45°)
       (0.0, 0.3),      # Up
       (0.0, -0.3),     # Down
   ]
   ```

2. **Conversation Tracking**
   ```python
   # Store conversation notes
   explorer.add_conversation_note(
       "Alice",
       "her dog Max",
       "Alice has a golden retriever who loves swimming"
   )

   # Greeting with recall
   "Hi Alice, it's been 3 hours since we last talked! "
   "Last time we talked about her dog Max."
   ```

3. **Person Profiles**
   - Extended metadata beyond face_db
   - Conversation history (last 10 conversations)
   - Preferences storage
   - JSON persistence

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  K-1 Face Interaction                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────┐      ┌──────────────┐                │
│  │   Camera    │──────▶│ YOLO Face    │                │
│  │  (ZED/OAK)  │      │  Detection   │                │
│  └─────────────┘      └──────┬───────┘                │
│                              │                          │
│                              ▼                          │
│                       ┌──────────────┐                 │
│                       │  DeepFace    │                 │
│                       │ Recognition  │                 │
│                       └──────┬───────┘                 │
│                              │                          │
│         ┌────────────────────┴────────────────┐        │
│         ▼                                     ▼        │
│  ┌─────────────┐                      ┌──────────────┐│
│  │   K-1       │                      │   Voice      ││
│  │   Head      │◀─────────────────────│ Interaction  ││
│  │  Control    │                      │ (Ask/Greet)  ││
│  └─────────────┘                      └──────┬───────┘│
│         │                                     │        │
│         ▼                                     ▼        │
│  ┌─────────────────────────────────────────────────┐  │
│  │        Face Explorer (Conversation Tracking)    │  │
│  │  - Person Profiles                              │  │
│  │  - Conversation Notes                           │  │
│  │  - Autonomous Scanning                          │  │
│  └─────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Usage

### Quick Start

```bash
# SSH to K-1
ssh booster@192.168.88.153

# Run demo
cd /home/user/whoami
python examples/k1_autonomous_face_interaction.py eth0
```

### Demo Mode (No Robot)

```bash
# Test without hardware
python examples/k1_autonomous_face_interaction.py
```

### Integration Example

```python
from booster_robotics_sdk_python import B1LocoClient, ChannelFactory
from whoami.k1_face_explorer import K1FaceExplorer
from whoami.voice_interaction import VoiceInteraction

# Initialize K-1
ChannelFactory.Instance().Init(0, 'eth0')
booster = B1LocoClient()
booster.Init()

# Create face explorer
explorer = K1FaceExplorer(booster_client=booster)
voice = VoiceInteraction()

# Greet someone
greeting = explorer.greet_person("Alice", include_conversation=True)
voice.say(greeting)

# Add conversation note
explorer.add_conversation_note(
    "Alice",
    "her vacation plans",
    "Alice is planning a trip to Japan next month"
)
```

### Autonomous Scanning

```python
# Start autonomous exploration
explorer.start_autonomous_exploration()

# Let it run...
time.sleep(60)

# Stop
explorer.stop_autonomous_exploration()
```

## Head Control

### K-1 Head Specification

From Booster K-1 documentation:
- **Degrees of Freedom**: 2 (Yaw, Pitch)
- **Yaw Range**: ±60° (±1.047 rad)
- **Pitch Range**: -30° to 45° (-0.524 to 0.785 rad)
- **API**: `RotateHead(pitch, yaw)` - note order!

### Movement Examples

```python
# Center
booster.RotateHead(0.0, 0.0)

# Look left
booster.RotateHead(0.0, 0.785)  # yaw = 45°

# Look right
booster.RotateHead(0.0, -0.785)  # yaw = -45°

# Look up
booster.RotateHead(0.3, 0.0)  # pitch = 17°

# Look down
booster.RotateHead(-0.3, 0.0)  # pitch = -17°
```

### Scan Pattern

The explorer uses a 9-position scan pattern:
1. Center
2. Left
3. Right
4. Up
5. Down
6. Upper left
7. Upper right
8. Lower left
9. Lower right

## Conversation Tracking

### Data Model

```python
@dataclass
class ConversationNote:
    timestamp: float
    topic: str
    note: str

@dataclass
class PersonProfile:
    name: str
    first_seen: float
    last_seen: float
    encounter_count: int
    conversations: List[ConversationNote]
    preferences: Dict[str, Any]
```

### Storage

Profiles are stored in `person_profiles.json`:

```json
{
  "Alice": {
    "name": "Alice",
    "first_seen": 1234567890.0,
    "last_seen": 1234567950.0,
    "encounter_count": 3,
    "conversations": [
      {
        "timestamp": 1234567890.0,
        "topic": "her dog Max",
        "note": "Alice has a golden retriever who loves swimming"
      }
    ],
    "preferences": {}
  }
}
```

### API

```python
# Add note
explorer.add_conversation_note(
    "Alice",
    "her garden",
    "Growing tomatoes and basil this summer"
)

# Get profile
profile = explorer.get_profile("Alice")
print(f"Encounters: {profile.encounter_count}")

# Get recent conversation
recent = profile.get_recent_conversation()
if recent:
    print(f"Last topic: {recent.topic}")
```

## Integration with Existing Systems

### With YOLO Face Recognition

```python
from whoami.yolo_face_recognition import K1FaceRecognitionSystem

# Create systems
face_system = K1FaceRecognitionSystem(enable_voice=True)
explorer = K1FaceExplorer(
    booster_client=booster,
    face_system=face_system
)

# Process detections
results = face_system.process_frame(frame)
for result in results:
    explorer.process_face_detection(result)
```

### With Vision Behaviors

```python
from whoami.vision_behaviors import VisionBehaviorSystem

# Combine scanning behaviors
vision = VisionBehaviorSystem(gimbal=..., config=...)
explorer = K1FaceExplorer(booster_client=booster)

# Use vision system's scan patterns + face explorer
vision.behaviors.scan_environment()
explorer.scan_for_faces(duration=30.0)
```

## Voice Interaction Flow

```
1. Scan environment → Find face
2. Recognize face
   ├─ Known person
   │  ├─ Check time since last seen
   │  ├─ Check conversation history
   │  └─ Greet with personalized message
   └─ Unknown person
      ├─ Say "Hello! I don't think we've met"
      ├─ Ask "What's your name?"
      ├─ Learn name
      └─ Greet new person
3. Continue conversation
4. Add conversation notes
5. Update profile
```

## Example Greetings

**First meeting:**
```
"Nice to meet you, Alice!"
```

**Recently seen (<5 min):**
```
"Hi again, Alice!"
```

**Longer time:**
```
"Hi Alice, it's been 2 days and 3 hours since we last talked!"
```

**With conversation recall:**
```
"Hi Alice, it's been 3 hours since we last talked!
Last time we talked about her dog Max."
```

## Configuration

### Adjust Scan Pattern

```python
# Custom scan positions
explorer.scan_positions = [
    (0.0, 0.0),      # Center
    (-1.0, 0.0),     # Far left (57°)
    (1.0, 0.0),      # Far right (57°)
    # Add more positions...
]
```

### Adjust Greet Cooldown

```python
# Don't greet same person within 2 minutes
explorer.greet_cooldown = 120.0
```

### Voice Engine

```python
# Use F5-TTS instead of pyttsx3
voice = VoiceInteraction(
    tts_engine='f5-tts',
    f5tts_ref_audio='/opt/whoami/voices/k1_default_voice.wav',
    f5tts_ref_text='This is the default voice for the K-1 robot.'
)
```

## Files

### Core Files
- `whoami/k1_face_explorer.py` - Main explorer class
- `whoami/yolo_face_recognition.py` - Face detection/recognition
- `whoami/voice_interaction.py` - Voice interaction

### Examples
- `examples/k1_autonomous_face_interaction.py` - Complete demo

### Config
- `config/k1_booster_config.json` - K-1 hardware config
- `person_profiles.json` - Person profiles (auto-generated)

## Troubleshooting

### Head not moving

**Check robot mode:**
```python
# Must be in PREP or WALK mode
booster.ChangeMode(RobotMode.kPrepare)
```

**Check ranges:**
```python
# K-1 head limits
assert -1.047 <= yaw <= 1.047  # ±60°
assert -0.524 <= pitch <= 0.785  # -30° to 45°
```

### Voice not working

**Check TTS engine:**
```bash
# Test pyttsx3
python -c "import pyttsx3; e = pyttsx3.init(); e.say('test'); e.runAndWait()"
```

**Check F5-TTS:**
```bash
# Test F5-TTS
python examples/f5tts_voice_demo.py
```

### Face detection slow

**Use smaller YOLO model:**
```python
yolo = YOLO('yolov8n.pt')  # Nano (fastest)
```

**Reduce resolution:**
```python
frame_small = cv2.resize(frame, (640, 480))
```

## Next Steps

1. **Add camera integration** - Connect to ZED/OAK camera feed
2. **Integrate with ROS2** - Publish/subscribe to camera topics
3. **Add gesture recognition** - Wave detection for greetings
4. **Multi-robot coordination** - Share profiles via Gun.js
5. **Advanced conversations** - Use LLM for dynamic responses

## References

- [K-1 SDK Documentation](https://github.com/BoosterRobotics/booster_robotics_sdk)
- [YOLO Documentation](https://docs.ultralytics.com/)
- [DeepFace](https://github.com/serengil/deepface)
- [F5-TTS Setup](docs/F5-TTS_SETUP.md)
