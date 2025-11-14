# Voice Interaction Guide

Complete guide for voice-based interaction with the WhoAmI face recognition system.

## Overview

The voice interaction system enables the robot to:
- **Ask for names** using text-to-speech
- **Listen for responses** using speech recognition
- **Link names to faces** in the database
- **Provide audio feedback** and greetings
- **Track conversation state** to avoid repetitive greetings

This creates a natural interaction flow:
1. Robot detects unknown face
2. Robot asks: "Hello! I don't think we've met before. What's your name?"
3. Person responds: "My name is John"
4. Robot confirms: "Did you say John? Please say yes or no."
5. Person confirms: "Yes"
6. Robot greets: "Nice to meet you, John!"
7. Robot adds face encoding + name to database
8. Next time: "Welcome back, John!"

## Installation

### Required Dependencies

```bash
# Text-to-Speech (pyttsx3)
pip install pyttsx3

# Speech Recognition
pip install SpeechRecognition

# Audio processing
pip install sounddevice numpy

# PyAudio (required by SpeechRecognition)
pip install pyaudio
```

### Platform-Specific Installation

#### Jetson (including K-1 Booster)

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y \
    portaudio19-dev \
    python3-pyaudio \
    espeak \
    flac \
    alsa-utils \
    pulseaudio

# Install Python packages
pip3 install pyttsx3 SpeechRecognition sounddevice numpy pyaudio

# Configure audio devices
aplay -l    # List playback devices
arecord -l  # List capture devices
```

#### Raspberry Pi

```bash
# Install dependencies
sudo apt-get install -y portaudio19-dev espeak flac alsa-utils

# Install Python packages
pip3 install pyttsx3 SpeechRecognition pyaudio
```

#### Mac

```bash
# Install PortAudio via Homebrew
brew install portaudio

# Install Python packages
pip3 install pyttsx3 SpeechRecognition pyaudio
```

### Optional: Offline Speech Recognition (Vosk)

For systems without internet connectivity or privacy-sensitive deployments:

```bash
# Install Vosk
pip install vosk

# Download language model
mkdir -p /opt/whoami/models
cd /opt/whoami/models

# Download small English model (50MB)
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip

# Or download large English model (1.8GB, more accurate)
wget https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip
unzip vosk-model-en-us-0.22.zip
```

## Quick Start

### Basic Voice Interaction

```python
from whoami.voice_interaction import VoiceInteraction

# Initialize voice system
voice = VoiceInteraction()

# Ask for a name
name = voice.ask_name()
if name:
    print(f"User said: {name}")
    voice.greet_person(name)

# Provide custom feedback
voice.say("How can I help you today?")

# Listen for response
response = voice.listen()
if response:
    print(f"User said: {response}")
```

### Integration with Face Recognition

```python
from whoami.voice_interaction import VoiceEnabledFaceRecognition, VoiceInteraction
from whoami.face_recognition_api import create_face_recognition_api

# Create face recognition API
face_api = create_face_recognition_api()

# Create voice-enabled wrapper
voice_enabled = VoiceEnabledFaceRecognition(
    face_recognizer=face_api,
    ask_unknown=True,      # Ask unknown people for names
    announce_known=True    # Greet known people
)

# Start camera
with face_api:
    face_api.start_camera()

    while True:
        # Process frame
        results = face_api.process_frame()

        for result in results:
            # Process with voice interaction
            final_name = voice_enabled.process_detection(
                name=result.name,
                confidence=result.confidence,
                face_encoding=result.encoding
            )

            if final_name and final_name != "Unknown":
                print(f"Person identified: {final_name}")
```

## Configuration

### Speech Recognition Engines

#### Google Speech Recognition (Online, Default)

```python
from whoami.voice_interaction import VoiceInteraction

# Use Google Speech Recognition (requires internet)
voice = VoiceInteraction(sr_engine='google')
```

**Pros:**
- High accuracy
- No model download required
- Supports many languages

**Cons:**
- Requires internet connection
- Privacy concerns (audio sent to Google)
- May have usage limits

#### Vosk (Offline)

```python
from whoami.voice_interaction import VoiceInteraction

# Use Vosk for offline recognition
voice = VoiceInteraction(
    sr_engine='vosk',
    vosk_model_path='/opt/whoami/models/vosk-model-small-en-us-0.15'
)
```

**Pros:**
- Works offline
- Privacy-friendly (all processing on-device)
- No usage limits
- Free and open source

**Cons:**
- Requires model download
- Lower accuracy than Google
- English model ~50MB-1.8GB

### Hardware-Specific Configuration

The voice system automatically detects audio devices from hardware profiles:

```python
from whoami.voice_interaction import VoiceInteraction

# Auto-detect audio from hardware profile
voice = VoiceInteraction()
# On K-1 booster, automatically uses hw:2,0

# Or manually specify
voice = VoiceInteraction(audio_device='hw:2,0')
```

### Customizing TTS Voice

```python
from whoami.voice_interaction import VoiceInteraction

voice = VoiceInteraction()

# Get TTS engine
engine = voice.tts_engine

# List available voices
voices = engine.getProperty('voices')
for v in voices:
    print(f"Voice: {v.name}")

# Set voice
engine.setProperty('voice', voices[0].id)

# Adjust speech rate (words per minute)
engine.setProperty('rate', 150)  # Default: 150

# Adjust volume (0.0 to 1.0)
engine.setProperty('volume', 0.9)  # Default: 0.9
```

### Timeout and Recognition Settings

```python
from whoami.voice_interaction import VoiceInteraction

voice = VoiceInteraction(
    timeout=5.0,               # Wait 5 seconds for speech to start
    phrase_time_limit=5.0,     # Max 5 seconds per phrase
    confidence_threshold=0.7   # Minimum 70% confidence
)
```

## Usage Patterns

### Pattern 1: Ask Unknown, Greet Known

```python
from whoami.voice_interaction import VoiceEnabledFaceRecognition

# Ask unknown people for names, greet known people
voice_face = VoiceEnabledFaceRecognition(
    face_recognizer=face_api,
    ask_unknown=True,
    announce_known=True
)
```

### Pattern 2: Silent Recognition with Optional Greeting

```python
# Don't ask unknown, only greet known people
voice_face = VoiceEnabledFaceRecognition(
    face_recognizer=face_api,
    ask_unknown=False,
    announce_known=True
)
```

### Pattern 3: Ask Only, No Automatic Greetings

```python
# Ask unknown people for names, but don't auto-greet
voice_face = VoiceEnabledFaceRecognition(
    face_recognizer=face_api,
    ask_unknown=True,
    announce_known=False
)

# Manually greet when appropriate
if name and some_condition:
    voice_face.voice_interaction.greet_person(name)
```

### Pattern 4: Custom Prompts

```python
from whoami.voice_interaction import VoiceInteraction

voice = VoiceInteraction()

# Custom name prompt
name = voice.ask_name(prompt="Hey there! What should I call you?")

# Custom listening prompts
response = voice.listen(prompt="What would you like to do?")

# Multi-step conversation
voice.say("Welcome to the lab.")
command = voice.listen(prompt="How can I assist you?")

if "tour" in command.lower():
    voice.say("Great! Let me show you around.")
elif "help" in command.lower():
    voice.say("I can recognize faces and answer questions.")
```

## Advanced Features

### Conversation State Management

The system tracks recently greeted people to avoid repetitive greetings:

```python
voice_face = VoiceEnabledFaceRecognition(
    face_recognizer=face_api,
    ask_unknown=True,
    announce_known=True
)

# Default cooldown: 60 seconds
voice_face.greet_cooldown = 120.0  # Don't greet same person within 2 minutes

# Manually clear cache
voice_face.recently_greeted.clear()

# Or clean up old entries
voice_face.cleanup_greeted_cache(max_age=300.0)
```

### Statistics Tracking

```python
from whoami.voice_interaction import VoiceInteraction

voice = VoiceInteraction()

# Use the system...
voice.ask_name()
voice.say("Hello!")

# Get statistics
stats = voice.get_stats()
print(f"Questions asked: {stats['questions_asked']}")
print(f"Responses received: {stats['responses_received']}")
print(f"Recognition failures: {stats['recognition_failures']}")
print(f"Names learned: {stats['names_learned']}")

# Reset statistics
voice.reset_stats()
```

### Error Handling and Retries

```python
from whoami.voice_interaction import VoiceInteraction

voice = VoiceInteraction()

# Ask with retries
name = voice.ask_name(max_attempts=3)

if name:
    print(f"Successfully recognized: {name}")
else:
    print("Failed to recognize name after 3 attempts")
    voice.say("Sorry, I'm having trouble hearing you. Let's try again later.")
```

### Name Confirmation

The system automatically confirms recognized names:

```python
# This happens automatically in ask_name()
# 1. Robot: "What's your name?"
# 2. Person: "John Smith"
# 3. Robot: "Did you say John Smith? Please say yes or no."
# 4. Person: "Yes"
# 5. Returns "John Smith"

# Or manually confirm
if voice.confirm_name("John Smith"):
    print("Name confirmed")
else:
    print("Name not confirmed")
```

## K-1 Booster Integration

### Full Integration Example

```python
from whoami.voice_interaction import VoiceEnabledFaceRecognition, VoiceInteraction
from whoami.face_recognition_api import create_face_recognition_api
from whoami.hardware_detector import get_hardware_detector, detect_hardware
import time

# Detect hardware
platform = detect_hardware()
print(f"Running on: {platform}")

# Check if audio is available
detector = get_hardware_detector()
if detector.has_audio_support():
    print("Audio support detected")
    audio_config = detector.get_audio_config()
    print(f"Input: {audio_config['input_device']}")
    print(f"Output: {audio_config['output_device']}")
else:
    print("No audio support on this platform")

# Create voice interaction
voice = VoiceInteraction(
    sr_engine='vosk',  # Offline for privacy
    vosk_model_path='/opt/whoami/models/vosk-model-small-en-us-0.15'
)

# Create face recognition API
face_api = create_face_recognition_api()

# Create voice-enabled wrapper
voice_face = VoiceEnabledFaceRecognition(
    face_recognizer=face_api,
    voice_interaction=voice,
    ask_unknown=True,
    announce_known=True
)

# Startup greeting
voice.say("Voice interaction system initialized. I am ready to meet people!")

# Main loop
with face_api:
    face_api.start_camera()

    try:
        while True:
            results = face_api.process_frame()

            for result in results:
                final_name = voice_face.process_detection(
                    name=result.name,
                    confidence=result.confidence,
                    face_encoding=result.encoding
                )

            # Clean up greeted cache periodically
            if int(time.time()) % 300 == 0:  # Every 5 minutes
                voice_face.cleanup_greeted_cache()

            time.sleep(0.1)

    except KeyboardInterrupt:
        voice.say("Shutting down voice interaction system. Goodbye!")
```

### Operational Modes

#### Remote VNC Mode

```python
# Operator can hear audio through VNC audio streaming
# Voice feedback provides status updates

voice.say("System starting up")
voice.say("Camera initialized")
voice.say("Face detection enabled")
```

#### Autonomous Mode

```python
# Robot operates independently with audio feedback

voice.say("Autonomous mode activated")

# Provide periodic status updates
import threading

def periodic_status():
    while True:
        time.sleep(600)  # Every 10 minutes
        stats = face_api.get_stats()
        voice.say(f"Status update. I have recognized {stats['total_faces']} people in the last 10 minutes.")

status_thread = threading.Thread(target=periodic_status, daemon=True)
status_thread.start()
```

#### Direct Access Mode

```python
# Local operator can hear audio directly
# Useful for testing and configuration

voice.say("Direct access mode. Audio output to local speaker.")
```

## Audio Device Configuration

### Testing Audio Devices

```bash
# List audio devices
python3 -c "
import sounddevice as sd
print(sd.query_devices())
"

# Test microphone
arecord -D hw:2,0 -f S16_LE -r 48000 -c 2 -d 5 test.wav
aplay test.wav

# Test speaker
speaker-test -D hw:2,0 -c 2

# Test with Python
python3 -c "
from whoami.voice_interaction import VoiceInteraction
voice = VoiceInteraction()
voice.say('Testing audio output')
"
```

### Configuring ALSA Devices

```bash
# Edit ALSA configuration
nano ~/.asoundrc
```

Add:
```
pcm.!default {
    type hw
    card 2
    device 0
}

ctl.!default {
    type hw
    card 2
}
```

### PulseAudio Configuration

```bash
# List PulseAudio devices
pacmd list-sources
pacmd list-sinks

# Set default source (microphone)
pacmd set-default-source alsa_input.usb-XXX

# Set default sink (speaker)
pacmd set-default-sink alsa_output.usb-XXX
```

## Troubleshooting

### TTS Not Working

```bash
# Check espeak installation
espeak "Test"

# Check pyttsx3
python3 -c "
import pyttsx3
engine = pyttsx3.init()
engine.say('Test')
engine.runAndWait()
"

# If fails, try alternative TTS
pip install gTTS
```

### Microphone Not Detected

```bash
# Check microphone
arecord -l

# Test recording
arecord -D hw:2,0 -f S16_LE -r 16000 -c 1 -d 3 test.wav
aplay test.wav

# Check permissions
ls -l /dev/snd/
groups  # Make sure user is in 'audio' group
sudo usermod -a -G audio $USER
```

### Speech Recognition Errors

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

from whoami.voice_interaction import VoiceInteraction
voice = VoiceInteraction()

# Check recognizer settings
print(f"Energy threshold: {voice.recognizer.energy_threshold}")
print(f"Dynamic energy: {voice.recognizer.dynamic_energy_threshold}")

# Adjust if needed
voice.recognizer.energy_threshold = 1000  # Increase for noisy environments
```

### Vosk Model Not Loading

```bash
# Check model exists
ls -l /opt/whoami/models/

# Test Vosk directly
python3 -c "
from vosk import Model
model = Model('/opt/whoami/models/vosk-model-small-en-us-0.15')
print('Model loaded successfully')
"

# Check permissions
sudo chmod -R 755 /opt/whoami/models/
```

### Low Recognition Accuracy

```python
# Adjust confidence threshold
voice = VoiceInteraction(confidence_threshold=0.5)  # Lower threshold

# Use larger Vosk model
voice = VoiceInteraction(
    sr_engine='vosk',
    vosk_model_path='/opt/whoami/models/vosk-model-en-us-0.22'  # Large model
)

# Or switch to Google
voice = VoiceInteraction(sr_engine='google')
```

## Performance Optimization

### Reduce Latency

```python
# Use faster TTS settings
voice = VoiceInteraction()
voice.tts_engine.setProperty('rate', 200)  # Speak faster

# Reduce listening timeout
voice.timeout = 3.0
voice.phrase_time_limit = 3.0
```

### Reduce CPU Usage

```python
# Use Vosk small model instead of large
vosk_model_path='/opt/whoami/models/vosk-model-small-en-us-0.15'

# Process fewer frames when combined with face recognition
face_api = create_face_recognition_api()
face_api.config.process_every_n_frames = 5  # Process every 5th frame
```

### Memory Optimization

```python
# Don't keep entire greeted cache
voice_face.cleanup_greeted_cache(max_age=60.0)  # Only keep last minute

# Limit max cache size
if len(voice_face.recently_greeted) > 100:
    voice_face.recently_greeted.clear()
```

## See Also

- [Hardware Configuration Guide](HARDWARE_CONFIG_GUIDE.md) - Audio device configuration
- [K-1 Booster Setup](K1_BOOSTER_SETUP.md) - K-1 audio setup
- [API Reference](API_REFERENCE.md) - Face recognition API
- [Usage Guide](USAGE_GUIDE.md) - General usage patterns

## Future Enhancements

Planned features for voice interaction:

- **Voice biometrics** - Identify people by voice
- **Speaker diarization** - Track multiple speakers
- **Emotion detection** - Detect emotions in speech
- **Multi-language support** - Support multiple languages
- **Wake word detection** - "Hey Robot" activation
- **Intent recognition** - Understand user commands
- **Dialog management** - Multi-turn conversations
- **Audio source localization** - Orient gimbal toward speaker
