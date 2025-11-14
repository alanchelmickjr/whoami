#!/usr/bin/env python3
"""
Voice Interaction Demo

Demonstrates voice-based interaction with face recognition:
1. Detect faces using camera
2. Ask unknown people for their names
3. Greet known people by name
4. Provide audio feedback

Usage:
    python examples/voice_interaction_demo.py

Requirements:
    pip install pyttsx3 SpeechRecognition pyaudio
"""

import sys
import time
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from whoami.voice_interaction import VoiceEnabledFaceRecognition, VoiceInteraction
from whoami.hardware_detector import detect_hardware, get_hardware_detector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main demo function"""
    print("="*60)
    print("Voice Interaction Demo")
    print("="*60)

    # Detect hardware
    print("\n1. Detecting hardware...")
    platform = detect_hardware()
    detector = get_hardware_detector()
    print(f"   Platform: {detector.get_display_name()}")

    # Check audio support
    if detector.has_audio_support():
        audio_config = detector.get_audio_config()
        print(f"   Audio input: {audio_config.get('input_device')}")
        print(f"   Audio output: {audio_config.get('output_device')}")
    else:
        print("   Warning: No audio configuration detected")

    # Initialize voice interaction
    print("\n2. Initializing voice system...")

    # Try offline (Vosk) first, fallback to online (Google)
    vosk_model_path = "/opt/whoami/models/vosk-model-small-en-us-0.15"
    if Path(vosk_model_path).exists():
        print("   Using offline speech recognition (Vosk)")
        voice = VoiceInteraction(sr_engine='vosk', vosk_model_path=vosk_model_path)
    else:
        print("   Using online speech recognition (Google)")
        print("   Note: Requires internet connection")
        voice = VoiceInteraction(sr_engine='google')

    # Test TTS
    print("\n3. Testing text-to-speech...")
    voice.say("Hello! Voice interaction system initialized.")

    # Demo: Simple conversation
    print("\n4. Demo: Simple conversation")
    print("   Say something after the prompt...")

    response = voice.listen(prompt="How are you today?")
    if response:
        print(f"   You said: {response}")
        voice.say(f"You said: {response}")
    else:
        print("   No response detected")

    # Demo: Ask for name
    print("\n5. Demo: Ask for name")
    print("   The robot will ask for your name...")

    name = voice.ask_name()
    if name:
        print(f"   Name learned: {name}")
        voice.greet_person(name)
    else:
        print("   Failed to learn name")

    # Demo: Voice-enabled face recognition (simulated)
    print("\n6. Demo: Voice-enabled face recognition")
    print("   Simulating face detection with voice interaction...")

    # Create mock face recognizer
    class MockFaceRecognizer:
        """Mock face recognizer for demo"""
        def __init__(self):
            self.database = {}

        def add_face(self, name, encoding):
            self.database[name] = encoding
            print(f"   [Mock] Added {name} to database")

    mock_recognizer = MockFaceRecognizer()

    # Create voice-enabled wrapper
    voice_face = VoiceEnabledFaceRecognition(
        face_recognizer=mock_recognizer,
        voice_interaction=voice,
        ask_unknown=True,
        announce_known=True
    )

    # Simulate unknown face detection
    print("\n   Simulating unknown face...")
    voice_face.process_detection(
        name="Unknown",
        confidence=0.0,
        face_encoding=[0.1, 0.2, 0.3]  # Mock encoding
    )

    # Simulate known face detection
    if name:
        print(f"\n   Simulating known face ({name})...")
        voice_face.process_detection(
            name=name,
            confidence=0.95,
            face_encoding=None
        )

    # Show statistics
    print("\n7. Voice Interaction Statistics")
    stats = voice.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # Cleanup
    print("\n8. Demo complete!")
    voice.say("Demo complete. Thank you!")

    print("\n" + "="*60)
    print("Voice Interaction Demo Finished")
    print("="*60)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\nError: {e}")
        print("Make sure audio devices are configured and dependencies are installed:")
        print("  pip install pyttsx3 SpeechRecognition pyaudio")
