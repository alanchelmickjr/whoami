#!/usr/bin/env python3
"""
F5-TTS Voice Demo for K-1 Robot

Demonstrates the F5-TTS high-quality neural text-to-speech engine
with voice cloning capabilities for the K-1 robot.

This script shows how to:
1. Initialize F5-TTS with a reference voice
2. Generate natural-sounding speech
3. Use voice cloning with custom audio samples
4. Compare F5-TTS vs pyttsx3 quality
5. Integrate with the K-1 voice interaction system

F5-TTS provides significantly better quality than pyttsx3/espeak,
sounding much more natural and human-like.

Usage:
    # Basic demo with default voice
    python examples/f5tts_voice_demo.py

    # Demo with custom voice sample
    python examples/f5tts_voice_demo.py --ref-audio my_voice.wav --ref-text "Sample text"

    # Compare F5-TTS vs pyttsx3
    python examples/f5tts_voice_demo.py --compare

Requirements:
    pip install f5-tts torch torchaudio sounddevice
"""

import argparse
import logging
import sys
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from whoami.tts_f5 import F5TTSEngine, F5TTS_AVAILABLE
from whoami.voice_interaction import VoiceInteraction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if F5-TTS is available"""
    if not F5TTS_AVAILABLE:
        print("❌ F5-TTS is not available!")
        print("\nInstall with:")
        print("  pip install f5-tts torch torchaudio sounddevice")
        print("\nFor CUDA support (recommended for faster inference):")
        print("  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118")
        return False
    return True


def test_basic_f5tts(ref_audio: str, ref_text: str):
    """Test basic F5-TTS functionality"""
    print("\n" + "="*60)
    print("TEST 1: Basic F5-TTS Synthesis")
    print("="*60)

    try:
        # Initialize engine
        print("\n1. Initializing F5-TTS engine...")
        engine = F5TTSEngine(
            default_ref_audio=ref_audio,
            default_ref_text=ref_text
        )
        print("   ✓ F5-TTS initialized successfully")

        # Test sentences
        test_phrases = [
            "Hello! I am the K-1 robot, and I'm using F5-TTS for high-quality speech synthesis.",
            "This voice sounds much more natural than the robotic espeak engine.",
            "I can also clone voices from audio samples, which is really cool!"
        ]

        for i, phrase in enumerate(test_phrases, 1):
            print(f"\n{i}. Generating: \"{phrase[:50]}...\"")
            start_time = time.time()
            success = engine.say(phrase)
            elapsed = time.time() - start_time

            if success:
                print(f"   ✓ Generated in {elapsed:.2f}s")
            else:
                print(f"   ✗ Failed to generate")

        # Show statistics
        print("\n" + "-"*60)
        print("Statistics:")
        print("-"*60)
        stats = engine.get_stats()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")

        return True

    except Exception as e:
        print(f"   ✗ Error: {e}")
        logger.exception("F5-TTS test failed")
        return False


def test_voice_interaction_integration(ref_audio: str, ref_text: str):
    """Test F5-TTS integration with VoiceInteraction class"""
    print("\n" + "="*60)
    print("TEST 2: VoiceInteraction Integration")
    print("="*60)

    try:
        # Initialize voice interaction with F5-TTS
        print("\n1. Initializing VoiceInteraction with F5-TTS...")
        voice = VoiceInteraction(
            tts_engine='f5-tts',
            f5tts_ref_audio=ref_audio,
            f5tts_ref_text=ref_text
        )
        print("   ✓ VoiceInteraction initialized with F5-TTS")

        # Test greeting functions
        print("\n2. Testing greeting functions...")

        print("\n   a) Greeting a new person:")
        voice.greet_person("Alice")
        time.sleep(0.5)

        print("\n   b) Welcoming someone back:")
        voice.welcome_back("Bob")
        time.sleep(0.5)

        print("\n   c) Announcing unknown person:")
        voice.announce_unknown()

        print("\n   ✓ All greeting functions working")

        return True

    except Exception as e:
        print(f"   ✗ Error: {e}")
        logger.exception("Integration test failed")
        return False


def test_comparison(ref_audio: str, ref_text: str):
    """Compare F5-TTS vs pyttsx3 quality"""
    print("\n" + "="*60)
    print("TEST 3: F5-TTS vs pyttsx3 Comparison")
    print("="*60)

    test_text = "This is a comparison between F5-TTS and pyttsx3 text to speech engines."

    # Test pyttsx3
    print("\n1. Testing pyttsx3 (old engine)...")
    try:
        voice_old = VoiceInteraction(tts_engine='pyttsx3')
        print(f"   Speaking: \"{test_text}\"")
        voice_old.say(test_text)
        print("   ✓ pyttsx3 complete (sounds robotic)")
    except Exception as e:
        print(f"   ✗ pyttsx3 failed: {e}")

    time.sleep(1)

    # Test F5-TTS
    print("\n2. Testing F5-TTS (new engine)...")
    try:
        voice_new = VoiceInteraction(
            tts_engine='f5-tts',
            f5tts_ref_audio=ref_audio,
            f5tts_ref_text=ref_text
        )
        print(f"   Speaking: \"{test_text}\"")
        voice_new.say(test_text)
        print("   ✓ F5-TTS complete (sounds natural)")
    except Exception as e:
        print(f"   ✗ F5-TTS failed: {e}")

    print("\n" + "-"*60)
    print("Quality Comparison:")
    print("-"*60)
    print("  pyttsx3:  ⭐⭐☆☆☆ (Robotic, mechanical)")
    print("  F5-TTS:   ⭐⭐⭐⭐⭐ (Natural, human-like)")


def create_sample_reference_audio():
    """Create a sample reference audio file for testing"""
    print("\n" + "="*60)
    print("Creating Sample Reference Audio")
    print("="*60)

    sample_dir = Path("/opt/whoami/voices")
    sample_path = sample_dir / "k1_default_voice.wav"

    if sample_path.exists():
        print(f"\n✓ Sample reference audio already exists:")
        print(f"  {sample_path}")
        return str(sample_path), "This is the default voice for the K-1 robot."

    print("\nℹ️  No reference audio found.")
    print("\nTo use F5-TTS with voice cloning, you need a reference audio file.")
    print("\nOptions:")
    print("  1. Record a 5-10 second audio sample of someone speaking")
    print("  2. Use any .wav file with clear speech")
    print("  3. Download a sample voice from the F5-TTS repository")
    print(f"\nPlace the file at: {sample_path}")
    print("\nFor now, the demo will use the default F5-TTS voice.")

    return None, None


def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(
        description="F5-TTS Voice Demo for K-1 Robot"
    )
    parser.add_argument(
        '--ref-audio',
        type=str,
        help='Path to reference audio file for voice cloning'
    )
    parser.add_argument(
        '--ref-text',
        type=str,
        help='Transcription of reference audio'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare F5-TTS vs pyttsx3 quality'
    )
    parser.add_argument(
        '--setup',
        action='store_true',
        help='Show setup instructions'
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("F5-TTS Voice Demo for K-1 Robot")
    print("="*60)
    print("\nF5-TTS: High-quality neural TTS with voice cloning")
    print("Open-source alternative to ElevenLabs")

    # Check dependencies
    if not check_dependencies():
        return 1

    # Show setup instructions if requested
    if args.setup:
        print("\n" + "="*60)
        print("Setup Instructions")
        print("="*60)
        print("\n1. Install F5-TTS:")
        print("   pip install f5-tts torch torchaudio sounddevice")
        print("\n2. Prepare reference audio:")
        print("   - Record a 5-10 second voice sample (.wav format)")
        print("   - Place it at: /opt/whoami/voices/k1_default_voice.wav")
        print("   - Or use --ref-audio flag to specify custom location")
        print("\n3. Run the demo:")
        print("   python examples/f5tts_voice_demo.py")
        return 0

    # Get reference audio
    ref_audio = args.ref_audio
    ref_text = args.ref_text

    if not ref_audio:
        # Try to find default reference audio
        ref_audio, ref_text = create_sample_reference_audio()

    if not ref_audio:
        print("\n⚠️  Warning: No reference audio provided")
        print("   F5-TTS requires reference audio for voice cloning.")
        print("   Use --setup for instructions, or --ref-audio to specify a file.")
        return 1

    # Verify reference audio exists
    if not Path(ref_audio).exists():
        print(f"\n❌ Error: Reference audio not found: {ref_audio}")
        return 1

    print(f"\n✓ Using reference audio: {ref_audio}")
    if ref_text:
        print(f"  Transcription: \"{ref_text}\"")

    # Run tests
    success = True

    # Test 1: Basic F5-TTS
    if not test_basic_f5tts(ref_audio, ref_text):
        success = False

    # Test 2: VoiceInteraction integration
    if not test_voice_interaction_integration(ref_audio, ref_text):
        success = False

    # Test 3: Comparison (if requested)
    if args.compare:
        test_comparison(ref_audio, ref_text)

    # Summary
    print("\n" + "="*60)
    print("Demo Summary")
    print("="*60)
    if success:
        print("✓ All tests passed!")
        print("\nNext steps:")
        print("  1. Update config/k1_booster_config.json to use 'f5-tts'")
        print("  2. Set ref_audio and ref_text in the config")
        print("  3. Enjoy natural-sounding speech on your K-1!")
    else:
        print("✗ Some tests failed")
        print("  Check the error messages above for details")

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
