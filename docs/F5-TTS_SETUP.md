# F5-TTS Setup Guide for K-1 Robot

## Overview

F5-TTS is a high-quality neural text-to-speech engine with voice cloning capabilities. It provides significantly better speech quality than the default pyttsx3/espeak engine, producing natural-sounding, human-like speech.

**Key Features:**
- 🎤 High-quality neural TTS (sounds natural, not robotic)
- 🔊 Zero-shot voice cloning (clone any voice from a sample)
- 💻 Offline/local inference (runs on your K-1)
- 🆓 Open-source (alternative to commercial services like ElevenLabs)
- ⚡ Fast inference on GPU (Jetson Orin NX)

## Installation

### 1. Install F5-TTS and Dependencies

On your K-1 robot (Jetson Orin NX), install F5-TTS:

```bash
# Install PyTorch (if not already installed)
pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install F5-TTS
pip3 install f5-tts

# Verify sounddevice is installed (should already be)
pip3 install sounddevice
```

### 2. Verify Installation

```bash
python3 -c "from f5_tts.api import F5TTS; print('F5-TTS installed successfully!')"
```

If you see "F5-TTS installed successfully!", you're good to go!

## Quick Start

### 1. Prepare Reference Audio

F5-TTS uses voice cloning, so you need a reference audio sample:

**Option A: Record Your Own Voice**
```bash
# Create voices directory
sudo mkdir -p /opt/whoami/voices

# Record a 5-10 second audio sample
arecord -D hw:2,0 -f S16_LE -r 48000 -c 2 -d 10 /opt/whoami/voices/k1_default_voice.wav

# Speak clearly during the recording:
# "This is the default voice for the K-1 robot. I am demonstrating natural speech synthesis."
```

**Option B: Use a Pre-recorded Sample**
Place any .wav file with clear speech at:
```
/opt/whoami/voices/k1_default_voice.wav
```

### 2. Test F5-TTS

Run the demo script:

```bash
cd /home/user/whoami
python3 examples/f5tts_voice_demo.py
```

### 3. Compare Quality (Optional)

Compare F5-TTS with the old pyttsx3 engine:

```bash
python3 examples/f5tts_voice_demo.py --compare
```

You'll hear a dramatic difference in quality!

## Configuration

### Enable F5-TTS on K-1

The K-1 configuration has already been updated to use F5-TTS. Check the config:

```json
// config/k1_booster_config.json
{
  "audio": {
    "features": {
      "voice_reporting": {
        "enabled": true,
        "engine": "f5-tts",
        "fallback_engine": "pyttsx3",
        "model_type": "F5-TTS",
        "ref_audio": "/opt/whoami/voices/k1_default_voice.wav",
        "ref_text": "This is the default voice for the K-1 robot.",
        ...
      }
    }
  }
}
```

### Using F5-TTS in Your Code

**Option 1: Direct F5-TTS Engine**
```python
from whoami.tts_f5 import F5TTSEngine

# Initialize with reference voice
engine = F5TTSEngine(
    default_ref_audio="/opt/whoami/voices/k1_default_voice.wav",
    default_ref_text="This is the default voice for the K-1 robot."
)

# Generate speech
engine.say("Hello! I'm using F5-TTS for natural speech.")
```

**Option 2: VoiceInteraction Class (Recommended)**
```python
from whoami.voice_interaction import VoiceInteraction

# Initialize with F5-TTS
voice = VoiceInteraction(
    tts_engine='f5-tts',
    f5tts_ref_audio="/opt/whoami/voices/k1_default_voice.wav",
    f5tts_ref_text="This is the default voice for the K-1 robot."
)

# Use as normal
voice.say("Welcome! Face recognition is active.")
voice.greet_person("Alice")
voice.welcome_back("Bob")
```

## Voice Cloning

### Clone a Specific Voice

To clone a different voice:

1. **Get a voice sample** (5-10 seconds of clear speech)
2. **Transcribe it** (write down exactly what was said)
3. **Use it in your code:**

```python
from whoami.tts_f5 import F5TTSEngine

engine = F5TTSEngine(
    default_ref_audio="path/to/voice_sample.wav",
    default_ref_text="Exact transcription of the voice sample."
)

# Now it will speak in the cloned voice!
engine.say("This is amazing, I sound just like the sample!")
```

### Multiple Voices

You can change voices on-the-fly:

```python
from whoami.tts_f5 import F5TTSEngine

engine = F5TTSEngine()

# Voice 1
engine.say(
    "Hello, I'm voice one.",
    ref_audio="voice1.wav",
    ref_text="Sample of voice one."
)

# Voice 2
engine.say(
    "And I'm voice two.",
    ref_audio="voice2.wav",
    ref_text="Sample of voice two."
)
```

## Performance Tuning

### GPU Acceleration

F5-TTS automatically uses CUDA if available. On Jetson Orin NX:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

### Caching

F5-TTS automatically caches generated audio. To manage the cache:

```python
from whoami.tts_f5 import F5TTSEngine

engine = F5TTSEngine(cache_dir="/tmp/f5tts_cache")

# Clear cache
files_deleted = engine.clear_cache()
print(f"Cleared {files_deleted} cached files")
```

### Speed Adjustment

Control speech speed:

```python
# Faster speech (1.5x speed)
engine.say("Speaking faster!", speed=1.5)

# Slower speech (0.8x speed)
engine.say("Speaking slower...", speed=0.8)
```

## Troubleshooting

### F5-TTS Not Available

**Error:** `F5-TTS is not available`

**Solution:**
```bash
pip3 install f5-tts torch torchaudio
```

### CUDA Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solution:** Force CPU mode:
```python
engine = F5TTSEngine(device="cpu")
```

### Audio Playback Issues

**Error:** Audio not playing on K-1

**Solution:** Specify audio device:
```python
engine = F5TTSEngine(output_device="hw:2,0")  # K-1 audio device
```

### Poor Voice Quality

**Issues:**
- Robotic or distorted sound
- Unnatural prosody

**Solutions:**
1. Use a higher-quality reference audio (clear, noise-free)
2. Ensure reference audio is at least 5 seconds long
3. Match the speaking style in ref_text and gen_text
4. Use accurate transcription in ref_text

## Advanced Usage

### Custom Model Checkpoints

Use a custom F5-TTS model:

```python
engine = F5TTSEngine(
    model_type="F5-TTS",
    ckpt_file="/path/to/custom_model.pt",
    vocab_file="/path/to/custom_vocab.txt"
)
```

### Statistics and Monitoring

Track TTS performance:

```python
# Get statistics
stats = engine.get_stats()
print(f"Utterances generated: {stats['utterances_generated']}")
print(f"Average generation time: {stats['avg_generation_time']:.2f}s")
print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")

# Reset statistics
engine.reset_stats()
```

## Integration with K-1 Systems

### Face Recognition Integration

F5-TTS is already integrated with the YOLO face recognition system:

```python
from whoami.yolo_face_recognition import K1FaceRecognitionSystem

# Initialize with F5-TTS enabled
system = K1FaceRecognitionSystem(
    enable_voice=True,  # Uses F5-TTS from config
)

# Voice interaction happens automatically:
# - Greets known people: "Welcome back, Alice!"
# - Asks unknown people: "Hello! I don't think we've met. What's your name?"
```

### Remote Operation

When operating K-1 remotely via VNC, F5-TTS audio streams through the remote connection.

## Files and Locations

### Code Files
- `whoami/tts_f5.py` - F5-TTS engine wrapper
- `whoami/voice_interaction.py` - Voice interaction with F5-TTS support
- `examples/f5tts_voice_demo.py` - Demo script

### Configuration
- `config/k1_booster_config.json` - K-1 configuration (F5-TTS settings)
- `/opt/whoami/voices/` - Voice reference audio directory

### Cache
- `/tmp/f5tts_cache/` - Default cache location for generated audio

## Next Steps

1. ✅ Install F5-TTS
2. ✅ Record reference audio
3. ✅ Test with demo script
4. ✅ Update K-1 configuration
5. 🎉 Enjoy natural-sounding speech!

## Resources

- **F5-TTS GitHub:** https://github.com/SWivid/F5-TTS
- **F5-TTS Paper:** https://huggingface.co/papers/2410.06885
- **Demo Space:** https://huggingface.co/spaces/mrfakename/E2-F5-TTS

## Support

For issues or questions:
1. Check this guide's troubleshooting section
2. Review the demo script: `examples/f5tts_voice_demo.py`
3. Check F5-TTS documentation: https://github.com/SWivid/F5-TTS
4. Open an issue on the WhoAmI repository
