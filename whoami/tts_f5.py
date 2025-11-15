"""
F5-TTS Engine Wrapper for WhoAmI System

Provides high-quality text-to-speech using F5-TTS, a flow-based TTS model
with voice cloning capabilities. This is a significant upgrade from the
robotic-sounding pyttsx3/espeak engine.

F5-TTS Features:
- High-quality neural TTS
- Zero-shot voice cloning
- Natural-sounding speech
- Offline/local inference
- Open-source alternative to ElevenLabs

Usage:
    from whoami.tts_f5 import F5TTSEngine

    # Initialize engine
    engine = F5TTSEngine()

    # Generate speech
    engine.say("Hello, I am using F5-TTS for natural speech!")

    # Generate with custom voice
    engine.say(
        text="This is a cloned voice.",
        ref_audio="path/to/voice_sample.wav",
        ref_text="Transcription of the voice sample"
    )
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, Union
import time

logger = logging.getLogger(__name__)

# F5-TTS imports
try:
    from f5_tts.api import F5TTS
    import torch
    import torchaudio
    import sounddevice as sd
    import numpy as np
    F5TTS_AVAILABLE = True
except ImportError as e:
    F5TTS_AVAILABLE = False
    logger.warning(f"F5-TTS not available: {e}")
    logger.warning("Install with: pip install f5-tts torch torchaudio")


class F5TTSEngine:
    """
    F5-TTS text-to-speech engine wrapper

    Provides a clean interface for high-quality TTS using F5-TTS.
    Supports voice cloning with reference audio.
    """

    def __init__(
        self,
        model_type: str = "F5-TTS",
        ckpt_file: Optional[str] = None,
        vocab_file: Optional[str] = None,
        device: Optional[str] = None,
        default_ref_audio: Optional[str] = None,
        default_ref_text: Optional[str] = None,
        cache_dir: Optional[str] = None,
        sample_rate: int = 24000,
        output_device: Optional[str] = None
    ):
        """
        Initialize F5-TTS engine

        Args:
            model_type: Model type ('F5-TTS' or 'E2-TTS')
            ckpt_file: Path to checkpoint file (downloads if None)
            vocab_file: Path to vocab file (uses default if None)
            device: Device for inference ('cuda', 'cpu', or None for auto)
            default_ref_audio: Default reference audio for voice cloning
            default_ref_text: Default reference text (transcription)
            cache_dir: Directory to cache generated audio files
            sample_rate: Audio sample rate
            output_device: Audio output device (e.g., 'hw:2,0' for K-1)
        """
        if not F5TTS_AVAILABLE:
            raise ImportError(
                "F5-TTS is not available. Install with:\n"
                "pip install f5-tts torch torchaudio"
            )

        self.model_type = model_type
        self.sample_rate = sample_rate
        self.output_device = output_device
        self.default_ref_audio = default_ref_audio
        self.default_ref_text = default_ref_text

        # Set up cache directory
        if cache_dir is None:
            cache_dir = tempfile.gettempdir()
        self.cache_dir = Path(cache_dir) / "f5tts_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device

        # Initialize F5-TTS model
        try:
            logger.info(f"Initializing F5-TTS on device: {device}")
            self.model = F5TTS(
                model_type=model_type,
                ckpt_file=ckpt_file,
                vocab_file=vocab_file,
                device=device
            )
            logger.info("F5-TTS initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize F5-TTS: {e}")
            raise

        # Statistics
        self.stats = {
            'utterances_generated': 0,
            'total_generation_time': 0.0,
            'total_characters': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }

    def say(
        self,
        text: str,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        wait: bool = True,
        save_path: Optional[str] = None,
        remove_silence: bool = True,
        speed: float = 1.0
    ) -> bool:
        """
        Speak text using F5-TTS

        Args:
            text: Text to synthesize
            ref_audio: Reference audio file for voice cloning (uses default if None)
            ref_text: Reference text/transcription (uses default if None)
            wait: Wait for speech to finish before returning
            save_path: Path to save generated audio (uses temp file if None)
            remove_silence: Remove silence from generated audio
            speed: Speech speed multiplier (1.0 = normal)

        Returns:
            True if successful, False otherwise
        """
        if not text or not text.strip():
            logger.warning("Empty text provided, skipping TTS")
            return False

        try:
            start_time = time.time()

            # Use defaults if not provided
            if ref_audio is None:
                ref_audio = self.default_ref_audio
            if ref_text is None:
                ref_text = self.default_ref_text

            # Validate reference audio
            if ref_audio is None:
                logger.error("No reference audio provided. F5-TTS requires reference audio for voice cloning.")
                return False

            if not Path(ref_audio).exists():
                logger.error(f"Reference audio not found: {ref_audio}")
                return False

            # Generate output path
            if save_path is None:
                # Use cache directory with hash of text
                text_hash = hash(text) & 0xFFFFFFFF  # 32-bit hash
                save_path = str(self.cache_dir / f"tts_{text_hash}.wav")

            # Check cache
            if Path(save_path).exists():
                logger.debug(f"Using cached audio: {save_path}")
                self.stats['cache_hits'] += 1
            else:
                # Generate audio
                logger.debug(f"Generating audio for: {text[:50]}...")
                self.stats['cache_misses'] += 1

                wav, sr, spect = self.model.infer(
                    ref_file=ref_audio,
                    ref_text=ref_text or "",
                    gen_text=text,
                    file_wave=save_path,
                    remove_silence=remove_silence,
                    speed=speed
                )

                logger.debug(f"Audio generated: {save_path}")

            # Play audio
            if wait:
                self._play_audio(save_path)

            # Update statistics
            generation_time = time.time() - start_time
            self.stats['utterances_generated'] += 1
            self.stats['total_generation_time'] += generation_time
            self.stats['total_characters'] += len(text)

            logger.debug(f"TTS generation took {generation_time:.2f}s")
            return True

        except Exception as e:
            logger.error(f"F5-TTS error: {e}")
            return False

    def _play_audio(self, audio_file: str) -> None:
        """
        Play audio file through speakers

        Args:
            audio_file: Path to audio file
        """
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_file)

            # Convert to numpy
            audio_data = waveform.numpy()

            # Handle stereo/mono
            if audio_data.shape[0] > 1:
                # Convert stereo to mono by averaging channels
                audio_data = audio_data.mean(axis=0)
            else:
                audio_data = audio_data[0]

            # Play using sounddevice
            if self.output_device:
                # Use specific device (e.g., 'hw:2,0' for K-1)
                sd.play(audio_data, samplerate=sample_rate, device=self.output_device)
            else:
                # Use default device
                sd.play(audio_data, samplerate=sample_rate)

            # Wait for playback to finish
            sd.wait()

            logger.debug(f"Audio playback complete: {audio_file}")

        except Exception as e:
            logger.error(f"Audio playback error: {e}")

    def set_default_voice(
        self,
        ref_audio: str,
        ref_text: Optional[str] = None
    ) -> bool:
        """
        Set default voice for all subsequent speech

        Args:
            ref_audio: Path to reference audio file
            ref_text: Transcription of reference audio

        Returns:
            True if successful, False otherwise
        """
        if not Path(ref_audio).exists():
            logger.error(f"Reference audio not found: {ref_audio}")
            return False

        self.default_ref_audio = ref_audio
        self.default_ref_text = ref_text
        logger.info(f"Default voice set: {ref_audio}")
        return True

    def clear_cache(self) -> int:
        """
        Clear cached audio files

        Returns:
            Number of files deleted
        """
        count = 0
        try:
            for file in self.cache_dir.glob("tts_*.wav"):
                file.unlink()
                count += 1
            logger.info(f"Cleared {count} cached audio files")
        except Exception as e:
            logger.error(f"Cache clear error: {e}")

        return count

    def get_stats(self) -> dict:
        """Get TTS statistics"""
        stats = self.stats.copy()

        # Calculate derived statistics
        if stats['utterances_generated'] > 0:
            stats['avg_generation_time'] = (
                stats['total_generation_time'] / stats['utterances_generated']
            )
            stats['avg_characters'] = (
                stats['total_characters'] / stats['utterances_generated']
            )
        else:
            stats['avg_generation_time'] = 0.0
            stats['avg_characters'] = 0

        # Cache statistics
        total_requests = stats['cache_hits'] + stats['cache_misses']
        if total_requests > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / total_requests
        else:
            stats['cache_hit_rate'] = 0.0

        return stats

    def reset_stats(self) -> None:
        """Reset TTS statistics"""
        self.stats = {
            'utterances_generated': 0,
            'total_generation_time': 0.0,
            'total_characters': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }


# Example usage
if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=== F5-TTS Engine Test ===\n")

    # Initialize engine
    print("Initializing F5-TTS engine...")
    engine = F5TTSEngine()

    # Test with reference audio (you'll need to provide your own)
    ref_audio = "path/to/reference_audio.wav"
    ref_text = "This is a sample of my voice."

    if Path(ref_audio).exists():
        # Set default voice
        engine.set_default_voice(ref_audio, ref_text)

        # Generate speech
        print("\nGenerating speech...")
        engine.say("Hello! I am using F5-TTS for high-quality text-to-speech synthesis.")

        # Show statistics
        print("\n=== Statistics ===")
        stats = engine.get_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")
    else:
        print(f"\nERROR: Reference audio not found: {ref_audio}")
        print("Please provide a reference audio file for voice cloning.")
