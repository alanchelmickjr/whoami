"""
F5-TTS Engine - Simplified for K-1 Robot

High-quality neural TTS with voice cloning, following K-1's simple architecture pattern.
"""

import logging
from pathlib import Path
from typing import Optional
import time

logger = logging.getLogger(__name__)

# F5-TTS imports
try:
    from f5_tts.api import F5TTS
    import torch
    import torchaudio
    import sounddevice as sd
    F5TTS_AVAILABLE = True
except ImportError as e:
    F5TTS_AVAILABLE = False
    logger.warning(f"F5-TTS not available: {e}")


class F5TTSEngine:
    """Simple F5-TTS wrapper for K-1 robot voice synthesis"""

    def __init__(
        self,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        output_device: Optional[str] = None
    ):
        """
        Initialize F5-TTS engine

        Args:
            ref_audio: Reference audio file for voice cloning
            ref_text: Transcription of reference audio
            output_device: Audio device (e.g., 'hw:2,0' for K-1)
        """
        if not F5TTS_AVAILABLE:
            raise ImportError("F5-TTS not available. Install: pip install f5-tts torch torchaudio")

        self.ref_audio = ref_audio
        self.ref_text = ref_text
        self.output_device = output_device

        # Auto-detect device (CUDA on Jetson Orin NX)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize model
        self.model = F5TTS(model_type="F5-TTS", device=device)
        logger.info(f"F5-TTS initialized on {device}")

    def say(self, text: str, wait: bool = True) -> bool:
        """
        Speak text using F5-TTS

        Args:
            text: Text to synthesize
            wait: Wait for playback to finish

        Returns:
            True if successful, False otherwise
        """
        if not text or not text.strip():
            return False

        if not self.ref_audio:
            logger.error("No reference audio set")
            return False

        try:
            # Generate audio
            wav, sr, _ = self.model.infer(
                ref_file=self.ref_audio,
                ref_text=self.ref_text or "",
                gen_text=text
            )

            # Play audio
            if wait:
                # Convert to numpy for sounddevice
                audio_data = wav[0] if wav.ndim > 1 else wav

                if self.output_device:
                    sd.play(audio_data, samplerate=sr, device=self.output_device)
                else:
                    sd.play(audio_data, samplerate=sr)

                sd.wait()

            return True

        except Exception as e:
            logger.error(f"F5-TTS error: {e}")
            return False

    def set_voice(self, ref_audio: str, ref_text: Optional[str] = None):
        """Change the voice"""
        if Path(ref_audio).exists():
            self.ref_audio = ref_audio
            self.ref_text = ref_text
            return True
        return False
