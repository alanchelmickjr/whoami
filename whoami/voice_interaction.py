"""
Voice Interaction Module for WhoAmI System

Provides voice-based interaction for identifying people:
- Ask for names using text-to-speech
- Listen for responses using speech recognition
- Link names to detected faces
- Provide audio feedback and confirmation

Supports both online (Google) and offline (Vosk) speech recognition.

Usage:
    from whoami.voice_interaction import VoiceInteraction

    # Initialize voice interaction
    voice = VoiceInteraction()

    # Ask for name and get response
    name = voice.ask_name()
    if name:
        print(f"User said: {name}")

    # Provide feedback
    voice.say(f"Nice to meet you, {name}!")

    # Integrate with face recognition
    voice.link_face_to_name(face_encoding, name)
"""

import logging
import time
from typing import Optional, Dict, Any, Callable
from pathlib import Path
import json

# Text-to-Speech
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    logging.warning("pyttsx3 not available.")

# F5-TTS (high-quality neural TTS)
try:
    from whoami.tts_f5 import F5TTSEngine
    F5TTS_AVAILABLE = True
except ImportError:
    F5TTS_AVAILABLE = False
    logging.warning("F5-TTS not available.")

# Speech Recognition
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False
    logging.warning("speech_recognition not available. Voice input disabled.")

# Audio processing
try:
    import sounddevice as sd
    import numpy as np
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    logging.warning("sounddevice not available. Audio features limited.")

# Vosk for offline speech recognition
try:
    from vosk import Model, KaldiRecognizer
    import wave
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False
    logging.warning("Vosk not available. Offline speech recognition disabled.")

# Hardware detection
try:
    from whoami.hardware_detector import get_hardware_detector
    HARDWARE_DETECTOR_AVAILABLE = True
except ImportError:
    HARDWARE_DETECTOR_AVAILABLE = False
    logging.warning("Hardware detector not available. Using default audio devices.")

logger = logging.getLogger(__name__)


class VoiceInteraction:
    """
    Voice interaction system for asking names and providing audio feedback

    Supports:
    - Text-to-speech (TTS) for questions and feedback
    - Speech recognition (online via Google or offline via Vosk)
    - Hardware-aware audio device selection
    - Customizable prompts and responses
    """

    def __init__(
        self,
        tts_engine: Optional[str] = 'pyttsx3',
        sr_engine: Optional[str] = 'google',  # 'google' or 'vosk'
        vosk_model_path: Optional[str] = None,
        audio_device: Optional[str] = None,
        sample_rate: int = 16000,
        timeout: float = 5.0,
        phrase_time_limit: float = 5.0,
        confidence_threshold: float = 0.7,
        f5tts_ref_audio: Optional[str] = None,
        f5tts_ref_text: Optional[str] = None,
        f5tts_model_type: str = "F5-TTS"
    ):
        """
        Initialize voice interaction system

        Args:
            tts_engine: Text-to-speech engine ('pyttsx3' or 'f5-tts')
            sr_engine: Speech recognition engine ('google' or 'vosk')
            vosk_model_path: Path to Vosk model for offline recognition
            audio_device: Audio device identifier (e.g., 'hw:2,0')
            sample_rate: Audio sample rate in Hz
            timeout: Timeout for voice input in seconds
            phrase_time_limit: Max duration for a single phrase
            confidence_threshold: Minimum confidence for speech recognition
            f5tts_ref_audio: Path to reference audio for F5-TTS voice cloning
            f5tts_ref_text: Transcription of reference audio for F5-TTS
            f5tts_model_type: F5-TTS model type ('F5-TTS' or 'E2-TTS')
        """
        self.tts_engine_name = tts_engine
        self.sr_engine_name = sr_engine
        self.vosk_model_path = vosk_model_path
        self.sample_rate = sample_rate
        self.timeout = timeout
        self.phrase_time_limit = phrase_time_limit
        self.confidence_threshold = confidence_threshold

        # Get audio device from hardware detector if available
        if audio_device is None and HARDWARE_DETECTOR_AVAILABLE:
            detector = get_hardware_detector()
            audio_config = detector.get_audio_config()
            if audio_config:
                self.audio_input_device = audio_config.get('input_device')
                self.audio_output_device = audio_config.get('output_device')
                logger.info(f"Using hardware-detected audio devices: in={self.audio_input_device}, out={self.audio_output_device}")
            else:
                self.audio_input_device = audio_device
                self.audio_output_device = audio_device
        else:
            self.audio_input_device = audio_device
            self.audio_output_device = audio_device

        # Initialize TTS engine
        self.tts_engine = None
        self.tts_engine_type = tts_engine

        if tts_engine == 'f5-tts' or tts_engine == 'f5tts':
            # Initialize F5-TTS (high-quality neural TTS)
            if F5TTS_AVAILABLE:
                try:
                    self.tts_engine = F5TTSEngine(
                        model_type=f5tts_model_type,
                        default_ref_audio=f5tts_ref_audio,
                        default_ref_text=f5tts_ref_text,
                        output_device=self.audio_output_device
                    )
                    logger.info("F5-TTS engine initialized (high-quality neural TTS)")
                except Exception as e:
                    logger.error(f"Failed to initialize F5-TTS: {e}")
                    logger.warning("Falling back to pyttsx3")
                    tts_engine = 'pyttsx3'
            else:
                logger.warning("F5-TTS not available, falling back to pyttsx3")
                tts_engine = 'pyttsx3'

        if tts_engine == 'pyttsx3' and self.tts_engine is None:
            # Initialize pyttsx3 (basic TTS)
            if PYTTSX3_AVAILABLE:
                try:
                    self.tts_engine = pyttsx3.init()
                    # Configure voice properties
                    self.tts_engine.setProperty('rate', 150)  # Speed
                    self.tts_engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
                    logger.info("Text-to-speech engine initialized (pyttsx3)")
                except Exception as e:
                    logger.error(f"Failed to initialize pyttsx3: {e}")
                    self.tts_engine = None
            else:
                logger.error("No TTS engine available")

        # Initialize speech recognition
        self.recognizer = None
        self.vosk_model = None

        if SR_AVAILABLE:
            self.recognizer = sr.Recognizer()
            # Adjust for ambient noise
            self.recognizer.energy_threshold = 300
            self.recognizer.dynamic_energy_threshold = True
            logger.info("Speech recognizer initialized")

        # Initialize Vosk model if offline recognition requested
        if sr_engine == 'vosk' and VOSK_AVAILABLE:
            if vosk_model_path is None:
                # Try default location
                vosk_model_path = "/opt/whoami/models/vosk-model-small-en-us-0.15"

            model_path = Path(vosk_model_path)
            if model_path.exists():
                try:
                    self.vosk_model = Model(str(model_path))
                    logger.info(f"Vosk model loaded from {vosk_model_path}")
                except Exception as e:
                    logger.error(f"Failed to load Vosk model: {e}")
                    self.vosk_model = None
            else:
                logger.warning(f"Vosk model not found at {vosk_model_path}")

        # Interaction statistics
        self.stats = {
            'questions_asked': 0,
            'responses_received': 0,
            'recognition_failures': 0,
            'names_learned': 0
        }

    def say(self, text: str, wait: bool = True) -> bool:
        """
        Speak text using TTS engine

        Args:
            text: Text to speak
            wait: Wait for speech to finish before returning

        Returns:
            True if speech succeeded, False otherwise
        """
        if not self.tts_engine:
            logger.warning(f"TTS not available. Would say: {text}")
            return False

        try:
            # Check if using F5-TTS or pyttsx3
            if isinstance(self.tts_engine, F5TTSEngine):
                # F5-TTS engine
                return self.tts_engine.say(text, wait=wait)
            else:
                # pyttsx3 engine
                self.tts_engine.say(text)
                if wait:
                    self.tts_engine.runAndWait()
                logger.debug(f"TTS: {text}")
                return True
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return False

    def listen(
        self,
        prompt: Optional[str] = None,
        timeout: Optional[float] = None,
        phrase_time_limit: Optional[float] = None
    ) -> Optional[str]:
        """
        Listen for speech input and return recognized text

        Args:
            prompt: Optional prompt to speak before listening
            timeout: Time to wait for speech to start (uses default if None)
            phrase_time_limit: Max time for a phrase (uses default if None)

        Returns:
            Recognized text, or None if recognition failed
        """
        if not self.recognizer:
            logger.error("Speech recognizer not available")
            return None

        # Speak prompt if provided
        if prompt:
            self.say(prompt)
            time.sleep(0.5)  # Brief pause after prompt

        # Use defaults if not specified
        if timeout is None:
            timeout = self.timeout
        if phrase_time_limit is None:
            phrase_time_limit = self.phrase_time_limit

        try:
            # Listen for audio input
            with sr.Microphone() as source:
                logger.info("Listening...")
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)

                # Listen for speech
                audio = self.recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit
                )

            # Recognize speech
            if self.sr_engine_name == 'vosk' and self.vosk_model:
                text = self._recognize_vosk(audio)
            else:
                # Use Google Speech Recognition (online)
                text = self._recognize_google(audio)

            if text:
                logger.info(f"Recognized: {text}")
                self.stats['responses_received'] += 1
                return text
            else:
                logger.warning("No speech recognized")
                self.stats['recognition_failures'] += 1
                return None

        except sr.WaitTimeoutError:
            logger.warning("Listening timed out - no speech detected")
            self.stats['recognition_failures'] += 1
            return None
        except Exception as e:
            logger.error(f"Speech recognition error: {e}")
            self.stats['recognition_failures'] += 1
            return None

    def _recognize_google(self, audio: 'sr.AudioData') -> Optional[str]:
        """Recognize speech using Google Speech Recognition (online)"""
        try:
            text = self.recognizer.recognize_google(audio)
            return text.strip()
        except sr.UnknownValueError:
            logger.debug("Google Speech Recognition could not understand audio")
            return None
        except sr.RequestError as e:
            logger.error(f"Google Speech Recognition error: {e}")
            return None

    def _recognize_vosk(self, audio: 'sr.AudioData') -> Optional[str]:
        """Recognize speech using Vosk (offline)"""
        if not self.vosk_model:
            logger.error("Vosk model not loaded")
            return None

        try:
            # Convert AudioData to raw audio for Vosk
            raw_data = audio.get_raw_data(convert_rate=16000, convert_width=2)

            # Create recognizer
            rec = KaldiRecognizer(self.vosk_model, 16000)
            rec.SetWords(True)

            # Process audio
            if rec.AcceptWaveform(raw_data):
                result = json.loads(rec.Result())
            else:
                result = json.loads(rec.FinalResult())

            # Extract text and confidence
            text = result.get('text', '').strip()

            # Check confidence if available
            confidence = result.get('confidence', 1.0)
            if confidence < self.confidence_threshold:
                logger.debug(f"Low confidence ({confidence:.2f}): {text}")
                return None

            return text if text else None

        except Exception as e:
            logger.error(f"Vosk recognition error: {e}")
            return None

    def ask_for_consent(self, prompt: Optional[str] = None) -> bool:
        """
        Ask if person wants to be remembered (opt-in for privacy)

        Args:
            prompt: Custom prompt to use

        Returns:
            True if person consents to be remembered, False if they opt out
        """
        if prompt is None:
            prompt = "Would you like me to remember you for next time? Say yes or no."

        response = self.listen(prompt=prompt, timeout=7.0, phrase_time_limit=3.0)

        if response:
            response_lower = response.lower()
            # Check for affirmative responses
            if any(word in response_lower for word in ['yes', 'yeah', 'yep', 'sure', 'okay', 'ok']):
                return True
            # Check for negative responses
            elif any(word in response_lower for word in ['no', 'nope', 'nah', 'not', "don't"]):
                # Friendly opt-out message
                self.say("Sure no problem, just remember I won't know you next time I see you so I will ask the same things again! :D")
                return False

        # If unclear, ask again once
        clarify = self.listen(
            prompt="Sorry, I didn't catch that. Do you want me to remember you? Yes or no?",
            timeout=7.0,
            phrase_time_limit=3.0
        )

        if clarify:
            clarify_lower = clarify.lower()
            if any(word in clarify_lower for word in ['yes', 'yeah', 'yep', 'sure', 'okay', 'ok']):
                return True
            elif any(word in clarify_lower for word in ['no', 'nope', 'nah', 'not', "don't"]):
                self.say("Sure no problem, just remember I won't know you next time I see you so I will ask the same things again! :D")
                return False

        # Default to not remembering if unclear (privacy-first)
        self.say("I'll ask you again next time!")
        return False

    def ask_name(
        self,
        prompt: Optional[str] = None,
        max_attempts: int = 3,
        ask_consent: bool = True
    ) -> Optional[str]:
        """
        Ask for a person's name and return the response

        Args:
            prompt: Custom prompt to use (default: "What's your name?")
            max_attempts: Maximum number of attempts to recognize speech
            ask_consent: Ask for consent before remembering (default: True)

        Returns:
            Recognized name if consent given, or None if failed/opted out
        """
        # Ask for consent first (privacy-first design)
        if ask_consent:
            if not self.ask_for_consent():
                # Person opted out of being remembered
                logger.info("Person opted out of being remembered")
                return None

        if prompt is None:
            prompt = "What's your name?"

        self.stats['questions_asked'] += 1

        for attempt in range(max_attempts):
            # Ask the question
            name = self.listen(prompt=prompt)

            if name:
                # Capitalize name properly
                name = self._format_name(name)

                # Confirm with user
                if self.confirm_name(name):
                    self.stats['names_learned'] += 1
                    return name
                else:
                    prompt = "Sorry, let me try again. What's your name?"
            else:
                if attempt < max_attempts - 1:
                    prompt = "I didn't catch that. Could you repeat your name?"

        # Failed to get name after max attempts
        self.say("Sorry, I'm having trouble hearing you. Let's try again later.")
        return None

    def confirm_name(self, name: str) -> bool:
        """
        Confirm the recognized name with the user

        Args:
            name: Name to confirm

        Returns:
            True if confirmed, False otherwise
        """
        # Ask for confirmation
        confirmation = self.listen(
            prompt=f"Did you say {name}? Please say yes or no.",
            timeout=5.0,
            phrase_time_limit=2.0
        )

        if confirmation:
            confirmation_lower = confirmation.lower()
            # Check for affirmative responses
            if any(word in confirmation_lower for word in ['yes', 'yeah', 'yep', 'correct', 'right']):
                return True
            # Check for negative responses
            elif any(word in confirmation_lower for word in ['no', 'nope', 'wrong', 'incorrect']):
                return False

        # If unclear, ask again
        return False

    def greet_person(self, name: str) -> None:
        """
        Greet a person by name

        Args:
            name: Person's name
        """
        self.say(f"Nice to meet you, {name}!")

    def welcome_back(self, name: str) -> None:
        """
        Welcome back a known person

        Args:
            name: Person's name
        """
        self.say(f"Welcome back, {name}!")

    def announce_unknown(self) -> None:
        """Announce that an unknown person was detected"""
        self.say("Hello! I don't think we've met before.")

    def _format_name(self, name: str) -> str:
        """
        Format a name with proper capitalization

        Args:
            name: Raw name string

        Returns:
            Formatted name
        """
        # Split on spaces and capitalize each word
        words = name.split()
        formatted = ' '.join(word.capitalize() for word in words)
        return formatted

    def get_stats(self) -> Dict[str, int]:
        """Get interaction statistics"""
        return self.stats.copy()

    def reset_stats(self) -> None:
        """Reset interaction statistics"""
        self.stats = {
            'questions_asked': 0,
            'responses_received': 0,
            'recognition_failures': 0,
            'names_learned': 0
        }


class VoiceEnabledFaceRecognition:
    """
    Face recognition system with voice interaction

    Combines face detection/recognition with voice interaction to:
    - Ask unknown people for their names
    - Link voices to faces
    - Greet known people by name
    - Provide audio feedback
    """

    def __init__(
        self,
        face_recognizer,
        voice_interaction: Optional[VoiceInteraction] = None,
        ask_unknown: bool = True,
        announce_known: bool = True
    ):
        """
        Initialize voice-enabled face recognition

        Args:
            face_recognizer: Face recognition system (e.g., FaceRecognitionAPI)
            voice_interaction: Voice interaction system (created if None)
            ask_unknown: Ask unknown people for their names
            announce_known: Announce known people by name
        """
        self.face_recognizer = face_recognizer
        self.voice_interaction = voice_interaction or VoiceInteraction()
        self.ask_unknown = ask_unknown
        self.announce_known = announce_known

        # Track recently greeted people to avoid repetition
        self.recently_greeted = {}  # {name: timestamp}
        self.greet_cooldown = 60.0  # Don't greet same person within 60 seconds

        logger.info("Voice-enabled face recognition initialized")

    def process_detection(
        self,
        name: str,
        confidence: float,
        face_encoding=None
    ) -> Optional[str]:
        """
        Process a face detection with voice interaction

        Args:
            name: Detected name ('Unknown' if not recognized)
            confidence: Recognition confidence
            face_encoding: Face encoding for linking to name

        Returns:
            Final name (may be different if user provides new name)
        """
        current_time = time.time()

        if name == "Unknown":
            if self.ask_unknown:
                # Ask for name
                self.voice_interaction.announce_unknown()
                new_name = self.voice_interaction.ask_name()

                if new_name and face_encoding is not None:
                    # Add to database
                    self.face_recognizer.add_face(new_name, face_encoding)
                    self.voice_interaction.greet_person(new_name)
                    self.recently_greeted[new_name] = current_time
                    logger.info(f"Learned new person: {new_name}")
                    return new_name
        else:
            # Known person
            if self.announce_known:
                # Check if we recently greeted this person
                last_greet = self.recently_greeted.get(name, 0)
                if current_time - last_greet > self.greet_cooldown:
                    self.voice_interaction.welcome_back(name)
                    self.recently_greeted[name] = current_time

        return name

    def cleanup_greeted_cache(self, max_age: float = 300.0) -> None:
        """
        Clean up old entries from greeted cache

        Args:
            max_age: Maximum age in seconds
        """
        current_time = time.time()
        self.recently_greeted = {
            name: timestamp
            for name, timestamp in self.recently_greeted.items()
            if current_time - timestamp < max_age
        }


# Example usage
if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Test voice interaction
    print("=== Voice Interaction Test ===\n")

    voice = VoiceInteraction()

    # Test TTS
    print("Testing text-to-speech...")
    voice.say("Hello! I am the WhoAmI voice interaction system.")
    time.sleep(1)

    # Test name asking
    print("\nTesting name recognition...")
    name = voice.ask_name()
    if name:
        print(f"Learned name: {name}")
        voice.greet_person(name)
    else:
        print("Failed to get name")

    # Show statistics
    print("\n=== Statistics ===")
    stats = voice.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
