#!/usr/bin/env python3
"""
Twiki - K-1 Greeter Bot

A friendly robot that:
- Identifies people using face recognition
- Asks for consent before remembering
- Remembers names and conversations
- Greets with personalized messages
- Waves when greeting (nub hands!)

Hardware: K-1 Booster robot ("Twiki" like from Buck Rogers, but without the stutter)
- Jetson Orin NX 16GB
- Zod camera for vision
- F5-TTS for natural voice
- Round nub hands (can wave, no fingers)

Usage:
    # Simulation mode (no hardware, uses webcam)
    python twiki_greeter_bot.py --simulate

    # On actual K-1 hardware
    python twiki_greeter_bot.py --interface wlan0

Author: Alan Helmick Jr + Claude
Based on: https://github.com/arminforoughi/booster_k1
"""

import sys
import time
import argparse
import logging
import cv2
from pathlib import Path

# Booster SDK (optional for simulation)
try:
    from booster_robotics_sdk_python import B1LocoClient, ChannelFactory, RobotMode
    BOOSTER_SDK_AVAILABLE = True
except ImportError:
    BOOSTER_SDK_AVAILABLE = False
    print("⚠️  Booster SDK not available - running in simulation mode")

# WhoAmI components
from whoami.yolo_face_recognition import K1FaceRecognitionSystem
from whoami.voice_interaction import VoiceInteraction
from whoami.k1_face_explorer import K1FaceExplorer
from whoami.k1_arm_controller import K1ArmController, HandSide
from whoami.gun_storage import GunStorageManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TwikiGreeterBot:
    """
    Twiki the Greeter Bot

    Friendly K-1 robot that remembers people and greets them with waves.
    """

    def __init__(
        self,
        simulate: bool = False,
        network_interface: str = '127.0.0.1',
        camera_index: int = 0,
        storage_dir: str = '/opt/whoami/gun_storage'
    ):
        """
        Initialize Twiki

        Args:
            simulate: Run without hardware (webcam only)
            network_interface: DDS interface for Booster SDK (127.0.0.1, eth0, wlan0)
            camera_index: Camera device index (0 for webcam, USB camera index for K-1)
            storage_dir: Gun.js storage directory for face/conversation data
        """
        self.simulate = simulate
        self.camera_index = camera_index
        self.running = False

        print("=" * 60)
        print("  Twiki - K-1 Greeter Bot")
        print("  'Bidibidibidi!' (just kidding, no stutter)")
        print("=" * 60)

        # Initialize Booster SDK (if not simulating)
        self.booster = None
        if not simulate and BOOSTER_SDK_AVAILABLE:
            print(f"\n1. Initializing Booster SDK (interface: {network_interface})...")
            try:
                ChannelFactory.Instance().Init(0, network_interface)
                self.booster = B1LocoClient()
                self.booster.Init()

                # Set to PREPARE mode (standing, head control active)
                print("   Setting PREPARE mode...")
                self.booster.ChangeMode(RobotMode.kPrepare)
                time.sleep(2)

                print("   ✓ Booster SDK connected")
            except Exception as e:
                logger.error(f"Failed to initialize Booster SDK: {e}")
                logger.warning("Falling back to simulation mode")
                self.simulate = True

        if self.simulate:
            print("\n⚠️  Running in SIMULATION mode (no hardware)")

        # Initialize Gun.js storage
        print("\n2. Initializing Gun.js storage (local, encrypted)...")
        try:
            Path(storage_dir).mkdir(parents=True, exist_ok=True)
            self.gun_storage = GunStorageManager(
                robot_id='twiki',
                config={'storage_dir': storage_dir}
            )
            print("   ✓ Gun.js storage ready")
        except Exception as e:
            logger.error(f"Failed to initialize Gun.js: {e}")
            self.gun_storage = None

        # Initialize voice interaction (F5-TTS if available)
        print("\n3. Initializing voice system (F5-TTS preferred)...")
        try:
            self.voice = VoiceInteraction(
                tts_engine='f5-tts',  # Will fall back to pyttsx3 if not available
                sr_engine='google',   # or 'vosk' for offline
                f5tts_ref_audio='/opt/whoami/voices/k1_voice.wav'  # Optional
            )
            print("   ✓ Voice system ready")
        except Exception as e:
            logger.error(f"Failed to initialize voice: {e}")
            self.voice = None

        # Initialize face recognition
        print("\n4. Initializing face recognition (YOLO + DeepFace)...")
        try:
            self.face_system = K1FaceRecognitionSystem(
                booster_client=self.booster if not self.simulate else None,
                enable_voice=False,  # We handle voice separately
                camera_index=camera_index
            )
            print("   ✓ Face recognition ready")
        except Exception as e:
            logger.error(f"Failed to initialize face recognition: {e}")
            self.face_system = None

        # Initialize arm controller (wave gesture)
        print("\n5. Initializing arm controller (wave gesture)...")
        self.arm_controller = K1ArmController(
            booster_client=self.booster if not self.simulate else None,
            simulate=self.simulate
        )
        print("   ✓ Arm controller ready (nub hands!)")

        # Initialize face explorer (conversation tracking)
        print("\n6. Initializing conversation tracker...")
        self.explorer = K1FaceExplorer(
            booster_client=self.booster if not self.simulate else None,
            face_system=self.face_system
        )
        print("   ✓ Conversation tracker ready")

        # Camera
        self.camera = None

        # Track recent greetings (avoid spamming)
        self.recently_greeted = {}
        self.greet_cooldown = 30.0  # Don't greet same person within 30 seconds

        print("\n" + "=" * 60)
        print("  Twiki is ready!")
        print("=" * 60)

    def start(self):
        """Start Twiki greeter bot"""
        print("\nStarting greeter bot...")

        if self.voice:
            self.voice.say("Hello! I'm Twiki, your friendly greeter bot!")

        # Open camera
        self.camera = cv2.VideoCapture(self.camera_index)
        if not self.camera.isOpened():
            logger.error(f"Failed to open camera {self.camera_index}")
            return

        self.running = True

        try:
            print("\nGreeting people... (Press 'q' to quit)")
            while self.running:
                ret, frame = self.camera.read()
                if not ret:
                    logger.warning("Failed to read camera frame")
                    continue

                # Process frame for face recognition
                result = self.face_system.process_frame(frame) if self.face_system else None

                if result and result.name:
                    # Someone detected
                    self.handle_person(result.name, frame)

                # Display frame (simulation mode)
                if self.simulate or True:  # Always show for debugging
                    cv2.imshow('Twiki Greeter Bot', frame)

                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                time.sleep(0.1)  # ~10 FPS

        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.stop()

    def handle_person(self, name: str, frame):
        """
        Handle detected person

        Args:
            name: Person's name ('Unknown' if not recognized)
            frame: Camera frame (for display)
        """
        current_time = time.time()

        # Check if we recently greeted this person
        last_greet = self.recently_greeted.get(name, 0)
        if current_time - last_greet < self.greet_cooldown:
            return  # Don't spam greetings

        if name == "Unknown":
            # New person - ask if they want to be remembered
            self.meet_new_person(frame)
        else:
            # Known person - greet them
            self.greet_known_person(name)

        # Mark as greeted
        self.recently_greeted[name] = current_time

    def meet_new_person(self, frame):
        """Meet a new person (with consent)"""
        print("\n👤 Unknown person detected!")

        if not self.voice:
            return

        # Greet
        self.voice.say("Hello! I don't think we've met before.")
        time.sleep(1)

        # Ask for name (includes consent check)
        name = self.voice.ask_name(ask_consent=True)

        if name:
            print(f"✓ Learned name: {name}")

            # Store in face system
            if self.face_system:
                # Extract face encoding from frame and store
                # (Simplified - actual implementation would extract encoding from result)
                pass

            # Greet and wave
            self.voice.say(f"Nice to meet you, {name}!")
            self.arm_controller.wave(hand=HandSide.RIGHT)

            # Update explorer profile
            greeting = self.explorer.greet_person(name, include_conversation=False)
            print(f"   Greeting: {greeting}")

        else:
            print("✗ Person opted out or recognition failed")
            self.voice.say("No problem! Enjoy your day!")

    def greet_known_person(self, name: str):
        """Greet a known person"""
        print(f"\n👋 Recognized: {name}")

        if not self.voice:
            return

        # Get personalized greeting from explorer
        greeting = self.explorer.greet_person(name, include_conversation=True)

        # Speak greeting
        self.voice.say(greeting)

        # Wave
        self.arm_controller.wave(hand=HandSide.RIGHT)

        print(f"   Greeting: {greeting}")

    def stop(self):
        """Stop Twiki greeter bot"""
        self.running = False

        if self.camera:
            self.camera.release()

        cv2.destroyAllWindows()

        # Return to DAMPING mode (safe shutdown)
        if self.booster and not self.simulate:
            print("\nReturning to DAMPING mode...")
            self.booster.ChangeMode(RobotMode.kDamping)

        if self.voice:
            self.voice.say("Goodbye! It was nice meeting everyone!")

        print("\nTwiki stopped.")


def main():
    parser = argparse.ArgumentParser(description='Twiki - K-1 Greeter Bot')

    parser.add_argument(
        '--simulate',
        action='store_true',
        help='Run in simulation mode (no hardware, uses webcam)'
    )

    parser.add_argument(
        '--interface',
        type=str,
        default='127.0.0.1',
        help='Network interface for Booster SDK (127.0.0.1, eth0, wlan0)'
    )

    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera device index (0 for webcam)'
    )

    parser.add_argument(
        '--storage',
        type=str,
        default='/opt/whoami/gun_storage',
        help='Gun.js storage directory'
    )

    args = parser.parse_args()

    # Create and start Twiki
    twiki = TwikiGreeterBot(
        simulate=args.simulate or not BOOSTER_SDK_AVAILABLE,
        network_interface=args.interface,
        camera_index=args.camera,
        storage_dir=args.storage
    )

    twiki.start()


if __name__ == '__main__':
    main()
