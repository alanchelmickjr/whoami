#!/usr/bin/env python3
"""
K-1 Autonomous Face Interaction Demo

Combines:
- Booster SDK head control
- YOLO face detection
- Voice interaction
- Conversation tracking

Based on K-1's working patterns (basic_controls.py, cam_yolo.py)
"""

import sys
import time
import threading
import cv2
import numpy as np
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from booster_robotics_sdk_python import B1LocoClient, ChannelFactory, RobotMode
    BOOSTER_AVAILABLE = True
except ImportError:
    BOOSTER_AVAILABLE = False
    print("⚠️  Booster SDK not available")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  YOLO not available. Install: pip install ultralytics")

from whoami.voice_interaction import VoiceInteraction
from whoami.k1_face_explorer import K1FaceExplorer, PersonProfile


class AutonomousFaceInteraction:
    """Simple autonomous face interaction for K-1"""

    def __init__(self, booster_client=None):
        self.booster = booster_client
        self.running = False

        # Initialize YOLO for face detection
        if YOLO_AVAILABLE:
            print("Loading YOLO model...")
            self.yolo = YOLO('yolov8n.pt')
            print("✓ YOLO loaded")
        else:
            self.yolo = None
            print("✗ YOLO not available")

        # Initialize voice
        print("Initializing voice interaction...")
        self.voice = VoiceInteraction(tts_engine='pyttsx3')  # or 'f5-tts'
        print("✓ Voice initialized")

        # Initialize face explorer
        print("Initializing face explorer...")
        self.explorer = K1FaceExplorer(booster_client=booster_client)
        print("✓ Face explorer initialized")

        # Scan positions for head (yaw, pitch in radians)
        self.scan_positions = [
            (0.0, 0.0),      # Center
            (-0.785, 0.0),   # Left
            (0.785, 0.0),    # Right
            (0.0, 0.3),      # Up
            (0.0, -0.3),     # Down
        ]
        self.scan_index = 0

        # Recently greeted (avoid spam)
        self.recently_greeted = {}
        self.greet_cooldown = 60.0  # seconds

    def move_head(self, yaw: float, pitch: float):
        """Move K-1 head"""
        if self.booster:
            # K-1 SDK: RotateHead(pitch, yaw)
            result = self.booster.RotateHead(pitch, yaw)
            if result != 0:
                print(f"⚠️  RotateHead failed: {result}")
            time.sleep(0.3)
        else:
            print(f"Simulate: Head -> yaw={yaw:.2f}, pitch={pitch:.2f}")
            time.sleep(0.3)

    def detect_faces(self, frame):
        """Detect faces in frame using YOLO"""
        if not self.yolo:
            return []

        try:
            # Run YOLO detection
            results = self.yolo(frame, conf=0.5, verbose=False)

            faces = []
            for r in results[0].boxes:
                # Check if it's a person (class 0)
                if int(r.cls) == 0:
                    x1, y1, x2, y2 = r.xyxy[0].cpu().numpy()
                    conf = float(r.conf)
                    faces.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': conf
                    })

            return faces
        except Exception as e:
            print(f"Detection error: {e}")
            return []

    def greet_person(self, name: str):
        """Greet person with conversation recall"""
        current_time = time.time()

        # Check cooldown
        last_greet = self.recently_greeted.get(name, 0)
        if current_time - last_greet < self.greet_cooldown:
            print(f"  [Skipping {name} - recently greeted]")
            return

        # Generate personalized greeting
        greeting = self.explorer.greet_person(name, include_conversation=True)

        print(f"\n🤖 K-1: {greeting}")
        self.voice.say(greeting)

        self.recently_greeted[name] = current_time

    def ask_for_name(self):
        """Ask unknown person for their name"""
        print("\n🤖 K-1: Hello! I don't think we've met before.")
        self.voice.say("Hello! I don't think we've met before.")
        time.sleep(0.5)

        print("🤖 K-1: What's your name?")
        name = self.voice.ask_name(prompt="What's your name?", max_attempts=3)

        if name:
            print(f"✓ Learned name: {name}")
            self.greet_person(name)
            return name

        print("✗ Could not learn name")
        return None

    def scan_and_interact(self):
        """Scan for faces and interact"""
        print("\n" + "="*60)
        print("Starting autonomous face interaction")
        print("Press Ctrl+C to stop")
        print("="*60)

        self.running = True

        try:
            while self.running:
                # Move to next scan position
                yaw, pitch = self.scan_positions[self.scan_index]
                print(f"\nScanning position {self.scan_index + 1}/{len(self.scan_positions)}")
                self.move_head(yaw, pitch)

                # In real implementation, would get camera frame here
                # For now, simulate
                print("  Checking for faces...")

                # Simulate face detection (replace with actual camera feed)
                # faces = self.detect_faces(camera_frame)

                # Demo: Simulate finding a face occasionally
                if self.scan_index == 0:  # Only check at center position
                    # In real code, process actual detections
                    # For demo, just show how it would work
                    print("  [Simulated: No faces detected]")

                # Next position
                self.scan_index = (self.scan_index + 1) % len(self.scan_positions)

                # Delay between scans
                time.sleep(2.0)

        except KeyboardInterrupt:
            print("\n\nStopping...")
        finally:
            self.running = False
            # Return head to center
            self.move_head(0.0, 0.0)

    def demo_interactions(self):
        """Demo showing interaction capabilities"""
        print("\n" + "="*60)
        print("K-1 AUTONOMOUS FACE INTERACTION DEMO")
        print("="*60)

        # Demo 1: First meeting
        print("\n--- Demo 1: Meeting Alice (first time) ---")
        self.greet_person("Alice")

        # Add conversation note
        self.explorer.add_conversation_note(
            "Alice",
            "her dog Max",
            "Alice has a golden retriever who loves swimming"
        )
        print("  [Added conversation note: her dog Max]")

        # Demo 2: Meeting Alice again
        print("\n--- Demo 2: Meeting Alice again ---")
        time.sleep(2)  # Simulate time passing
        self.greet_person("Alice")

        # Demo 3: Meeting Bob
        print("\n--- Demo 3: Meeting Bob (first time) ---")
        self.greet_person("Bob")

        self.explorer.add_conversation_note(
            "Bob",
            "his new job",
            "Bob just started working at a robotics company"
        )
        print("  [Added conversation note: his new job]")

        # Demo 4: Show profiles
        print("\n" + "="*60)
        print("PERSON PROFILES")
        print("="*60)

        for name in ["Alice", "Bob"]:
            profile = self.explorer.get_profile(name)
            if profile:
                print(f"\n{name}:")
                print(f"  Encounters: {profile.encounter_count}")
                print(f"  Conversations:")
                for conv in profile.conversations:
                    print(f"    - {conv.topic}: {conv.note}")

        print("\n" + "="*60)
        print("Demo complete!")
        print("="*60)


def main():
    if len(sys.argv) < 2 and BOOSTER_AVAILABLE:
        print(f"Usage: {sys.argv[0]} <network_interface>")
        print("Example: python k1_autonomous_face_interaction.py eth0")
        print("\nOr run without arguments for demo mode (no real robot)")
        sys.exit(1)

    # Initialize Booster SDK if available
    booster_client = None
    if BOOSTER_AVAILABLE and len(sys.argv) >= 2:
        print("\n" + "="*60)
        print("Initializing Booster SDK...")
        print("="*60)

        ChannelFactory.Instance().Init(0, sys.argv[1])
        booster_client = B1LocoClient()
        booster_client.Init()
        time.sleep(1.0)

        print("✓ Booster SDK initialized")

        # Set robot to PREP mode (standing)
        print("Setting robot to PREP mode...")
        result = booster_client.ChangeMode(RobotMode.kPrepare)
        if result != 0:
            print(f"⚠️  Mode change failed: {result}")
        else:
            print("✓ Robot ready")

    # Create interaction system
    system = AutonomousFaceInteraction(booster_client)

    # Run demo
    print("\nChoose mode:")
    print("  1) Demo interactions (simulated)")
    print("  2) Live scanning (requires camera)")

    try:
        choice = input("\nChoice [1]: ").strip() or "1"

        if choice == "1":
            system.demo_interactions()
        else:
            system.scan_and_interact()

    except KeyboardInterrupt:
        print("\n\nExiting...")

    # Cleanup
    if booster_client:
        print("\nReturning head to center...")
        booster_client.RotateHead(0.0, 0.0)


if __name__ == '__main__':
    main()
