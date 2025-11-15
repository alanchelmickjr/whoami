"""
K-1 Autonomous Face Explorer

Integrates K-1 Booster SDK head control with face recognition for autonomous
face exploration, greetings, and conversation tracking.

Features:
- Autonomous head scanning to find faces
- Personalized greetings with time since last seen
- Conversation topic tracking and recall
- K-1 Booster SDK head control integration
"""

import time
import logging
import random
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import threading

try:
    from booster_robotics_sdk_python import B1LocoClient
    BOOSTER_SDK_AVAILABLE = True
except ImportError:
    BOOSTER_SDK_AVAILABLE = False
    logging.warning("Booster SDK not available")

from whoami.yolo_face_recognition import (
    K1FaceRecognitionSystem,
    FaceRecognitionResult,
    format_time_delta
)

logger = logging.getLogger(__name__)


@dataclass
class ConversationNote:
    """Note from a conversation"""
    timestamp: float
    topic: str
    note: str


@dataclass
class PersonProfile:
    """Extended profile for a person"""
    name: str
    first_seen: float
    last_seen: float
    encounter_count: int
    conversations: List[ConversationNote] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)

    def add_conversation_note(self, topic: str, note: str):
        """Add a conversation note"""
        self.conversations.append(ConversationNote(
            timestamp=time.time(),
            topic=topic,
            note=note
        ))
        # Keep only recent conversations (last 10)
        if len(self.conversations) > 10:
            self.conversations = self.conversations[-10:]

    def get_recent_conversation(self) -> Optional[ConversationNote]:
        """Get most recent conversation note"""
        if self.conversations:
            return self.conversations[-1]
        return None


class K1FaceExplorer:
    """
    Autonomous face exploration system for K-1 robot

    Scans environment for faces, recognizes people, and provides
    personalized greetings with conversation recall.
    """

    def __init__(
        self,
        booster_client: Optional['B1LocoClient'] = None,
        face_system: Optional[K1FaceRecognitionSystem] = None,
        profiles_path: str = "person_profiles.json"
    ):
        """
        Initialize K-1 Face Explorer

        Args:
            booster_client: Booster SDK client for head control
            face_system: Face recognition system
            profiles_path: Path to store person profiles
        """
        self.booster = booster_client
        self.face_system = face_system
        self.profiles_path = Path(profiles_path)

        # Person profiles with conversation history
        self.profiles: Dict[str, PersonProfile] = {}
        self.load_profiles()

        # Scanning state
        self.scanning = False
        self.scan_thread = None

        # Head scan positions (yaw, pitch) in radians
        # K-1 head range: yaw ±60°, pitch -30° to 45°
        self.scan_positions = [
            (0.0, 0.0),      # Center
            (-0.785, 0.0),   # Left (-45°)
            (0.785, 0.0),    # Right (45°)
            (0.0, 0.3),      # Up (17°)
            (0.0, -0.3),     # Down (-17°)
            (-0.5, 0.2),     # Upper left
            (0.5, 0.2),      # Upper right
            (-0.5, -0.2),    # Lower left
            (0.5, -0.2),     # Lower right
        ]
        self.current_scan_index = 0

        logger.info("K-1 Face Explorer initialized")

    def load_profiles(self):
        """Load person profiles from disk"""
        if self.profiles_path.exists():
            try:
                with open(self.profiles_path, 'r') as f:
                    data = json.load(f)

                for name, profile_data in data.items():
                    # Reconstruct conversation notes
                    conversations = [
                        ConversationNote(**conv)
                        for conv in profile_data.get('conversations', [])
                    ]

                    self.profiles[name] = PersonProfile(
                        name=name,
                        first_seen=profile_data.get('first_seen', time.time()),
                        last_seen=profile_data.get('last_seen', time.time()),
                        encounter_count=profile_data.get('encounter_count', 0),
                        conversations=conversations,
                        preferences=profile_data.get('preferences', {})
                    )

                logger.info(f"Loaded {len(self.profiles)} person profiles")
            except Exception as e:
                logger.error(f"Failed to load profiles: {e}")

    def save_profiles(self):
        """Save person profiles to disk"""
        try:
            data = {}
            for name, profile in self.profiles.items():
                data[name] = {
                    'name': profile.name,
                    'first_seen': profile.first_seen,
                    'last_seen': profile.last_seen,
                    'encounter_count': profile.encounter_count,
                    'conversations': [
                        {
                            'timestamp': conv.timestamp,
                            'topic': conv.topic,
                            'note': conv.note
                        }
                        for conv in profile.conversations
                    ],
                    'preferences': profile.preferences
                }

            with open(self.profiles_path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved {len(self.profiles)} person profiles")
        except Exception as e:
            logger.error(f"Failed to save profiles: {e}")

    def move_head(self, yaw: float, pitch: float, smooth: bool = True):
        """
        Move K-1 head to position

        Args:
            yaw: Yaw angle in radians (-0.785 to 0.785 for ±45°)
            pitch: Pitch angle in radians (-0.5 to 0.8)
            smooth: Use smooth movement
        """
        if not self.booster:
            logger.warning("Booster client not available")
            return

        try:
            # K-1 SDK uses RotateHead(pitch, yaw)
            result = self.booster.RotateHead(pitch, yaw)
            if result != 0:
                logger.warning(f"RotateHead failed: {result}")

            # Brief delay for movement
            if smooth:
                time.sleep(0.3)
            else:
                time.sleep(0.1)

        except Exception as e:
            logger.error(f"Head movement error: {e}")

    def scan_for_faces(self, duration: float = 30.0):
        """
        Scan environment for faces autonomously

        Args:
            duration: How long to scan in seconds
        """
        logger.info(f"Starting face scan for {duration}s")
        start_time = time.time()

        while time.time() - start_time < duration and self.scanning:
            # Move to next scan position
            yaw, pitch = self.scan_positions[self.current_scan_index]
            logger.debug(f"Scanning position {self.current_scan_index}: yaw={yaw:.2f}, pitch={pitch:.2f}")

            self.move_head(yaw, pitch)

            # Pause to let camera capture and process
            time.sleep(1.0)

            # Check if face system detected anyone
            # (Face system runs in parallel, processing camera feed)

            # Move to next position
            self.current_scan_index = (self.current_scan_index + 1) % len(self.scan_positions)

        # Return to center
        self.move_head(0.0, 0.0)
        logger.info("Face scan complete")

    def start_autonomous_exploration(self):
        """Start autonomous face exploration"""
        if self.scanning:
            logger.warning("Already scanning")
            return

        self.scanning = True
        self.scan_thread = threading.Thread(
            target=self._exploration_loop,
            daemon=True
        )
        self.scan_thread.start()
        logger.info("Autonomous exploration started")

    def stop_autonomous_exploration(self):
        """Stop autonomous face exploration"""
        self.scanning = False
        if self.scan_thread:
            self.scan_thread.join(timeout=2.0)
        logger.info("Autonomous exploration stopped")

    def _exploration_loop(self):
        """Main exploration loop"""
        while self.scanning:
            # Scan for faces for 30 seconds
            self.scan_for_faces(duration=30.0)

            # Rest period
            time.sleep(5.0)

    def greet_person(
        self,
        name: str,
        include_conversation: bool = True
    ) -> str:
        """
        Generate personalized greeting

        Args:
            name: Person's name
            include_conversation: Include reference to last conversation

        Returns:
            Greeting message
        """
        # Update or create profile
        if name not in self.profiles:
            self.profiles[name] = PersonProfile(
                name=name,
                first_seen=time.time(),
                last_seen=time.time(),
                encounter_count=1
            )
        else:
            profile = self.profiles[name]
            profile.last_seen = time.time()
            profile.encounter_count += 1

        profile = self.profiles[name]

        # Calculate time since last seen
        time_since = time.time() - profile.first_seen
        if profile.encounter_count > 1:
            time_since = time.time() - profile.last_seen

        # Build greeting
        if profile.encounter_count == 1:
            greeting = f"Nice to meet you, {name}!"
        elif time_since < 300:  # < 5 minutes
            greeting = f"Hi again, {name}!"
        else:
            time_str = format_time_delta(time_since)
            greeting = f"Hi {name}, it's been {time_str} since we last talked!"

            # Add conversation recall
            if include_conversation:
                recent_conv = profile.get_recent_conversation()
                if recent_conv:
                    # Reference the topic
                    greeting += f" Last time we talked about {recent_conv.topic}."

        self.save_profiles()
        return greeting

    def add_conversation_note(
        self,
        name: str,
        topic: str,
        note: str
    ):
        """
        Add a conversation note for a person

        Args:
            name: Person's name
            topic: Conversation topic (e.g., "their dog", "their job")
            note: Brief note about conversation
        """
        if name not in self.profiles:
            logger.warning(f"No profile for {name}")
            return

        self.profiles[name].add_conversation_note(topic, note)
        self.save_profiles()
        logger.info(f"Added conversation note for {name}: {topic}")

    def get_profile(self, name: str) -> Optional[PersonProfile]:
        """Get person profile"""
        return self.profiles.get(name)

    def process_face_detection(self, result: FaceRecognitionResult):
        """
        Process a face detection result

        Args:
            result: Face recognition result
        """
        name = result.name

        if name == "Unknown":
            return

        # Generate greeting
        greeting = self.greet_person(name, include_conversation=True)

        # Speak greeting (via face system's voice)
        if self.face_system and self.face_system.voice:
            self.face_system.voice.say(greeting)


# Example usage
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("K-1 Face Explorer Demo")
    print("=" * 60)

    if not BOOSTER_SDK_AVAILABLE:
        print("⚠️  Booster SDK not available. Running in simulation mode.")
        print("   Install: pip install booster_robotics_sdk_python")
        booster_client = None
    else:
        print("Initializing Booster SDK...")
        # Note: Requires network interface parameter
        # booster_client = B1LocoClient()
        # booster_client.Init()
        booster_client = None

    # Create explorer
    explorer = K1FaceExplorer(booster_client=booster_client)

    # Demonstrate profile management
    print("\nAdding sample profiles...")

    # Simulate greeting Alice
    greeting1 = explorer.greet_person("Alice")
    print(f"\nFirst meeting: {greeting1}")

    # Add conversation note
    explorer.add_conversation_note(
        "Alice",
        "her dog Max",
        "Alice has a golden retriever named Max who loves to play fetch"
    )

    # Simulate seeing Alice again later
    time.sleep(2)
    greeting2 = explorer.greet_person("Alice", include_conversation=True)
    print(f"\nSecond meeting: {greeting2}")

    # Show profile
    profile = explorer.get_profile("Alice")
    if profile:
        print(f"\n{'='*60}")
        print(f"Profile for {profile.name}:")
        print(f"  Encounter count: {profile.encounter_count}")
        print(f"  Conversations: {len(profile.conversations)}")
        if profile.conversations:
            for conv in profile.conversations:
                print(f"    - {conv.topic}: {conv.note}")

    print("\nDemo complete!")
