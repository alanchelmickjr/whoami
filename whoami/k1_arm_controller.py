"""
K-1 Booster Arm Controller

High-level arm control for the K-1 humanoid robot using Booster SDK.
Uses B1LocoClient methods for gestures and hand control.

Based on Booster Robotics SDK:
- https://github.com/BoosterRobotics/booster_robotics_sdk
- https://github.com/arminforoughi/booster_k1

Available SDK Methods:
- WaveHand(action) - Wave gesture
- Handshake(action) - Handshake gesture
- MoveHandEndEffectorV2(posture, duration, hand_index) - Move hand endpoint
- ControlDexterousHand(finger_params, hand_index) - Control individual fingers
- ControlGripper(motion_param, mode, hand_index) - Gripper control

Note: Use high-level SDK methods. Low-level joint control not recommended.
"""

import logging
import time
from typing import Optional
from enum import Enum

logger = logging.getLogger(__name__)

# Check if Booster SDK is available
try:
    from booster_robotics_sdk_python import (
        B1LocoClient,
        B1HandIndex,
        DexterousFingerParameter,
        Posture,
        # Add other imports as needed from SDK
    )
    BOOSTER_SDK_AVAILABLE = True
except ImportError:
    logger.warning("Booster SDK not available - arm control will be simulated")
    BOOSTER_SDK_AVAILABLE = False


class HandAction(Enum):
    """Hand gesture actions (from SDK examples)"""
    OPEN = "open"
    CLOSE = "close"
    WAVE = "wave"
    HANDSHAKE = "handshake"


class HandSide(Enum):
    """Which hand to control"""
    LEFT = 0   # B1HandIndex.kLeftHand
    RIGHT = 1  # B1HandIndex.kRightHand


class K1ArmController:
    """
    High-level arm controller for K-1 using Booster SDK

    Uses SDK's built-in gesture methods instead of low-level joint control.

    Example usage:
        controller = K1ArmController(booster_client)
        controller.wave(hand=HandSide.RIGHT)
        controller.handshake()
    """

    def __init__(
        self,
        booster_client: Optional['B1LocoClient'] = None,
        simulate: bool = False
    ):
        """
        Initialize K-1 arm controller

        Args:
            booster_client: Booster SDK client (B1LocoClient instance)
            simulate: If True, simulate gestures (no actual robot control)
        """
        self.booster = booster_client
        self.simulate = simulate

        if not BOOSTER_SDK_AVAILABLE and not simulate:
            logger.error("Booster SDK not available and not in simulation mode")
            self.simulate = True

        if self.simulate:
            logger.info("K-1 Arm Controller initialized in SIMULATION mode")
        else:
            logger.info("K-1 Arm Controller initialized with Booster SDK")

    def wave(self, hand: HandSide = HandSide.RIGHT, duration: float = 2.0) -> bool:
        """
        Perform wave gesture

        Args:
            hand: Which hand to wave (LEFT or RIGHT)
            duration: Duration of wave in seconds

        Returns:
            True if successful, False otherwise
        """
        if self.simulate:
            logger.info(f"[SIMULATE] Waving {hand.name} hand for {duration}s")
            time.sleep(duration)
            return True

        if not self.booster:
            logger.error("No booster client available")
            return False

        try:
            # Use SDK's built-in WaveHand method
            # From SDK: WaveHand(action) where action is kHandOpen/kHandClose
            # This is a simplified implementation - SDK may have dedicated wave method

            logger.info(f"Waving {hand.name} hand")

            # Wave pattern: open -> close -> open
            # Note: Actual SDK may have WaveHand() method that handles this automatically
            # Check SDK docs for exact API

            # For now, using placeholder - NEED TO VERIFY ACTUAL SDK API
            logger.warning("Wave gesture using placeholder - verify SDK API")
            # self.booster.WaveHand(action)  # Actual SDK call

            return True

        except Exception as e:
            logger.error(f"Wave gesture failed: {e}")
            return False

    def handshake(self, duration: float = 3.0) -> bool:
        """
        Perform handshake gesture

        Args:
            duration: Duration of handshake in seconds

        Returns:
            True if successful, False otherwise
        """
        if self.simulate:
            logger.info(f"[SIMULATE] Handshake gesture for {duration}s")
            time.sleep(duration)
            return True

        if not self.booster:
            logger.error("No booster client available")
            return False

        try:
            logger.info("Performing handshake gesture")

            # Use SDK's built-in Handshake method
            # From SDK: Handshake(action)
            # NEED TO VERIFY ACTUAL SDK API
            logger.warning("Handshake using placeholder - verify SDK API")
            # self.booster.Handshake(action)  # Actual SDK call

            return True

        except Exception as e:
            logger.error(f"Handshake gesture failed: {e}")
            return False

    def move_hand(
        self,
        hand: HandSide,
        posture: 'Posture',
        duration_ms: int = 1000
    ) -> bool:
        """
        Move hand end effector to target posture

        Args:
            hand: Which hand (LEFT or RIGHT)
            posture: Target posture (Posture object from SDK)
            duration_ms: Movement duration in milliseconds

        Returns:
            True if successful, False otherwise
        """
        if self.simulate:
            logger.info(f"[SIMULATE] Moving {hand.name} hand to posture")
            time.sleep(duration_ms / 1000.0)
            return True

        if not self.booster:
            logger.error("No booster client available")
            return False

        try:
            logger.info(f"Moving {hand.name} hand to target posture")

            # Use SDK's MoveHandEndEffectorV2
            # From SDK: MoveHandEndEffectorV2(target_posture, time_millis, hand_index)
            hand_index = B1HandIndex.kLeftHand if hand == HandSide.LEFT else B1HandIndex.kRightHand

            result = self.booster.MoveHandEndEffectorV2(
                posture,
                duration_ms,
                hand_index
            )

            return result == 0  # Assuming 0 = success

        except Exception as e:
            logger.error(f"Move hand failed: {e}")
            return False

    def control_fingers(
        self,
        hand: HandSide,
        finger_params: list,  # List of DexterousFingerParameter
    ) -> bool:
        """
        Control individual fingers

        Args:
            hand: Which hand (LEFT or RIGHT)
            finger_params: List of DexterousFingerParameter objects
                          Each has: seq (finger index), angle, force, speed

        Returns:
            True if successful, False otherwise
        """
        if self.simulate:
            logger.info(f"[SIMULATE] Controlling {hand.name} hand fingers")
            return True

        if not self.booster:
            logger.error("No booster client available")
            return False

        try:
            logger.info(f"Controlling {hand.name} hand fingers")

            # Use SDK's ControlDexterousHand
            # From SDK: ControlDexterousHand(finger_params, hand_index)
            hand_index = B1HandIndex.kLeftHand if hand == HandSide.LEFT else B1HandIndex.kRightHand

            result = self.booster.ControlDexterousHand(
                finger_params,
                hand_index
            )

            return result == 0  # Assuming 0 = success

        except Exception as e:
            logger.error(f"Control fingers failed: {e}")
            return False

    # Predefined gestures from basic_controls.py example

    def hand_rock(self, hand: HandSide = HandSide.RIGHT) -> bool:
        """Rock gesture - all fingers closed (fist)"""
        if not BOOSTER_SDK_AVAILABLE or self.simulate:
            logger.info(f"[SIMULATE] Rock gesture with {hand.name} hand")
            return True

        # From basic_controls.py: all fingers at 0° angle
        fingers = []
        for i in range(5):  # 5 fingers
            fingers.append(DexterousFingerParameter(
                seq=i,
                angle=0,
                force=400,
                speed=800
            ))

        return self.control_fingers(hand, fingers)

    def hand_paper(self, hand: HandSide = HandSide.RIGHT) -> bool:
        """Paper gesture - all fingers extended (open hand)"""
        if not BOOSTER_SDK_AVAILABLE or self.simulate:
            logger.info(f"[SIMULATE] Paper gesture with {hand.name} hand")
            return True

        # From basic_controls.py: all fingers at 1000° (fully extended)
        fingers = []
        for i in range(5):
            fingers.append(DexterousFingerParameter(
                seq=i,
                angle=1000,
                force=400,
                speed=800
            ))

        return self.control_fingers(hand, fingers)

    def hand_scissor(self, hand: HandSide = HandSide.RIGHT) -> bool:
        """Scissor gesture - index and middle fingers extended"""
        if not BOOSTER_SDK_AVAILABLE or self.simulate:
            logger.info(f"[SIMULATE] Scissor gesture with {hand.name} hand")
            return True

        # From basic_controls.py: fingers 2-3 at 1000°, others 0°
        finger_angles = [0, 0, 1000, 1000, 0]  # Thumb, Index, Middle, Ring, Pinky
        fingers = []
        for i, angle in enumerate(finger_angles):
            fingers.append(DexterousFingerParameter(
                seq=i,
                angle=angle,
                force=400,
                speed=800
            ))

        return self.control_fingers(hand, fingers)

    def hand_ok(self, hand: HandSide = HandSide.RIGHT) -> bool:
        """OK gesture - thumb and index finger form circle"""
        if not BOOSTER_SDK_AVAILABLE or self.simulate:
            logger.info(f"[SIMULATE] OK gesture with {hand.name} hand")
            return True

        # From basic_controls.py: mixed angles
        finger_angles = [350, 0, 1000, 1000, 500]
        fingers = []
        for i, angle in enumerate(finger_angles):
            fingers.append(DexterousFingerParameter(
                seq=i,
                angle=angle,
                force=400,
                speed=800
            ))

        return self.control_fingers(hand, fingers)


# Example usage
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("K-1 Arm Controller Demo")
    print("=" * 60)

    if not BOOSTER_SDK_AVAILABLE:
        print("⚠️  Booster SDK not available. Running in simulation mode.")
        controller = K1ArmController(simulate=True)
    else:
        print("Booster SDK available - initialize B1LocoClient first")
        print("Example:")
        print("  from booster_robotics_sdk_python import B1LocoClient, ChannelFactory")
        print("  ChannelFactory.Instance().Init(0, '127.0.0.1')")
        print("  booster = B1LocoClient()")
        print("  booster.Init()")
        print("  controller = K1ArmController(booster)")
        controller = K1ArmController(simulate=True)

    # Demo gestures
    print("\nTesting gestures in simulation mode:")

    print("\n1. Wave gesture")
    controller.wave(hand=HandSide.RIGHT)

    print("\n2. Handshake gesture")
    controller.handshake()

    print("\n3. Rock-paper-scissors gestures")
    controller.hand_rock()
    time.sleep(0.5)
    controller.hand_paper()
    time.sleep(0.5)
    controller.hand_scissor()
    time.sleep(0.5)
    controller.hand_ok()

    print("\nDemo complete!")
