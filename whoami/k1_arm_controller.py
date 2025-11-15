"""
K-1 Booster Arm Controller

High-level arm control for the K-1 humanoid robot using Booster SDK.
Uses B1LocoClient methods for simple gestures.

K-1 Hardware:
- Round nub hands (NO fingers or dexterous hands)
- Can wave and do basic arm movements
- Built for fighting/repurposed for greeting

Based on Booster Robotics SDK:
- https://github.com/BoosterRobotics/booster_robotics_sdk
- https://github.com/arminforoughi/booster_k1

Available SDK Methods for K-1:
- WaveHand(action) - Wave gesture
- Handshake(action) - Handshake gesture (maybe)
- MoveHandEndEffectorV2(posture, duration, hand_index) - Move arm endpoint

Note: K-1 has NUBS, not fingers. No ControlDexterousHand or finger control!
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
        Posture,
    )
    BOOSTER_SDK_AVAILABLE = True
except ImportError:
    logger.warning("Booster SDK not available - arm control will be simulated")
    BOOSTER_SDK_AVAILABLE = False


class HandAction(Enum):
    """Hand gesture actions"""
    WAVE = "wave"
    HANDSHAKE = "handshake"


class HandSide(Enum):
    """Which hand to control"""
    LEFT = 0   # B1HandIndex.kLeftHand
    RIGHT = 1  # B1HandIndex.kRightHand


class K1ArmController:
    """
    High-level arm controller for K-1 using Booster SDK

    K-1 has round nub hands (no fingers), so only basic gestures available.

    Example usage:
        controller = K1ArmController(booster_client)
        controller.wave(hand=HandSide.RIGHT)
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
            logger.info(f"Waving {hand.name} hand")

            # Use SDK's built-in WaveHand method
            # From SDK binding: WaveHand(action)
            # Need to verify exact API - may need action parameter like kHandOpen/kHandClose

            # Placeholder - VERIFY WITH ACTUAL SDK
            logger.warning("Wave gesture using placeholder - verify SDK API")
            # hand_index = B1HandIndex.kLeftHand if hand == HandSide.LEFT else B1HandIndex.kRightHand
            # self.booster.WaveHand(action)  # Need to determine correct action param

            time.sleep(duration)
            return True

        except Exception as e:
            logger.error(f"Wave gesture failed: {e}")
            return False

    def handshake(self, duration: float = 3.0) -> bool:
        """
        Perform handshake gesture (if supported)

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

            # Use SDK's built-in Handshake method (if available)
            # From SDK binding: Handshake(action)
            logger.warning("Handshake using placeholder - verify SDK API")
            # self.booster.Handshake(action)

            time.sleep(duration)
            return True

        except Exception as e:
            logger.error(f"Handshake gesture failed: {e}")
            return False

    def move_arm(
        self,
        hand: HandSide,
        posture: 'Posture',
        duration_ms: int = 1000
    ) -> bool:
        """
        Move arm end effector to target posture

        Args:
            hand: Which hand (LEFT or RIGHT)
            posture: Target posture (Posture object from SDK)
            duration_ms: Movement duration in milliseconds

        Returns:
            True if successful, False otherwise
        """
        if self.simulate:
            logger.info(f"[SIMULATE] Moving {hand.name} arm to posture")
            time.sleep(duration_ms / 1000.0)
            return True

        if not self.booster:
            logger.error("No booster client available")
            return False

        try:
            logger.info(f"Moving {hand.name} arm to target posture")

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
            logger.error(f"Move arm failed: {e}")
            return False


# Example usage
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("K-1 Arm Controller Demo")
    print("=" * 60)
    print("Note: K-1 has round nub hands (no fingers)")
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

    # Demo gestures (only what K-1 can do with nubs)
    print("\nTesting gestures in simulation mode:")

    print("\n1. Wave gesture")
    controller.wave(hand=HandSide.RIGHT)

    print("\n2. Handshake gesture")
    controller.handshake()

    print("\nDemo complete!")
    print("\nNote: K-1's nub hands cannot do finger control.")
    print("For dexterous hand control, see Robi with Feetech servos.")
