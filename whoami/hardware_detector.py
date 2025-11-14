"""
Hardware Detection Module for WhoAmI System

Automatically detects hardware platform and loads appropriate configuration.
Supports Jetson modules, Raspberry Pi, Mac, and generic Linux desktops.

Usage:
    from whoami.hardware_detector import detect_hardware, get_serial_port

    # Auto-detect hardware
    profile_name = detect_hardware()
    print(f"Running on: {profile_name}")

    # Get serial port for this hardware
    serial_port = get_serial_port()
    print(f"Using serial port: {serial_port}")

Environment Variables:
    WHOAMI_HARDWARE_PROFILE - Override auto-detection with specific profile name
    WHOAMI_SERIAL_PORT - Override serial port detection

See docs/HARDWARE_CONFIG_GUIDE.md for details.
"""

import os
import json
import platform
import logging
from pathlib import Path
from typing import Dict, Optional, Any, List

logger = logging.getLogger(__name__)


class HardwareDetector:
    """
    Detects hardware platform and loads appropriate configuration profile

    The detector examines:
    - Device tree model (/proc/device-tree/model) for embedded systems
    - Platform and architecture for desktops
    - Carrier board detection (GPIO, device tree compatible strings)
    - Environment variable overrides
    """

    def __init__(self, config_dir: Path = None):
        """
        Initialize hardware detector

        Args:
            config_dir: Path to hardware config directory
                       Defaults to /config/hardware relative to this file
        """
        if config_dir is None:
            # Default to config/hardware directory
            config_dir = Path(__file__).parent.parent / "config" / "hardware"

        self.config_dir = Path(config_dir)
        self.profiles_file = self.config_dir / "hardware_profiles.json"
        self.profiles = self._load_profiles()
        self._detected_profile = None
        self._cache = {}

    def _load_profiles(self) -> Dict:
        """Load hardware profiles from JSON file"""
        if not self.profiles_file.exists():
            logger.warning(f"Hardware profiles file not found: {self.profiles_file}")
            logger.warning("Using minimal fallback configuration")
            return {
                "profiles": {
                    "desktop_linux": {
                        "display_name": "Generic Linux Desktop",
                        "peripherals": {
                            "serial": {"primary": "/dev/ttyUSB0", "available": ["/dev/ttyUSB0"]}
                        }
                    }
                },
                "fallback_profile": "desktop_linux"
            }

        try:
            with open(self.profiles_file) as f:
                profiles = json.load(f)
                logger.debug(f"Loaded {len(profiles.get('profiles', {}))} hardware profiles")
                return profiles
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing hardware profiles: {e}")
            return {"profiles": {}, "fallback_profile": "desktop_linux"}

    def detect_hardware(self) -> str:
        """
        Detect current hardware platform and return profile name

        Returns:
            str: Profile name (e.g., 'jetson_orin_nx_k1', 'desktop_linux')

        The detection process:
        1. Check WHOAMI_HARDWARE_PROFILE environment variable
        2. Detect Jetson platforms via device tree
        3. Detect Raspberry Pi via device tree
        4. Detect Mac via platform info
        5. Fall back to generic desktop Linux
        """
        # Return cached result if available
        if self._detected_profile:
            return self._detected_profile

        # Check environment variable override
        env_profile = os.getenv('WHOAMI_HARDWARE_PROFILE')
        if env_profile:
            if env_profile in self.profiles['profiles']:
                logger.info(f"Using hardware profile from environment: {env_profile}")
                self._detected_profile = env_profile
                return env_profile
            else:
                logger.warning(f"Environment profile '{env_profile}' not found, auto-detecting")

        # Detect Jetson platforms
        if self._is_jetson():
            profile = self._detect_jetson_variant()
            if profile:
                logger.info(f"Detected Jetson platform: {profile}")
                self._detected_profile = profile
                return profile

        # Detect Raspberry Pi
        if self._is_raspberry_pi():
            logger.info("Detected Raspberry Pi platform")
            self._detected_profile = "raspberry_pi_4"
            return "raspberry_pi_4"

        # Detect Mac
        if self._is_mac():
            logger.info("Detected Mac platform")
            self._detected_profile = "mac_m_series"
            return "mac_m_series"

        # Fallback to generic desktop
        fallback = self.profiles.get('fallback_profile', 'desktop_linux')
        logger.info(f"Using fallback profile: {fallback}")
        self._detected_profile = fallback
        return fallback

    def _is_jetson(self) -> bool:
        """Check if running on NVIDIA Jetson platform"""
        model_file = Path('/proc/device-tree/model')
        if model_file.exists():
            try:
                model = model_file.read_text().strip('\x00')
                return 'Jetson' in model or 'NVIDIA' in model
            except Exception as e:
                logger.debug(f"Error reading device tree model: {e}")
        return False

    def _detect_jetson_variant(self) -> Optional[str]:
        """
        Detect specific Jetson variant and carrier board

        Returns:
            Profile name if detected, None otherwise
        """
        try:
            model_file = Path('/proc/device-tree/model')
            if not model_file.exists():
                return None

            model = model_file.read_text().strip('\x00')
            logger.debug(f"Device tree model: {model}")

            # Read compatible strings
            compatible_file = Path('/proc/device-tree/compatible')
            compatible = []
            if compatible_file.exists():
                compatible = compatible_file.read_text().strip('\x00').split('\x00')
                logger.debug(f"Compatible strings: {compatible}")

            # Check profiles in priority order
            detection_order = self.profiles.get('detection_priority', [])
            profiles_to_check = [p for p in detection_order if p.startswith('jetson_')]

            # If no priority order, check all jetson profiles
            if not profiles_to_check:
                profiles_to_check = [p for p in self.profiles['profiles'].keys()
                                    if p.startswith('jetson_')]

            for profile_name in profiles_to_check:
                profile_data = self.profiles['profiles'][profile_name]
                detection = profile_data.get('detection', {})

                # Check device tree model
                dt_model = detection.get('device_tree_model', '')
                if dt_model and dt_model in model:
                    # Check carrier board detection if specified
                    carrier_detect = detection.get('carrier_detection')
                    if carrier_detect:
                        if self._verify_carrier_board(carrier_detect):
                            display_name = profile_data.get('display_name', profile_name)
                            logger.info(f"Detected hardware: {display_name}")
                            return profile_name
                    else:
                        # No carrier detection needed, match found
                        display_name = profile_data.get('display_name', profile_name)
                        logger.info(f"Detected hardware: {display_name}")
                        return profile_name

                # Check compatible strings
                compat_list = detection.get('compatible', [])
                if compat_list and any(c in compatible for c in compat_list):
                    display_name = profile_data.get('display_name', profile_name)
                    logger.info(f"Detected hardware: {display_name}")
                    return profile_name

        except Exception as e:
            logger.error(f"Error detecting Jetson variant: {e}", exc_info=True)

        return None

    def _verify_carrier_board(self, carrier_detect: Dict) -> bool:
        """
        Verify carrier board using detection method

        Args:
            carrier_detect: Carrier detection configuration

        Returns:
            True if carrier board detected, False otherwise
        """
        method = carrier_detect.get('method')

        if method == 'gpio_probe':
            # Check specific GPIO pin value
            gpio_pin = carrier_detect.get('gpio_pin')
            expected = carrier_detect.get('expected_value')

            if gpio_pin is None or expected is None:
                logger.warning("GPIO probe requires gpio_pin and expected_value")
                return False

            try:
                # Try to read GPIO value via sysfs
                gpio_path = Path(f'/sys/class/gpio/gpio{gpio_pin}/value')

                # If GPIO not exported, try to export it
                if not gpio_path.exists():
                    export_path = Path('/sys/class/gpio/export')
                    if export_path.exists():
                        with open(export_path, 'w') as f:
                            f.write(str(gpio_pin))
                        # Give it a moment to appear
                        import time
                        time.sleep(0.1)

                if gpio_path.exists():
                    value = int(gpio_path.read_text().strip())
                    matches = (value == expected)
                    logger.debug(f"GPIO {gpio_pin} = {value}, expected {expected}: {matches}")
                    return matches
                else:
                    logger.debug(f"GPIO {gpio_pin} not available for probing")
                    return False

            except Exception as e:
                logger.debug(f"GPIO probe failed: {e}")
                return False

        elif method == 'device_tree':
            # Check for specific device tree node or property
            dt_path = carrier_detect.get('device_tree_path')
            if dt_path:
                path = Path(f'/proc/device-tree/{dt_path}')
                exists = path.exists()
                logger.debug(f"Device tree path {dt_path}: {exists}")
                return exists

        logger.warning(f"Unknown carrier detection method: {method}")
        return False

    def _is_raspberry_pi(self) -> bool:
        """Check if running on Raspberry Pi"""
        model_file = Path('/proc/device-tree/model')
        if model_file.exists():
            try:
                model = model_file.read_text().strip('\x00')
                return 'Raspberry Pi' in model
            except Exception as e:
                logger.debug(f"Error reading device tree model: {e}")
        return False

    def _is_mac(self) -> bool:
        """Check if running on macOS"""
        return platform.system() == 'Darwin'

    def get_profile(self, profile_name: str = None) -> Dict[str, Any]:
        """
        Get hardware profile configuration

        Args:
            profile_name: Specific profile to load, or None to auto-detect

        Returns:
            Dict containing profile configuration
        """
        if profile_name is None:
            profile_name = self.detect_hardware()

        # Check cache
        if profile_name in self._cache:
            return self._cache[profile_name]

        # Get profile
        profile = self.profiles['profiles'].get(profile_name)
        if not profile:
            logger.warning(f"Profile '{profile_name}' not found, using fallback")
            profile_name = self.profiles.get('fallback_profile', 'desktop_linux')
            profile = self.profiles['profiles'].get(profile_name, {})

        # Cache and return
        self._cache[profile_name] = profile
        return profile

    def get_serial_port(self, profile_name: str = None) -> str:
        """
        Get primary serial port for hardware profile

        Checks WHOAMI_SERIAL_PORT environment variable first.

        Args:
            profile_name: Specific profile, or None to auto-detect

        Returns:
            Serial port path (e.g., '/dev/ttyTHS0')
        """
        # Check environment override
        env_port = os.getenv('WHOAMI_SERIAL_PORT')
        if env_port:
            logger.info(f"Using serial port from environment: {env_port}")
            return env_port

        # Get from profile
        profile = self.get_profile(profile_name)
        serial_port = profile.get('peripherals', {}).get('serial', {}).get('primary', '/dev/ttyUSB0')
        return serial_port

    def get_available_serial_ports(self, profile_name: str = None) -> List[str]:
        """
        Get list of available serial ports for hardware profile

        Args:
            profile_name: Specific profile, or None to auto-detect

        Returns:
            List of serial port paths
        """
        profile = self.get_profile(profile_name)
        ports = profile.get('peripherals', {}).get('serial', {}).get('available', [])
        return ports

    def get_gpio_chip(self, profile_name: str = None) -> Optional[str]:
        """
        Get GPIO chip identifier for hardware profile

        Args:
            profile_name: Specific profile, or None to auto-detect

        Returns:
            GPIO chip name (e.g., 'gpiochip0') or None if not available
        """
        profile = self.get_profile(profile_name)
        return profile.get('peripherals', {}).get('gpio', {}).get('chip')

    def get_i2c_buses(self, profile_name: str = None) -> List[int]:
        """
        Get list of I2C buses for hardware profile

        Args:
            profile_name: Specific profile, or None to auto-detect

        Returns:
            List of I2C bus numbers
        """
        profile = self.get_profile(profile_name)
        buses = profile.get('peripherals', {}).get('i2c', {}).get('buses', [])
        return buses

    def get_display_name(self, profile_name: str = None) -> str:
        """
        Get human-readable hardware name

        Args:
            profile_name: Specific profile, or None to auto-detect

        Returns:
            Display name (e.g., 'Jetson Orin NX on K-1 Booster')
        """
        profile = self.get_profile(profile_name)
        return profile.get('display_name', profile_name or 'Unknown Hardware')

    def get_audio_config(self, profile_name: str = None) -> Optional[Dict[str, Any]]:
        """
        Get audio configuration for hardware profile

        Args:
            profile_name: Specific profile, or None to auto-detect

        Returns:
            Audio configuration dict or None if not available
        """
        profile = self.get_profile(profile_name)
        return profile.get('peripherals', {}).get('audio')

    def get_gimbal_config(self, profile_name: str = None) -> Optional[Dict[str, Any]]:
        """
        Get gimbal system configuration for hardware profile

        Args:
            profile_name: Specific profile, or None to auto-detect

        Returns:
            Gimbal configuration dict or None if not available
        """
        profile = self.get_profile(profile_name)
        return profile.get('gimbal_system')

    def has_audio_support(self, profile_name: str = None) -> bool:
        """
        Check if hardware profile has audio support

        Args:
            profile_name: Specific profile, or None to auto-detect

        Returns:
            True if audio is supported
        """
        return self.get_audio_config(profile_name) is not None

    def has_dual_gimbal(self, profile_name: str = None) -> bool:
        """
        Check if hardware profile has dual gimbal system (head/neck)

        Args:
            profile_name: Specific profile, or None to auto-detect

        Returns:
            True if dual gimbal system is available
        """
        gimbal_config = self.get_gimbal_config(profile_name)
        if gimbal_config:
            return gimbal_config.get('type') == 'head_neck_dual'
        return False

    def get_remote_access_config(self, profile_name: str = None) -> Optional[Dict[str, Any]]:
        """
        Get remote access configuration for hardware profile

        Args:
            profile_name: Specific profile, or None to auto-detect

        Returns:
            Remote access configuration dict or None if not available
        """
        profile = self.get_profile(profile_name)
        return profile.get('peripherals', {}).get('display', {}).get('remote_access')

    def get_operational_modes(self, profile_name: str = None) -> Optional[Dict[str, Any]]:
        """
        Get supported operational modes for hardware profile

        Args:
            profile_name: Specific profile, or None to auto-detect

        Returns:
            Operational modes dict or None if not available
        """
        profile = self.get_profile(profile_name)
        return profile.get('operational_modes')


# Global singleton instance
_detector: Optional[HardwareDetector] = None


def get_hardware_detector() -> HardwareDetector:
    """
    Get global hardware detector instance (singleton pattern)

    Returns:
        HardwareDetector instance
    """
    global _detector
    if _detector is None:
        _detector = HardwareDetector()
    return _detector


def detect_hardware() -> str:
    """
    Convenience function to detect hardware platform

    Returns:
        Profile name (e.g., 'jetson_orin_nx_k1')
    """
    return get_hardware_detector().detect_hardware()


def get_serial_port(profile_name: str = None) -> str:
    """
    Convenience function to get primary serial port

    Args:
        profile_name: Specific profile, or None to auto-detect

    Returns:
        Serial port path
    """
    return get_hardware_detector().get_serial_port(profile_name)


def get_profile(profile_name: str = None) -> Dict[str, Any]:
    """
    Convenience function to get hardware profile

    Args:
        profile_name: Specific profile, or None to auto-detect

    Returns:
        Profile configuration dict
    """
    return get_hardware_detector().get_profile(profile_name)


# Example usage
if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Detect hardware
    print("=== Hardware Detection ===")
    detector = get_hardware_detector()
    profile_name = detector.detect_hardware()
    profile = detector.get_profile()

    print(f"\nDetected Hardware: {profile.get('display_name', profile_name)}")
    print(f"Profile Name: {profile_name}")
    print(f"Module: {profile.get('module', 'unknown')}")
    print(f"Carrier: {profile.get('carrier', 'unknown')}")

    print("\n=== Peripherals ===")
    print(f"Serial Port: {detector.get_serial_port()}")
    print(f"Available Serial Ports: {detector.get_available_serial_ports()}")
    print(f"GPIO Chip: {detector.get_gpio_chip()}")
    print(f"I2C Buses: {detector.get_i2c_buses()}")

    if 'performance' in profile:
        print("\n=== Performance ===")
        perf = profile['performance']
        for key, value in perf.items():
            print(f"{key}: {value}")

    # Audio configuration
    audio_config = detector.get_audio_config()
    if audio_config:
        print("\n=== Audio Configuration ===")
        print(f"Input Device: {audio_config.get('input_device')}")
        print(f"Output Device: {audio_config.get('output_device')}")
        print(f"Sample Rate: {audio_config.get('sample_rate')}Hz")
        print(f"Supported Features: {', '.join(audio_config.get('supported_features', []))}")

    # Gimbal configuration
    gimbal_config = detector.get_gimbal_config()
    if gimbal_config:
        print("\n=== Gimbal Configuration ===")
        print(f"Type: {gimbal_config.get('type')}")
        if detector.has_dual_gimbal():
            head = gimbal_config.get('head_gimbal', {})
            neck = gimbal_config.get('neck_gimbal', {})
            print(f"Head Gimbal Port: {head.get('serial_port')}")
            print(f"Head Gimbal Axes: {', '.join(head.get('axes', []))}")
            print(f"Neck Gimbal Port: {neck.get('serial_port')}")
            print(f"Neck Gimbal Axes: {', '.join(neck.get('axes', []))}")

    # Remote access
    remote_access = detector.get_remote_access_config()
    if remote_access:
        print("\n=== Remote Access ===")
        print(f"VNC Enabled: {remote_access.get('vnc_enabled')}")
        print(f"VNC Port: {remote_access.get('vnc_port')}")
        print(f"Direct HDMI: {remote_access.get('direct_hdmi')}")
        print(f"Resolution: {remote_access.get('resolution')}")

    # Operational modes
    op_modes = detector.get_operational_modes()
    if op_modes:
        print("\n=== Operational Modes ===")
        for mode_name, mode_config in op_modes.items():
            if mode_config.get('enabled'):
                print(f"- {mode_name}: {mode_config.get('description')}")

    if 'setup_notes' in profile:
        print(f"\n=== Setup Notes ===")
        print(profile['setup_notes'])
