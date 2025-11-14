# Hardware Configuration Strategy Guide

## Overview

This guide outlines the recommended approach for adding hardware configurations to the WhoAmI system, including support for different Jetson variants (Orin NX, AGX Orin, etc.) and carrier boards (K-1 booster, custom carriers, etc.).

## Current Configuration Architecture

### Existing Structure
```
/config/
├── brain_config.json          # Robot brain and learning system
├── gimbal_config.json         # 2-axis gimbal control
└── gimbal_3dof_config.json    # 3-axis gimbal control
```

### Platform Detection Pattern
Currently uses basic platform detection:
- **Jetson** (any variant) → `/dev/ttyTHS0`
- **Raspberry Pi** → `/dev/ttyAMA0`
- **Desktop/Laptop** → `/dev/ttyUSB0`

### Limitations
1. **No carrier board detection** - Cannot differentiate between carrier boards
2. **No module variant detection** - Orin Nano, Orin NX, AGX Orin all treated the same
3. **No peripheral mapping** - Serial ports, GPIOs, I2C buses not mapped per hardware
4. **Manual configuration required** - Users must manually edit configs for different hardware

## Recommended Architecture

### Option 1: Hierarchical Hardware Profiles (Recommended)

Create a new hardware configuration system with three levels:

```
Hardware Configuration Hierarchy:
└── Module Type (Orin Nano, Orin NX, AGX Orin, etc.)
    └── Carrier Board (K-1, DevKit, Custom, etc.)
        └── Peripheral Configuration (Serial, GPIO, I2C, etc.)
```

**Structure:**
```
/config/
├── brain_config.json
├── gimbal_config.json
├── gimbal_3dof_config.json
└── hardware/                           # New hardware profiles directory
    ├── hardware_profiles.json          # Master hardware profile definitions
    ├── jetson_orin_nano_devkit.json   # Specific hardware config
    ├── jetson_orin_nx_k1.json         # Orin NX on K-1 booster
    ├── jetson_agx_orin_devkit.json    # AGX Orin config
    └── README.md                       # Hardware config documentation
```

**Benefits:**
- ✅ **Modular**: Easy to add new hardware variants
- ✅ **Maintainable**: Each hardware profile is self-contained
- ✅ **Auto-detection**: Can detect and load appropriate profile
- ✅ **Override-friendly**: User can manually specify hardware profile
- ✅ **Extensible**: Easy to add new hardware characteristics

### Option 2: Extended Platform-Specific Sections

Extend the existing `platform_specific` sections in each config file:

```json
"platform_specific": {
  "jetson_orin_nano_devkit": {
    "serial_port": "/dev/ttyTHS0",
    "gpio_chip": "gpiochip0",
    "i2c_buses": [0, 1, 7, 8]
  },
  "jetson_orin_nx_k1": {
    "serial_port": "/dev/ttyTHS1",
    "gpio_chip": "gpiochip1",
    "i2c_buses": [0, 2]
  },
  "jetson_agx_orin_devkit": {
    "serial_port": "/dev/ttyTHS0",
    "gpio_chip": "gpiochip0",
    "i2c_buses": [0, 1, 2, 7, 8, 9]
  }
}
```

**Benefits:**
- ✅ **Simple**: Minimal changes to existing structure
- ✅ **Backward compatible**: Existing configs still work
- ❌ **Duplication**: Must update all config files
- ❌ **Less maintainable**: Hardware info scattered across files

### Option 3: Environment-Based Configuration

Use environment variables and detection scripts:

```bash
# Set via environment or auto-detection
export WHOAMI_HARDWARE_PROFILE="jetson_orin_nx_k1"
export WHOAMI_SERIAL_PORT="/dev/ttyTHS1"
export WHOAMI_GPIO_CHIP="gpiochip1"
```

**Benefits:**
- ✅ **Flexible**: Easy runtime configuration
- ✅ **Docker-friendly**: Works well in containers
- ❌ **Less discoverable**: Hardware profiles not in code
- ❌ **Error-prone**: Easy to misconfigure

## Recommended Implementation: Option 1 (Hardware Profiles)

### 1. Create Hardware Profile Schema

**File: `/config/hardware/hardware_profiles.json`**

This master file defines all supported hardware configurations:

```json
{
  "schema_version": "1.0",
  "description": "Hardware profile definitions for WhoAmI system",

  "profiles": {
    "jetson_orin_nano_devkit": {
      "display_name": "Jetson Orin Nano DevKit",
      "module": "orin_nano",
      "carrier": "devkit",
      "detection": {
        "device_tree_model": "NVIDIA Orin Nano Developer Kit",
        "compatible": ["nvidia,p3768-0000+p3767-0000"]
      },
      "peripherals": {
        "serial": {
          "primary": "/dev/ttyTHS0",
          "available": ["/dev/ttyTHS0", "/dev/ttyTHS1"]
        },
        "gpio": {
          "chip": "gpiochip0",
          "pins": 40
        },
        "i2c": {
          "buses": [0, 1, 7, 8]
        },
        "usb": {
          "ports": 4,
          "usb3_ports": 4
        }
      },
      "performance": {
        "cpu_cores": 6,
        "gpu_sm_count": 512,
        "max_power_mode": "MAXN",
        "memory_gb": 8
      }
    },

    "jetson_orin_nx_k1": {
      "display_name": "Jetson Orin NX on K-1 Booster",
      "module": "orin_nx",
      "carrier": "k1_booster",
      "detection": {
        "device_tree_model": "NVIDIA Orin NX",
        "compatible": ["nvidia,p3509-0000+p3767-0000"],
        "carrier_detection": {
          "method": "gpio_probe",
          "gpio_pin": 194,
          "expected_value": 1
        }
      },
      "peripherals": {
        "serial": {
          "primary": "/dev/ttyTHS1",
          "available": ["/dev/ttyTHS0", "/dev/ttyTHS1", "/dev/ttyTHS2"]
        },
        "gpio": {
          "chip": "gpiochip1",
          "pins": 40
        },
        "i2c": {
          "buses": [0, 2, 8]
        },
        "usb": {
          "ports": 6,
          "usb3_ports": 4
        },
        "notes": "K-1 booster provides additional USB and serial ports"
      },
      "performance": {
        "cpu_cores": 8,
        "gpu_sm_count": 1024,
        "max_power_mode": "MAXN",
        "memory_gb": 16
      }
    },

    "jetson_agx_orin_devkit": {
      "display_name": "Jetson AGX Orin Developer Kit",
      "module": "agx_orin",
      "carrier": "devkit",
      "detection": {
        "device_tree_model": "NVIDIA AGX Orin Developer Kit",
        "compatible": ["nvidia,p3737-0000+p3701-0000"]
      },
      "peripherals": {
        "serial": {
          "primary": "/dev/ttyTHS0",
          "available": ["/dev/ttyTHS0", "/dev/ttyTHS1", "/dev/ttyTHS2", "/dev/ttyTHS3"]
        },
        "gpio": {
          "chip": "gpiochip0",
          "pins": 40
        },
        "i2c": {
          "buses": [0, 1, 2, 7, 8, 9]
        },
        "usb": {
          "ports": 8,
          "usb3_ports": 6
        }
      },
      "performance": {
        "cpu_cores": 12,
        "gpu_sm_count": 2048,
        "max_power_mode": "MAXN",
        "memory_gb": 64
      }
    },

    "desktop_linux": {
      "display_name": "Desktop Linux (x86_64)",
      "module": "generic",
      "carrier": "desktop",
      "detection": {
        "platform": "linux",
        "architecture": "x86_64"
      },
      "peripherals": {
        "serial": {
          "primary": "/dev/ttyUSB0",
          "available": ["/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyACM0"]
        },
        "gpio": {
          "chip": null,
          "notes": "GPIO not typically available on desktop"
        },
        "i2c": {
          "buses": []
        },
        "usb": {
          "ports": "varies",
          "usb3_ports": "varies"
        }
      }
    },

    "raspberry_pi_4": {
      "display_name": "Raspberry Pi 4",
      "module": "bcm2711",
      "carrier": "rpi4",
      "detection": {
        "device_tree_model": "Raspberry Pi 4",
        "compatible": ["raspberrypi,4-model-b"]
      },
      "peripherals": {
        "serial": {
          "primary": "/dev/ttyAMA0",
          "available": ["/dev/ttyAMA0", "/dev/serial0"]
        },
        "gpio": {
          "chip": "gpiochip0",
          "pins": 40
        },
        "i2c": {
          "buses": [0, 1]
        },
        "usb": {
          "ports": 4,
          "usb3_ports": 2
        }
      }
    }
  },

  "fallback_profile": "desktop_linux"
}
```

### 2. Create Hardware Detection Module

**File: `/whoami/hardware_detector.py`**

```python
"""
Hardware Detection Module
Automatically detects hardware platform and loads appropriate configuration
"""

import os
import json
import platform
import logging
from pathlib import Path
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


class HardwareDetector:
    """Detects hardware platform and loads appropriate configuration profile"""

    def __init__(self, config_dir: Path = None):
        self.config_dir = config_dir or Path(__file__).parent.parent / "config" / "hardware"
        self.profiles_file = self.config_dir / "hardware_profiles.json"
        self.profiles = self._load_profiles()
        self._detected_profile = None

    def _load_profiles(self) -> Dict:
        """Load hardware profiles from JSON file"""
        if not self.profiles_file.exists():
            logger.warning(f"Hardware profiles file not found: {self.profiles_file}")
            return {"profiles": {}, "fallback_profile": "desktop_linux"}

        with open(self.profiles_file) as f:
            return json.load(f)

    def detect_hardware(self) -> str:
        """
        Detect current hardware platform and return profile name

        Returns:
            str: Profile name (e.g., 'jetson_orin_nx_k1')
        """
        if self._detected_profile:
            return self._detected_profile

        # Check environment variable override
        env_profile = os.getenv('WHOAMI_HARDWARE_PROFILE')
        if env_profile and env_profile in self.profiles['profiles']:
            logger.info(f"Using hardware profile from environment: {env_profile}")
            self._detected_profile = env_profile
            return env_profile

        # Detect Jetson platforms
        if self._is_jetson():
            profile = self._detect_jetson_variant()
            if profile:
                self._detected_profile = profile
                return profile

        # Detect Raspberry Pi
        if self._is_raspberry_pi():
            self._detected_profile = "raspberry_pi_4"
            return "raspberry_pi_4"

        # Fallback to generic desktop
        fallback = self.profiles.get('fallback_profile', 'desktop_linux')
        logger.info(f"Using fallback profile: {fallback}")
        self._detected_profile = fallback
        return fallback

    def _is_jetson(self) -> bool:
        """Check if running on Jetson platform"""
        model_file = Path('/proc/device-tree/model')
        if model_file.exists():
            model = model_file.read_text().strip('\x00')
            return 'Jetson' in model or 'NVIDIA' in model
        return False

    def _detect_jetson_variant(self) -> Optional[str]:
        """Detect specific Jetson variant and carrier board"""
        try:
            model_file = Path('/proc/device-tree/model')
            if not model_file.exists():
                return None

            model = model_file.read_text().strip('\x00')
            compatible_file = Path('/proc/device-tree/compatible')
            compatible = compatible_file.read_text().strip('\x00').split('\x00') if compatible_file.exists() else []

            # Check each profile for matching detection criteria
            for profile_name, profile_data in self.profiles['profiles'].items():
                if not profile_name.startswith('jetson_'):
                    continue

                detection = profile_data.get('detection', {})

                # Check device tree model
                dt_model = detection.get('device_tree_model', '')
                if dt_model and dt_model in model:
                    # Check carrier board detection if specified
                    carrier_detect = detection.get('carrier_detection')
                    if carrier_detect:
                        if self._verify_carrier_board(carrier_detect):
                            logger.info(f"Detected hardware: {profile_data['display_name']}")
                            return profile_name
                    else:
                        logger.info(f"Detected hardware: {profile_data['display_name']}")
                        return profile_name

                # Check compatible strings
                compat_list = detection.get('compatible', [])
                if compat_list and any(c in compatible for c in compat_list):
                    logger.info(f"Detected hardware: {profile_data['display_name']}")
                    return profile_name

        except Exception as e:
            logger.error(f"Error detecting Jetson variant: {e}")

        return None

    def _verify_carrier_board(self, carrier_detect: Dict) -> bool:
        """Verify carrier board using detection method"""
        method = carrier_detect.get('method')

        if method == 'gpio_probe':
            # Check specific GPIO pin value
            gpio_pin = carrier_detect.get('gpio_pin')
            expected = carrier_detect.get('expected_value')
            try:
                # Read GPIO value (simplified - real implementation would use GPIO library)
                gpio_path = Path(f'/sys/class/gpio/gpio{gpio_pin}/value')
                if gpio_path.exists():
                    value = int(gpio_path.read_text().strip())
                    return value == expected
            except Exception as e:
                logger.debug(f"GPIO probe failed: {e}")
                return False

        return False

    def _is_raspberry_pi(self) -> bool:
        """Check if running on Raspberry Pi"""
        model_file = Path('/proc/device-tree/model')
        if model_file.exists():
            model = model_file.read_text().strip('\x00')
            return 'Raspberry Pi' in model
        return False

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

        profile = self.profiles['profiles'].get(profile_name)
        if not profile:
            logger.warning(f"Profile '{profile_name}' not found, using fallback")
            profile_name = self.profiles['fallback_profile']
            profile = self.profiles['profiles'].get(profile_name)

        return profile

    def get_serial_port(self, profile_name: str = None) -> str:
        """Get primary serial port for hardware profile"""
        profile = self.get_profile(profile_name)
        return profile['peripherals']['serial']['primary']

    def get_available_serial_ports(self, profile_name: str = None) -> list:
        """Get list of available serial ports for hardware profile"""
        profile = self.get_profile(profile_name)
        return profile['peripherals']['serial'].get('available', [])

    def get_gpio_chip(self, profile_name: str = None) -> Optional[str]:
        """Get GPIO chip identifier for hardware profile"""
        profile = self.get_profile(profile_name)
        return profile['peripherals']['gpio'].get('chip')

    def get_i2c_buses(self, profile_name: str = None) -> list:
        """Get list of I2C buses for hardware profile"""
        profile = self.get_profile(profile_name)
        return profile['peripherals']['i2c'].get('buses', [])


# Global instance
_detector = None

def get_hardware_detector() -> HardwareDetector:
    """Get global hardware detector instance"""
    global _detector
    if _detector is None:
        _detector = HardwareDetector()
    return _detector


def detect_hardware() -> str:
    """Convenience function to detect hardware"""
    return get_hardware_detector().detect_hardware()


def get_serial_port() -> str:
    """Convenience function to get primary serial port"""
    return get_hardware_detector().get_serial_port()
```

### 3. Update Existing Configurations

Modify existing config files to use hardware detection:

**File: `/whoami/gimbal_controller.py` (example integration)**

```python
from whoami.hardware_detector import get_hardware_detector

class GimbalController:
    def __init__(self, config_path=None):
        # Load gimbal config
        self.config = self._load_config(config_path)

        # Get hardware profile
        self.hw_detector = get_hardware_detector()
        self.hw_profile = self.hw_detector.detect_hardware()

        # Override serial port with hardware-specific value
        serial_port = self.hw_detector.get_serial_port()
        self.config['gimbal']['communication']['serial_port'] = serial_port

        logger.info(f"Hardware: {self.hw_detector.get_profile()['display_name']}")
        logger.info(f"Using serial port: {serial_port}")
```

### 4. Add Hardware Profile Override

Allow users to manually specify hardware profile:

**Environment variable:**
```bash
export WHOAMI_HARDWARE_PROFILE="jetson_orin_nx_k1"
```

**Command-line argument:**
```bash
python -m whoami.gui --hardware jetson_orin_nx_k1
```

**Config file:**
```json
{
  "hardware_profile": "jetson_orin_nx_k1",
  "gimbal": {
    ...
  }
}
```

## Implementation Steps for Adding Orin NX + K-1 Booster

### Step 1: Create Hardware Profiles Directory
```bash
mkdir -p /home/user/whoami/config/hardware
```

### Step 2: Create `hardware_profiles.json`
Use the master profile file shown above, ensuring the `jetson_orin_nx_k1` profile is included.

### Step 3: Create Hardware Detector Module
Add `/whoami/hardware_detector.py` as shown above.

### Step 4: Update Jetson Setup Script
Modify `jetson_setup_v2.sh` to detect and configure based on hardware profile:

```bash
# Add to jetson_setup_v2.sh

detect_jetson_variant() {
    if [ -f /proc/device-tree/model ]; then
        model=$(tr -d '\0' < /proc/device-tree/model)
        echo "Detected: $model"

        # Detect Orin NX
        if [[ $model == *"Orin NX"* ]]; then
            # Check for K-1 booster (example: check for specific GPIO)
            if [ -f /sys/class/gpio/gpio194/value ]; then
                export WHOAMI_HARDWARE_PROFILE="jetson_orin_nx_k1"
                echo "Detected Orin NX with K-1 booster"
            else
                export WHOAMI_HARDWARE_PROFILE="jetson_orin_nx_devkit"
                echo "Detected Orin NX on DevKit"
            fi
        fi
    fi
}
```

### Step 5: Update Configuration Loader
Modify the config loader to use hardware detection:

```python
# In whoami/config.py

from whoami.hardware_detector import get_hardware_detector

class Config:
    def __init__(self, config_path=None):
        self.hw_detector = get_hardware_detector()
        self.hw_profile_name = self.hw_detector.detect_hardware()
        self.hw_profile = self.hw_detector.get_profile()

        # Load base config
        self.config = self._load_config(config_path)

        # Apply hardware-specific overrides
        self._apply_hardware_overrides()

    def _apply_hardware_overrides(self):
        """Apply hardware-specific configuration overrides"""
        # Override serial port
        if 'gimbal' in self.config:
            serial_port = self.hw_detector.get_serial_port()
            self.config['gimbal']['communication']['serial_port'] = serial_port
```

### Step 6: Document Hardware Profile
Create documentation for the new hardware configuration in the profile itself and in user-facing docs.

## Testing New Hardware Profiles

### Manual Testing
```bash
# Test hardware detection
python -c "from whoami.hardware_detector import detect_hardware; print(detect_hardware())"

# Test with override
WHOAMI_HARDWARE_PROFILE=jetson_orin_nx_k1 python -c "from whoami.hardware_detector import get_hardware_detector; print(get_hardware_detector().get_profile())"

# Test serial port detection
python -c "from whoami.hardware_detector import get_serial_port; print(get_serial_port())"
```

### Integration Testing
```bash
# Run gimbal controller with auto-detection
python -m whoami.gimbal_controller

# Run with specific profile
WHOAMI_HARDWARE_PROFILE=jetson_orin_nx_k1 python -m whoami.gimbal_controller
```

## Adding Future Hardware Configurations

To add new hardware configurations in the future:

1. **Add profile to `hardware_profiles.json`**
   - Define detection criteria
   - Specify peripherals
   - Document performance characteristics

2. **Test detection logic**
   - Verify device tree model matching
   - Test carrier board detection if needed
   - Confirm serial port and GPIO assignments

3. **Update documentation**
   - Add to supported hardware list
   - Document any special setup steps
   - Include performance benchmarks

4. **Create setup script variant** (if needed)
   - Add hardware-specific installation steps
   - Configure power modes
   - Set up peripherals

## Migration Path from Current System

### Phase 1: Add Hardware Detection (Non-Breaking)
- Create hardware profiles
- Add hardware detector module
- Keep existing platform_specific sections as fallback

### Phase 2: Update Configuration Loaders
- Modify config loaders to use hardware detection
- Apply hardware-specific overrides automatically
- Maintain backward compatibility

### Phase 3: Deprecate Old System
- Mark platform_specific sections as deprecated
- Update documentation
- Provide migration guide for users

### Phase 4: Remove Old System
- Remove platform_specific sections
- Fully migrate to hardware profiles
- Clean up redundant code

## Summary

**Recommended Approach:** Option 1 (Hardware Profiles)

**Key Benefits:**
- ✅ Clean separation of hardware configuration
- ✅ Auto-detection with manual override capability
- ✅ Easy to add new hardware variants
- ✅ Centralized hardware knowledge
- ✅ Testable and maintainable

**For Orin NX + K-1 Booster specifically:**
1. Create `jetson_orin_nx_k1` profile in `hardware_profiles.json`
2. Define carrier board detection method (GPIO probe or device tree)
3. Specify serial ports, GPIO chips, I2C buses
4. Test detection and configuration
5. Document setup process

This approach provides a scalable, maintainable way to support multiple hardware configurations while maintaining backward compatibility with the existing system.
