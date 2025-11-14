# Hardware Configuration Profiles

This directory contains hardware configuration profiles for the WhoAmI system. The hardware detector automatically identifies your platform and loads the appropriate configuration.

## Quick Start

### Auto-Detection (Recommended)

The system automatically detects your hardware:

```python
from whoami.hardware_detector import detect_hardware, get_serial_port

# Detect hardware platform
platform = detect_hardware()
print(f"Running on: {platform}")

# Get serial port for this platform
port = get_serial_port()
print(f"Serial port: {port}")
```

### Manual Override

Override auto-detection if needed:

```bash
# Set hardware profile via environment
export WHOAMI_HARDWARE_PROFILE="jetson_orin_nx_k1"

# Override serial port
export WHOAMI_SERIAL_PORT="/dev/ttyTHS1"

# Run your application
python -m whoami.gui
```

## Supported Hardware

### NVIDIA Jetson Platforms

| Profile | Module | Carrier | Serial Port | Notes |
|---------|--------|---------|-------------|-------|
| `jetson_orin_nano_devkit` | Orin Nano | DevKit | /dev/ttyTHS0 | 8GB RAM, 6 CPU cores |
| `jetson_orin_nx_devkit` | Orin NX | DevKit | /dev/ttyTHS0 | 16GB RAM, 8 CPU cores |
| `jetson_orin_nx_k1` | Orin NX | K-1 Booster | /dev/ttyTHS1 | Expanded I/O, 2x Ethernet |
| `jetson_agx_orin_devkit` | AGX Orin | DevKit | /dev/ttyTHS0 | 64GB RAM, 12 CPU cores |

### Other Platforms

| Profile | Description | Serial Port |
|---------|-------------|-------------|
| `raspberry_pi_4` | Raspberry Pi 4 Model B | /dev/ttyAMA0 |
| `mac_m_series` | Apple Silicon Mac | /dev/cu.usbserial-0 |
| `desktop_linux` | Generic x86_64 Linux | /dev/ttyUSB0 |

## Configuration File Structure

### `hardware_profiles.json`

Master configuration file containing all hardware profile definitions.

**Structure:**
```json
{
  "schema_version": "1.0",
  "profiles": {
    "profile_name": {
      "display_name": "Human-readable name",
      "module": "Module type",
      "carrier": "Carrier board type",
      "detection": {
        "device_tree_model": "Model string to match",
        "compatible": ["device-tree-compatible-strings"]
      },
      "peripherals": {
        "serial": {
          "primary": "/dev/ttyXXX",
          "available": ["/dev/ttyXXX", ...]
        },
        "gpio": {
          "chip": "gpiochipN",
          "pins": 40
        },
        "i2c": {
          "buses": [0, 1, 7, 8]
        }
      },
      "performance": {
        "cpu_cores": 8,
        "memory_gb": 16
      }
    }
  },
  "fallback_profile": "desktop_linux"
}
```

## Detection Logic

The hardware detector follows this sequence:

1. **Environment Variable Check**
   - `WHOAMI_HARDWARE_PROFILE` → Use specified profile

2. **Jetson Detection**
   - Read `/proc/device-tree/model`
   - Match against profile detection criteria
   - Check carrier board (if specified)

3. **Raspberry Pi Detection**
   - Check device tree for "Raspberry Pi"

4. **Mac Detection**
   - Check platform.system() == 'Darwin'

5. **Fallback**
   - Use `desktop_linux` profile

## Adding New Hardware Profiles

### Step 1: Identify Hardware Characteristics

Collect information about your hardware:

```bash
# Device tree model
cat /proc/device-tree/model

# Compatible strings
cat /proc/device-tree/compatible | tr '\0' '\n'

# Serial ports
ls -l /dev/tty*

# GPIO chips
ls /dev/gpiochip*

# I2C buses
ls /dev/i2c-*
```

### Step 2: Add Profile to `hardware_profiles.json`

```json
{
  "profiles": {
    "my_custom_hardware": {
      "display_name": "My Custom Hardware",
      "module": "module_name",
      "carrier": "carrier_name",
      "detection": {
        "device_tree_model": "Model String",
        "compatible": ["compatible-string"]
      },
      "peripherals": {
        "serial": {
          "primary": "/dev/ttyXXX",
          "available": ["/dev/ttyXXX"]
        },
        "gpio": {
          "chip": "gpiochipN",
          "pins": 40
        },
        "i2c": {
          "buses": [0, 1]
        }
      }
    }
  }
}
```

### Step 3: Test Detection

```python
from whoami.hardware_detector import get_hardware_detector

detector = get_hardware_detector()
profile = detector.detect_hardware()
print(f"Detected: {detector.get_display_name()}")
print(f"Serial: {detector.get_serial_port()}")
```

### Step 4: Test with Override

```bash
WHOAMI_HARDWARE_PROFILE=my_custom_hardware python -c \
  "from whoami.hardware_detector import detect_hardware; print(detect_hardware())"
```

## Carrier Board Detection

For hardware with multiple carrier board options (like Orin NX with K-1 booster), use carrier detection:

### GPIO Probe Method

Checks a specific GPIO pin value:

```json
{
  "detection": {
    "device_tree_model": "NVIDIA Orin NX",
    "carrier_detection": {
      "method": "gpio_probe",
      "gpio_pin": 194,
      "expected_value": 1,
      "description": "K-1 carrier board ID pin"
    }
  }
}
```

### Device Tree Method

Checks for a specific device tree node:

```json
{
  "detection": {
    "device_tree_model": "NVIDIA Orin NX",
    "carrier_detection": {
      "method": "device_tree",
      "device_tree_path": "carrier-board/k1-booster"
    }
  }
}
```

## Jetson Orin NX + K-1 Booster Example

The K-1 booster profile demonstrates advanced carrier board detection:

```json
{
  "jetson_orin_nx_k1": {
    "display_name": "Jetson Orin NX on K-1 Booster",
    "module": "orin_nx",
    "carrier": "k1_booster",
    "detection": {
      "device_tree_model": "NVIDIA Orin NX",
      "carrier_detection": {
        "method": "gpio_probe",
        "gpio_pin": 194,
        "expected_value": 1
      }
    },
    "peripherals": {
      "serial": {
        "primary": "/dev/ttyTHS1",
        "notes": "K-1 provides additional UART on ttyTHS1"
      },
      "gpio": {
        "chip": "gpiochip1",
        "notes": "K-1 uses different GPIO chip"
      },
      "additional": {
        "m2_slots": 2,
        "ethernet_ports": 2
      }
    }
  }
}
```

**Key Differences from DevKit:**
- **Serial Port:** `/dev/ttyTHS1` instead of `/dev/ttyTHS0`
- **GPIO Chip:** `gpiochip1` instead of `gpiochip0`
- **I2C Buses:** Different bus numbering
- **Additional I/O:** Extra USB, Ethernet, M.2 slots

## Integration with Existing Code

### Gimbal Controller Example

```python
from whoami.hardware_detector import get_hardware_detector

class GimbalController:
    def __init__(self, config_path=None):
        # Load configuration
        self.config = self._load_config(config_path)

        # Detect hardware
        self.hw_detector = get_hardware_detector()

        # Override serial port with hardware-specific value
        serial_port = self.hw_detector.get_serial_port()
        self.config['communication']['serial_port'] = serial_port

        print(f"Hardware: {self.hw_detector.get_display_name()}")
        print(f"Serial: {serial_port}")
```

### Brain System Example

```python
from whoami.hardware_detector import detect_hardware, get_profile

# Check hardware capabilities
hardware = detect_hardware()
profile = get_profile()

# Adjust settings based on performance
if profile.get('performance', {}).get('memory_gb', 0) < 8:
    print("Low memory system, using conservative settings")
    batch_size = 1
else:
    print("High memory system, using optimal settings")
    batch_size = 4
```

## Troubleshooting

### Hardware Not Detected

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

from whoami.hardware_detector import detect_hardware
platform = detect_hardware()
```

### Wrong Profile Selected

```bash
# Override with correct profile
export WHOAMI_HARDWARE_PROFILE="jetson_orin_nx_k1"
```

### Serial Port Not Working

```bash
# Override serial port
export WHOAMI_SERIAL_PORT="/dev/ttyTHS1"

# Check port permissions
ls -l /dev/ttyTHS1

# Add user to dialout group if needed
sudo usermod -a -G dialout $USER
```

### Test Hardware Detection

```bash
# Run hardware detector standalone
python -m whoami.hardware_detector
```

This will print:
- Detected hardware name
- Profile details
- Serial ports
- GPIO/I2C configuration
- Performance characteristics

## See Also

- **[Hardware Configuration Guide](../../docs/HARDWARE_CONFIG_GUIDE.md)** - Comprehensive guide
- **[Installation Guide](../../INSTALLATION.md)** - Setup instructions
- **[Jetson Setup Guide](../../SETUP_JETSON_M4.md)** - Jetson-specific setup

## Contributing

When adding support for new hardware:

1. Test on actual hardware
2. Document detection criteria
3. Verify serial port and peripheral assignments
4. Update this README with the new profile
5. Submit a pull request

For questions or issues, please open a GitHub issue.
