# WhoAmI - Jetson Orin Nano Installation Guide

Complete installation guide for setting up the WhoAmI Face Recognition System on NVIDIA Jetson Orin Nano with OAK-D Series 3 camera.

**Privacy Philosophy:** WhoAmI gives robots the right to remember people they meet through local, encrypted storage - NOT cloud-based face farming. See [README](README.md#security--privacy) for our privacy-first design principles.

## Table of Contents

- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Installation Methods](#installation-methods)
- [Post-Installation](#post-installation)
- [Verification](#verification)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Advanced Setup](#advanced-setup)

---

## Quick Start

For a complete automated installation:

```bash
# Clone the repository
git clone <repository-url>
cd whoami

# Run the setup script
./jetson_setup_v2.sh
```

That's it! The script will guide you through the entire process.

---

## Prerequisites

### Hardware Requirements

- **NVIDIA Jetson Orin Nano** (8GB recommended)
- **OAK-D Series 3 Camera** (Luxonis DepthAI)
- **USB 3.0 Cable** (for camera connection)
- **32GB+ Storage** (64GB+ recommended)
- **Power Supply** (Official Jetson power adapter)

#### Optional Hardware

- **Feetech Servo Gimbal** (2-axis pan/tilt for camera)
- **USB-to-Serial Adapter** (for gimbal control)

### Software Requirements

- **JetPack 5.1.2 or later**
- **Ubuntu 20.04/22.04 LTS** (comes with JetPack)
- **Python 3.8-3.10**
- **Internet connection** (for downloading packages)
- **Sudo privileges** (for system configuration)

### System Preparation

Before installation, ensure your Jetson is:

1. **Flashed with JetPack** (latest version recommended)
2. **Connected to internet** (ethernet or WiFi)
3. **Updated**:
   ```bash
   sudo apt update
   sudo apt upgrade -y
   ```
4. **Has sufficient disk space** (at least 5GB free)

---

## Installation Methods

### Method 1: Full Automated Installation (Recommended)

Complete installation including all features:

```bash
./jetson_setup_v2.sh --full
```

**Features included:**
- ✅ System dependencies
- ✅ Camera permissions and drivers
- ✅ Python virtual environment
- ✅ Face recognition libraries
- ✅ Gun.js storage (hardware-backed encryption) - See [Gun.js Storage Guide](docs/GUN_JS_STORAGE.md)
- ✅ 3D scanning support (Open3D, trimesh)
- ✅ Gimbal control (pyserial)
- ✅ Robot brain and learning systems
- ✅ GUI support (Tkinter)
- ✅ Desktop shortcuts
- ✅ Performance optimization

**Duration:** 30-60 minutes (depending on internet speed)

---

### Method 2: Minimal Installation

Install only core facial recognition features (no 3D scanning, no gimbal):

```bash
./jetson_setup_v2.sh --minimal
```

**Features included:**
- ✅ System dependencies
- ✅ Camera permissions and drivers
- ✅ Python virtual environment
- ✅ Face recognition libraries
- ✅ GUI support
- ❌ 3D scanning (Open3D, trimesh)
- ❌ Gimbal control
- ✅ Desktop shortcuts

**Duration:** 15-30 minutes

**Use when:**
- You don't need 3D scanning
- You don't have a gimbal
- You want faster installation
- You have limited disk space

---

### Method 3: Modular Installation

Install specific components only:

#### Camera Setup Only
```bash
./jetson_setup_v2.sh --camera-only
```
Sets up OAK-D camera permissions and udev rules.

#### Python Environment Only
```bash
./jetson_setup_v2.sh --python-only
```
Creates virtual environment and installs Python packages.

---

## Installation Process Details

### What the Script Does

1. **System Detection**
   - Verifies Jetson hardware
   - Checks Python version
   - Validates disk space

2. **Performance Configuration**
   - Sets Jetson to MAXN mode (maximum performance)
   - Enables jetson_clocks
   - Makes settings persistent

3. **System Dependencies**
   - Build tools (gcc, cmake, git)
   - Python development packages
   - OpenCV libraries
   - USB/camera libraries
   - Math libraries (BLAS, LAPACK, ATLAS)
   - GUI libraries (Gtk, Qt)

4. **Hardware Permissions**
   - Creates udev rules for OAK-D (vendor ID: 03e7)
   - Adds user to groups: `video`, `dialout`, `plugdev`, `i2c`
   - Configures serial ports for gimbal

5. **Python Environment**
   - Creates virtual environment at `~/whoami_env`
   - Upgrades pip, setuptools, wheel
   - Creates activation helper script

6. **Python Packages**
   - numpy (ARM64 optimized)
   - OpenCV (with contrib modules)
   - DepthAI (OAK-D SDK)
   - dlib (compiled from source)
   - face-recognition
   - scikit-learn
   - cryptography
   - pyserial (gimbal)
   - open3d, trimesh (3D scanning)

7. **Application Setup**
   - Installs WhoAmI package
   - Creates configuration files
   - Sets up data directories
   - Configures logging

8. **Desktop Integration**
   - Creates GUI launcher
   - Creates CLI launcher
   - Creates terminal launcher

9. **Verification**
   - Tests all imports
   - Detects cameras
   - Verifies permissions
   - Runs comprehensive checks

---

## Post-Installation

### Required Steps

After installation completes, you **MUST**:

1. **Logout and Login**
   ```bash
   # Logout to apply group changes
   logout
   # Or reboot
   sudo reboot
   ```

   This applies the group permission changes (video, dialout, etc.).

   **Alternative:** Without logout:
   ```bash
   newgrp video
   newgrp dialout
   ```

2. **Activate Virtual Environment**
   ```bash
   # Method 1: Direct activation
   source ~/whoami_env/bin/activate

   # Method 2: Helper script
   source ~/activate_whoami.sh
   ```

3. **Navigate to Project**
   ```bash
   cd ~/whoami  # Or wherever you cloned the repo
   ```

### Optional Steps

**Test Camera Connection:**
```bash
python -c "import depthai as dai; print(f'Devices: {len(dai.Device.getAllAvailableDevices())}')"
```

**Check Serial Ports** (for gimbal):
```bash
ls -la /dev/ttyUSB* /dev/ttyTHS*
```

**Monitor System Performance:**
```bash
jtop  # If jetson-stats is installed
```

---

## Verification

### Quick Verification

```bash
# Activate environment
source ~/whoami_env/bin/activate

# Run verification script
python verify_install_v2.py
```

### Manual Verification

**Check Python packages:**
```bash
pip list | grep -E "depthai|opencv|numpy|face-recognition|open3d|trimesh"
```

**Test camera:**
```bash
python -c "
import depthai as dai
devices = dai.Device.getAllAvailableDevices()
print(f'Found {len(devices)} camera(s)')
for d in devices:
    print(f'  - {d.getMxId()[:16]}... [{d.state.name}]')
"
```

**Test face recognition:**
```bash
python -c "
import face_recognition
import numpy as np
img = np.zeros((100,100,3), dtype=np.uint8)
encodings = face_recognition.face_encodings(img)
print('Face recognition OK')
"
```

**Check groups:**
```bash
groups | grep -E "video|dialout|plugdev"
```

---

## Configuration

### Main Configuration

Edit `config.json`:

```json
{
  "camera": {
    "preview_width": 640,
    "preview_height": 480,
    "fps": 30
  },
  "recognition": {
    "tolerance": 0.6,
    "model": "hog"
  },
  "database": {
    "path": "face_database.db"
  }
}
```

### Gimbal Configuration

Edit `config/gimbal_config.json`:

```json
{
  "gimbal": {
    "communication": {
      "serial_port": "/dev/ttyUSB0",  // Change for your setup
      "baudrate": 1000000
    }
  }
}
```

**Jetson-specific serial port:** Use `/dev/ttyTHS0` for Jetson's built-in UART

### Robot Brain Configuration

Edit `config/brain_config.json` to customize:
- Learning parameters
- Personality traits
- Memory settings
- Emotion states

---

## Running WhoAmI

### GUI Mode

```bash
# Activate environment
source ~/whoami_env/bin/activate

# Launch GUI
python run_gui.py

# Or use desktop shortcut (double-click "WhoAmI-GUI" on desktop)
```

### CLI Mode

```bash
# Activate environment
source ~/whoami_env/bin/activate

# Launch CLI
python run_cli.py --help

# Example commands
python run_cli.py list                    # List known faces
python run_cli.py add John                # Add new face
python run_cli.py recognize               # Recognize faces
```

### API Mode

```python
from whoami.face_recognition_api import FaceRecognitionAPI, CameraType

# Initialize
api = FaceRecognitionAPI(camera_type=CameraType.OAK_D)

# Start camera
api.start_camera()

# Recognize faces
faces = api.recognize_faces()
for face in faces:
    print(f"Found: {face.name} ({face.confidence:.2f})")

# Stop camera
api.stop_camera()
```

---

## Troubleshooting

### Camera Not Detected

**Symptoms:**
```
⚠ No OAK-D devices detected
```

**Solutions:**

1. **Check USB connection:**
   ```bash
   lsusb | grep 03e7
   ```
   Should show: `Movidius MyriadX`

2. **Try different USB port:**
   - Use USB 3.0 port (blue port)
   - Avoid USB hubs
   - Try different cable

3. **Check permissions:**
   ```bash
   groups | grep video
   ```
   If not in video group:
   ```bash
   sudo usermod -a -G video $USER
   logout  # Then login again
   ```

4. **Reload udev rules:**
   ```bash
   sudo udevadm control --reload-rules
   sudo udevadm trigger
   ```

5. **Check dmesg:**
   ```bash
   dmesg | grep -i usb | tail -20
   ```

---

### Import Errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'depthai'
```

**Solutions:**

1. **Ensure virtual environment is activated:**
   ```bash
   source ~/whoami_env/bin/activate
   ```
   Prompt should show: `(whoami_env)`

2. **Reinstall package:**
   ```bash
   pip install depthai
   ```

3. **Check Python version:**
   ```bash
   python --version  # Should be 3.8+
   ```

4. **Reinstall all requirements:**
   ```bash
   pip install -r requirements.txt
   ```

---

### Gimbal Not Responding

**Symptoms:**
- Gimbal doesn't move
- Serial port errors

**Solutions:**

1. **Check serial port exists:**
   ```bash
   ls /dev/ttyUSB* /dev/ttyTHS*
   ```

2. **Check permissions:**
   ```bash
   groups | grep dialout
   ```
   If not in dialout group:
   ```bash
   sudo usermod -a -G dialout $USER
   logout  # Then login again
   ```

3. **Test serial port:**
   ```bash
   python -c "
   import serial
   ser = serial.Serial('/dev/ttyUSB0', 1000000)
   print('Serial port OK')
   ser.close()
   "
   ```

4. **Update gimbal config:**
   Edit `config/gimbal_config.json` with correct port

5. **Check servo power:**
   - Ensure servos are powered
   - Check voltage (usually 6-8V)

---

### Performance Issues

**Symptoms:**
- Slow frame rate
- High latency
- Stuttering

**Solutions:**

1. **Enable performance mode:**
   ```bash
   sudo nvpmodel -m 0      # MAXN mode
   sudo jetson_clocks      # Max frequencies
   ```

2. **Check current mode:**
   ```bash
   sudo nvpmodel -q
   jetson_clocks --show
   ```

3. **Monitor resources:**
   ```bash
   jtop
   ```

4. **Reduce camera resolution:**
   Edit `config.json`:
   ```json
   {
     "camera": {
       "preview_width": 320,
       "preview_height": 240
     }
   }
   ```

5. **Close other applications:**
   ```bash
   htop  # Check running processes
   ```

---

### Memory Issues

**Symptoms:**
```
MemoryError: Unable to allocate array
```

**Solutions:**

1. **Check available memory:**
   ```bash
   free -h
   ```

2. **Enable swap:**
   ```bash
   sudo systemctl enable nvzramconfig
   ```

3. **Reduce batch processing:**
   - Process fewer faces at once
   - Lower camera resolution

4. **Close other applications:**
   ```bash
   sudo systemctl stop gdm3  # Stop GUI (if using headless)
   ```

---

### GUI Not Starting

**Symptoms:**
```
TclError: no display name and no $DISPLAY environment variable
```

**Solutions:**

1. **If using SSH:**
   ```bash
   # Enable X11 forwarding
   ssh -X user@jetson
   ```

2. **If headless, use CLI:**
   ```bash
   python run_cli.py
   ```

3. **Check DISPLAY:**
   ```bash
   echo $DISPLAY  # Should show :0 or :1
   ```

4. **Install Tkinter:**
   ```bash
   sudo apt install python3-tk
   ```

---

## Advanced Setup

### Custom Python Version

Use specific Python version:

```bash
# Install Python 3.9
sudo apt install python3.9 python3.9-venv python3.9-dev

# Create venv with Python 3.9
python3.9 -m venv ~/whoami_env

# Continue with installation
source ~/whoami_env/bin/activate
pip install -r requirements.txt
```

### Development Installation

For development:

```bash
# Activate environment
source ~/whoami_env/bin/activate

# Install in editable mode
pip install -e .

# Install dev dependencies
pip install pytest black flake8 mypy
```

### CUDA Acceleration

Enable CUDA for faster processing (if applicable):

```bash
# Check CUDA availability
python -c "
import cv2
print(f'CUDA: {cv2.cuda.getCudaEnabledDeviceCount()}')
"

# Install PyTorch with CUDA support
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
```

### Remote Access

Setup remote access:

```bash
# Enable SSH
sudo systemctl enable ssh
sudo systemctl start ssh

# Install VNC server
sudo apt install vino

# Configure VNC
gsettings set org.gnome.Vino require-encryption false
```

### Autostart on Boot

Make WhoAmI start automatically:

```bash
# Create systemd service
sudo nano /etc/systemd/system/whoami.service
```

Add:
```ini
[Unit]
Description=WhoAmI Face Recognition
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/home/youruser/whoami
ExecStart=/home/youruser/whoami_env/bin/python run_gui.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Enable:
```bash
sudo systemctl enable whoami
sudo systemctl start whoami
```

---

## Logging and Debugging

### Enable Debug Logging

Edit `config.json`:
```json
{
  "logging": {
    "level": "DEBUG",
    "file": "logs/whoami_debug.log"
  }
}
```

### View Logs

```bash
# Application logs
tail -f logs/whoami.log

# System logs
journalctl -u whoami -f

# Setup logs
cat ~/whoami_setup_*.log
```

---

## Updating

### Update WhoAmI

```bash
# Activate environment
source ~/whoami_env/bin/activate

# Pull latest code
git pull

# Update dependencies
pip install -r requirements.txt --upgrade

# Reinstall package
pip install -e .
```

### Update System

```bash
# Update system packages
sudo apt update
sudo apt upgrade -y

# Update Python packages
source ~/whoami_env/bin/activate
pip list --outdated
pip install --upgrade <package-name>
```

---

## Uninstallation

### Remove WhoAmI

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
rm -rf ~/whoami_env

# Remove project directory
rm -rf ~/whoami

# Remove desktop shortcuts
rm ~/Desktop/WhoAmI-*.desktop

# Remove activation script
rm ~/activate_whoami.sh

# Remove user from groups (optional)
sudo gpasswd -d $USER video
sudo gpasswd -d $USER dialout
sudo gpasswd -d $USER plugdev
```

### Remove udev Rules

```bash
sudo rm /etc/udev/rules.d/80-movidius.rules
sudo rm /etc/udev/rules.d/81-realsense.rules
sudo rm /etc/udev/rules.d/82-serial.rules
sudo udevadm control --reload-rules
```

---

## Support

### Resources

- **README:** `README.md`
- **API Reference:** `docs/API_REFERENCE.md`
- **Usage Guide:** `docs/USAGE_GUIDE.md`
- **Jetson Setup:** `SETUP_JETSON_M4.md`

### Getting Help

1. Check this installation guide
2. Review troubleshooting section
3. Check setup logs: `~/whoami_setup_*.log`
4. Run verification: `python verify_install_v2.py`
5. Check GitHub issues
6. Create new issue with:
   - Setup log file
   - Verification output
   - System info (`uname -a`, `jetson_release`, etc.)

---

## Appendix

### File Locations

```
~/whoami/                      # Project root
  ├── whoami/                  # Python package
  ├── config/                  # Configuration files
  ├── data/                    # Data directory
  │   ├── faces/              # Face database
  │   ├── scans/              # 3D scans
  │   └── logs/               # Application logs
  ├── logs/                    # System logs
  ├── run_gui.py              # GUI launcher
  ├── run_cli.py              # CLI launcher
  └── config.json             # Main config

~/whoami_env/                  # Virtual environment
~/activate_whoami.sh           # Activation helper
~/Desktop/WhoAmI-*.desktop    # Desktop shortcuts
~/.bashrc                      # Performance settings added here
```

### Network Ports

If using network features:

- Robot peer-to-peer: TCP 9999
- Web API (if enabled): HTTP 8080

Configure firewall:
```bash
sudo ufw allow 9999/tcp
sudo ufw allow 8080/tcp
```

### Performance Benchmarks

Expected performance on Jetson Orin Nano:

| Task | FPS | Latency |
|------|-----|---------|
| Face Detection | 20-30 | <50ms |
| Face Recognition | 15-25 | <100ms |
| 3D Scanning | 10-15 | <200ms |
| Gimbal Tracking | 30+ | <30ms |

### Package Versions

Tested and verified package versions:

```
numpy==1.23.5
opencv-python==4.8.1.78
depthai==2.24.0
face-recognition==1.3.0
dlib==19.24.2
scikit-learn>=1.3.0
cryptography>=41.0.0
pyserial>=3.5
open3d>=0.17.0
trimesh>=4.0.0
```

---

## License

This installation guide is part of the WhoAmI project.
See LICENSE file for details.

---

**Last Updated:** 2025-11-06
**Version:** 2.0
**For:** Jetson Orin Nano with JetPack 5.1.2+
