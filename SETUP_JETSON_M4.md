# Setup Guide for Jetson Orin Nano & M4 Mac with OAK-D S3 Camera

This comprehensive guide covers the setup process for the WhoAmI face recognition system on both NVIDIA Jetson Orin Nano (Linux ARM64) and Apple M4 Mac (macOS ARM64) platforms with the OAK-D S3 camera.

## Table of Contents
- [Platform Requirements](#platform-requirements)
- [Jetson Orin Nano Setup](#jetson-orin-nano-setup)
- [M4 Mac Setup](#m4-mac-setup)
- [OAK-D S3 Camera Configuration](#oak-d-s3-camera-configuration)
- [Troubleshooting](#troubleshooting)
- [Performance Optimization](#performance-optimization)

## Platform Requirements

### Jetson Orin Nano
- JetPack 5.1.2 or later
- Ubuntu 20.04/22.04 LTS
- 8GB RAM minimum
- 32GB storage minimum
- Python 3.8-3.10

### M4 Mac
- macOS 14.0 (Sonoma) or later
- Python 3.8-3.10 (via Homebrew recommended)
- Xcode Command Line Tools
- 8GB RAM minimum

## Jetson Orin Nano Setup

### 1. System Preparation

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install essential build tools
sudo apt install -y build-essential cmake git pkg-config
sudo apt install -y python3-pip python3-dev python3-venv
sudo apt install -y libopencv-dev python3-opencv
sudo apt install -y libusb-1.0-0-dev libudev-dev
sudo apt install -y v4l-utils

# Install performance tools
sudo apt install -y htop jtop nano curl wget
```

### 2. Configure Performance Mode

```bash
# Set Jetson to maximum performance mode
sudo nvpmodel -m 0
sudo jetson_clocks

# Make settings persistent
echo "sudo nvpmodel -m 0" >> ~/.bashrc
echo "sudo jetson_clocks" >> ~/.bashrc
```

### 3. Setup USB/Camera Permissions

Create udev rules for OAK-D S3:

```bash
# Create udev rules file
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules

# Add user to video and dialout groups
sudo usermod -a -G video,dialout $USER

# Reload udev rules
sudo udevadm control --reload-rules && sudo udevadm trigger

# Reboot to apply changes
sudo reboot
```

### 4. Python Environment Setup

```bash
# Create virtual environment
python3 -m venv ~/whoami_env
source ~/whoami_env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install numpy first (important for ARM64)
pip install numpy==1.23.5

# Install OpenCV with contrib modules
pip install opencv-python==4.8.1.78
pip install opencv-contrib-python==4.8.1.78
```

### 5. Install DepthAI and Dependencies

```bash
# Install DepthAI
pip install depthai==2.24.0
pip install depthai-sdk==1.14.0

# Install face recognition dependencies
pip install face-recognition==1.3.0
pip install dlib==19.24.2

# Install other dependencies
pip install Pillow==10.1.0
pip install pyqt6==6.5.3
pip install colorama==0.4.6
pip install tqdm==4.66.1
```

### 6. Clone and Install WhoAmI

```bash
# Clone repository
git clone https://github.com/yourusername/whoami-1.git
cd whoami-1

# Install in development mode
pip install -e .

# Verify installation
python verify_install.py
```

### 7. Run Automated Setup Script

```bash
# Make setup script executable
chmod +x jetson_setup.sh

# Run setup script
./jetson_setup.sh
```

## M4 Mac Setup

### 1. System Preparation

```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Update Homebrew
brew update && brew upgrade
```

### 2. Install System Dependencies

```bash
# Install Python 3.10
brew install python@3.10

# Install development tools
brew install cmake git pkg-config

# Install OpenCV
brew install opencv

# Install libusb for camera support
brew install libusb

# Install other useful tools
brew install htop tree wget
```

### 3. Python Environment Setup

```bash
# Create virtual environment
python3.10 -m venv ~/whoami_env
source ~/whoami_env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install numpy (optimized for M4)
pip install numpy==1.24.3
```

### 4. Install DepthAI and Dependencies

```bash
# Install DepthAI
pip install depthai==2.24.0
pip install depthai-sdk==1.14.0

# Install OpenCV
pip install opencv-python==4.8.1.78
pip install opencv-contrib-python==4.8.1.78

# Install face recognition (may require building from source on M4)
pip install cmake
pip install dlib --verbose
pip install face-recognition==1.3.0

# Install GUI and other dependencies
pip install PyQt6==6.5.3
pip install Pillow==10.1.0
pip install colorama==0.4.6
pip install tqdm==4.66.1
```

### 5. Handle Camera Permissions

On macOS, you may need to grant camera permissions:

```bash
# Test camera access (may require sudo initially)
sudo python3 -c "import depthai as dai; dai.Device()"

# Grant terminal camera access in System Settings:
# System Settings > Privacy & Security > Camera > Terminal
```

### 6. Clone and Install WhoAmI

```bash
# Clone repository
git clone https://github.com/yourusername/whoami-1.git
cd whoami-1

# Install in development mode
pip install -e .

# Verify installation
python verify_install.py
```

## OAK-D S3 Camera Configuration

### Camera Detection Test

```bash
# Test camera detection
python3 -c "
import depthai as dai
import sys

try:
    # Get available devices
    devices = dai.Device.getAllAvailableDevices()
    if len(devices) == 0:
        print('No devices found!')
        sys.exit(1)
    
    print(f'Found {len(devices)} device(s):')
    for device in devices:
        print(f'  - {device.getMxId()} [{device.state.name}]')
        
    # Try to connect
    with dai.Device() as device:
        print(f'Successfully connected to: {device.getMxId()}')
        print(f'Camera model: {device.getDeviceName()}')
        
except Exception as e:
    print(f'Error: {e}')
    sys.exit(1)
"
```

### Configure Camera Pipeline

Create a test configuration file:

```python
# save as test_camera_config.py
import depthai as dai
import cv2

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
xoutRgb = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")

# Properties
camRgb.setPreviewSize(640, 480)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

# Linking
camRgb.preview.link(xoutRgb.input)

# Connect and start pipeline
with dai.Device(pipeline) as device:
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    
    print("Camera configured successfully. Press 'q' to quit.")
    
    while True:
        inRgb = qRgb.get()
        frame = inRgb.getCvFrame()
        
        cv2.imshow("OAK-D S3 Camera", frame)
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    cv2.destroyAllWindows()
```

## Troubleshooting

### Jetson Orin Nano Issues

#### 1. Camera Not Detected
```bash
# Check USB devices
lsusb | grep Movidius

# Check kernel messages
dmesg | grep -i usb

# Reset USB subsystem
sudo modprobe -r uvcvideo
sudo modprobe uvcvideo
```

#### 2. Performance Issues
```bash
# Monitor system resources
jtop  # or htop

# Check thermal throttling
cat /sys/devices/virtual/thermal/thermal_zone*/temp

# Ensure maximum performance
sudo nvpmodel -q  # Check current mode
sudo jetson_clocks --show
```

#### 3. Python Import Errors
```bash
# Verify library installations
pip list | grep -E "depthai|opencv|face-recognition"

# Reinstall problematic packages
pip uninstall depthai depthai-sdk -y
pip install --no-cache-dir depthai==2.24.0 depthai-sdk==1.14.0
```

### M4 Mac Issues

#### 1. Permission Denied Errors
```bash
# Run with sudo (temporary fix)
sudo python run_gui.py

# Or grant camera permissions in System Settings
# System Settings > Privacy & Security > Camera
```

#### 2. Library Loading Issues
```bash
# Fix dylib issues
export DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH

# Add to ~/.zshrc for persistence
echo 'export DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH' >> ~/.zshrc
```

#### 3. Face Recognition Build Errors
```bash
# Install with verbose output
pip install --verbose --no-cache-dir dlib

# If fails, build from source
git clone https://github.com/davisking/dlib.git
cd dlib
python setup.py install
```

## Performance Optimization

### Jetson Orin Nano Optimization

```bash
# Enable GPU acceleration for OpenCV
export OPENCV_CUDA_BACKEND=1

# Optimize memory usage
echo 1 | sudo tee /proc/sys/vm/overcommit_memory

# Set process priority
nice -n -10 python run_gui.py
```

### M4 Mac Optimization

```bash
# Enable Metal Performance Shaders
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Use optimized BLAS
export OPENBLAS_NUM_THREADS=8

# Run with performance mode
caffeinate -i python run_gui.py
```

## Testing the Installation

### Basic Functionality Test

```bash
# Test CLI mode
python run_cli.py

# Test GUI mode
python run_gui.py

# Run comprehensive tests
python test_oak_camera_full.py
```

### Performance Benchmark

```python
# Save as benchmark.py
import time
import depthai as dai
import cv2
import numpy as np

def benchmark_camera():
    pipeline = dai.Pipeline()
    camRgb = pipeline.create(dai.node.ColorCamera)
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    
    xoutRgb.setStreamName("rgb")
    camRgb.setPreviewSize(640, 480)
    camRgb.setFps(30)
    camRgb.preview.link(xoutRgb.input)
    
    with dai.Device(pipeline) as device:
        q = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        
        frame_count = 0
        start_time = time.time()
        
        while frame_count < 300:  # Process 300 frames
            if q.has():
                q.get().getCvFrame()
                frame_count += 1
        
        elapsed = time.time() - start_time
        fps = frame_count / elapsed
        
        print(f"Processed {frame_count} frames in {elapsed:.2f} seconds")
        print(f"Average FPS: {fps:.2f}")
        
        return fps

if __name__ == "__main__":
    benchmark_camera()
```

## Next Steps

1. Run the comprehensive test suite: `python test_oak_camera_full.py`
2. Configure face encodings database
3. Test face recognition accuracy
4. Optimize for your specific use case
5. Deploy in production environment

## Support

For platform-specific issues:
- Jetson Forums: https://forums.developer.nvidia.com/
- Luxonis Support: https://discuss.luxonis.com/
- GitHub Issues: https://github.com/yourusername/whoami-1/issues

## Version Compatibility Matrix

| Component | Jetson Orin Nano | M4 Mac |
|-----------|------------------|---------|
| Python | 3.8-3.10 | 3.8-3.10 |
| DepthAI | 2.24.0 | 2.24.0 |
| DepthAI-SDK | 1.14.0 | 1.14.0 |
| OpenCV | 4.8.1.78 | 4.8.1.78 |
| NumPy | 1.23.5 | 1.24.3 |
| face-recognition | 1.3.0 | 1.3.0 |
| dlib | 19.24.2 | 19.24.2 |
| PyQt6 | 6.5.3 | 6.5.3 |

---

Last Updated: November 2024