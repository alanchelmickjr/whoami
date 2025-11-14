#!/bin/bash
#
# K-1 Booster Setup Script
# Installs YOLO face recognition with voice interaction
# Optimized for Jetson Orin NX on K-1 Booster carrier board
#

set -e

echo "========================================="
echo "K-1 Booster Setup Script"
echo "========================================="
echo ""

# Detect if running on Jetson
if [ -f /etc/nv_tegra_release ]; then
    echo "✓ Jetson platform detected"
    IS_JETSON=true
else
    echo "⚠ Not running on Jetson (will use CPU mode)"
    IS_JETSON=false
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $PYTHON_VERSION"

# Update system
echo ""
echo "Updating system packages..."
sudo apt-get update

# Install system dependencies
echo ""
echo "Installing system dependencies..."
sudo apt-get install -y \
    portaudio19-dev \
    espeak \
    flac \
    alsa-utils \
    pulseaudio \
    python3-pip \
    python3-dev \
    libportaudio2 \
    libportaudiocpp0 \
    ffmpeg \
    libsm6 \
    libxext6 \
    build-essential

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip3 install --upgrade pip

# Install core requirements
echo ""
echo "Installing core packages..."
pip3 install -r requirements.txt

# Install YOLO and DeepFace (with special handling for Jetson)
echo ""
echo "Installing YOLO and DeepFace..."

if [ "$IS_JETSON" = true ]; then
    echo "Installing for Jetson (CUDA-enabled)..."

    # Install PyTorch for Jetson (if not already installed)
    if ! python3 -c "import torch" 2>/dev/null; then
        echo "Installing PyTorch for Jetson..."
        pip3 install --no-cache-dir torch torchvision torchaudio
    fi

    # Install ultralytics with CUDA support
    pip3 install ultralytics

    # Install TensorFlow for Jetson
    pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v511 tensorflow==2.12.0+nv23.06

else
    echo "Installing for x86 (CPU mode)..."
    pip3 install ultralytics
    pip3 install tensorflow
fi

# Install DeepFace
pip3 install deepface

# Download YOLO model
echo ""
echo "Downloading YOLOv8 nano model..."
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Download Vosk model for offline speech recognition
echo ""
echo "Setting up offline speech recognition (Vosk)..."
VOSK_DIR="/opt/whoami/models"
sudo mkdir -p "$VOSK_DIR"
sudo chown $USER:$USER "$VOSK_DIR"

if [ ! -d "$VOSK_DIR/vosk-model-small-en-us-0.15" ]; then
    echo "Downloading Vosk model..."
    cd "$VOSK_DIR"
    wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
    unzip vosk-model-small-en-us-0.15.zip
    rm vosk-model-small-en-us-0.15.zip
    echo "✓ Vosk model installed"
else
    echo "✓ Vosk model already installed"
fi

# Configure audio (K-1 specific)
echo ""
echo "Configuring audio for K-1..."

# Check if K-1 audio device exists
if arecord -l | grep -q "hw:2,0"; then
    echo "✓ K-1 audio device detected (hw:2,0)"

    # Set as default
    cat > ~/.asoundrc << 'EOF'
pcm.!default {
    type hw
    card 2
    device 0
}

ctl.!default {
    type hw
    card 2
}
EOF
    echo "✓ Audio configured for K-1"
else
    echo "⚠ K-1 audio device not detected, using system default"
fi

# Configure serial ports for gimbal
echo ""
echo "Configuring serial ports for gimbal control..."
sudo usermod -a -G dialout $USER
sudo chmod 666 /dev/ttyTHS0 /dev/ttyTHS1 /dev/ttyTHS2 2>/dev/null || true

# Set hardware environment variable
echo ""
echo "Setting K-1 hardware profile..."
if ! grep -q "WHOAMI_HARDWARE_PROFILE" ~/.bashrc; then
    echo 'export WHOAMI_HARDWARE_PROFILE="jetson_orin_nx_k1"' >> ~/.bashrc
    echo "✓ Hardware profile set to jetson_orin_nx_k1"
fi

# Test installation
echo ""
echo "Testing installation..."
python3 -c "
import sys
try:
    import ultralytics
    import deepface
    import cv2
    import pyttsx3
    import speech_recognition
    import depthai
    print('✓ All critical packages imported successfully')
except ImportError as e:
    print(f'✗ Import error: {e}')
    sys.exit(1)
"

echo ""
echo "========================================="
echo "K-1 Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Reload your shell: source ~/.bashrc"
echo "  2. Test the system: python3 examples/k1_yolo_demo.py"
echo ""
echo "For K-1 Booster documentation, see:"
echo "  - docs/K1_BOOSTER_SETUP.md"
echo "  - docs/VOICE_INTERACTION_GUIDE.md"
echo ""
echo "Note: You may need to log out and back in for group"
echo "      permissions (dialout) to take effect."
echo ""
