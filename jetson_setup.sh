#!/bin/bash

#############################################################################
# Jetson Orin Nano Setup Script for WhoAmI Face Recognition System
# 
# This script automates the setup process for NVIDIA Jetson Orin Nano
# including system configuration, dependency installation, and camera setup
#
# Usage: ./jetson_setup.sh [--full] [--python-only] [--camera-only]
#############################################################################

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script version
VERSION="1.0.0"

# Configuration
PYTHON_VERSION="3.9"
VENV_PATH="$HOME/whoami_env"
WORKSPACE_DIR="$HOME/whoami-1"

# Function to print colored output
print_color() {
    local color=$1
    shift
    echo -e "${color}$@${NC}"
}

# Function to print section headers
print_header() {
    echo ""
    print_color "$CYAN" "=========================================="
    print_color "$CYAN" "$1"
    print_color "$CYAN" "=========================================="
    echo ""
}

# Function to check if running on Jetson
check_jetson() {
    if [ -f /proc/device-tree/model ]; then
        MODEL=$(tr -d '\0' < /proc/device-tree/model)
        if [[ $MODEL == *"Jetson"* ]] || [[ $MODEL == *"NVIDIA"* ]]; then
            print_color "$GREEN" "✓ Detected Jetson Device: $MODEL"
            return 0
        fi
    fi
    
    print_color "$YELLOW" "⚠ Warning: This doesn't appear to be a Jetson device"
    print_color "$YELLOW" "  Detected: $(uname -m) - $(cat /etc/os-release | grep PRETTY_NAME | cut -d '"' -f2)"
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
}

# Function to check current user permissions
check_permissions() {
    if [ "$EUID" -eq 0 ]; then 
        print_color "$RED" "✗ Please do not run this script as root/sudo"
        print_color "$YELLOW" "  The script will request sudo when needed"
        exit 1
    fi
    
    print_color "$GREEN" "✓ Running as user: $USER"
}

# Function to configure Jetson performance mode
configure_performance() {
    print_header "Configuring Jetson Performance Mode"
    
    # Check current power mode
    if command -v nvpmodel &> /dev/null; then
        CURRENT_MODE=$(sudo nvpmodel -q | grep "NV Power Mode" | awk '{print $4}')
        print_color "$BLUE" "Current power mode: $CURRENT_MODE"
        
        # Set to maximum performance (mode 0)
        print_color "$YELLOW" "Setting to maximum performance mode (MAXN)..."
        sudo nvpmodel -m 0
        
        # Enable jetson_clocks for maximum frequency
        print_color "$YELLOW" "Maximizing clock frequencies..."
        sudo jetson_clocks
        
        # Show new status
        sudo jetson_clocks --show
        
        print_color "$GREEN" "✓ Performance mode configured"
        
        # Make persistent
        print_color "$YELLOW" "Making performance settings persistent..."
        
        # Check if already in bashrc
        if ! grep -q "nvpmodel -m 0" ~/.bashrc; then
            echo "" >> ~/.bashrc
            echo "# Jetson Performance Settings (added by whoami setup)" >> ~/.bashrc
            echo "sudo nvpmodel -m 0 2>/dev/null" >> ~/.bashrc
            echo "sudo jetson_clocks 2>/dev/null" >> ~/.bashrc
            print_color "$GREEN" "✓ Added to ~/.bashrc"
        else
            print_color "$BLUE" "ℹ Performance settings already in ~/.bashrc"
        fi
    else
        print_color "$YELLOW" "⚠ nvpmodel not found - skipping performance configuration"
    fi
}

# Function to install system dependencies
install_system_deps() {
    print_header "Installing System Dependencies"
    
    print_color "$YELLOW" "Updating package lists..."
    sudo apt update
    
    # Essential build tools
    print_color "$YELLOW" "Installing build essentials..."
    sudo apt install -y \
        build-essential \
        cmake \
        git \
        pkg-config \
        wget \
        curl \
        nano \
        htop
    
    # Python development packages
    print_color "$YELLOW" "Installing Python development packages..."
    sudo apt install -y \
        python3-pip \
        python3-dev \
        python3-venv \
        python3-wheel \
        python3-setuptools
    
    # OpenCV dependencies
    print_color "$YELLOW" "Installing OpenCV dependencies..."
    sudo apt install -y \
        libopencv-dev \
        python3-opencv \
        libgtk-3-0 \
        libgtk-3-dev \
        libcanberra-gtk3-module
    
    # USB and camera libraries
    print_color "$YELLOW" "Installing USB and camera libraries..."
    sudo apt install -y \
        libusb-1.0-0-dev \
        libudev-dev \
        libusb-dev \
        usbutils \
        v4l-utils
    
    # Additional libraries for face recognition
    print_color "$YELLOW" "Installing face recognition dependencies..."
    sudo apt install -y \
        libopenblas-dev \
        liblapack-dev \
        libatlas-base-dev \
        gfortran \
        libhdf5-dev
    
    # Qt dependencies for GUI
    print_color "$YELLOW" "Installing Qt dependencies..."
    sudo apt install -y \
        libxcb-xinerama0 \
        libxcb-cursor0 \
        libxkbcommon-x11-0 \
        libxcb-keysyms1 \
        libxcb-render-util0 \
        libxcb-icccm4 \
        libxcb-image0
    
    # Jetson-specific tools
    if command -v jetson_release &> /dev/null; then
        print_color "$YELLOW" "Installing Jetson utilities..."
        sudo apt install -y jetson-stats
    fi
    
    print_color "$GREEN" "✓ System dependencies installed"
}

# Function to setup camera permissions
setup_camera_permissions() {
    print_header "Setting Up Camera Permissions"
    
    # Create udev rules for Movidius devices
    print_color "$YELLOW" "Creating udev rules for OAK-D cameras..."
    
    UDEV_RULE='SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"'
    echo "$UDEV_RULE" | sudo tee /etc/udev/rules.d/80-movidius.rules > /dev/null
    
    # Additional rule for Intel RealSense (if needed)
    REALSENSE_RULE='SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", MODE="0666"'
    echo "$REALSENSE_RULE" | sudo tee /etc/udev/rules.d/81-realsense.rules > /dev/null
    
    print_color "$GREEN" "✓ Udev rules created"
    
    # Add user to necessary groups
    print_color "$YELLOW" "Adding user to video and dialout groups..."
    sudo usermod -a -G video,dialout,plugdev,i2c $USER
    
    # Reload udev rules
    print_color "$YELLOW" "Reloading udev rules..."
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    
    print_color "$GREEN" "✓ Camera permissions configured"
    print_color "$YELLOW" "ℹ Note: You may need to logout and login for group changes to take effect"
}

# Function to setup Python environment
setup_python_env() {
    print_header "Setting Up Python Environment"
    
    # Check Python version
    PYTHON_CMD="python$PYTHON_VERSION"
    if ! command -v $PYTHON_CMD &> /dev/null; then
        print_color "$YELLOW" "Python $PYTHON_VERSION not found, trying python3..."
        PYTHON_CMD="python3"
    fi
    
    PYTHON_ACTUAL_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    print_color "$BLUE" "Using Python: $PYTHON_ACTUAL_VERSION"
    
    # Create virtual environment
    if [ -d "$VENV_PATH" ]; then
        print_color "$YELLOW" "Virtual environment already exists at $VENV_PATH"
        read -p "Remove and recreate? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_PATH"
            print_color "$YELLOW" "Creating new virtual environment..."
            $PYTHON_CMD -m venv "$VENV_PATH"
        fi
    else
        print_color "$YELLOW" "Creating virtual environment at $VENV_PATH..."
        $PYTHON_CMD -m venv "$VENV_PATH"
    fi
    
    # Activate virtual environment
    source "$VENV_PATH/bin/activate"
    
    # Upgrade pip
    print_color "$YELLOW" "Upgrading pip, setuptools, and wheel..."
    pip install --upgrade pip setuptools wheel
    
    print_color "$GREEN" "✓ Python environment ready"
}

# Function to install Python packages
install_python_packages() {
    print_header "Installing Python Packages"
    
    # Ensure we're in virtual environment
    if [ -z "$VIRTUAL_ENV" ]; then
        source "$VENV_PATH/bin/activate"
    fi
    
    # Install numpy first (important for ARM64)
    print_color "$YELLOW" "Installing numpy..."
    pip install numpy==1.23.5
    
    # Install OpenCV
    print_color "$YELLOW" "Installing OpenCV..."
    pip install opencv-python==4.8.1.78
    pip install opencv-contrib-python==4.8.1.78
    
    # Install DepthAI
    print_color "$YELLOW" "Installing DepthAI..."
    pip install depthai==2.24.0
    
    # Try to install depthai-sdk (may have issues on some Jetson setups)
    print_color "$YELLOW" "Installing DepthAI SDK..."
    pip install depthai-sdk==1.14.0 || {
        print_color "$YELLOW" "⚠ depthai-sdk installation failed, trying alternative..."
        pip install depthai-sdk || print_color "$YELLOW" "⚠ depthai-sdk not installed"
    }
    
    # Install dlib and face_recognition
    print_color "$YELLOW" "Installing dlib (this may take a while)..."
    pip install dlib==19.24.2
    
    print_color "$YELLOW" "Installing face_recognition..."
    pip install face-recognition==1.3.0
    
    # Install GUI dependencies
    print_color "$YELLOW" "Installing PyQt6..."
    pip install PyQt6==6.5.3 || {
        print_color "$YELLOW" "⚠ PyQt6 installation failed, trying PyQt5..."
        pip install PyQt5
    }
    
    # Install other dependencies
    print_color "$YELLOW" "Installing additional packages..."
    pip install \
        Pillow==10.1.0 \
        colorama==0.4.6 \
        tqdm==4.66.1 \
        psutil \
        pyyaml
    
    print_color "$GREEN" "✓ Python packages installed"
}

# Function to clone and setup WhoAmI
setup_whoami() {
    print_header "Setting Up WhoAmI Application"
    
    # Check if directory exists
    if [ -d "$WORKSPACE_DIR" ]; then
        print_color "$BLUE" "WhoAmI directory already exists at $WORKSPACE_DIR"
        cd "$WORKSPACE_DIR"
        
        # Check if it's a git repo and pull latest
        if [ -d ".git" ]; then
            print_color "$YELLOW" "Pulling latest changes..."
            git pull || print_color "$YELLOW" "⚠ Could not pull latest changes"
        fi
    else
        print_color "$YELLOW" "Cloning WhoAmI repository..."
        git clone https://github.com/yourusername/whoami-1.git "$WORKSPACE_DIR"
        cd "$WORKSPACE_DIR"
    fi
    
    # Activate virtual environment if not active
    if [ -z "$VIRTUAL_ENV" ]; then
        source "$VENV_PATH/bin/activate"
    fi
    
    # Install in development mode
    print_color "$YELLOW" "Installing WhoAmI in development mode..."
    pip install -e .
    
    print_color "$GREEN" "✓ WhoAmI application setup complete"
}

# Function to verify installation
verify_installation() {
    print_header "Verifying Installation"
    
    # Activate virtual environment if not active
    if [ -z "$VIRTUAL_ENV" ]; then
        source "$VENV_PATH/bin/activate"
    fi
    
    # Test Python imports
    print_color "$YELLOW" "Testing Python imports..."
    
    python3 -c "
import sys
print(f'Python: {sys.version}')

try:
    import cv2
    print(f'✓ OpenCV: {cv2.__version__}')
except ImportError as e:
    print(f'✗ OpenCV: {e}')

try:
    import numpy as np
    print(f'✓ NumPy: {np.__version__}')
except ImportError as e:
    print(f'✗ NumPy: {e}')

try:
    import depthai as dai
    print(f'✓ DepthAI: {dai.__version__}')
except ImportError as e:
    print(f'✗ DepthAI: {e}')

try:
    import face_recognition
    print(f'✓ face_recognition: {face_recognition.__version__}')
except ImportError as e:
    print(f'✗ face_recognition: {e}')

try:
    from PyQt6 import QtCore
    print(f'✓ PyQt6: {QtCore.qVersion()}')
except ImportError:
    try:
        from PyQt5 import QtCore
        print(f'✓ PyQt5: {QtCore.qVersion()}')
    except ImportError as e:
        print(f'✗ Qt: {e}')
"
    
    # Test camera detection
    print_color "$YELLOW" "Testing camera detection..."
    
    python3 -c "
import depthai as dai
try:
    devices = dai.Device.getAllAvailableDevices()
    if len(devices) > 0:
        print(f'✓ Found {len(devices)} OAK device(s)')
        for device in devices:
            print(f'  - {device.getMxId()} [{device.state.name}]')
    else:
        print('⚠ No OAK devices detected')
        print('  Make sure your OAK-D camera is connected')
except Exception as e:
    print(f'✗ Camera detection failed: {e}')
"
    
    # Check groups
    print_color "$YELLOW" "Checking user groups..."
    groups | grep -q video && print_color "$GREEN" "✓ User in video group" || print_color "$YELLOW" "⚠ User not in video group"
    groups | grep -q dialout && print_color "$GREEN" "✓ User in dialout group" || print_color "$YELLOW" "⚠ User not in dialout group"
    
    # Test WhoAmI installation
    if [ -f "$WORKSPACE_DIR/verify_install.py" ]; then
        print_color "$YELLOW" "Running WhoAmI verification..."
        cd "$WORKSPACE_DIR"
        python verify_install.py || print_color "$YELLOW" "⚠ WhoAmI verification had issues"
    fi
    
    print_color "$GREEN" "✓ Installation verification complete"
}

# Function to create desktop shortcuts
create_shortcuts() {
    print_header "Creating Desktop Shortcuts"
    
    DESKTOP_DIR="$HOME/Desktop"
    if [ ! -d "$DESKTOP_DIR" ]; then
        print_color "$YELLOW" "Desktop directory not found, skipping shortcuts"
        return
    fi
    
    # Create launcher for GUI
    cat > "$DESKTOP_DIR/WhoAmI-GUI.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=WhoAmI Face Recognition
Comment=Launch WhoAmI GUI
Exec=bash -c "source $VENV_PATH/bin/activate && cd $WORKSPACE_DIR && python run_gui.py"
Icon=$WORKSPACE_DIR/icon.png
Terminal=false
Categories=Application;
EOF
    
    # Create launcher for CLI
    cat > "$DESKTOP_DIR/WhoAmI-CLI.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=WhoAmI CLI
Comment=Launch WhoAmI in terminal
Exec=bash -c "source $VENV_PATH/bin/activate && cd $WORKSPACE_DIR && python run_cli.py; read -p 'Press Enter to close...'"
Icon=utilities-terminal
Terminal=true
Categories=Application;
EOF
    
    # Make executable
    chmod +x "$DESKTOP_DIR/WhoAmI-GUI.desktop"
    chmod +x "$DESKTOP_DIR/WhoAmI-CLI.desktop"
    
    print_color "$GREEN" "✓ Desktop shortcuts created"
}

# Function to show post-installation instructions
show_instructions() {
    print_header "Installation Complete!"
    
    print_color "$GREEN" "✓ Jetson Orin Nano setup completed successfully!"
    echo ""
    print_color "$CYAN" "Next Steps:"
    echo ""
    print_color "$YELLOW" "1. Logout and login again for group permissions to take effect"
    print_color "$YELLOW" "   Or run: newgrp video"
    echo ""
    print_color "$YELLOW" "2. Activate the virtual environment:"
    print_color "$WHITE" "   source $VENV_PATH/bin/activate"
    echo ""
    print_color "$YELLOW" "3. Navigate to the project directory:"
    print_color "$WHITE" "   cd $WORKSPACE_DIR"
    echo ""
    print_color "$YELLOW" "4. Test the camera:"
    print_color "$WHITE" "   python test_oak_camera_full.py"
    echo ""
    print_color "$YELLOW" "5. Run WhoAmI:"
    print_color "$WHITE" "   python run_gui.py    # For GUI mode"
    print_color "$WHITE" "   python run_cli.py    # For CLI mode"
    echo ""
    
    if [ -d "$HOME/Desktop" ]; then
        print_color "$BLUE" "Desktop shortcuts have been created for easy access"
    fi
    
    echo ""
    print_color "$MAGENTA" "For troubleshooting, check:"
    print_color "$WHITE" "  - Camera connection: lsusb | grep 03e7"
    print_color "$WHITE" "  - System performance: jtop"
    print_color "$WHITE" "  - Logs: dmesg | grep -i usb"
    echo ""
}

# Function to run quick setup (camera permissions only)
quick_camera_setup() {
    print_header "Quick Camera Setup"
    setup_camera_permissions
    print_color "$GREEN" "✓ Quick camera setup complete"
    print_color "$YELLOW" "Remember to logout and login for group changes to take effect"
}

# Function to run Python-only setup
python_only_setup() {
    print_header "Python Environment Setup"
    setup_python_env
    install_python_packages
    print_color "$GREEN" "✓ Python setup complete"
}

# Main function
main() {
    print_color "$CYAN" "╔════════════════════════════════════════════════╗"
    print_color "$CYAN" "║   Jetson Orin Nano Setup Script for WhoAmI    ║"
    print_color "$CYAN" "║                 Version $VERSION                  ║"
    print_color "$CYAN" "╚════════════════════════════════════════════════╝"
    echo ""
    
    # Parse arguments
    case "${1:-}" in
        --camera-only)
            check_permissions
            quick_camera_setup
            exit 0
            ;;
        --python-only)
            check_permissions
            python_only_setup
            exit 0
            ;;
        --help|-h)
            print_color "$CYAN" "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --full         Run complete setup (default)"
            echo "  --camera-only  Setup camera permissions only"
            echo "  --python-only  Setup Python environment only"
            echo "  --help         Show this help message"
            exit 0
            ;;
    esac
    
    # Run full setup
    check_permissions
    check_jetson
    
    # Full installation
    configure_performance
    install_system_deps
    setup_camera_permissions
    setup_python_env
    install_python_packages
    setup_whoami
    verify_installation
    create_shortcuts
    show_instructions
    
    # Ask about reboot
    echo ""
    read -p "$(print_color "$YELLOW" "Would you like to reboot now for all changes to take effect? (y/n): ")" -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_color "$YELLOW" "Rebooting in 5 seconds..."
        sleep 5
        sudo reboot
    fi
}

# Run main function
main "$@"