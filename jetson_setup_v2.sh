#!/bin/bash

#############################################################################
# Jetson Orin Nano Setup Script v2.0 for WhoAmI Face Recognition System
#
# Enhanced version with improved error handling, comprehensive verification,
# and support for all system components including:
# - OAK-D Series 3 camera
# - Feetech servo gimbal control
# - 3D scanning capabilities
# - Robot brain and learning systems
# - Encrypted memory storage
#
# Usage: ./jetson_setup_v2.sh [OPTIONS]
# Options:
#   --full              Complete installation (default)
#   --minimal           Minimal install (no 3D scanning, no gimbal)
#   --camera-only       Setup camera permissions only
#   --python-only       Setup Python environment only
#   --verify-only       Run verification tests only
#   --skip-performance  Skip Jetson performance configuration
#   --help              Show this help message
#############################################################################

set -e  # Exit on error
set -o pipefail  # Pipe failures propagate

# Script metadata
VERSION="2.0.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$HOME/whoami_setup_$(date +%Y%m%d_%H%M%S).log"

# Configuration
PYTHON_MIN_VERSION="3.8"
PYTHON_PREFERRED_VERSION="3.9"
VENV_PATH="$HOME/whoami_env"
WORKSPACE_DIR="$SCRIPT_DIR"

# Installation options
INSTALL_FULL=true
INSTALL_3D_SCANNING=true
INSTALL_GIMBAL=true
SKIP_PERFORMANCE=false

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Error tracking
ERROR_COUNT=0
WARNING_COUNT=0

#############################################################################
# Utility Functions
#############################################################################

# Logging function
log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
}

# Print colored output
print_color() {
    local color=$1
    shift
    echo -e "${color}$@${NC}"
}

# Print section headers
print_header() {
    echo "" | tee -a "$LOG_FILE"
    print_color "$CYAN" "=========================================="
    print_color "$CYAN" "$1"
    print_color "$CYAN" "=========================================="
    echo "" | tee -a "$LOG_FILE"
    log "INFO" "Starting: $1"
}

# Print status messages
print_success() {
    print_color "$GREEN" "✓ $@"
    log "SUCCESS" "$@"
}

print_error() {
    print_color "$RED" "✗ $@"
    log "ERROR" "$@"
    ((ERROR_COUNT++))
}

print_warning() {
    print_color "$YELLOW" "⚠ $@"
    log "WARNING" "$@"
    ((WARNING_COUNT++))
}

print_info() {
    print_color "$BLUE" "ℹ $@"
    log "INFO" "$@"
}

# Command existence check
command_exists() {
    command -v "$1" &> /dev/null
}

# Version comparison (returns 0 if version1 >= version2)
version_ge() {
    printf '%s\n%s\n' "$2" "$1" | sort -V -C
}

# Pause and continue
pause_continue() {
    local message=${1:-"Press Enter to continue..."}
    read -p "$(print_color "$YELLOW" "$message")"
}

# Check if running in virtual environment
check_venv() {
    if [ -z "$VIRTUAL_ENV" ]; then
        return 1
    fi
    return 0
}

#############################################################################
# System Detection and Validation
#############################################################################

# Check if running on Jetson
check_jetson() {
    print_header "Detecting Platform"

    local is_jetson=false
    local model="Unknown"

    # Check device tree model
    if [ -f /proc/device-tree/model ]; then
        model=$(tr -d '\0' < /proc/device-tree/model)
        if [[ $model == *"Jetson"* ]] || [[ $model == *"NVIDIA"* ]]; then
            is_jetson=true
        fi
    fi

    # Check for Jetson-specific files
    if [ -d /sys/devices/platform/gpu.0 ] || command_exists tegrastats; then
        is_jetson=true
    fi

    if $is_jetson; then
        print_success "Detected Jetson Device: $model"

        # Get JetPack version if available
        if command_exists jetson_release; then
            local jetpack_version=$(jetson_release -v 2>/dev/null | grep "L4T" || echo "Unknown")
            print_info "JetPack Info: $jetpack_version"
        fi

        return 0
    else
        print_warning "This doesn't appear to be a Jetson device"
        print_info "Detected: $(uname -m) - $(grep PRETTY_NAME /etc/os-release | cut -d '"' -f2 2>/dev/null || echo "Unknown")"

        echo ""
        read -p "$(print_color "$YELLOW" "Continue installation anyway? (y/n): ")" -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Installation cancelled by user"
            exit 0
        fi
        return 1
    fi
}

# Check current user permissions
check_permissions() {
    print_header "Checking Permissions"

    if [ "$EUID" -eq 0 ]; then
        print_error "Please do not run this script as root/sudo"
        print_warning "The script will request sudo privileges when needed"
        exit 1
    fi

    print_success "Running as user: $USER"

    # Check sudo access
    if sudo -n true 2>/dev/null; then
        print_success "Sudo access available"
    else
        print_info "Testing sudo access (password may be required)..."
        if sudo true; then
            print_success "Sudo access confirmed"
        else
            print_error "Sudo access required for system configuration"
            exit 1
        fi
    fi
}

# Check disk space
check_disk_space() {
    print_header "Checking Disk Space"

    local required_mb=5000  # 5GB minimum
    local available_mb=$(df -m "$HOME" | tail -1 | awk '{print $4}')

    print_info "Available space: $((available_mb / 1024))GB"
    print_info "Required space: $((required_mb / 1024))GB"

    if [ $available_mb -lt $required_mb ]; then
        print_error "Insufficient disk space"
        print_warning "Please free up at least $((required_mb / 1024))GB of space"
        return 1
    fi

    print_success "Sufficient disk space available"
    return 0
}

# Check Python version
check_python_version() {
    local python_cmd=""

    # Try different Python commands
    for cmd in python$PYTHON_PREFERRED_VERSION python3 python; do
        if command_exists $cmd; then
            python_cmd=$cmd
            break
        fi
    done

    if [ -z "$python_cmd" ]; then
        print_error "Python not found"
        return 1
    fi

    local version=$($python_cmd --version 2>&1 | awk '{print $2}')
    local major=$(echo $version | cut -d. -f1)
    local minor=$(echo $version | cut -d. -f2)

    print_info "Found Python: $version ($python_cmd)"

    if [ "$major" -ge 3 ] && [ "$minor" -ge 8 ]; then
        print_success "Python version is compatible"
        echo "$python_cmd"
        return 0
    else
        print_error "Python $PYTHON_MIN_VERSION or higher required (found $version)"
        return 1
    fi
}

#############################################################################
# Jetson Performance Configuration
#############################################################################

configure_performance() {
    if $SKIP_PERFORMANCE; then
        print_info "Skipping performance configuration (--skip-performance)"
        return 0
    fi

    print_header "Configuring Jetson Performance Mode"

    if ! command_exists nvpmodel; then
        print_warning "nvpmodel not found - not a Jetson or not properly configured"
        return 1
    fi

    # Get current power mode
    local current_mode=$(sudo nvpmodel -q 2>/dev/null | grep "NV Power Mode" | awk '{print $4}' || echo "unknown")
    print_info "Current power mode: $current_mode"

    # Set to maximum performance (mode 0 = MAXN)
    print_info "Setting to maximum performance mode (MAXN)..."
    if sudo nvpmodel -m 0; then
        print_success "Power mode set to MAXN"
    else
        print_warning "Failed to set power mode"
    fi

    # Enable jetson_clocks for maximum frequency
    print_info "Maximizing clock frequencies..."
    if sudo jetson_clocks; then
        print_success "Clock frequencies maximized"
    else
        print_warning "Failed to set clock frequencies"
    fi

    # Show current status
    print_info "Current clock status:"
    sudo jetson_clocks --show 2>&1 | head -10 | tee -a "$LOG_FILE"

    # Make persistent in bashrc
    print_info "Making performance settings persistent..."
    local bashrc="$HOME/.bashrc"

    if ! grep -q "nvpmodel -m 0" "$bashrc" 2>/dev/null; then
        cat >> "$bashrc" << 'EOF'

# Jetson Performance Settings (added by WhoAmI setup)
if command -v nvpmodel &> /dev/null; then
    sudo nvpmodel -m 0 2>/dev/null
    sudo jetson_clocks 2>/dev/null
fi
EOF
        print_success "Added to ~/.bashrc"
    else
        print_info "Performance settings already in ~/.bashrc"
    fi

    print_success "Performance configuration complete"
}

#############################################################################
# System Dependencies Installation
#############################################################################

install_system_deps() {
    print_header "Installing System Dependencies"

    print_info "Updating package lists..."
    if ! sudo apt update; then
        print_error "Failed to update package lists"
        return 1
    fi

    # Essential build tools
    print_info "Installing build essentials..."
    sudo apt install -y \
        build-essential \
        cmake \
        git \
        pkg-config \
        wget \
        curl \
        nano \
        htop \
        net-tools \
        || print_warning "Some build tools failed to install"

    # Python development packages
    print_info "Installing Python development packages..."
    sudo apt install -y \
        python3-pip \
        python3-dev \
        python3-venv \
        python3-wheel \
        python3-setuptools \
        || print_warning "Some Python packages failed to install"

    # OpenCV dependencies
    print_info "Installing OpenCV dependencies..."
    sudo apt install -y \
        libopencv-dev \
        python3-opencv \
        libgtk-3-0 \
        libgtk-3-dev \
        libcanberra-gtk3-module \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libv4l-dev \
        || print_warning "Some OpenCV dependencies failed to install"

    # USB and camera libraries
    print_info "Installing USB and camera libraries..."
    sudo apt install -y \
        libusb-1.0-0-dev \
        libudev-dev \
        libusb-dev \
        usbutils \
        v4l-utils \
        || print_warning "Some USB libraries failed to install"

    # Face recognition dependencies (ARM64 optimized)
    print_info "Installing face recognition dependencies..."
    sudo apt install -y \
        libopenblas-dev \
        liblapack-dev \
        libatlas-base-dev \
        gfortran \
        libhdf5-dev \
        libhdf5-serial-dev \
        || print_warning "Some math libraries failed to install"

    # 3D scanning dependencies
    if $INSTALL_3D_SCANNING; then
        print_info "Installing 3D scanning dependencies..."
        sudo apt install -y \
            libeigen3-dev \
            libglfw3-dev \
            libglew-dev \
            libpng-dev \
            libjpeg-dev \
            || print_warning "Some 3D scanning dependencies failed to install"
    fi

    # Serial communication for gimbal
    if $INSTALL_GIMBAL; then
        print_info "Installing serial communication dependencies..."
        sudo apt install -y \
            libserial-dev \
            setserial \
            || print_warning "Serial libraries installation had issues"
    fi

    # GUI dependencies (Tkinter)
    print_info "Installing GUI dependencies..."
    sudo apt install -y \
        python3-tk \
        tk-dev \
        libxcb-xinerama0 \
        libxcb-cursor0 \
        libxkbcommon-x11-0 \
        libxcb-keysyms1 \
        libxcb-render-util0 \
        libxcb-icccm4 \
        libxcb-image0 \
        || print_warning "Some GUI dependencies failed to install"

    # Jetson-specific tools
    if command_exists jetson_release; then
        print_info "Installing Jetson utilities..."
        sudo apt install -y jetson-stats 2>/dev/null || print_warning "jetson-stats installation failed"
    fi

    # Clean up
    print_info "Cleaning up package cache..."
    sudo apt autoremove -y
    sudo apt clean

    print_success "System dependencies installation complete"
}

#############################################################################
# Camera and Hardware Permissions
#############################################################################

setup_camera_permissions() {
    print_header "Setting Up Camera and Hardware Permissions"

    # Create udev rules for OAK-D cameras (Movidius vendor ID: 03e7)
    print_info "Creating udev rules for OAK-D cameras..."

    local movidius_rule='SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666", GROUP="plugdev"'
    echo "$movidius_rule" | sudo tee /etc/udev/rules.d/80-movidius.rules > /dev/null
    print_success "Movidius udev rule created"

    # Additional rule for Intel RealSense (optional, for future compatibility)
    local realsense_rule='SUBSYSTEM=="usb", ATTRS{idVendor}=="8086", MODE="0666", GROUP="plugdev"'
    echo "$realsense_rule" | sudo tee /etc/udev/rules.d/81-realsense.rules > /dev/null
    print_success "RealSense udev rule created (optional)"

    # Serial port udev rules for gimbal control
    if $INSTALL_GIMBAL; then
        print_info "Creating udev rules for serial devices..."
        local serial_rule='KERNEL=="ttyUSB[0-9]*", MODE="0666", GROUP="dialout"'
        echo "$serial_rule" | sudo tee /etc/udev/rules.d/82-serial.rules > /dev/null

        # Jetson-specific UART
        local jetson_uart_rule='KERNEL=="ttyTHS[0-9]*", MODE="0666", GROUP="dialout"'
        echo "$jetson_uart_rule" | sudo tee -a /etc/udev/rules.d/82-serial.rules > /dev/null
        print_success "Serial device udev rules created"
    fi

    # Add user to necessary groups
    print_info "Adding user to hardware access groups..."
    local groups_to_add="video dialout plugdev i2c"

    for group in $groups_to_add; do
        if getent group $group > /dev/null 2>&1; then
            sudo usermod -a -G $group $USER
            print_success "Added to group: $group"
        else
            print_warning "Group not found: $group"
        fi
    done

    # Reload udev rules
    print_info "Reloading udev rules..."
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    print_success "Udev rules reloaded"

    # Detect connected OAK-D cameras
    print_info "Scanning for OAK-D cameras..."
    local oak_devices=$(lsusb | grep -i "03e7" | wc -l)

    if [ $oak_devices -gt 0 ]; then
        print_success "Found $oak_devices OAK-D device(s)"
        lsusb | grep "03e7" | tee -a "$LOG_FILE"
    else
        print_warning "No OAK-D cameras detected"
        print_info "Make sure your camera is connected"
    fi

    # Detect serial ports for gimbal
    if $INSTALL_GIMBAL; then
        print_info "Scanning for serial devices..."
        local serial_devices=$(ls /dev/ttyUSB* /dev/ttyTHS* 2>/dev/null || echo "")

        if [ -n "$serial_devices" ]; then
            print_success "Found serial device(s):"
            echo "$serial_devices" | tee -a "$LOG_FILE"
        else
            print_warning "No serial devices detected (gimbal may not be connected)"
        fi
    fi

    print_success "Camera and hardware permissions configured"
    print_warning "NOTE: You must logout and login for group changes to take effect"
    print_info "      Or run: newgrp video && newgrp dialout"
}

#############################################################################
# Python Environment Setup
#############################################################################

setup_python_env() {
    print_header "Setting Up Python Virtual Environment"

    # Find Python
    local python_cmd=$(check_python_version)
    if [ -z "$python_cmd" ]; then
        print_error "Compatible Python version not found"
        return 1
    fi

    local python_version=$($python_cmd --version 2>&1 | awk '{print $2}')
    print_info "Using Python: $python_version ($python_cmd)"

    # Check if virtual environment exists
    if [ -d "$VENV_PATH" ]; then
        print_warning "Virtual environment already exists at $VENV_PATH"
        echo ""
        read -p "$(print_color "$YELLOW" "Remove and recreate? (y/n): ")" -n 1 -r
        echo

        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "Removing existing virtual environment..."
            rm -rf "$VENV_PATH"
        else
            print_info "Using existing virtual environment"
            source "$VENV_PATH/bin/activate"
            print_success "Virtual environment activated"
            return 0
        fi
    fi

    # Create new virtual environment
    print_info "Creating virtual environment at $VENV_PATH..."
    if $python_cmd -m venv "$VENV_PATH"; then
        print_success "Virtual environment created"
    else
        print_error "Failed to create virtual environment"
        return 1
    fi

    # Activate virtual environment
    print_info "Activating virtual environment..."
    source "$VENV_PATH/bin/activate"

    if check_venv; then
        print_success "Virtual environment activated: $VIRTUAL_ENV"
    else
        print_error "Failed to activate virtual environment"
        return 1
    fi

    # Upgrade pip, setuptools, wheel
    print_info "Upgrading pip, setuptools, and wheel..."
    pip install --upgrade pip setuptools wheel || print_warning "Upgrade had issues"

    local pip_version=$(pip --version | awk '{print $2}')
    print_success "pip version: $pip_version"

    # Create activation helper script
    print_info "Creating activation helper script..."
    cat > "$HOME/activate_whoami.sh" << EOF
#!/bin/bash
# WhoAmI Environment Activation Script
source "$VENV_PATH/bin/activate"
cd "$WORKSPACE_DIR"
echo "WhoAmI environment activated"
echo "Python: \$(python --version)"
echo "Working directory: \$(pwd)"
EOF
    chmod +x "$HOME/activate_whoami.sh"
    print_success "Created: ~/activate_whoami.sh"

    print_success "Python environment setup complete"
}

#############################################################################
# Python Package Installation
#############################################################################

install_python_packages() {
    print_header "Installing Python Packages"

    # Ensure we're in virtual environment
    if ! check_venv; then
        print_info "Activating virtual environment..."
        source "$VENV_PATH/bin/activate"
    fi

    if ! check_venv; then
        print_error "Not in virtual environment"
        return 1
    fi

    print_info "Virtual environment: $VIRTUAL_ENV"

    # Install numpy first (critical for ARM64)
    print_info "Installing numpy (ARM64 optimized version)..."
    pip install numpy==1.23.5 || {
        print_warning "Specific numpy version failed, trying latest..."
        pip install numpy
    }

    # Install OpenCV
    print_info "Installing OpenCV..."
    pip install opencv-python==4.8.1.78 || pip install opencv-python
    pip install opencv-contrib-python==4.8.1.78 || pip install opencv-contrib-python

    # Install DepthAI for OAK-D camera
    print_info "Installing DepthAI..."
    pip install depthai==2.24.0 || pip install depthai

    # Install PIL/Pillow
    print_info "Installing Pillow..."
    pip install Pillow>=10.0.0

    # Install dlib (takes a while to compile on ARM)
    print_info "Installing dlib (this may take 5-10 minutes on ARM64)..."
    print_warning "Please be patient, dlib is compiling from source..."

    if ! pip install dlib==19.24.2; then
        print_warning "Specific dlib version failed, trying latest..."
        if ! pip install dlib; then
            print_error "dlib installation failed"
            print_warning "Face recognition features may not work"
        fi
    fi

    # Install face_recognition
    print_info "Installing face-recognition..."
    pip install face-recognition>=1.3.0 || print_warning "face-recognition installation had issues"

    # Install Tkinter support (usually from system, but check)
    print_info "Verifying Tkinter support..."
    python -c "import tkinter" 2>/dev/null && print_success "Tkinter available" || {
        print_warning "Tkinter not available - GUI may not work"
        print_info "Install with: sudo apt install python3-tk"
    }

    # Install scikit-learn for machine learning
    print_info "Installing scikit-learn..."
    pip install scikit-learn>=1.3.0

    # Install cryptography for encrypted storage
    print_info "Installing cryptography..."
    pip install cryptography>=41.0.0

    # Install serial communication for gimbal
    if $INSTALL_GIMBAL; then
        print_info "Installing pyserial for gimbal control..."
        pip install pyserial>=3.5
    fi

    # Install 3D scanning packages
    if $INSTALL_3D_SCANNING; then
        print_info "Installing 3D scanning packages..."
        print_warning "open3d compilation may take 10-20 minutes on ARM64..."

        # Try to install open3d
        if ! pip install open3d>=0.17.0; then
            print_warning "open3d installation failed (may not have ARM64 wheel)"
            print_info "3D scanning features may be limited"
        fi

        # Install trimesh
        pip install trimesh>=4.0.0 || print_warning "trimesh installation failed"
    fi

    # Install additional utilities
    print_info "Installing additional utilities..."
    pip install colorama tqdm psutil pyyaml || print_warning "Some utilities failed to install"

    # Verify critical packages
    print_info "Verifying critical package installations..."
    python -c "
import sys
packages = {
    'cv2': 'opencv-python',
    'numpy': 'numpy',
    'depthai': 'depthai',
    'PIL': 'Pillow',
    'sklearn': 'scikit-learn',
    'cryptography': 'cryptography',
}

failed = []
for module, package in packages.items():
    try:
        __import__(module)
        print(f'✓ {package}')
    except ImportError:
        print(f'✗ {package}')
        failed.append(package)

if failed:
    print(f'\nFailed packages: {failed}')
    sys.exit(1)
" || print_warning "Some package verifications failed"

    print_success "Python packages installation complete"
}

#############################################################################
# WhoAmI Application Setup
#############################################################################

setup_whoami_app() {
    print_header "Setting Up WhoAmI Application"

    # Ensure we're in the workspace directory
    cd "$WORKSPACE_DIR"
    print_info "Working directory: $WORKSPACE_DIR"

    # Ensure we're in virtual environment
    if ! check_venv; then
        source "$VENV_PATH/bin/activate"
    fi

    # Install in development mode
    print_info "Installing WhoAmI package in development mode..."
    if [ -f "setup.py" ]; then
        pip install -e . || print_warning "Development installation had issues"
        print_success "WhoAmI package installed"
    else
        print_warning "setup.py not found, skipping package installation"
    fi

    # Initialize configuration files
    print_info "Checking configuration files..."

    if [ ! -d "config" ]; then
        print_warning "config directory not found, creating..."
        mkdir -p config
    fi

    # Create default config.json if it doesn't exist
    if [ ! -f "config.json" ]; then
        print_info "Creating default config.json..."
        cat > config.json << 'EOF'
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
  },
  "gui": {
    "window_width": 800,
    "window_height": 600
  }
}
EOF
        print_success "Created config.json"
    else
        print_info "config.json already exists"
    fi

    # Create data directories
    print_info "Creating data directories..."
    mkdir -p data/faces
    mkdir -p data/scans
    mkdir -p data/logs
    mkdir -p data/backups
    print_success "Data directories created"

    # Set up logging
    print_info "Configuring logging..."
    mkdir -p logs
    touch logs/whoami.log
    print_success "Logging configured"

    print_success "WhoAmI application setup complete"
}

#############################################################################
# Installation Verification
#############################################################################

verify_installation() {
    print_header "Verifying Installation"

    # Ensure we're in virtual environment
    if ! check_venv; then
        source "$VENV_PATH/bin/activate"
    fi

    local verification_failed=false

    # Test Python version
    print_info "Checking Python version..."
    local py_version=$(python --version 2>&1)
    print_success "Python: $py_version"

    # Test critical imports
    print_info "Testing Python package imports..."
    python << 'PYEOF'
import sys

# Critical packages
packages = [
    ('cv2', 'OpenCV'),
    ('numpy', 'NumPy'),
    ('depthai', 'DepthAI'),
    ('PIL', 'Pillow'),
    ('face_recognition', 'face-recognition'),
    ('sklearn', 'scikit-learn'),
    ('cryptography', 'cryptography'),
]

# Optional packages
optional_packages = [
    ('serial', 'pyserial'),
    ('open3d', 'Open3D'),
    ('trimesh', 'trimesh'),
    ('tkinter', 'Tkinter'),
]

print("\nCritical packages:")
critical_ok = True
for module, name in packages:
    try:
        mod = __import__(module)
        version = getattr(mod, '__version__', 'unknown')
        print(f'  ✓ {name}: {version}')
    except ImportError as e:
        print(f'  ✗ {name}: {e}')
        critical_ok = False

print("\nOptional packages:")
for module, name in optional_packages:
    try:
        mod = __import__(module)
        version = getattr(mod, '__version__', 'OK')
        print(f'  ✓ {name}: {version}')
    except ImportError:
        print(f'  ⚠ {name}: Not installed')

if not critical_ok:
    sys.exit(1)
PYEOF

    if [ $? -ne 0 ]; then
        print_error "Package import tests failed"
        verification_failed=true
    else
        print_success "Package imports successful"
    fi

    # Test camera detection
    print_info "Testing OAK-D camera detection..."
    python << 'PYEOF'
import depthai as dai
import sys

try:
    devices = dai.Device.getAllAvailableDevices()
    if len(devices) > 0:
        print(f'  ✓ Found {len(devices)} OAK-D device(s)')
        for device in devices:
            print(f'    - MxID: {device.getMxId()[:16]}... [{device.state.name}]')
    else:
        print('  ⚠ No OAK-D devices detected')
        print('    Make sure your OAK-D camera is connected')
except Exception as e:
    print(f'  ✗ Camera detection error: {e}')
    sys.exit(1)
PYEOF

    if [ $? -ne 0 ]; then
        print_warning "Camera detection had issues"
    fi

    # Check user groups
    print_info "Checking user group memberships..."
    local required_groups="video dialout"

    for group in $required_groups; do
        if groups | grep -q "\b$group\b"; then
            print_success "User in $group group"
        else
            print_warning "User NOT in $group group (logout required)"
        fi
    done

    # Check serial ports (if gimbal enabled)
    if $INSTALL_GIMBAL; then
        print_info "Checking serial ports for gimbal..."
        if ls /dev/ttyUSB* /dev/ttyTHS* &>/dev/null; then
            print_success "Serial ports available:"
            ls -la /dev/ttyUSB* /dev/ttyTHS* 2>/dev/null | tee -a "$LOG_FILE"
        else
            print_warning "No serial ports detected (gimbal not connected?)"
        fi
    fi

    # Run WhoAmI verification script if available
    if [ -f "verify_install.py" ]; then
        print_info "Running WhoAmI verification script..."
        if python verify_install.py; then
            print_success "WhoAmI verification passed"
        else
            print_warning "WhoAmI verification had issues"
            verification_failed=true
        fi
    fi

    # System information
    print_info "System Information:"
    echo "  Platform: $(uname -m)"
    echo "  Kernel: $(uname -r)"
    echo "  Memory: $(free -h | awk '/^Mem:/ {print $2}')"

    if command_exists tegrastats; then
        print_info "  Jetson stats available (run 'jtop' for monitoring)"
    fi

    # Summary
    echo ""
    if $verification_failed; then
        print_warning "Verification completed with some issues"
        return 1
    else
        print_success "All verification tests passed!"
        return 0
    fi
}

#############################################################################
# Desktop Integration
#############################################################################

create_desktop_shortcuts() {
    print_header "Creating Desktop Shortcuts"

    local desktop_dir="$HOME/Desktop"

    if [ ! -d "$desktop_dir" ]; then
        print_warning "Desktop directory not found, skipping shortcuts"
        return 0
    fi

    # Create GUI launcher
    print_info "Creating GUI launcher..."
    cat > "$desktop_dir/WhoAmI-GUI.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=WhoAmI Face Recognition
Comment=Launch WhoAmI GUI Application
Exec=bash -c "source '$VENV_PATH/bin/activate' && cd '$WORKSPACE_DIR' && python run_gui.py"
Icon=$WORKSPACE_DIR/icon.png
Terminal=false
Categories=Application;Video;
Keywords=face;recognition;camera;ai;
StartupNotify=true
EOF

    chmod +x "$desktop_dir/WhoAmI-GUI.desktop"
    print_success "GUI launcher created"

    # Create CLI launcher
    print_info "Creating CLI launcher..."
    cat > "$desktop_dir/WhoAmI-CLI.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=WhoAmI CLI
Comment=Launch WhoAmI Command Line Interface
Exec=bash -c "source '$VENV_PATH/bin/activate' && cd '$WORKSPACE_DIR' && python run_cli.py; echo ''; read -p 'Press Enter to close...'"
Icon=utilities-terminal
Terminal=true
Categories=System;Terminal;
Keywords=face;recognition;cli;
StartupNotify=true
EOF

    chmod +x "$desktop_dir/WhoAmI-CLI.desktop"
    print_success "CLI launcher created"

    # Create environment activation launcher
    cat > "$desktop_dir/WhoAmI-Terminal.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=WhoAmI Terminal
Comment=Open terminal with WhoAmI environment activated
Exec=bash -c "source '$VENV_PATH/bin/activate' && cd '$WORKSPACE_DIR' && bash"
Icon=utilities-terminal
Terminal=true
Categories=System;Terminal;
StartupNotify=true
EOF

    chmod +x "$desktop_dir/WhoAmI-Terminal.desktop"
    print_success "Terminal launcher created"

    print_success "Desktop shortcuts created in ~/Desktop"
}

#############################################################################
# Post-Installation Instructions
#############################################################################

show_post_install_instructions() {
    print_header "Installation Complete!"

    echo ""
    print_color "$GREEN" "╔════════════════════════════════════════════════════════════╗"
    print_color "$GREEN" "║          WhoAmI Setup Completed Successfully!              ║"
    print_color "$GREEN" "╚════════════════════════════════════════════════════════════╝"
    echo ""

    if [ $ERROR_COUNT -gt 0 ]; then
        print_color "$RED" "⚠ Completed with $ERROR_COUNT error(s) and $WARNING_COUNT warning(s)"
        print_color "$YELLOW" "  Check the log file for details: $LOG_FILE"
        echo ""
    elif [ $WARNING_COUNT -gt 0 ]; then
        print_color "$YELLOW" "⚠ Completed with $WARNING_COUNT warning(s)"
        print_color "$YELLOW" "  Check the log file for details: $LOG_FILE"
        echo ""
    fi

    print_color "$CYAN" "═══════════════════════════════════════════════════════════"
    print_color "$CYAN" "IMPORTANT NEXT STEPS:"
    print_color "$CYAN" "═══════════════════════════════════════════════════════════"
    echo ""

    print_color "$YELLOW" "1. Apply Group Permissions:"
    print_color "$WHITE" "   You MUST logout and login again for group changes to take effect"
    print_color "$WHITE" "   OR run: newgrp video"
    echo ""

    print_color "$YELLOW" "2. Activate Virtual Environment:"
    print_color "$WHITE" "   source $VENV_PATH/bin/activate"
    print_color "$WHITE" "   OR use the helper: source ~/activate_whoami.sh"
    echo ""

    print_color "$YELLOW" "3. Navigate to Project Directory:"
    print_color "$WHITE" "   cd $WORKSPACE_DIR"
    echo ""

    print_color "$YELLOW" "4. Test Camera Connection:"
    print_color "$WHITE" "   python -c 'import depthai as dai; print(f\"Devices: {len(dai.Device.getAllAvailableDevices())}\")')"
    echo ""

    print_color "$YELLOW" "5. Run WhoAmI Application:"
    print_color "$WHITE" "   python run_gui.py    # Launch GUI"
    print_color "$WHITE" "   python run_cli.py    # Launch CLI"
    echo ""

    if [ -d "$HOME/Desktop" ]; then
        print_color "$BLUE" "Desktop shortcuts have been created for easy access!"
        echo ""
    fi

    print_color "$CYAN" "═══════════════════════════════════════════════════════════"
    print_color "$CYAN" "USEFUL COMMANDS:"
    print_color "$CYAN" "═══════════════════════════════════════════════════════════"
    echo ""

    print_color "$WHITE" "  Check camera:         lsusb | grep 03e7"
    print_color "$WHITE" "  Check serial ports:   ls -la /dev/ttyUSB* /dev/ttyTHS*"
    print_color "$WHITE" "  Monitor system:       jtop  (if jetson-stats installed)"
    print_color "$WHITE" "  Check groups:         groups"
    print_color "$WHITE" "  View logs:            cat $LOG_FILE"
    print_color "$WHITE" "  Test installation:    python verify_install.py"
    echo ""

    print_color "$CYAN" "═══════════════════════════════════════════════════════════"
    print_color "$CYAN" "CONFIGURATION:"
    print_color "$CYAN" "═══════════════════════════════════════════════════════════"
    echo ""

    print_color "$WHITE" "  Main config:          config.json"
    print_color "$WHITE" "  Gimbal config:        config/gimbal_config.json"
    print_color "$WHITE" "  Robot brain config:   config/brain_config.json"
    print_color "$WHITE" "  Data directory:       data/"
    print_color "$WHITE" "  Logs directory:       logs/"
    echo ""

    print_color "$CYAN" "═══════════════════════════════════════════════════════════"
    print_color "$CYAN" "TROUBLESHOOTING:"
    print_color "$CYAN" "═══════════════════════════════════════════════════════════"
    echo ""

    print_color "$WHITE" "  Camera not detected:"
    print_color "$WHITE" "    - Check USB connection: lsusb | grep 03e7"
    print_color "$WHITE" "    - Try different USB port (USB 3.0 preferred)"
    print_color "$WHITE" "    - Check permissions: groups | grep video"
    print_color "$WHITE" "    - Reload udev: sudo udevadm trigger"
    echo ""

    print_color "$WHITE" "  Import errors:"
    print_color "$WHITE" "    - Ensure virtual environment is activated"
    print_color "$WHITE" "    - Reinstall package: pip install <package-name>"
    print_color "$WHITE" "    - Check Python version: python --version"
    echo ""

    print_color "$WHITE" "  Gimbal not responding:"
    print_color "$WHITE" "    - Check serial connection: ls /dev/ttyUSB* /dev/ttyTHS*"
    print_color "$WHITE" "    - Check permissions: groups | grep dialout"
    print_color "$WHITE" "    - Verify config: config/gimbal_config.json"
    echo ""

    print_color "$MAGENTA" "═══════════════════════════════════════════════════════════"
    print_color "$MAGENTA" "For more help, visit:"
    print_color "$MAGENTA" "  GitHub: https://github.com/yourusername/whoami"
    print_color "$MAGENTA" "  Docs: $WORKSPACE_DIR/README.md"
    print_color "$MAGENTA" "═══════════════════════════════════════════════════════════"
    echo ""

    print_color "$GREEN" "Setup log saved to: $LOG_FILE"
    echo ""
}

#############################################################################
# Main Installation Flow
#############################################################################

run_full_installation() {
    print_color "$CYAN" "╔════════════════════════════════════════════════════════════╗"
    print_color "$CYAN" "║     Jetson Orin Nano Setup Script for WhoAmI v$VERSION       ║"
    print_color "$CYAN" "╚════════════════════════════════════════════════════════════╝"
    echo ""

    log "INFO" "Starting full installation"
    log "INFO" "Installation options: Full=$INSTALL_FULL, 3D=$INSTALL_3D_SCANNING, Gimbal=$INSTALL_GIMBAL"

    # Pre-flight checks
    check_permissions
    check_jetson
    check_disk_space || {
        print_error "Pre-flight checks failed"
        exit 1
    }

    # System configuration
    configure_performance

    # Install dependencies
    install_system_deps || {
        print_error "System dependencies installation failed"
        exit 1
    }

    # Setup hardware
    setup_camera_permissions

    # Setup Python environment
    setup_python_env || {
        print_error "Python environment setup failed"
        exit 1
    }

    # Install Python packages
    install_python_packages || {
        print_error "Python package installation failed"
        exit 1
    }

    # Setup application
    setup_whoami_app

    # Verify installation
    verify_installation

    # Create desktop shortcuts
    create_desktop_shortcuts

    # Show final instructions
    show_post_install_instructions

    # Offer reboot
    echo ""
    read -p "$(print_color "$YELLOW" "Would you like to reboot now? (recommended) (y/n): ")" -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Rebooting in 5 seconds... (Ctrl+C to cancel)"
        sleep 5
        sudo reboot
    else
        print_info "Remember to logout and login for group changes to take effect"
    fi
}

#############################################################################
# Command-line Argument Parsing
#############################################################################

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --full)
                INSTALL_FULL=true
                INSTALL_3D_SCANNING=true
                INSTALL_GIMBAL=true
                shift
                ;;
            --minimal)
                INSTALL_FULL=false
                INSTALL_3D_SCANNING=false
                INSTALL_GIMBAL=false
                shift
                ;;
            --camera-only)
                check_permissions
                setup_camera_permissions
                exit 0
                ;;
            --python-only)
                check_permissions
                setup_python_env
                install_python_packages
                exit 0
                ;;
            --verify-only)
                source "$VENV_PATH/bin/activate" 2>/dev/null || true
                verify_installation
                exit $?
                ;;
            --skip-performance)
                SKIP_PERFORMANCE=true
                shift
                ;;
            --help|-h)
                print_color "$CYAN" "WhoAmI Jetson Orin Nano Setup Script v$VERSION"
                echo ""
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --full              Complete installation with all features (default)"
                echo "  --minimal           Minimal installation (no 3D scanning, no gimbal)"
                echo "  --camera-only       Setup camera permissions only"
                echo "  --python-only       Setup Python environment and packages only"
                echo "  --verify-only       Run verification tests only"
                echo "  --skip-performance  Skip Jetson performance configuration"
                echo "  --help, -h          Show this help message"
                echo ""
                echo "Examples:"
                echo "  $0                  # Full installation"
                echo "  $0 --minimal        # Minimal installation"
                echo "  $0 --camera-only    # Just setup camera permissions"
                echo "  $0 --verify-only    # Just run verification"
                echo ""
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
}

#############################################################################
# Script Entry Point
#############################################################################

main() {
    # Start logging
    log "INFO" "========================================="
    log "INFO" "WhoAmI Setup Script v$VERSION"
    log "INFO" "Started at: $(date)"
    log "INFO" "User: $USER"
    log "INFO" "Platform: $(uname -m)"
    log "INFO" "========================================="

    # Parse command line arguments
    parse_arguments "$@"

    # Run full installation
    run_full_installation

    # Log completion
    log "INFO" "========================================="
    log "INFO" "Setup completed at: $(date)"
    log "INFO" "Errors: $ERROR_COUNT, Warnings: $WARNING_COUNT"
    log "INFO" "========================================="
}

# Run main function with all arguments
main "$@"
