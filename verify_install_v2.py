#!/usr/bin/env python3
"""
Enhanced Installation Verification Script for WhoAmI
Comprehensive checking of all system components including:
- Python environment
- Core dependencies
- Optional features (3D scanning, gimbal control)
- Hardware detection
- Configuration files
"""

import sys
import os
import importlib
from pathlib import Path

# ANSI color codes
class Colors:
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    WHITE = '\033[1;37m'
    NC = '\033[0m'  # No Color

def print_color(color, text):
    """Print colored text"""
    print(f"{color}{text}{Colors.NC}")

def print_header(text):
    """Print section header"""
    print()
    print_color(Colors.CYAN, "=" * 60)
    print_color(Colors.CYAN, text)
    print_color(Colors.CYAN, "=" * 60)
    print()

def check_module(module_name, package_name=None, optional=False):
    """Check if a module can be imported"""
    display_name = package_name or module_name
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, '__version__', 'unknown')

        # Get version info for specific packages
        if module_name == 'cv2':
            version = mod.__version__
        elif module_name == 'numpy':
            version = mod.__version__
        elif module_name == 'depthai':
            version = mod.__version__

        status = "✓" if not optional else "✓"
        color = Colors.GREEN if not optional else Colors.BLUE
        print_color(color, f"  {status} {display_name}: {version}")
        return True
    except ImportError as e:
        if optional:
            print_color(Colors.YELLOW, f"  ⚠ {display_name}: Not installed (optional)")
        else:
            print_color(Colors.RED, f"  ✗ {display_name}: {e}")
        return not optional  # Return True for optional packages

def check_python_version():
    """Check Python version"""
    print_header("1. Python Environment")

    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    if version >= (3, 8):
        print_color(Colors.GREEN, f"  ✓ Python {version_str}")
        print_color(Colors.BLUE, f"    Path: {sys.executable}")

        # Check if in virtual environment
        in_venv = sys.prefix != sys.base_prefix
        if in_venv:
            print_color(Colors.GREEN, f"  ✓ Virtual environment: {sys.prefix}")
        else:
            print_color(Colors.YELLOW, "  ⚠ Not in virtual environment")

        return True
    else:
        print_color(Colors.RED, f"  ✗ Python {version_str}")
        print_color(Colors.RED, "    Requires Python 3.8 or higher")
        return False

def check_core_dependencies():
    """Check core dependencies"""
    print_header("2. Core Dependencies")

    dependencies = [
        ('depthai', 'depthai - OAK-D Camera SDK', False),
        ('cv2', 'opencv-python - Computer Vision', False),
        ('numpy', 'numpy - Numerical Computing', False),
        ('PIL', 'Pillow - Image Processing', False),
    ]

    all_ok = True
    for module, package, optional in dependencies:
        if not check_module(module, package, optional):
            all_ok = False

    return all_ok

def check_face_recognition():
    """Check face recognition dependencies"""
    print_header("3. Face Recognition")

    dependencies = [
        ('face_recognition', 'face-recognition - Face Detection & Encoding', False),
        ('dlib', 'dlib - Machine Learning Library', False),
    ]

    all_ok = True
    for module, package, optional in dependencies:
        if not check_module(module, package, optional):
            all_ok = False

    # Test face recognition functionality
    if all_ok:
        try:
            import face_recognition
            import numpy as np

            # Create a dummy image and test encoding
            dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
            encodings = face_recognition.face_encodings(dummy_image)
            print_color(Colors.GREEN, "  ✓ Face recognition functionality verified")
        except Exception as e:
            print_color(Colors.YELLOW, f"  ⚠ Face recognition test failed: {e}")
            all_ok = False

    return all_ok

def check_machine_learning():
    """Check machine learning dependencies"""
    print_header("4. Machine Learning & AI")

    dependencies = [
        ('sklearn', 'scikit-learn - ML Algorithms', False),
        ('cryptography', 'cryptography - Encrypted Storage', False),
    ]

    all_ok = True
    for module, package, optional in dependencies:
        if not check_module(module, package, optional):
            all_ok = False

    return all_ok

def check_3d_scanning():
    """Check 3D scanning dependencies"""
    print_header("5. 3D Scanning (Optional)")

    dependencies = [
        ('open3d', 'Open3D - Point Cloud & Mesh Processing', True),
        ('trimesh', 'trimesh - Advanced Mesh Operations', True),
    ]

    all_ok = True
    any_installed = False

    for module, package, optional in dependencies:
        result = check_module(module, package, optional)
        if result and module in ['open3d', 'trimesh']:
            any_installed = True

    if any_installed:
        print_color(Colors.GREEN, "  ✓ 3D scanning features available")
    else:
        print_color(Colors.YELLOW, "  ⚠ 3D scanning features not available")

    return True  # Optional, so always return True

def check_gimbal_control():
    """Check gimbal control dependencies"""
    print_header("6. Gimbal Control (Optional)")

    dependencies = [
        ('serial', 'pyserial - Serial Communication', True),
    ]

    all_ok = True
    for module, package, optional in dependencies:
        if not check_module(module, package, optional):
            all_ok = False

    # Check for serial ports
    import glob
    serial_ports = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyTHS*')

    if serial_ports:
        print_color(Colors.GREEN, f"  ✓ Found {len(serial_ports)} serial port(s):")
        for port in serial_ports:
            print_color(Colors.WHITE, f"    - {port}")
    else:
        print_color(Colors.YELLOW, "  ⚠ No serial ports detected (gimbal may not be connected)")

    return True  # Optional, so always return True

def check_gui_support():
    """Check GUI dependencies"""
    print_header("7. GUI Support")

    all_ok = True

    # Check tkinter
    try:
        import tkinter
        print_color(Colors.GREEN, f"  ✓ tkinter: {tkinter.TkVersion}")

        # Test GUI creation (headless safe)
        try:
            root = tkinter.Tk()
            root.withdraw()  # Hide window
            root.destroy()
            print_color(Colors.GREEN, "  ✓ GUI functionality verified")
        except tkinter.TclError:
            print_color(Colors.YELLOW, "  ⚠ GUI creation failed (headless environment?)")
    except ImportError:
        print_color(Colors.RED, "  ✗ tkinter: Not installed")
        print_color(Colors.YELLOW, "    Install with: sudo apt install python3-tk")
        all_ok = False

    return all_ok

def check_whoami_package():
    """Check WhoAmI package installation"""
    print_header("8. WhoAmI Package")

    whoami_modules = [
        ('whoami', 'Main package'),
        ('whoami.face_recognition_api', 'Face Recognition API'),
        ('whoami.config', 'Configuration'),
    ]

    optional_modules = [
        ('whoami.gui', 'GUI Module'),
        ('whoami.cli', 'CLI Module'),
        ('whoami.gimbal_control', 'Gimbal Control'),
        ('whoami.scanner_3d', '3D Scanner'),
        ('whoami.robot_brain', 'Robot Brain'),
    ]

    # Check core modules
    all_ok = True
    for module, description in whoami_modules:
        try:
            importlib.import_module(module)
            print_color(Colors.GREEN, f"  ✓ {description}")
        except ImportError as e:
            print_color(Colors.RED, f"  ✗ {description}: {e}")
            all_ok = False

    # Check optional modules
    print()
    print_color(Colors.BLUE, "  Optional modules:")
    for module, description in optional_modules:
        try:
            importlib.import_module(module)
            print_color(Colors.GREEN, f"    ✓ {description}")
        except ImportError:
            print_color(Colors.YELLOW, f"    ⚠ {description}: Not available")

    return all_ok

def check_camera_hardware():
    """Check camera hardware detection"""
    print_header("9. Camera Hardware Detection")

    try:
        import depthai as dai

        # Get all available devices
        devices = dai.Device.getAllAvailableDevices()

        if len(devices) > 0:
            print_color(Colors.GREEN, f"  ✓ Found {len(devices)} OAK-D device(s):")
            for i, device in enumerate(devices, 1):
                mxid = device.getMxId()
                state = device.state.name
                print_color(Colors.WHITE, f"    {i}. MxID: {mxid[:16]}... [State: {state}]")

            # Try to initialize a device
            try:
                with dai.Device() as device:
                    print_color(Colors.GREEN, f"  ✓ Successfully connected to device")

                    # Get device info
                    calib = device.readCalibration()
                    eeprom = calib.getEepromData()
                    print_color(Colors.BLUE, f"    Board: {eeprom.boardName}")
                    print_color(Colors.BLUE, f"    Product: {eeprom.productName}")

            except Exception as e:
                print_color(Colors.YELLOW, f"  ⚠ Device connection test failed: {e}")
        else:
            print_color(Colors.YELLOW, "  ⚠ No OAK-D cameras detected")
            print_color(Colors.WHITE, "    Make sure your OAK-D camera is:")
            print_color(Colors.WHITE, "    1. Connected via USB")
            print_color(Colors.WHITE, "    2. Using a USB 3.0 port (blue port)")
            print_color(Colors.WHITE, "    3. Using a good quality USB cable")
            return False

    except ImportError:
        print_color(Colors.RED, "  ✗ DepthAI not installed")
        return False
    except Exception as e:
        print_color(Colors.RED, f"  ✗ Camera detection error: {e}")
        return False

    return True

def check_user_groups():
    """Check user group memberships"""
    print_header("10. User Permissions")

    import subprocess

    try:
        # Get user groups
        groups_output = subprocess.check_output(['groups'], text=True).strip()
        user_groups = groups_output.split()

        required_groups = {
            'video': 'Camera access',
            'dialout': 'Serial port access (gimbal)',
            'plugdev': 'USB device access',
        }

        all_ok = True
        for group, description in required_groups.items():
            if group in user_groups:
                print_color(Colors.GREEN, f"  ✓ User in '{group}' group ({description})")
            else:
                print_color(Colors.YELLOW, f"  ⚠ User NOT in '{group}' group ({description})")
                print_color(Colors.WHITE, f"    Add with: sudo usermod -a -G {group} $USER")
                print_color(Colors.WHITE, f"    Then logout and login")
                all_ok = False

        return all_ok
    except Exception as e:
        print_color(Colors.RED, f"  ✗ Error checking groups: {e}")
        return False

def check_configuration_files():
    """Check configuration files"""
    print_header("11. Configuration Files")

    config_files = [
        ('config.json', 'Main configuration', False),
        ('config/gimbal_config.json', 'Gimbal configuration', True),
        ('config/brain_config.json', 'Robot brain configuration', True),
        ('requirements.txt', 'Python dependencies list', False),
    ]

    all_ok = True
    for file_path, description, optional in config_files:
        full_path = Path(file_path)
        if full_path.exists():
            size = full_path.stat().st_size
            print_color(Colors.GREEN, f"  ✓ {description}: {file_path} ({size} bytes)")
        else:
            if optional:
                print_color(Colors.YELLOW, f"  ⚠ {description}: {file_path} (not found, optional)")
            else:
                print_color(Colors.RED, f"  ✗ {description}: {file_path} (not found)")
                all_ok = False

    # Check data directories
    print()
    print_color(Colors.BLUE, "  Data directories:")
    data_dirs = ['data', 'data/faces', 'data/scans', 'logs']
    for dir_path in data_dirs:
        if Path(dir_path).exists():
            print_color(Colors.GREEN, f"    ✓ {dir_path}/")
        else:
            print_color(Colors.YELLOW, f"    ⚠ {dir_path}/ (not found)")

    return all_ok

def check_system_info():
    """Display system information"""
    print_header("12. System Information")

    import platform
    import subprocess

    # Platform info
    print_color(Colors.BLUE, f"  Platform: {platform.system()} {platform.machine()}")
    print_color(Colors.BLUE, f"  Kernel: {platform.release()}")

    # Memory info
    try:
        mem_output = subprocess.check_output(['free', '-h'], text=True)
        mem_lines = mem_output.strip().split('\n')
        if len(mem_lines) > 1:
            mem_parts = mem_lines[1].split()
            if len(mem_parts) >= 2:
                print_color(Colors.BLUE, f"  Total Memory: {mem_parts[1]}")
    except:
        pass

    # Disk space
    try:
        df_output = subprocess.check_output(['df', '-h', str(Path.home())], text=True)
        df_lines = df_output.strip().split('\n')
        if len(df_lines) > 1:
            df_parts = df_lines[1].split()
            if len(df_parts) >= 4:
                print_color(Colors.BLUE, f"  Disk Available: {df_parts[3]}")
    except:
        pass

    # Check for Jetson-specific info
    if Path('/proc/device-tree/model').exists():
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read().strip('\x00')
                if 'Jetson' in model or 'NVIDIA' in model:
                    print_color(Colors.GREEN, f"  ✓ Jetson Device: {model}")
        except:
            pass

    # Check for jetson_clocks
    try:
        import subprocess
        result = subprocess.run(['jetson_clocks', '--show'],
                              capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            print_color(Colors.GREEN, "  ✓ Jetson performance tools available")
    except:
        pass

def print_summary(results):
    """Print verification summary"""
    print_header("Verification Summary")

    total_checks = len(results)
    passed_checks = sum(1 for r in results.values() if r)
    failed_checks = total_checks - passed_checks

    if failed_checks == 0:
        print_color(Colors.GREEN, "╔════════════════════════════════════════════════════════╗")
        print_color(Colors.GREEN, "║          ✅ ALL CHECKS PASSED!                          ║")
        print_color(Colors.GREEN, "║          WhoAmI is ready to use!                       ║")
        print_color(Colors.GREEN, "╚════════════════════════════════════════════════════════╝")
        print()
        print_color(Colors.CYAN, "Next steps:")
        print_color(Colors.WHITE, "  1. Connect your OAK-D camera (if not already connected)")
        print_color(Colors.WHITE, "  2. Run the GUI: python run_gui.py")
        print_color(Colors.WHITE, "  3. Or CLI: python run_cli.py --help")
        print()
    else:
        print_color(Colors.YELLOW, "╔════════════════════════════════════════════════════════╗")
        print_color(Colors.YELLOW, f"║     ⚠ {passed_checks}/{total_checks} checks passed ({failed_checks} issues found)          ║")
        print_color(Colors.YELLOW, "╚════════════════════════════════════════════════════════╝")
        print()
        print_color(Colors.CYAN, "Issues found:")
        for check_name, result in results.items():
            if not result:
                print_color(Colors.RED, f"  ✗ {check_name}")
        print()
        print_color(Colors.CYAN, "To fix issues:")
        print_color(Colors.WHITE, "  1. Review the output above for specific errors")
        print_color(Colors.WHITE, "  2. Run setup script: ./jetson_setup_v2.sh")
        print_color(Colors.WHITE, "  3. Install missing packages: pip install -r requirements.txt")
        print()

    return failed_checks == 0

def main():
    """Main verification routine"""
    print_color(Colors.CYAN, """
╔════════════════════════════════════════════════════════════╗
║       WhoAmI Installation Verification v2.0                ║
║       Comprehensive System Check                           ║
╚════════════════════════════════════════════════════════════╝
    """)

    # Run all checks
    results = {}

    results['Python Version'] = check_python_version()
    results['Core Dependencies'] = check_core_dependencies()
    results['Face Recognition'] = check_face_recognition()
    results['Machine Learning'] = check_machine_learning()
    results['3D Scanning'] = check_3d_scanning()
    results['Gimbal Control'] = check_gimbal_control()
    results['GUI Support'] = check_gui_support()
    results['WhoAmI Package'] = check_whoami_package()
    results['Camera Hardware'] = check_camera_hardware()
    results['User Permissions'] = check_user_groups()
    results['Configuration Files'] = check_configuration_files()

    # System info (always succeeds)
    check_system_info()

    # Print summary
    success = print_summary(results)

    return 0 if success else 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print()
        print_color(Colors.YELLOW, "Verification interrupted by user")
        sys.exit(1)
    except Exception as e:
        print()
        print_color(Colors.RED, f"Verification failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
