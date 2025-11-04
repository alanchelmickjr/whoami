#!/usr/bin/env python3
"""
Comprehensive OAK-D S3 Camera Test Suite for WhoAmI Face Recognition System

This script provides thorough testing of:
- Camera detection and connection
- Face recognition functionality  
- GUI and CLI modes
- Permission handling
- Platform-specific issues
"""

import sys
import os
import platform
import subprocess
import time
import json
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

# Color output support
try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False
    # Fallback color definitions
    class Fore:
        RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = RESET = ''
    class Style:
        BRIGHT = DIM = RESET_ALL = ''

class CameraTestSuite:
    """Comprehensive test suite for OAK-D S3 camera"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results = {}
        self.platform_info = self.get_platform_info()
        self.test_passed = 0
        self.test_failed = 0
        self.test_skipped = 0
        
    def get_platform_info(self) -> Dict:
        """Gather platform information"""
        info = {
            'system': platform.system(),
            'machine': platform.machine(),
            'platform': platform.platform(),
            'python_version': sys.version,
            'is_jetson': self._is_jetson(),
            'is_mac': platform.system() == 'Darwin',
            'is_m_series': platform.machine() == 'arm64' and platform.system() == 'Darwin'
        }
        return info
    
    def _is_jetson(self) -> bool:
        """Check if running on NVIDIA Jetson"""
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read().lower()
                return 'jetson' in model or 'nvidia' in model
        except:
            return False
    
    def print_header(self, text: str):
        """Print formatted header"""
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN}{text:^60}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    
    def print_test(self, name: str, status: str, details: str = ""):
        """Print test result"""
        status_color = {
            'PASS': Fore.GREEN,
            'FAIL': Fore.RED,
            'SKIP': Fore.YELLOW,
            'WARN': Fore.MAGENTA
        }.get(status, Fore.WHITE)
        
        print(f"  {name:40} [{status_color}{status:^6}{Style.RESET_ALL}] {details}")
    
    def test_python_environment(self) -> bool:
        """Test Python environment and dependencies"""
        self.print_header("Python Environment Test")
        
        all_passed = True
        
        # Check Python version
        py_version = sys.version_info
        if py_version.major == 3 and 8 <= py_version.minor <= 10:
            self.print_test("Python Version", "PASS", f"{py_version.major}.{py_version.minor}.{py_version.micro}")
        else:
            self.print_test("Python Version", "WARN", f"{py_version.major}.{py_version.minor} (3.8-3.10 recommended)")
        
        # Check required packages
        required_packages = {
            'depthai': '2.24.0',
            'cv2': None,  # opencv-python
            'numpy': None,
            'PIL': None,  # Pillow
        }
        
        optional_packages = {
            'face_recognition': '1.3.0',
            'PyQt6': None,
            'depthai_sdk': '1.14.0'
        }
        
        print(f"\n{Fore.BLUE}Required Packages:{Style.RESET_ALL}")
        for package, min_version in required_packages.items():
            try:
                if package == 'cv2':
                    import cv2
                    version = cv2.__version__
                elif package == 'PIL':
                    from PIL import Image
                    import PIL
                    version = PIL.__version__
                else:
                    module = __import__(package)
                    version = getattr(module, '__version__', 'unknown')
                
                self.print_test(f"  {package}", "PASS", f"v{version}")
            except ImportError:
                self.print_test(f"  {package}", "FAIL", "Not installed")
                all_passed = False
        
        print(f"\n{Fore.BLUE}Optional Packages:{Style.RESET_ALL}")
        for package, min_version in optional_packages.items():
            try:
                if package == 'PyQt6':
                    from PyQt6 import QtCore
                    version = QtCore.qVersion()
                elif package == 'depthai_sdk':
                    import depthai_sdk
                    version = getattr(depthai_sdk, '__version__', 'unknown')
                else:
                    module = __import__(package)
                    version = getattr(module, '__version__', 'unknown')
                
                self.print_test(f"  {package}", "PASS", f"v{version}")
            except ImportError:
                self.print_test(f"  {package}", "SKIP", "Not installed")
        
        return all_passed
    
    def test_camera_detection(self) -> Tuple[bool, Optional[str]]:
        """Test camera detection and connection"""
        self.print_header("Camera Detection Test")
        
        try:
            import depthai as dai
            
            # Check for available devices
            devices = dai.Device.getAllAvailableDevices()
            
            if len(devices) == 0:
                self.print_test("Device Detection", "FAIL", "No OAK devices found")
                
                # Platform-specific troubleshooting
                if self.platform_info['is_mac']:
                    print(f"\n{Fore.YELLOW}macOS Troubleshooting:")
                    print("  1. Check USB connection")
                    print("  2. Try running with sudo: sudo python test_oak_camera_full.py")
                    print("  3. Grant camera permissions in System Settings")
                elif self.platform_info['is_jetson']:
                    print(f"\n{Fore.YELLOW}Jetson Troubleshooting:")
                    print("  1. Check udev rules: ls -la /etc/udev/rules.d/*movidius*")
                    print("  2. Verify user groups: groups $USER")
                    print("  3. Try: sudo udevadm control --reload-rules && sudo udevadm trigger")
                
                return False, None
            
            self.print_test("Device Detection", "PASS", f"Found {len(devices)} device(s)")
            
            # List all devices
            mx_id = None
            for i, device in enumerate(devices):
                state = device.state.name
                mx_id = device.getMxId()
                self.print_test(f"  Device {i+1}", "INFO", f"ID: {mx_id} [{state}]")
            
            # Try to connect to first device
            print(f"\n{Fore.BLUE}Connection Test:{Style.RESET_ALL}")
            try:
                with dai.Device() as device:
                    device_name = device.getDeviceName()
                    device_mx_id = device.getMxId()
                    
                    # Get device info
                    calibration = device.readCalibration()
                    has_imu = device.getConnectedIMU() is not None
                    
                    self.print_test("Device Connection", "PASS", f"Connected to {device_name}")
                    self.print_test("  Device ID", "INFO", device_mx_id)
                    self.print_test("  Calibration", "PASS" if calibration else "WARN", "")
                    self.print_test("  IMU Available", "INFO", "Yes" if has_imu else "No")
                    
                    return True, device_mx_id
                    
            except PermissionError as e:
                self.print_test("Device Connection", "FAIL", "Permission denied")
                
                if self.platform_info['is_mac']:
                    print(f"\n{Fore.YELLOW}macOS Permission Fix:")
                    print("  Run with sudo: sudo python test_oak_camera_full.py")
                    print("  Or grant camera access in System Settings > Privacy & Security > Camera")
                else:
                    print(f"\n{Fore.YELLOW}Linux Permission Fix:")
                    print("  Add user to video group: sudo usermod -a -G video $USER")
                    print("  Then logout and login again")
                
                return False, None
                
        except ImportError:
            self.print_test("DepthAI Import", "FAIL", "DepthAI not installed")
            print(f"\n{Fore.YELLOW}Install with: pip install depthai==2.24.0")
            return False, None
        except Exception as e:
            self.print_test("Camera Detection", "FAIL", str(e))
            if self.verbose:
                traceback.print_exc()
            return False, None
    
    def test_camera_stream(self) -> bool:
        """Test camera streaming capabilities"""
        self.print_header("Camera Stream Test")
        
        try:
            import depthai as dai
            import cv2
            import numpy as np
            
            # Create pipeline
            pipeline = dai.Pipeline()
            
            # Define sources
            camRgb = pipeline.create(dai.node.ColorCamera)
            xoutRgb = pipeline.create(dai.node.XLinkOut)
            xoutRgb.setStreamName("rgb")
            
            # Configure camera
            camRgb.setPreviewSize(640, 480)
            camRgb.setInterleaved(False)
            camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
            camRgb.setFps(30)
            
            # Link
            camRgb.preview.link(xoutRgb.input)
            
            # Connect and test
            with dai.Device(pipeline) as device:
                q = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
                
                self.print_test("Pipeline Creation", "PASS", "")
                self.print_test("Queue Setup", "PASS", "")
                
                # Capture frames for 2 seconds
                start_time = time.time()
                frame_count = 0
                frame_times = []
                
                print(f"\n{Fore.BLUE}Capturing frames for 2 seconds...{Style.RESET_ALL}")
                
                while time.time() - start_time < 2.0:
                    if q.has():
                        frame_start = time.time()
                        frame = q.get().getCvFrame()
                        frame_times.append(time.time() - frame_start)
                        frame_count += 1
                
                # Calculate statistics
                if frame_count > 0:
                    avg_fps = frame_count / 2.0
                    avg_latency = sum(frame_times) / len(frame_times) * 1000
                    
                    self.print_test("Frame Capture", "PASS", f"{frame_count} frames")
                    self.print_test("Average FPS", "PASS" if avg_fps > 20 else "WARN", f"{avg_fps:.1f} fps")
                    self.print_test("Average Latency", "PASS" if avg_latency < 50 else "WARN", f"{avg_latency:.1f} ms")
                    
                    return True
                else:
                    self.print_test("Frame Capture", "FAIL", "No frames received")
                    return False
                    
        except Exception as e:
            self.print_test("Camera Stream", "FAIL", str(e))
            if self.verbose:
                traceback.print_exc()
            return False
    
    def test_face_detection(self) -> bool:
        """Test face detection capabilities"""
        self.print_header("Face Detection Test")
        
        try:
            import depthai as dai
            import cv2
            import numpy as np
            
            # Create pipeline with face detection
            pipeline = dai.Pipeline()
            
            # Camera
            camRgb = pipeline.create(dai.node.ColorCamera)
            camRgb.setPreviewSize(640, 480)
            camRgb.setInterleaved(False)
            
            # Face detection using MobileNet
            faceDetection = pipeline.create(dai.node.MobileNetDetectionNetwork)
            faceDetection.setConfidenceThreshold(0.5)
            
            # Use a simple blob for testing (we'll check if model exists)
            model_path = Path("models/face-detection-retail-0004.blob")
            if model_path.exists():
                faceDetection.setBlobPath(str(model_path))
                self.print_test("Face Detection Model", "PASS", "Model loaded")
            else:
                self.print_test("Face Detection Model", "SKIP", "Model not found")
                print(f"{Fore.YELLOW}  Download models from: https://github.com/luxonis/depthai-model-zoo")
                return False
            
            # Link nodes
            camRgb.preview.link(faceDetection.input)
            
            # Outputs
            xoutRgb = pipeline.create(dai.node.XLinkOut)
            xoutRgb.setStreamName("rgb")
            camRgb.preview.link(xoutRgb.input)
            
            xoutDetection = pipeline.create(dai.node.XLinkOut)
            xoutDetection.setStreamName("detections")
            faceDetection.out.link(xoutDetection.input)
            
            # Test pipeline
            with dai.Device(pipeline) as device:
                qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
                qDet = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
                
                self.print_test("Detection Pipeline", "PASS", "")
                
                # Process frames for 3 seconds
                start_time = time.time()
                detections_count = 0
                
                print(f"\n{Fore.BLUE}Looking for faces for 3 seconds...{Style.RESET_ALL}")
                
                while time.time() - start_time < 3.0:
                    if qDet.has():
                        detections = qDet.get().detections
                        detections_count += len(detections)
                
                if detections_count > 0:
                    self.print_test("Face Detection", "PASS", f"{detections_count} face(s) detected")
                else:
                    self.print_test("Face Detection", "WARN", "No faces detected (try showing your face)")
                
                return True
                
        except Exception as e:
            self.print_test("Face Detection", "FAIL", str(e))
            if self.verbose:
                traceback.print_exc()
            return False
    
    def test_whoami_cli(self) -> bool:
        """Test WhoAmI CLI mode"""
        self.print_header("WhoAmI CLI Mode Test")
        
        try:
            # Check if CLI script exists
            cli_script = Path("run_cli.py")
            if not cli_script.exists():
                self.print_test("CLI Script", "FAIL", "run_cli.py not found")
                return False
            
            self.print_test("CLI Script", "PASS", "Found")
            
            # Try to import and test basic functionality
            try:
                # Test import
                import whoami.cli
                self.print_test("CLI Module Import", "PASS", "")
                
                # Test with --help flag
                result = subprocess.run(
                    [sys.executable, "run_cli.py", "--help"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.returncode == 0:
                    self.print_test("CLI Help Command", "PASS", "")
                else:
                    self.print_test("CLI Help Command", "FAIL", result.stderr[:100])
                    return False
                
                # Test with --list-cameras
                result = subprocess.run(
                    [sys.executable, "run_cli.py", "--list-cameras"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if "No cameras found" in result.stdout or "Found" in result.stdout:
                    self.print_test("CLI Camera List", "PASS", "")
                else:
                    self.print_test("CLI Camera List", "WARN", "Unexpected output")
                
                return True
                
            except ImportError as e:
                self.print_test("CLI Module Import", "FAIL", str(e))
                return False
            except subprocess.TimeoutExpired:
                self.print_test("CLI Test", "FAIL", "Command timed out")
                return False
                
        except Exception as e:
            self.print_test("CLI Mode Test", "FAIL", str(e))
            if self.verbose:
                traceback.print_exc()
            return False
    
    def test_whoami_gui(self) -> bool:
        """Test WhoAmI GUI mode (basic import test)"""
        self.print_header("WhoAmI GUI Mode Test")
        
        try:
            # Check if GUI script exists
            gui_script = Path("run_gui.py")
            if not gui_script.exists():
                self.print_test("GUI Script", "FAIL", "run_gui.py not found")
                return False
            
            self.print_test("GUI Script", "PASS", "Found")
            
            # Test PyQt6 availability
            try:
                from PyQt6 import QtCore, QtWidgets
                self.print_test("PyQt6 Import", "PASS", f"v{QtCore.qVersion()}")
            except ImportError:
                self.print_test("PyQt6 Import", "FAIL", "PyQt6 not installed")
                print(f"{Fore.YELLOW}  Install with: pip install PyQt6")
                return False
            
            # Test GUI module import
            try:
                import whoami.gui
                self.print_test("GUI Module Import", "PASS", "")
                
                # Check if running headless
                if os.environ.get('DISPLAY') is None and not self.platform_info['is_mac']:
                    self.print_test("Display Available", "SKIP", "No display (headless)")
                    return True
                
                return True
                
            except ImportError as e:
                self.print_test("GUI Module Import", "FAIL", str(e))
                return False
                
        except Exception as e:
            self.print_test("GUI Mode Test", "FAIL", str(e))
            if self.verbose:
                traceback.print_exc()
            return False
    
    def test_permissions(self) -> bool:
        """Test and diagnose permission issues"""
        self.print_header("Permission Diagnostics")
        
        all_good = True
        
        if self.platform_info['is_mac']:
            # macOS specific checks
            print(f"\n{Fore.BLUE}macOS Permission Checks:{Style.RESET_ALL}")
            
            # Check if running with sudo
            if os.geteuid() == 0:
                self.print_test("Running as root", "WARN", "Currently using sudo")
            else:
                self.print_test("Running as user", "INFO", f"UID: {os.getuid()}")
            
            # Check camera permissions (approximation)
            try:
                import subprocess
                result = subprocess.run(
                    ["tccutil", "check", "Camera"],
                    capture_output=True,
                    text=True
                )
                # This might not work, but we try
                self.print_test("Camera Permission", "INFO", "Check System Settings > Privacy")
            except:
                self.print_test("Camera Permission", "INFO", "Manual check required")
            
        elif self.platform_info['is_jetson'] or platform.system() == 'Linux':
            # Linux/Jetson specific checks
            print(f"\n{Fore.BLUE}Linux Permission Checks:{Style.RESET_ALL}")
            
            # Check user groups
            import grp
            import pwd
            
            username = pwd.getpwuid(os.getuid()).pw_name
            groups = [grp.getgrgid(g).gr_name for g in os.getgroups()]
            
            required_groups = ['video', 'dialout']
            for group in required_groups:
                if group in groups:
                    self.print_test(f"Group '{group}'", "PASS", "User in group")
                else:
                    self.print_test(f"Group '{group}'", "FAIL", "User not in group")
                    print(f"{Fore.YELLOW}    Fix: sudo usermod -a -G {group} $USER")
                    all_good = False
            
            # Check udev rules
            udev_file = Path("/etc/udev/rules.d/80-movidius.rules")
            if udev_file.exists():
                self.print_test("Udev Rules", "PASS", "Rules file exists")
            else:
                self.print_test("Udev Rules", "FAIL", "Rules file missing")
                print(f"{Fore.YELLOW}    Fix: echo 'SUBSYSTEM==\"usb\", ATTRS{{idVendor}}==\"03e7\", MODE=\"0666\"' | sudo tee {udev_file}")
                all_good = False
            
            # Check USB devices
            try:
                result = subprocess.run(
                    ["lsusb"],
                    capture_output=True,
                    text=True
                )
                if "03e7" in result.stdout:
                    self.print_test("USB Device", "PASS", "Movidius device detected")
                else:
                    self.print_test("USB Device", "WARN", "No Movidius device in lsusb")
            except:
                self.print_test("USB Device", "SKIP", "lsusb not available")
        
        return all_good
    
    def run_comprehensive_test(self) -> None:
        """Run all tests in sequence"""
        print(f"{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN}{'OAK-D S3 Camera Comprehensive Test Suite':^60}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        
        # Display platform info
        print(f"\n{Fore.BLUE}Platform Information:{Style.RESET_ALL}")
        print(f"  System: {self.platform_info['system']}")
        print(f"  Machine: {self.platform_info['machine']}")
        print(f"  Python: {sys.version.split()[0]}")
        if self.platform_info['is_jetson']:
            print(f"  Device: {Fore.GREEN}NVIDIA Jetson Detected{Style.RESET_ALL}")
        elif self.platform_info['is_m_series']:
            print(f"  Device: {Fore.GREEN}Apple M-Series Mac{Style.RESET_ALL}")
        
        # Run tests
        tests = [
            ("Python Environment", self.test_python_environment),
            ("Camera Detection", lambda: self.test_camera_detection()[0]),
            ("Camera Stream", self.test_camera_stream),
            ("Face Detection", self.test_face_detection),
            ("WhoAmI CLI", self.test_whoami_cli),
            ("WhoAmI GUI", self.test_whoami_gui),
            ("Permissions", self.test_permissions),
        ]
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                if result:
                    self.test_passed += 1
                else:
                    self.test_failed += 1
                self.results[test_name] = result
            except Exception as e:
                self.test_failed += 1
                self.results[test_name] = False
                print(f"\n{Fore.RED}Test '{test_name}' crashed: {e}{Style.RESET_ALL}")
                if self.verbose:
                    traceback.print_exc()
        
        # Summary
        self.print_summary()
    
    def print_summary(self) -> None:
        """Print test summary"""
        self.print_header("Test Summary")
        
        total_tests = self.test_passed + self.test_failed + self.test_skipped
        
        print(f"\n{Fore.BLUE}Results:{Style.RESET_ALL}")
        print(f"  Total Tests: {total_tests}")
        print(f"  {Fore.GREEN}Passed: {self.test_passed}{Style.RESET_ALL}")
        print(f"  {Fore.RED}Failed: {self.test_failed}{Style.RESET_ALL}")
        print(f"  {Fore.YELLOW}Skipped: {self.test_skipped}{Style.RESET_ALL}")
        
        if self.test_failed > 0:
            print(f"\n{Fore.RED}Some tests failed. Review the output above for details.{Style.RESET_ALL}")
            
            # Platform-specific recommendations
            if self.platform_info['is_mac']:
                print(f"\n{Fore.YELLOW}macOS Recommendations:")
                print("  1. Try running with sudo if permission issues persist")
                print("  2. Check System Settings > Privacy & Security > Camera")
                print("  3. Ensure OAK-D S3 is properly connected via USB")
            elif self.platform_info['is_jetson']:
                print(f"\n{Fore.YELLOW}Jetson Recommendations:")
                print("  1. Run jetson_setup.sh for automated setup")
                print("  2. Ensure user is in video and dialout groups")
                print("  3. Check power mode: sudo nvpmodel -q")
        else:
            print(f"\n{Fore.GREEN}All tests passed! Your system is ready for WhoAmI.{Style.RESET_ALL}")
        
        # Save results to file
        results_file = Path("test_results.json")
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'platform': self.platform_info,
                'results': self.results,
                'summary': {
                    'passed': self.test_passed,
                    'failed': self.test_failed,
                    'skipped': self.test_skipped
                }
            }, f, indent=2)
        
        print(f"\n{Fore.BLUE}Results saved to: {results_file}{Style.RESET_ALL}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Comprehensive OAK-D S3 Camera Test Suite"
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick tests only (skip streaming tests)'
    )
    
    args = parser.parse_args()
    
    # Create and run test suite
    suite = CameraTestSuite(verbose=args.verbose)
    
    try:
        suite.run_comprehensive_test()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Tests interrupted by user{Style.RESET_ALL}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Fore.RED}Fatal error: {e}{Style.RESET_ALL}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()