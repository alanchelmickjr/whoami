#!/usr/bin/env python3
"""
Simple test script to verify OAK-D S3 camera connectivity
Updated for DepthAI 3.1.0 API with enhanced macOS permission handling
"""

import sys
import os
import platform
import time
import traceback
import subprocess

# Platform detection
IS_MAC = platform.system() == 'Darwin'
IS_LINUX = platform.system() == 'Linux'
IS_M_SERIES = platform.machine() == 'arm64' and IS_MAC
IS_JETSON = False

# Check if running on Jetson
try:
    with open('/proc/device-tree/model', 'r') as f:
        IS_JETSON = 'jetson' in f.read().lower()
except:
    pass

# Color codes for output
if sys.stdout.isatty():
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'  # No Color
else:
    RED = GREEN = YELLOW = BLUE = CYAN = NC = ''

try:
    import depthai as dai
    import cv2
    import numpy as np
    print(f"{GREEN}✓ Required libraries imported successfully (DepthAI {dai.__version__}){NC}")
except ImportError as e:
    print(f"{RED}✗ Failed to import required libraries: {e}{NC}")
    print(f"{YELLOW}Please install: pip install depthai opencv-python numpy{NC}")
    sys.exit(1)

def check_permissions():
    """Check and report permission status"""
    print(f"\n{CYAN}Permission Status:{NC}")
    
    if IS_MAC:
        # Check if running with sudo
        if os.geteuid() == 0:
            print(f"   {YELLOW}⚠ Running with sudo (temporary solution){NC}")
        else:
            print(f"   ℹ️  Running as regular user (UID: {os.getuid()})")
        
        # Provide macOS-specific guidance
        print(f"\n   {BLUE}macOS Permission Tips:{NC}")
        print("   • Camera access may require System Settings approval")
        print("   • Go to: System Settings > Privacy & Security > Camera")
        print("   • Grant permission to Terminal or your IDE")
        print(f"   • Or run with: {YELLOW}sudo python test_oak_d_s3.py{NC}")
        
    elif IS_LINUX or IS_JETSON:
        # Check user groups
        try:
            import grp, pwd
            username = pwd.getpwuid(os.getuid()).pw_name
            groups = [grp.getgrgid(g).gr_name for g in os.getgroups()]
            
            print(f"   User: {username}")
            print(f"   Groups: {', '.join(groups)}")
            
            if 'video' not in groups:
                print(f"   {YELLOW}⚠ User not in 'video' group{NC}")
                print(f"     Fix: sudo usermod -a -G video $USER")
            if 'dialout' not in groups:
                print(f"   {YELLOW}⚠ User not in 'dialout' group{NC}")
                print(f"     Fix: sudo usermod -a -G dialout $USER")
        except:
            pass
        
        # Check udev rules
        if os.path.exists("/etc/udev/rules.d/80-movidius.rules"):
            print(f"   {GREEN}✓ Movidius udev rules found{NC}")
        else:
            print(f"   {YELLOW}⚠ Movidius udev rules not found{NC}")
            print(f"     Fix: echo 'SUBSYSTEM==\"usb\", ATTRS{{idVendor}}==\"03e7\", MODE=\"0666\"' | sudo tee /etc/udev/rules.d/80-movidius.rules")

def test_oak_d_s3(retry_with_sudo=False):
    """Test OAK-D S3 camera connectivity with DepthAI 3.1.0"""
    print(f"\n{CYAN}" + "="*60)
    print("OAK-D S3 CAMERA TEST")
    print("="*60 + f"{NC}")
    
    # Display platform info
    print(f"\n{BLUE}Platform Information:{NC}")
    print(f"   System: {platform.system()} {platform.machine()}")
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   DepthAI: {dai.__version__}")
    if IS_M_SERIES:
        print(f"   Device: {GREEN}Apple M-Series Mac detected{NC}")
    elif IS_JETSON:
        print(f"   Device: {GREEN}NVIDIA Jetson detected{NC}")
    
    # Check permissions first
    check_permissions()
    
    # Step 1: Check for available devices
    print(f"\n{CYAN}1. Checking for available devices...{NC}")
    try:
        devices = dai.Device.getAllAvailableDevices()
        if not devices:
            print(f"   {RED}❌ No DepthAI devices found!{NC}")
            print(f"   {YELLOW}Please check:{NC}")
            print("   - Camera is connected via USB")
            print("   - Camera is powered on")
            print("   - USB cable supports data (not charge-only)")
            
            # Platform-specific troubleshooting
            if IS_MAC:
                print(f"\n   {YELLOW}macOS Troubleshooting:{NC}")
                print("   1. Check USB connection in System Information:")
                print(f"      {CYAN}system_profiler SPUSBDataType | grep -i movidius{NC}")
                print("   2. Try different USB ports (prefer direct connection)")
                print("   3. Restart the camera (unplug/replug)")
                if not retry_with_sudo and os.geteuid() != 0:
                    print(f"   4. Try running with sudo: {YELLOW}sudo python test_oak_d_s3.py{NC}")
            elif IS_LINUX or IS_JETSON:
                print(f"\n   {YELLOW}Linux Troubleshooting:{NC}")
                print("   1. Check USB devices:")
                print(f"      {CYAN}lsusb | grep 03e7{NC}")
                print("   2. Check kernel messages:")
                print(f"      {CYAN}dmesg | tail -20{NC}")
                print("   3. Verify udev rules and reload:")
                print(f"      {CYAN}sudo udevadm control --reload-rules && sudo udevadm trigger{NC}")
            
            return False
        
        print(f"   {GREEN}✅ Found {len(devices)} device(s){NC}")
        for i, info in enumerate(devices):
            # Use mxid property instead of getMxId() for DepthAI 3.1.0 compatibility
            mxid = info.mxid if hasattr(info, 'mxid') else 'N/A'
            print(f"   Device {i+1}: {mxid} [{info.state.name}]")
    except PermissionError as e:
        print(f"   {RED}❌ Permission Error: {e}{NC}")
        
        if IS_MAC and not retry_with_sudo and os.geteuid() != 0:
            print(f"\n   {YELLOW}Attempting to run with elevated permissions...{NC}")
            print(f"   You may be prompted for your password.")
            
            # Retry with sudo
            try:
                result = subprocess.run(
                    ['sudo', sys.executable, __file__],
                    check=True
                )
                return result.returncode == 0
            except subprocess.CalledProcessError:
                print(f"   {RED}Failed to run with sudo{NC}")
                return False
        else:
            print(f"\n   {YELLOW}Permission denied. Please check the troubleshooting tips above.{NC}")
            return False
    except Exception as e:
        print(f"   {RED}❌ Error: {e}{NC}")
        return False
    
    # Step 2: Create pipeline (compatible with multiple DepthAI versions)
    print(f"\n{CYAN}2. Creating pipeline...{NC}")
    pipeline = None
    
    # Try different pipeline configurations
    methods = [
        ("DepthAI 2.24+ API", lambda: create_pipeline_v2_24(dai)),
        ("DepthAI 2.x API", lambda: create_pipeline_v2(dai)),
        ("Legacy API", lambda: create_pipeline_legacy(dai))
    ]
    
    for method_name, method_func in methods:
        try:
            print(f"   Trying {method_name}...")
            pipeline = method_func()
            if pipeline:
                print(f"   {GREEN}✅ Pipeline created with {method_name}{NC}")
                break
        except Exception as e:
            print(f"   {YELLOW}⚠ {method_name} failed: {e}{NC}")
    
    if not pipeline:
        print(f"   {RED}❌ Failed to create pipeline with any method{NC}")
        return False

def create_pipeline_v2_24(dai):
    """Create pipeline for DepthAI 3.1.0+ using Camera node"""
    pipeline = dai.Pipeline()
    
    # Use Camera node for DepthAI 3.1.0 compatibility
    cam = pipeline.create(dai.node.Camera)
    cam.setPreviewSize(640, 480)
    cam.setInterleaved(False)
    
    # Set color order if the property exists
    if hasattr(cam, 'setColorOrder'):
        cam.setColorOrder(dai.CameraProperties.ColorOrder.RGB)
    
    # Create XLinkOut
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("rgb")
    
    # Link - use video output for Camera node
    cam.video.link(xout.input)
    
    return pipeline

def create_pipeline_v2(dai):
    """Create pipeline for DepthAI 2.x"""
    pipeline = dai.Pipeline()
    
    # Use createColorCamera
    cam_rgb = pipeline.createColorCamera()
    cam_rgb.setPreviewSize(640, 480)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    
    # Create XLinkOut
    xout = pipeline.createXLinkOut()
    xout.setStreamName("rgb")
    
    # Link
    cam_rgb.preview.link(xout.input)
    
    return pipeline

def create_pipeline_legacy(dai):
    """Create pipeline for older DepthAI versions"""
    pipeline = dai.Pipeline()
    
    # Try Camera node
    cam = pipeline.create(dai.node.Camera)
    cam.setPreviewSize(640, 480)
    cam.setInterleaved(False)
    
    # Create XLinkOut
    xout = pipeline.createXLinkOut()
    xout.setStreamName("rgb")
    
    # Link
    cam.preview.link(xout.input)
    
    return pipeline
    
    # Step 3: Connect to device and start pipeline
    print(f"\n{CYAN}3. Connecting to device...{NC}")
    device = None
    try:
        device = dai.Device(pipeline)
        print(f"   {GREEN}✅ Connected to device successfully{NC}")
        
        # Get device info (using DepthAI 3.1.0 compatible API)
        try:
            device_name = device.getDeviceName() if hasattr(device, 'getDeviceName') else 'Unknown'
            print(f"   Device Name: {device_name}")
        except:
            print(f"   Device Name: N/A")
        
        try:
            # Try different methods for getting MxId based on DepthAI version
            if hasattr(device, 'getMxId'):
                mxid = device.getMxId()
            elif hasattr(device, 'getDeviceInfo'):
                mxid = device.getDeviceInfo().getMxId() if hasattr(device.getDeviceInfo(), 'getMxId') else 'N/A'
            else:
                mxid = 'N/A'
            print(f"   MxId: {mxid}")
        except:
            print(f"   MxId: N/A")
        
        try:
            usb_speed = device.getUsbSpeed().name if hasattr(device, 'getUsbSpeed') else 'Unknown'
            print(f"   USB Speed: {usb_speed}")
        except:
            print(f"   USB Speed: N/A")
        
    except PermissionError as e:
        print(f"   {RED}❌ Permission denied: {e}{NC}")
        
        if IS_MAC:
            print(f"\n   {YELLOW}macOS Permission Fix:{NC}")
            print(f"   Option 1: Run with sudo: {CYAN}sudo python test_oak_d_s3.py{NC}")
            print(f"   Option 2: Grant camera access in System Settings > Privacy & Security > Camera")
            
            if os.geteuid() != 0:
                print(f"\n   {YELLOW}Retrying with sudo...{NC}")
                try:
                    result = subprocess.run(
                        ['sudo', sys.executable, __file__],
                        check=False
                    )
                    sys.exit(result.returncode)
                except:
                    pass
        else:
            print(f"\n   {YELLOW}Linux Permission Fix:{NC}")
            print(f"   1. Add user to video group: {CYAN}sudo usermod -a -G video $USER{NC}")
            print(f"   2. Setup udev rules: See setup instructions")
            print(f"   3. Logout and login again")
        
        return False
    except Exception as e:
        print(f"   {RED}❌ Failed to connect: {e}{NC}")
        if "Permission" in str(e):
            print(f"   {YELLOW}💡 TIP: This appears to be a permission issue{NC}")
            check_permissions()
        return False
    
    # Step 4: Test getting frames
    print(f"\n{CYAN}4. Testing camera stream...{NC}")
    try:
        queue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        
        frame_count = 0
        max_frames = 5
        start_time = time.time()
        
        print(f"   Capturing {max_frames} frames...")
        
        while frame_count < max_frames:
            in_rgb = queue.tryGet()
            if in_rgb is not None:
                frame_count += 1
                frame = in_rgb.getCvFrame()
                print(f"   {GREEN}✓{NC} Frame {frame_count}: {frame.shape}")
                
                # Save first frame as test
                if frame_count == 1:
                    cv2.imwrite("test_frame.jpg", frame)
                    print(f"   💾 Test frame saved as 'test_frame.jpg'")
            
            time.sleep(0.1)
        
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        
        print(f"   {GREEN}✅ Successfully captured {frame_count} frames{NC}")
        print(f"   📊 Effective FPS: {fps:.2f}")
        
    except Exception as e:
        print(f"   {RED}❌ Error getting frames: {e}{NC}")
        if device:
            device.close()
        return False
    
    # Step 5: Check additional capabilities
    print(f"\n{CYAN}5. Checking camera capabilities...{NC}")
    try:
        # Get camera features
        calibData = device.readCalibration()
        if calibData:
            print(f"   {GREEN}✅ Calibration data available{NC}")
        
        # Check available cameras
        connected_cameras = device.getConnectedCameras()
        print(f"   Connected cameras: {connected_cameras}")
        
        # Check IMU if available
        try:
            imu = device.getConnectedIMU()
            if imu:
                print(f"   {GREEN}✅ IMU available{NC}")
        except:
            pass
        
    except Exception as e:
        print(f"   ℹ️  Some features unavailable: {e}")
    
    # Step 6: Clean up
    print(f"\n{CYAN}6. Cleaning up...{NC}")
    try:
        if device:
            device.close()
            print(f"   {GREEN}✅ Device closed properly{NC}")
    except Exception as e:
        print(f"   {YELLOW}⚠️  Warning during cleanup: {e}{NC}")
    
    print(f"\n{GREEN}" + "="*60)
    print("✅ OAK-D S3 CAMERA TEST COMPLETED SUCCESSFULLY!")
    print("="*60 + f"{NC}")
    print(f"\n{GREEN}The camera is working and ready for use with the WhoAmI system.{NC}")
    
    return True

def main():
    """Main entry point"""
    print(f"{CYAN}Starting OAK-D S3 Camera Test...{NC}")
    print("Make sure the camera is connected via USB")
    
    # Check for --help flag
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print(f"\n{CYAN}Usage:{NC} python test_oak_d_s3.py [options]")
        print(f"\n{CYAN}Options:{NC}")
        print("  -h, --help     Show this help message")
        print("  --no-sudo      Don't attempt sudo retry on macOS")
        print(f"\n{CYAN}Platform:{NC} {platform.system()} {platform.machine()}")
        return 0
    
    try:
        # Check if --no-sudo flag is present
        retry_with_sudo = '--no-sudo' not in sys.argv
        
        success = test_oak_d_s3(retry_with_sudo=retry_with_sudo)
        
        if not success:
            print(f"\n{RED}❌ Camera test failed!{NC}")
            print(f"\n{YELLOW}Troubleshooting:{NC}")
            
            if IS_MAC:
                print("1. Verify USB connection (try different ports)")
                print(f"2. Check device recognition: {CYAN}system_profiler SPUSBDataType | grep -i movidius{NC}")
                print("3. Grant camera permissions in System Settings")
                print(f"4. Try with sudo: {CYAN}sudo python test_oak_d_s3.py{NC}")
            else:
                print("1. Verify USB connection (try different ports)")
                print(f"2. Check device recognition: {CYAN}lsusb | grep 03e7{NC}")
                print(f"3. Add user to video group: {CYAN}sudo usermod -a -G video $USER{NC}")
                print("4. Setup udev rules (see SETUP_JETSON_M4.md)")
            
            print(f"\n5. Restart camera (unplug/replug)")
            print(f"6. Check DepthAI version: {CYAN}python -c 'import depthai; print(depthai.__version__)'{NC}")
            return 1
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}Test interrupted by user{NC}")
        return 1
    except Exception as e:
        print(f"\n{RED}❌ Unexpected error: {e}{NC}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())