#!/usr/bin/env python3
"""
Simple OAK-D S3 Camera Test Script
Using DepthAI v3 API - compatible with DepthAI 3.1.0
"""

import sys
import os
import platform
import time

# Check for required libraries
try:
    import depthai as dai
    import cv2
    import numpy as np
    print(f"✓ DepthAI version: {dai.__version__}")
except ImportError as e:
    print(f"✗ Missing required library: {e}")
    print("Install with: pip install depthai opencv-python")
    sys.exit(1)

def main():
    """Main function to test OAK-D S3 camera with DepthAI v3 API"""
    print("\n" + "="*50)
    print("OAK-D S3 CAMERA TEST (DepthAI v3 API)")
    print("="*50)
    print(f"System: {platform.system()} {platform.machine()}")
    print(f"Python: {sys.version.split()[0]}")
    
    # Check if running with sudo on macOS
    if platform.system() == 'Darwin':
        if os.geteuid() == 0:
            print("✓ Running with sudo (elevated permissions)")
        else:
            print("ℹ Running as regular user")
            print("  If you encounter permission errors, run with:")
            print(f"  sudo python {os.path.basename(__file__)}")
    
    # Step 1: Check for devices
    print("\n1. Searching for OAK-D devices...")
    try:
        devices = dai.Device.getAllAvailableDevices()
        if not devices:
            print("✗ No devices found!")
            print("\nTroubleshooting:")
            print("- Check USB connection")
            print("- Try different USB port")
            print("- Ensure camera is powered")
            if platform.system() == 'Darwin':
                print("- On macOS, try: sudo python test_oak_simple.py")
            return 1
        
        print(f"✓ Found {len(devices)} device(s)")
        for i, info in enumerate(devices):
            # Use mxid property for DepthAI 3.1.0
            mxid = info.mxid if hasattr(info, 'mxid') else 'N/A'
            print(f"  Device {i+1}: {mxid} [{info.state.name}]")
    except Exception as e:
        print(f"✗ Error checking devices: {e}")
        if "Permission" in str(e):
            print("\n⚠ Permission issue detected!")
            print("  On macOS: Run with sudo")
            print("  On Linux: Add user to video group")
        return 1
    
    # Step 2: Connect to device and create pipeline using v3 API
    print("\n2. Connecting to device and creating pipeline...")
    try:
        # Create device first (v3 API approach)
        device = dai.Device()
        print("✓ Connected to device")
        
        # Use Pipeline as context manager with the device
        with dai.Pipeline(device) as pipeline:
            print("✓ Pipeline created")
            
            # Get connected cameras
            sockets = device.getConnectedCameras()
            print(f"✓ Found {len(sockets)} camera(s): {[str(s) for s in sockets]}")
            
            if not sockets:
                print("✗ No cameras found on the device")
                return 1
            
            # Use the first available camera (usually CAM_A for RGB)
            socket = sockets[0]
            print(f"  Using camera: {socket}")
            
            # Create camera node using v3 API with build()
            cam = pipeline.create(dai.node.Camera).build(socket)
            
            # Request output with specific resolution
            # For OAK-D S3, we'll request 640x480 RGB output
            cam_out = cam.requestOutput((640, 480), dai.ImgFrame.Type.BGR888p)
            
            # Create output queue
            queue = cam_out.createOutputQueue()
            
            # Start the pipeline
            print("\n3. Starting pipeline...")
            pipeline.start()
            print("✓ Pipeline started")
            
            print("\n4. Starting video feed...")
            print("Press 'q' to quit, 's' to save a frame")
            print("-"*50)
            
            frame_count = 0
            fps_start = time.time()
            
            while pipeline.isRunning():
                # Get frame from queue
                videoIn = queue.get()
                assert isinstance(videoIn, dai.ImgFrame)
                frame = videoIn.getCvFrame()
                
                # Calculate FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    elapsed = time.time() - fps_start
                    fps = frame_count / elapsed
                    print(f"FPS: {fps:.2f}")
                
                # Add text overlay
                cv2.putText(frame, f"OAK-D S3 v3 API - Press 'q' to quit, 's' to save", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Frame: {frame_count}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow("OAK-D S3 Camera Feed", frame)
                
                # Handle key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('s'):
                    filename = f"oak_frame_{frame_count}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Saved frame to {filename}")
        
        print("\n5. Pipeline stopped")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Cleanup
        print("\n6. Cleaning up...")
        cv2.destroyAllWindows()
        print("✓ Cleanup complete")
    
    print("\n" + "="*50)
    print("✓ TEST COMPLETED SUCCESSFULLY")
    print("✓ OAK-D S3 camera is working with DepthAI v3 API!")
    print("="*50)
    return 0

if __name__ == "__main__":
    sys.exit(main())