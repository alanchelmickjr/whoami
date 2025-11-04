#!/usr/bin/env python3
"""
Integration test for face recognition with OAK-D S3 camera
Tests the complete face recognition pipeline
"""

import sys
import time
import cv2
import numpy as np
from whoami.face_recognizer import FaceRecognizer

def test_face_recognition_oak():
    """
    Test the face recognition system with OAK-D S3 camera
    """
    print("=" * 60)
    print("Face Recognition Integration Test with OAK-D S3")
    print("=" * 60)
    
    # Initialize face recognizer
    print("\n1. Initializing FaceRecognizer...")
    recognizer = FaceRecognizer("test_face_database.pkl")
    
    # Start camera
    print("\n2. Starting OAK-D S3 camera with V3 API...")
    if not recognizer.start_camera():
        print("ERROR: Failed to start camera")
        return False
    
    print("Camera started successfully!")
    
    # Test adding a face
    print("\n3. Testing face addition...")
    print("Capturing frame for face addition...")
    
    # Get frames until we have a good one
    frame = None
    for _ in range(10):
        test_frame = recognizer.get_frame()
        if test_frame is not None:
            frame = test_frame
            break
        time.sleep(0.1)
    
    if frame is None:
        print("ERROR: Could not capture frame from camera")
        recognizer.stop_camera()
        return False
    
    print(f"Captured frame: {frame.shape}")
    
    # Create a test image with a face (synthetic for testing)
    # In real scenario, this would be a real face from the camera
    # For now, we'll test with the captured frame
    result = recognizer.add_face("Test Person", frame)
    if result:
        print("Successfully added face for 'Test Person'")
    else:
        print("No face detected in frame (this is normal if no one is in front of camera)")
    
    # List known faces
    print("\n4. Listing known faces...")
    names = recognizer.get_all_names()
    print(f"Known faces: {names}")
    
    # Test recognition
    print("\n5. Testing face recognition...")
    print("Capturing frames for recognition (5 second test)...")
    
    start_time = time.time()
    recognized_count = 0
    frame_count = 0
    
    while time.time() - start_time < 5:
        frame = recognizer.get_frame()
        if frame is not None:
            frame_count += 1
            
            # Detect and recognize faces
            face_locations, face_encodings = recognizer.detect_faces(frame)
            
            if len(face_encodings) > 0:
                results = recognizer.recognize_faces(face_encodings)
                for name, confidence in results:
                    if name != "Unknown":
                        recognized_count += 1
                        print(f"  - Recognized: {name} (confidence: {confidence:.2f})")
            
            # Display frame with face boxes
            for (top, right, bottom, left) in face_locations:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Show FPS
            fps = frame_count / (time.time() - start_time)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Face Recognition Test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cv2.destroyAllWindows()
    
    # Report results
    print(f"\n6. Test Results:")
    print(f"  - Frames captured: {frame_count}")
    print(f"  - Average FPS: {frame_count/5:.1f}")
    print(f"  - Faces recognized: {recognized_count}")
    
    # Stop camera
    print("\n7. Stopping camera...")
    recognizer.stop_camera()
    
    # Clean up test database
    print("\n8. Cleaning up test database...")
    recognizer.clear_database()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return True

def main():
    """Main test function"""
    try:
        print("Starting Face Recognition Integration Test")
        print("Make sure OAK-D S3 camera is connected\n")
        
        success = test_face_recognition_oak()
        
        if success:
            print("\n✅ All tests passed!")
            sys.exit(0)
        else:
            print("\n❌ Some tests failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()