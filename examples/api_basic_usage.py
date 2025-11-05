#!/usr/bin/env python3
"""
Basic Face Recognition API Usage Example

This script demonstrates the fundamental features of the Face Recognition API:
- Starting/stopping camera
- Adding faces to database
- Real-time face recognition
- Saving/loading database

Requirements:
- OAK-D camera or webcam
- Python 3.7+
- Required packages: numpy, opencv-python, face-recognition, depthai
"""

import sys
import os
import time
import cv2

# Add parent directory to path to import whoami module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whoami.face_recognition_api import (
    create_face_recognition_api,
    FaceRecognitionAPI,
    RecognitionConfig,
    CameraType,
    RecognitionModel
)


def basic_recognition_demo():
    """
    Basic demonstration of face recognition capabilities
    """
    print("=" * 60)
    print("Basic Face Recognition API Demo")
    print("=" * 60)
    
    # Create API with default settings
    print("\n1. Creating Face Recognition API...")
    api = create_face_recognition_api(
        database_path="demo_faces.pkl",
        camera_type=CameraType.OAK_D,  # Change to CameraType.WEBCAM if no OAK-D
        tolerance=0.6
    )
    print("✓ API created successfully")
    
    # Start camera
    print("\n2. Starting camera...")
    if not api.start_camera():
        print("✗ Failed to start camera. Please check your camera connection.")
        print("If using OAK-D, ensure it's connected via USB 3.0")
        print("To use webcam instead, change camera_type to CameraType.WEBCAM")
        return
    print("✓ Camera started successfully")
    
    # Get API statistics
    stats = api.get_statistics()
    print(f"\n3. Current Statistics:")
    print(f"   - Total faces in database: {stats['total_faces']}")
    print(f"   - Unique people: {stats['unique_people']}")
    print(f"   - Camera type: {stats['config']['camera_type']}")
    
    # Add faces to database
    print("\n4. Adding faces to database...")
    print("   Press Enter to capture a face, or 'q' to skip")
    
    while True:
        user_input = input("   Enter name (or 'q' to continue): ").strip()
        if user_input.lower() == 'q':
            break
        
        if user_input:
            print(f"   Position your face in front of the camera and press Enter...")
            input()
            
            # Capture and add face
            if api.add_face(user_input):
                print(f"   ✓ Successfully added {user_input} to database")
            else:
                print(f"   ✗ Failed to add face. Make sure your face is visible.")
    
    # Show updated statistics
    stats = api.get_statistics()
    print(f"\n5. Updated Database:")
    print(f"   - Total faces: {stats['total_faces']}")
    print(f"   - People in database: {api.get_all_names()}")
    
    # Real-time recognition
    print("\n6. Starting real-time recognition...")
    print("   Press 'q' to quit, 'a' to add a new face, 's' to save database")
    print("   Press 'c' to clear database, 'd' to display statistics")
    print()
    
    frame_count = 0
    fps_start_time = time.time()
    fps = 0
    
    try:
        while True:
            # Get frame from camera
            frame = api.get_frame()
            if frame is None:
                continue
            
            # Process frame for recognition
            results = api.process_frame(frame)
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_start_time)
                fps_start_time = time.time()
            
            # Draw results on frame
            for result in results:
                # Get face location
                top, right, bottom, left = result.location
                
                # Determine color based on recognition
                if result.name == "Unknown":
                    color = (255, 0, 0)  # Red for unknown
                    label = "Unknown"
                else:
                    color = (0, 255, 0)  # Green for recognized
                    label = f"{result.name} ({result.confidence:.2f})"
                
                # Draw rectangle around face
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                
                # Draw label background
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                cv2.rectangle(frame, (left, top - label_size[1] - 10),
                            (left + label_size[0], top), color, -1)
                
                # Draw label text
                cv2.putText(frame, label, (left, top - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Display FPS
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display frame count
            cv2.putText(frame, f"Faces: {len(results)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Convert RGB to BGR for OpenCV display
            display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Show frame
            cv2.imshow('Face Recognition - Basic Demo', display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n   Quitting...")
                break
            elif key == ord('a'):
                # Add new face
                print("\n   Adding new face...")
                name = input("   Enter name: ").strip()
                if name:
                    print("   Position face and press Enter...")
                    input()
                    if api.add_face(name):
                        print(f"   ✓ Added {name} to database")
                    else:
                        print("   ✗ Failed to add face")
            elif key == ord('s'):
                # Save database
                if api.save_database():
                    print("\n   ✓ Database saved successfully")
            elif key == ord('c'):
                # Clear database
                confirm = input("\n   Clear entire database? (y/n): ").strip().lower()
                if confirm == 'y':
                    api.clear_database()
                    print("   ✓ Database cleared")
            elif key == ord('d'):
                # Display statistics
                stats = api.get_statistics()
                print(f"\n   Database Statistics:")
                print(f"   - Total faces: {stats['total_faces']}")
                print(f"   - Unique people: {stats['unique_people']}")
                print(f"   - Frames processed: {stats['frames_processed']}")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        # Cleanup
        cv2.destroyAllWindows()
        
        # Save database
        print("\n7. Saving database...")
        if api.save_database():
            print("✓ Database saved successfully")
        
        # Stop camera
        print("\n8. Stopping camera...")
        api.stop_camera()
        print("✓ Camera stopped")
        
        # Final statistics
        stats = api.get_statistics()
        print(f"\nFinal Statistics:")
        print(f"  - Total frames processed: {stats['frames_processed']}")
        print(f"  - Total faces in database: {stats['total_faces']}")
        print(f"  - Unique people: {stats['unique_people']}")


def simple_context_manager_example():
    """
    Demonstrate using the API with context manager for automatic cleanup
    """
    print("\n" + "=" * 60)
    print("Context Manager Example")
    print("=" * 60)
    
    # Using context manager ensures proper cleanup
    with create_face_recognition_api() as api:
        print("\n1. API created and entered context")
        
        # Start camera
        if api.start_camera():
            print("2. Camera started")
            
            # Process a few frames
            print("3. Processing frames...")
            for i in range(10):
                frame = api.get_frame()
                if frame is not None:
                    results = api.process_frame(frame)
                    print(f"   Frame {i+1}: Found {len(results)} face(s)")
                time.sleep(0.1)
            
            print("4. Exiting context - camera will stop automatically")
        else:
            print("Failed to start camera")
    
    print("5. Context exited - all resources cleaned up")


def configuration_examples():
    """
    Demonstrate different configuration options
    """
    print("\n" + "=" * 60)
    print("Configuration Examples")
    print("=" * 60)
    
    # Example 1: High-performance configuration
    print("\n1. High Performance Configuration:")
    config_fast = RecognitionConfig(
        model=RecognitionModel.HOG,           # Faster model
        process_every_n_frames=3,              # Skip frames
        face_detection_scale=0.5,              # Scale down for speed
        tolerance=0.6,                         # Standard tolerance
        log_level="WARNING"                   # Less logging
    )
    api_fast = FaceRecognitionAPI(config_fast)
    print(f"   Model: {config_fast.model.value}")
    print(f"   Frame skip: {config_fast.process_every_n_frames}")
    print(f"   Detection scale: {config_fast.face_detection_scale}")
    
    # Example 2: High-accuracy configuration
    print("\n2. High Accuracy Configuration:")
    config_accurate = RecognitionConfig(
        model=RecognitionModel.CNN,           # More accurate model
        process_every_n_frames=1,              # Process all frames
        face_detection_scale=1.0,              # Full resolution
        tolerance=0.4,                         # Stricter matching
        num_jitters=2,                         # Better encodings
        log_level="INFO"
    )
    api_accurate = FaceRecognitionAPI(config_accurate)
    print(f"   Model: {config_accurate.model.value}")
    print(f"   Tolerance: {config_accurate.tolerance}")
    print(f"   Num jitters: {config_accurate.num_jitters}")
    
    # Example 3: Webcam configuration
    print("\n3. Webcam Configuration:")
    config_webcam = RecognitionConfig(
        camera_type=CameraType.WEBCAM,
        camera_resolution=(640, 480),
        camera_fps=30,
        database_path="webcam_faces.pkl",
        auto_save=True
    )
    api_webcam = FaceRecognitionAPI(config_webcam)
    print(f"   Camera type: {config_webcam.camera_type.value}")
    print(f"   Resolution: {config_webcam.camera_resolution}")
    print(f"   Auto-save: {config_webcam.auto_save}")


def database_operations_example():
    """
    Demonstrate database operations
    """
    print("\n" + "=" * 60)
    print("Database Operations Example")
    print("=" * 60)
    
    api = create_face_recognition_api(database_path="test_database.pkl")
    
    # Load existing database
    print("\n1. Loading database...")
    if api.load_database():
        print(f"   ✓ Loaded {api.get_face_count()} faces")
        print(f"   People: {api.get_all_names()}")
    else:
        print("   No existing database found")
    
    # Add test faces (using dummy encodings for demonstration)
    print("\n2. Adding test entries...")
    import numpy as np
    
    # Note: In real usage, you would get encodings from actual face images
    # This is just for demonstration
    test_encoding = np.random.randn(128)  # Face encodings are 128-dimensional
    
    api.add_face("Test Person 1", encoding=test_encoding)
    api.add_face("Test Person 2", encoding=test_encoding + 0.1)
    print(f"   Added 2 test entries")
    
    # Query database
    print("\n3. Database queries:")
    print(f"   Total faces: {api.get_face_count()}")
    print(f"   Unique people: {len(api.get_all_names())}")
    print(f"   'Test Person 1' count: {api.get_face_count('Test Person 1')}")
    
    # Remove face
    print("\n4. Removing faces...")
    if api.remove_face("Test Person 1"):
        print("   ✓ Removed 'Test Person 1'")
    print(f"   Remaining faces: {api.get_face_count()}")
    
    # Save database
    print("\n5. Saving database...")
    if api.save_database():
        print("   ✓ Database saved")
    
    # Clear database
    print("\n6. Clearing database...")
    api.clear_database()
    print(f"   Database cleared. Faces: {api.get_face_count()}")


def main():
    """
    Main function to run examples
    """
    print("\n" + "=" * 60)
    print("Face Recognition API - Basic Usage Examples")
    print("=" * 60)
    print("\nSelect an example to run:")
    print("1. Basic Recognition Demo (interactive)")
    print("2. Context Manager Example")
    print("3. Configuration Examples")
    print("4. Database Operations")
    print("5. Run All Examples")
    print("0. Exit")
    
    try:
        choice = input("\nEnter your choice (0-5): ").strip()
        
        if choice == '1':
            basic_recognition_demo()
        elif choice == '2':
            simple_context_manager_example()
        elif choice == '3':
            configuration_examples()
        elif choice == '4':
            database_operations_example()
        elif choice == '5':
            # Run all non-interactive examples
            simple_context_manager_example()
            configuration_examples()
            database_operations_example()
            print("\n" + "=" * 60)
            print("To run the interactive demo, please select option 1")
        elif choice == '0':
            print("Exiting...")
        else:
            print("Invalid choice. Please run the script again.")
    
    except Exception as e:
        print(f"\n✗ Error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Example completed")
    print("=" * 60)


if __name__ == "__main__":
    main()