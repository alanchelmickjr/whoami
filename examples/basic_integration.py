"""
Example: Basic Integration
Shows how to integrate WhoAmI into your own application
"""

from whoami.face_recognizer import FaceRecognizer
import time

def main():
    """Example of using WhoAmI in your application"""
    
    # Initialize the face recognizer
    print("Initializing face recognizer...")
    recognizer = FaceRecognizer(database_path="my_app_faces.pkl")
    
    # Check current database
    known_faces = recognizer.get_all_names()
    print(f"Known faces in database: {known_faces}")
    
    # Start the camera
    print("\nStarting Oak D camera...")
    if not recognizer.start_camera():
        print("Error: Could not start camera")
        return
    
    print("Camera started successfully!")
    print("Running face recognition for 30 seconds...")
    print("Press Ctrl+C to stop early\n")
    
    try:
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < 30:
            # Get frame from camera
            frame = recognizer.get_frame()
            
            if frame is not None:
                frame_count += 1
                
                # Only process every 5th frame for efficiency
                if frame_count % 5 == 0:
                    # Detect faces
                    face_locations, face_encodings = recognizer.detect_faces(frame)
                    
                    if face_encodings:
                        # Recognize faces
                        results = recognizer.recognize_faces(face_encodings)
                        
                        # Print results
                        for i, (name, confidence) in enumerate(results):
                            if name != "Unknown":
                                print(f"Recognized: {name} (confidence: {confidence:.2f})")
                            else:
                                print(f"Unknown face detected (location: {face_locations[i]})")
            
            # Small delay to reduce CPU usage
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    
    finally:
        # Clean up
        print("\nStopping camera...")
        recognizer.stop_camera()
        print("Done!")


if __name__ == "__main__":
    main()
