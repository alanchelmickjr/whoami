"""
Command Line Interface for Face Recognition System
Alternative to GUI for headless operation
"""

import argparse
import cv2
from .face_recognizer import FaceRecognizer


class FaceRecognitionCLI:
    """CLI interface for face recognition"""
    
    def __init__(self):
        """Initialize CLI"""
        self.recognizer = FaceRecognizer()
    
    def list_faces(self):
        """List all known faces"""
        names = self.recognizer.get_all_names()
        if names:
            print("\nKnown faces:")
            for name in sorted(names):
                print(f"  - {name}")
        else:
            print("\nNo faces in database")
    
    def add_face(self, name: str):
        """
        Add a face to the database
        
        Args:
            name: Name of the person
        """
        print(f"\nAdding face for {name}")
        print("Starting camera...")
        
        if not self.recognizer.start_camera():
            print("Error: Failed to start camera")
            return
        
        print("Position face in camera view and press SPACE to capture, ESC to cancel")
        
        try:
            while True:
                frame = self.recognizer.get_frame()
                if frame is None:
                    continue
                
                # Show preview
                display_frame = frame.copy()
                face_locations, _ = self.recognizer.detect_faces(frame)
                
                for (top, right, bottom, left) in face_locations:
                    cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
                cv2.imshow('Add Face - Press SPACE to capture, ESC to cancel', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    print("Cancelled")
                    break
                elif key == 32:  # SPACE
                    if self.recognizer.add_face(name, frame):
                        print(f"Successfully added face for {name}")
                    else:
                        print("Error: Could not detect face in frame")
                    break
        finally:
            cv2.destroyAllWindows()
            self.recognizer.stop_camera()
    
    def remove_face(self, name: str):
        """
        Remove a face from the database
        
        Args:
            name: Name of the person to remove
        """
        if self.recognizer.remove_face(name):
            print(f"Successfully removed {name}")
        else:
            print(f"Error: {name} not found in database")
    
    def clear_database(self):
        """Clear all faces from database"""
        self.recognizer.clear_database()
        print("All faces cleared from database")
    
    def recognize(self):
        """Run real-time face recognition"""
        print("\nStarting face recognition...")
        print("Press ESC to exit")
        
        if not self.recognizer.start_camera():
            print("Error: Failed to start camera")
            return
        
        try:
            while True:
                frame = self.recognizer.get_frame()
                if frame is None:
                    continue
                
                # Detect and recognize faces
                face_locations, face_encodings = self.recognizer.detect_faces(frame)
                recognized_faces = self.recognizer.recognize_faces(face_encodings)
                
                # Draw results
                for (top, right, bottom, left), (name, confidence) in zip(
                    face_locations, recognized_faces
                ):
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    
                    label = f"{name}"
                    if name != "Unknown":
                        label += f" ({confidence:.2f})"
                    
                    cv2.rectangle(frame, (left, bottom - 25), (right, bottom), color, cv2.FILLED)
                    cv2.putText(
                        frame, label, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1
                    )
                
                cv2.imshow('Face Recognition - Press ESC to exit', frame)
                
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break
        finally:
            cv2.destroyAllWindows()
            self.recognizer.stop_camera()


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="WhoAmI - Facial Recognition System")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # List command
    subparsers.add_parser('list', help='List all known faces')
    
    # Add command
    add_parser = subparsers.add_parser('add', help='Add a new face')
    add_parser.add_argument('name', help='Name of the person')
    
    # Remove command
    remove_parser = subparsers.add_parser('remove', help='Remove a face')
    remove_parser.add_argument('name', help='Name of the person')
    
    # Clear command
    subparsers.add_parser('clear', help='Clear all faces from database')
    
    # Recognize command
    subparsers.add_parser('recognize', help='Run real-time face recognition')
    
    args = parser.parse_args()
    
    cli = FaceRecognitionCLI()
    
    if args.command == 'list':
        cli.list_faces()
    elif args.command == 'add':
        cli.add_face(args.name)
    elif args.command == 'remove':
        cli.remove_face(args.name)
    elif args.command == 'clear':
        cli.clear_database()
    elif args.command == 'recognize':
        cli.recognize()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
