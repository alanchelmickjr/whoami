"""
Demo Mode for WhoAmI
Simulates camera input for testing without Oak D hardware
"""

import cv2
import numpy as np
from typing import Optional
import time


class DemoCamera:
    """Simulated camera for testing without Oak D hardware"""
    
    def __init__(self):
        """Initialize demo camera"""
        self.running = False
        self.cap = None
        self.use_webcam = True
        
    def start(self) -> bool:
        """
        Start the demo camera
        Uses webcam if available, otherwise generates synthetic frames
        
        Returns:
            True if successful
        """
        # Try to use webcam first
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.use_webcam = True
            self.running = True
            print("Demo mode: Using webcam")
            return True
        else:
            self.use_webcam = False
            self.running = True
            print("Demo mode: Using synthetic frames (no webcam detected)")
            return True
    
    def stop(self):
        """Stop the demo camera"""
        self.running = False
        if self.cap is not None:
            self.cap.release()
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get current frame
        
        Returns:
            Frame as numpy array or None
        """
        if not self.running:
            return None
        
        if self.use_webcam and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                # Resize to standard size
                frame = cv2.resize(frame, (640, 480))
                return frame
        
        # Generate synthetic frame
        frame = self._generate_synthetic_frame()
        return frame
    
    def _generate_synthetic_frame(self) -> np.ndarray:
        """
        Generate a synthetic test frame
        
        Returns:
            Synthetic frame with text
        """
        # Create blank frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (50, 50, 50)  # Dark gray background
        
        # Add text
        text_lines = [
            "DEMO MODE",
            "",
            "No Oak D camera detected",
            "Using simulated camera",
            "",
            "Connect Oak D for real",
            "facial recognition"
        ]
        
        y_offset = 150
        for i, line in enumerate(text_lines):
            cv2.putText(
                frame, line, (120, y_offset + i * 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
            )
        
        # Add timestamp
        timestamp = time.strftime("%H:%M:%S")
        cv2.putText(
            frame, timestamp, (250, 450),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 100), 2
        )
        
        return frame


def create_demo_recognizer():
    """
    Create a face recognizer with demo camera fallback
    
    Returns:
        Modified FaceRecognizer that uses demo camera if Oak D not available
    """
    from whoami.face_recognizer import FaceRecognizer
    import depthai as dai
    
    class DemoFaceRecognizer(FaceRecognizer):
        """Face recognizer with demo camera fallback"""
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.demo_camera = None
            self.is_demo_mode = False
        
        def start_camera(self) -> bool:
            """Start camera with fallback to demo mode"""
            try:
                # Try to start real Oak D camera
                return super().start_camera()
            except Exception as e:
                print(f"Oak D not available: {e}")
                print("Switching to demo mode...")
                
                # Fall back to demo camera
                self.demo_camera = DemoCamera()
                self.is_demo_mode = True
                return self.demo_camera.start()
        
        def stop_camera(self):
            """Stop camera"""
            if self.is_demo_mode and self.demo_camera:
                self.demo_camera.stop()
            else:
                super().stop_camera()
        
        def get_frame(self) -> Optional[np.ndarray]:
            """Get frame from camera"""
            if self.is_demo_mode and self.demo_camera:
                return self.demo_camera.get_frame()
            else:
                return super().get_frame()
    
    return DemoFaceRecognizer


if __name__ == "__main__":
    # Test demo camera
    print("Testing demo camera...")
    demo = DemoCamera()
    demo.start()
    
    print("Press ESC to exit...")
    while True:
        frame = demo.get_frame()
        if frame is not None:
            cv2.imshow('Demo Camera Test', frame)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    demo.stop()
    cv2.destroyAllWindows()
