#!/usr/bin/env python3
"""
Advanced Face Recognition API Features Example

This script demonstrates advanced features of the Face Recognition API:
- Event callbacks and custom handlers
- Multi-threaded processing
- Recognition caching and performance optimization
- Face landmarks detection
- Custom configuration and dynamic adjustment
- Error handling and recovery
- Advanced database management

Requirements:
- OAK-D camera or webcam
- Python 3.7+
- Required packages: numpy, opencv-python, face-recognition, depthai
"""

import sys
import os
import time
import cv2
import numpy as np
import threading
import queue
import json
from datetime import datetime
from collections import deque, defaultdict
from typing import List, Dict, Any

# Add parent directory to path to import whoami module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whoami.face_recognition_api import (
    FaceRecognitionAPI,
    RecognitionConfig,
    CameraType,
    RecognitionModel,
    RecognitionResult,
    create_face_recognition_api
)


class AdvancedRecognitionSystem:
    """
    Advanced face recognition system with custom features
    """
    
    def __init__(self):
        # Create API with optimized configuration
        self.config = RecognitionConfig(
            tolerance=0.5,
            model=RecognitionModel.HOG,
            camera_type=CameraType.OAK_D,
            process_every_n_frames=2,
            face_detection_scale=0.5,
            database_path="advanced_demo.pkl",
            auto_save=True,
            enable_threading=True,
            log_level="DEBUG"
        )
        
        self.api = FaceRecognitionAPI(self.config)
        
        # Event tracking
        self.event_log = deque(maxlen=100)
        self.recognition_history = defaultdict(list)
        self.performance_metrics = {
            'frames_processed': 0,
            'faces_detected': 0,
            'successful_recognitions': 0,
            'processing_times': deque(maxlen=100)
        }
        
        # Register callbacks
        self._register_callbacks()
        
        # Threading components
        self.processing_queue = queue.Queue(maxsize=30)
        self.result_queue = queue.Queue()
        self.running = False
        
    def _register_callbacks(self):
        """Register event callbacks"""
        self.api.register_callback('on_face_detected', self._on_face_detected)
        self.api.register_callback('on_face_recognized', self._on_face_recognized)
        self.api.register_callback('on_face_added', self._on_face_added)
        self.api.register_callback('on_face_removed', self._on_face_removed)
        self.api.register_callback('on_error', self._on_error)
    
    def _on_face_detected(self, detections):
        """Handle face detection events"""
        event = {
            'type': 'face_detected',
            'timestamp': datetime.now().isoformat(),
            'count': len(detections)
        }
        self.event_log.append(event)
        self.performance_metrics['faces_detected'] += len(detections)
        print(f"[EVENT] Detected {len(detections)} face(s)")
    
    def _on_face_recognized(self, results: List[RecognitionResult]):
        """Handle face recognition events"""
        for result in results:
            event = {
                'type': 'face_recognized',
                'timestamp': datetime.now().isoformat(),
                'name': result.name,
                'confidence': result.confidence
            }
            self.event_log.append(event)
            
            if result.name != "Unknown":
                self.recognition_history[result.name].append({
                    'timestamp': datetime.now(),
                    'confidence': result.confidence
                })
                self.performance_metrics['successful_recognitions'] += 1
                print(f"[RECOGNIZED] {result.name} (confidence: {result.confidence:.2f})")
    
    def _on_face_added(self, name):
        """Handle face addition events"""
        event = {
            'type': 'face_added',
            'timestamp': datetime.now().isoformat(),
            'name': name
        }
        self.event_log.append(event)
        print(f"[DATABASE] Added {name}")
    
    def _on_face_removed(self, name):
        """Handle face removal events"""
        event = {
            'type': 'face_removed',
            'timestamp': datetime.now().isoformat(),
            'name': name
        }
        self.event_log.append(event)
        print(f"[DATABASE] Removed {name}")
    
    def _on_error(self, error):
        """Handle error events"""
        event = {
            'type': 'error',
            'timestamp': datetime.now().isoformat(),
            'error': str(error)
        }
        self.event_log.append(event)
        print(f"[ERROR] {error}")


class MultiThreadedProcessor:
    """
    Multi-threaded face recognition processor
    """
    
    def __init__(self, api: FaceRecognitionAPI):
        self.api = api
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue()
        self.running = False
        self.threads = []
        
    def start(self, num_workers=2):
        """Start processing threads"""
        self.running = True
        
        # Start capture thread
        capture_thread = threading.Thread(target=self._capture_thread, daemon=True)
        capture_thread.start()
        self.threads.append(capture_thread)
        
        # Start worker threads
        for i in range(num_workers):
            worker_thread = threading.Thread(
                target=self._worker_thread, 
                args=(i,), 
                daemon=True
            )
            worker_thread.start()
            self.threads.append(worker_thread)
        
        # Start display thread
        display_thread = threading.Thread(target=self._display_thread, daemon=True)
        display_thread.start()
        self.threads.append(display_thread)
        
        print(f"Started {num_workers} worker threads")
    
    def _capture_thread(self):
        """Capture frames from camera"""
        print("[CAPTURE] Thread started")
        frame_count = 0
        
        while self.running:
            frame = self.api.get_frame()
            if frame is not None:
                try:
                    self.frame_queue.put((frame_count, frame), timeout=0.01)
                    frame_count += 1
                except queue.Full:
                    pass  # Drop frame if queue is full
            time.sleep(0.033)  # ~30 FPS
    
    def _worker_thread(self, worker_id):
        """Process frames for recognition"""
        print(f"[WORKER-{worker_id}] Thread started")
        
        while self.running:
            try:
                frame_id, frame = self.frame_queue.get(timeout=0.1)
                start_time = time.time()
                
                # Process frame
                results = self.api.process_frame(frame)
                
                processing_time = time.time() - start_time
                
                # Queue results
                self.result_queue.put({
                    'frame_id': frame_id,
                    'worker_id': worker_id,
                    'results': results,
                    'processing_time': processing_time,
                    'frame': frame
                })
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[WORKER-{worker_id}] Error: {e}")
    
    def _display_thread(self):
        """Display results"""
        print("[DISPLAY] Thread started")
        
        while self.running:
            try:
                data = self.result_queue.get(timeout=0.1)
                
                # Display processing info
                print(f"[FRAME-{data['frame_id']}] "
                      f"Worker-{data['worker_id']} "
                      f"processed in {data['processing_time']*1000:.1f}ms - "
                      f"Found {len(data['results'])} face(s)")
                
                # Draw results on frame
                frame = data['frame']
                for result in data['results']:
                    self._draw_result(frame, result)
                
                # Show frame
                cv2.imshow('Multi-threaded Recognition', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop()
                    
            except queue.Empty:
                continue
    
    def _draw_result(self, frame, result):
        """Draw recognition result on frame"""
        top, right, bottom, left = result.location
        
        if result.name == "Unknown":
            color = (255, 0, 0)  # Red
        else:
            color = (0, 255, 0)  # Green
        
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        
        label = f"{result.name} ({result.confidence:.2f})"
        cv2.putText(frame, label, (left, top - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def stop(self):
        """Stop all threads"""
        print("Stopping threads...")
        self.running = False
        for thread in self.threads:
            thread.join(timeout=1.0)
        cv2.destroyAllWindows()


class AdaptiveRecognition:
    """
    Adaptive recognition with dynamic parameter adjustment
    """
    
    def __init__(self, api: FaceRecognitionAPI):
        self.api = api
        self.base_tolerance = 0.6
        self.recognition_stats = deque(maxlen=50)
        
    def analyze_lighting(self, frame: np.ndarray) -> float:
        """Analyze frame lighting conditions"""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        # Calculate lighting score (0-1)
        lighting_score = min(brightness / 127.5, 1.0)
        
        return lighting_score
    
    def adjust_parameters(self, frame: np.ndarray):
        """Dynamically adjust recognition parameters"""
        lighting_score = self.analyze_lighting(frame)
        
        # Adjust tolerance based on lighting
        if lighting_score < 0.3:  # Dark
            new_tolerance = self.base_tolerance * 1.2
            scale = 0.5
            print(f"[ADAPTIVE] Dark conditions - tolerance: {new_tolerance:.2f}")
        elif lighting_score > 0.8:  # Bright
            new_tolerance = self.base_tolerance * 1.1
            scale = 0.75
            print(f"[ADAPTIVE] Bright conditions - tolerance: {new_tolerance:.2f}")
        else:  # Normal
            new_tolerance = self.base_tolerance
            scale = 1.0
        
        # Update configuration
        self.api.update_config(
            tolerance=min(new_tolerance, 0.8),
            face_detection_scale=scale
        )
        
        return lighting_score
    
    def adaptive_recognition(self, frame: np.ndarray) -> List[RecognitionResult]:
        """Perform adaptive recognition"""
        # Adjust parameters
        lighting_score = self.adjust_parameters(frame)
        
        # Perform recognition
        results = self.api.process_frame(frame)
        
        # Track statistics
        self.recognition_stats.append({
            'timestamp': time.time(),
            'lighting': lighting_score,
            'faces_found': len(results),
            'recognized': sum(1 for r in results if r.name != "Unknown")
        })
        
        # Analyze performance
        if len(self.recognition_stats) >= 10:
            recent_stats = list(self.recognition_stats)[-10:]
            avg_recognized = np.mean([s['recognized'] for s in recent_stats])
            
            if avg_recognized < 0.3:  # Poor recognition rate
                print("[ADAPTIVE] Poor recognition rate - adjusting tolerance")
                self.base_tolerance = min(self.base_tolerance + 0.05, 0.8)
        
        return results


class FaceLandmarkAnalyzer:
    """
    Analyze facial landmarks for additional features
    """
    
    def __init__(self, api: FaceRecognitionAPI):
        self.api = api
    
    def analyze_landmarks(self, frame: np.ndarray):
        """Analyze facial landmarks"""
        # Detect faces
        detections = self.api.detect_faces(frame, compute_encodings=False)
        
        for detection in detections:
            # Get landmarks
            landmarks = self.api.detector.detect_face_landmarks(frame, detection.location)
            
            if landmarks:
                # Analyze facial features
                self._analyze_expression(landmarks)
                self._analyze_pose(landmarks)
                
                # Draw landmarks
                self._draw_landmarks(frame, landmarks)
        
        return frame
    
    def _analyze_expression(self, landmarks: Dict):
        """Analyze facial expression from landmarks"""
        if 'left_eye' in landmarks and 'right_eye' in landmarks:
            # Simple eye aspect ratio for blink detection
            left_eye = landmarks['left_eye']
            right_eye = landmarks['right_eye']
            
            # Calculate eye aspect ratios
            left_ear = self._eye_aspect_ratio(left_eye)
            right_ear = self._eye_aspect_ratio(right_eye)
            
            avg_ear = (left_ear + right_ear) / 2.0
            
            if avg_ear < 0.2:
                print("[LANDMARKS] Eyes appear closed (possible blink)")
        
        if 'top_lip' in landmarks and 'bottom_lip' in landmarks:
            # Simple mouth aspect ratio for smile detection
            mar = self._mouth_aspect_ratio(landmarks['top_lip'], landmarks['bottom_lip'])
            
            if mar > 0.5:
                print("[LANDMARKS] Possible smile detected")
    
    def _analyze_pose(self, landmarks: Dict):
        """Analyze head pose from landmarks"""
        if 'nose_tip' in landmarks and 'nose_bridge' in landmarks:
            nose_tip = landmarks['nose_tip'][0]
            nose_bridge = landmarks['nose_bridge'][0]
            
            # Simple pose estimation
            dx = nose_tip[0] - nose_bridge[0]
            dy = nose_tip[1] - nose_bridge[1]
            
            if abs(dx) > 20:
                direction = "left" if dx < 0 else "right"
                print(f"[LANDMARKS] Head turned {direction}")
    
    def _eye_aspect_ratio(self, eye_points):
        """Calculate eye aspect ratio"""
        if len(eye_points) < 6:
            return 0.3
        
        # Simplified EAR calculation
        vertical_dist = np.linalg.norm(
            np.array(eye_points[1]) - np.array(eye_points[5])
        )
        horizontal_dist = np.linalg.norm(
            np.array(eye_points[0]) - np.array(eye_points[3])
        )
        
        if horizontal_dist == 0:
            return 0
        
        return vertical_dist / horizontal_dist
    
    def _mouth_aspect_ratio(self, top_lip, bottom_lip):
        """Calculate mouth aspect ratio"""
        if not top_lip or not bottom_lip:
            return 0
        
        # Simplified MAR calculation
        top_center = np.mean(top_lip, axis=0)
        bottom_center = np.mean(bottom_lip, axis=0)
        
        vertical_dist = np.linalg.norm(top_center - bottom_center)
        
        left = top_lip[0]
        right = top_lip[-1]
        horizontal_dist = np.linalg.norm(np.array(left) - np.array(right))
        
        if horizontal_dist == 0:
            return 0
        
        return vertical_dist / horizontal_dist
    
    def _draw_landmarks(self, frame: np.ndarray, landmarks: Dict):
        """Draw facial landmarks on frame"""
        colors = {
            'chin': (255, 0, 0),
            'left_eyebrow': (0, 255, 0),
            'right_eyebrow': (0, 255, 0),
            'nose_bridge': (0, 0, 255),
            'nose_tip': (0, 0, 255),
            'left_eye': (255, 255, 0),
            'right_eye': (255, 255, 0),
            'top_lip': (255, 0, 255),
            'bottom_lip': (255, 0, 255)
        }
        
        for feature, points in landmarks.items():
            color = colors.get(feature, (255, 255, 255))
            for point in points:
                cv2.circle(frame, point, 2, color, -1)


def advanced_callbacks_demo():
    """
    Demonstrate advanced callback usage
    """
    print("=" * 60)
    print("Advanced Callbacks Demo")
    print("=" * 60)
    
    system = AdvancedRecognitionSystem()
    
    # Start camera
    if not system.api.start_camera():
        print("Failed to start camera")
        return
    
    print("\nProcessing frames for 10 seconds...")
    print("Watch for event notifications...\n")
    
    start_time = time.time()
    
    try:
        while time.time() - start_time < 10:
            frame = system.api.get_frame()
            if frame is not None:
                system.api.process_frame(frame)
                system.performance_metrics['frames_processed'] += 1
            time.sleep(0.1)
    
    finally:
        system.api.stop_camera()
        
        # Display statistics
        print("\n" + "=" * 60)
        print("Performance Metrics:")
        print(f"  Frames processed: {system.performance_metrics['frames_processed']}")
        print(f"  Faces detected: {system.performance_metrics['faces_detected']}")
        print(f"  Successful recognitions: {system.performance_metrics['successful_recognitions']}")
        
        print("\nEvent Log (last 10 events):")
        for event in list(system.event_log)[-10:]:
            print(f"  {event['timestamp']}: {event['type']}")
        
        # Save event log
        with open('event_log.json', 'w') as f:
            json.dump(list(system.event_log), f, indent=2)
        print("\nEvent log saved to event_log.json")


def multi_threaded_demo():
    """
    Demonstrate multi-threaded processing
    """
    print("=" * 60)
    print("Multi-threaded Processing Demo")
    print("=" * 60)
    
    api = create_face_recognition_api()
    
    if not api.start_camera():
        print("Failed to start camera")
        return
    
    processor = MultiThreadedProcessor(api)
    
    print("\nStarting multi-threaded processor...")
    print("Press 'q' in the window to quit\n")
    
    processor.start(num_workers=3)
    
    # Run for up to 30 seconds or until quit
    timeout = 30
    start_time = time.time()
    
    try:
        while processor.running and (time.time() - start_time) < timeout:
            time.sleep(1)
            elapsed = time.time() - start_time
            print(f"Running... ({elapsed:.0f}/{timeout}s)")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        processor.stop()
        api.stop_camera()
        print("Multi-threaded demo completed")


def adaptive_recognition_demo():
    """
    Demonstrate adaptive recognition
    """
    print("=" * 60)
    print("Adaptive Recognition Demo")
    print("=" * 60)
    
    api = create_face_recognition_api()
    
    if not api.start_camera():
        print("Failed to start camera")
        return
    
    adaptive = AdaptiveRecognition(api)
    
    print("\nRunning adaptive recognition...")
    print("The system will adjust parameters based on lighting conditions")
    print("Press 'q' to quit\n")
    
    try:
        while True:
            frame = api.get_frame()
            if frame is None:
                continue
            
            # Perform adaptive recognition
            results = adaptive.adaptive_recognition(frame)
            
            # Display frame with results
            for result in results:
                top, right, bottom, left = result.location
                color = (0, 255, 0) if result.name != "Unknown" else (255, 0, 0)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                label = f"{result.name}"
                cv2.putText(frame, label, (left, top - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Display lighting score
            lighting_score = adaptive.analyze_lighting(frame)
            cv2.putText(frame, f"Lighting: {lighting_score:.2f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Adaptive Recognition', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cv2.destroyAllWindows()
        api.stop_camera()
        
        # Display adaptive statistics
        if adaptive.recognition_stats:
            stats = list(adaptive.recognition_stats)
            avg_lighting = np.mean([s['lighting'] for s in stats])
            avg_faces = np.mean([s['faces_found'] for s in stats])
            avg_recognized = np.mean([s['recognized'] for s in stats])
            
            print("\nAdaptive Recognition Statistics:")
            print(f"  Average lighting score: {avg_lighting:.2f}")
            print(f"  Average faces found: {avg_faces:.1f}")
            print(f"  Average recognized: {avg_recognized:.1f}")


def landmarks_demo():
    """
    Demonstrate facial landmarks analysis
    """
    print("=" * 60)
    print("Facial Landmarks Demo")
    print("=" * 60)
    
    api = create_face_recognition_api()
    
    if not api.start_camera():
        print("Failed to start camera")
        return
    
    analyzer = FaceLandmarkAnalyzer(api)
    
    print("\nAnalyzing facial landmarks...")
    print("Press 'q' to quit\n")
    
    try:
        while True:
            frame = api.get_frame()
            if frame is None:
                continue
            
            # Analyze landmarks
            frame_with_landmarks = analyzer.analyze_landmarks(frame)
            
            # Display frame
            cv2.imshow('Facial Landmarks', cv2.cvtColor(frame_with_landmarks, cv2.COLOR_RGB2BGR))
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cv2.destroyAllWindows()
        api.stop_camera()
        print("Landmarks demo completed")


def main():
    """
    Main function to run advanced demos
    """
    print("\n" + "=" * 60)
    print("Face Recognition API - Advanced Features")
    print("=" * 60)
    print("\nSelect a demo to run:")
    print("1. Advanced Callbacks")
    print("2. Multi-threaded Processing")
    print("3. Adaptive Recognition")
    print("4. Facial Landmarks Analysis")
    print("5. Run All Demos")
    print("0. Exit")
    
    try:
        choice = input("\nEnter your choice (0-5): ").strip()
        
        if choice == '1':
            advanced_callbacks_demo()
        elif choice == '2':
            multi_threaded_demo()
        elif choice == '3':
            adaptive_recognition_demo()
        elif choice == '4':
            landmarks_demo()
        elif choice == '5':
            # Run all demos
            advanced_callbacks_demo()
            print("\n" + "=" * 60 + "\n")
            multi_threaded_demo()
            print("\n" + "=" * 60 + "\n")
            adaptive_recognition_demo()
            print("\n" + "=" * 60 + "\n")
            landmarks_demo()
        elif choice == '0':
            print("Exiting...")
        else:
            print("Invalid choice. Please run the script again.")
    
    except Exception as e:
        print(f"\n✗ Error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Advanced features demo completed")
    print("=" * 60)


if __name__ == "__main__":
    main()