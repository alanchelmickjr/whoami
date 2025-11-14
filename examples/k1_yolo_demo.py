#!/usr/bin/env python3
"""
K-1 Booster YOLO Face Recognition Demo

This script demonstrates the complete face recognition workflow for the K-1:
1. Detect faces using YOLO
2. Ask unknown people for their names (voice)
3. Recognize known people and greet them (voice)
4. Display live video feed with annotations

Usage:
    python examples/k1_yolo_demo.py
    python examples/k1_yolo_demo.py --no-display  # Headless mode for VNC
    python examples/k1_yolo_demo.py --no-voice    # Silent mode
"""

import sys
import os
import cv2
import logging
import argparse
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from whoami.yolo_face_recognition import K1FaceRecognitionSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def draw_results(frame, results):
    """
    Draw face recognition results on frame

    Args:
        frame: Input frame
        results: List of FaceRecognitionResult
    """
    for result in results:
        x1, y1, x2, y2 = result.bbox

        # Choose color based on recognition
        if result.name == "Unknown":
            color = (0, 0, 255)  # Red for unknown
            label_color = (255, 255, 255)
        else:
            color = (0, 255, 0)  # Green for known
            label_color = (255, 255, 255)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        label = f"{result.name} ({result.confidence:.2f})"
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )

        # Draw background rectangle for text
        cv2.rectangle(
            frame,
            (x1, y1 - label_height - baseline - 10),
            (x1 + label_width, y1),
            color,
            -1
        )

        # Draw text
        cv2.putText(
            frame,
            label,
            (x1, y1 - baseline - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            label_color,
            2
        )

    return frame


def main():
    parser = argparse.ArgumentParser(
        description='K-1 Booster YOLO Face Recognition Demo'
    )
    parser.add_argument(
        '--no-voice',
        action='store_true',
        help='Disable voice interaction'
    )
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Disable video display (headless mode)'
    )
    parser.add_argument(
        '--yolo-model',
        default='yolov8n.pt',
        help='YOLO model path (default: yolov8n.pt)'
    )
    parser.add_argument(
        '--deepface-model',
        default='Facenet',
        choices=['Facenet', 'ArcFace', 'VGG-Face', 'Facenet512'],
        help='DeepFace model for recognition'
    )
    parser.add_argument(
        '--resolution',
        default='1280x720',
        help='Camera resolution (e.g., 1280x720)'
    )
    parser.add_argument(
        '--fps-limit',
        type=int,
        default=10,
        help='FPS limit for processing (default: 10)'
    )
    args = parser.parse_args()

    # Parse resolution
    width, height = map(int, args.resolution.split('x'))

    logger.info("=" * 60)
    logger.info("K-1 Booster Face Recognition System")
    logger.info("=" * 60)
    logger.info(f"YOLO Model: {args.yolo_model}")
    logger.info(f"DeepFace Model: {args.deepface_model}")
    logger.info(f"Resolution: {width}x{height}")
    logger.info(f"Voice: {'Enabled' if not args.no_voice else 'Disabled'}")
    logger.info(f"Display: {'Enabled' if not args.no_display else 'Disabled'}")
    logger.info("=" * 60)

    # Create system
    try:
        system = K1FaceRecognitionSystem(
            yolo_model=args.yolo_model,
            deepface_model=args.deepface_model,
            enable_voice=not args.no_voice,
            camera_resolution=(width, height)
        )
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        logger.error("Make sure you have installed: pip install ultralytics deepface tf-keras")
        return 1

    # Start system
    if not system.start():
        logger.error("Failed to start camera")
        return 1

    logger.info("\n" + "=" * 60)
    logger.info("System ready!")
    logger.info("- Unknown faces will be asked for their names")
    logger.info("- Known faces will be greeted")
    logger.info("- Press 'q' to quit")
    logger.info("- Press 'a' to manually add a face")
    logger.info("- Press 's' to show statistics")
    logger.info("=" * 60 + "\n")

    # Main loop
    try:
        frame_count = 0
        fps_start_time = time.time()
        fps = 0
        frame_interval = 1.0 / args.fps_limit

        while True:
            loop_start = time.time()

            # Get frame
            frame = system.camera.get_frame()

            if frame is None:
                logger.warning("No frame received")
                time.sleep(0.1)
                continue

            # Process frame
            results = system.process_frame(
                frame=frame,
                ask_unknown=not args.no_voice,
                greet_known=not args.no_voice
            )

            # Draw results if display enabled
            if not args.no_display:
                display_frame = frame.copy()
                display_frame = draw_results(display_frame, results)

                # Add FPS counter
                cv2.putText(
                    display_frame,
                    f"FPS: {fps:.1f} | Faces: {len(results)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2
                )

                # Add known people count
                stats = system.get_statistics()
                cv2.putText(
                    display_frame,
                    f"Known: {stats['known_people']} people",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                # Convert RGB to BGR for display
                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)

                # Show frame
                cv2.imshow('K-1 Face Recognition', display_frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    logger.info("Quit requested")
                    break
                elif key == ord('a'):
                    logger.info("Manual face add mode")
                    name = input("Enter name: ").strip()
                    if name:
                        if system.add_face_from_frame(name, frame):
                            logger.info(f"Added face for {name}")
                        else:
                            logger.error("Failed to add face")
                elif key == ord('s'):
                    stats = system.get_statistics()
                    logger.info(f"Statistics: {stats}")

            # Log results
            if results:
                for result in results:
                    logger.info(
                        f"Frame {frame_count}: {result.name} "
                        f"(conf: {result.confidence:.2f})"
                    )

            # Update FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_start_time)
                fps_start_time = time.time()

            # Throttle FPS
            elapsed = time.time() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Error in main loop: {e}", exc_info=True)
    finally:
        logger.info("Shutting down...")
        system.stop()

        if not args.no_display:
            cv2.destroyAllWindows()

        # Print final statistics
        stats = system.get_statistics()
        logger.info("\n" + "=" * 60)
        logger.info("Final Statistics")
        logger.info("=" * 60)
        logger.info(f"Known people: {stats['known_people']}")
        logger.info(f"Names: {', '.join(stats['people']) if stats['people'] else 'None'}")
        logger.info(f"Frames processed: {frame_count}")
        logger.info(f"Average FPS: {frame_count / (time.time() - fps_start_time):.1f}")
        logger.info("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
