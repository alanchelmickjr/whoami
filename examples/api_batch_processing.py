#!/usr/bin/env python3
"""
Face Recognition API - Batch Processing Example

This script demonstrates batch processing capabilities:
- Processing directories of images
- Video file processing
- CSV/JSON export of results
- Performance optimization for batch operations
- Progress tracking and reporting
- Parallel processing options
- Statistical analysis of results

Requirements:
- Python 3.7+
- Required packages: numpy, opencv-python, face-recognition, depthai
"""

import sys
import os
import time
import cv2
import json
import csv
import glob
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter
import concurrent.futures
from tqdm import tqdm
import numpy as np

# Add parent directory to path to import whoami module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whoami.face_recognition_api import (
    FaceRecognitionAPI,
    RecognitionConfig,
    RecognitionModel,
    RecognitionResult,
    create_face_recognition_api
)


class BatchProcessor:
    """
    Batch processor for face recognition tasks
    """
    
    def __init__(self, api: Optional[FaceRecognitionAPI] = None):
        """
        Initialize batch processor
        
        Args:
            api: Face Recognition API instance. If None, creates default.
        """
        self.api = api or create_face_recognition_api(
            model=RecognitionModel.HOG,  # Faster for batch processing
            tolerance=0.6,
            process_every_n_frames=1
        )
        
        self.results = []
        self.statistics = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'total_faces': 0,
            'recognized_faces': 0,
            'unknown_faces': 0,
            'processing_time': 0,
            'people_found': Counter()
        }
    
    def process_image(self, image_path: str, save_annotated: bool = False) -> Dict[str, Any]:
        """
        Process a single image file
        
        Args:
            image_path: Path to image file
            save_annotated: Whether to save annotated image
        
        Returns:
            Processing results dictionary
        """
        result = {
            'file': image_path,
            'timestamp': datetime.now().isoformat(),
            'faces': [],
            'error': None
        }
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Process image
            start_time = time.time()
            recognition_results = self.api.recognize_faces_in_image(image_path)
            processing_time = time.time() - start_time
            
            result['processing_time'] = processing_time
            
            # Process results
            for rec_result in recognition_results:
                face_data = {
                    'name': rec_result.name,
                    'confidence': float(rec_result.confidence),
                    'location': rec_result.location
                }
                result['faces'].append(face_data)
                
                # Update statistics
                self.statistics['total_faces'] += 1
                if rec_result.name != "Unknown":
                    self.statistics['recognized_faces'] += 1
                    self.statistics['people_found'][rec_result.name] += 1
                else:
                    self.statistics['unknown_faces'] += 1
            
            # Save annotated image if requested
            if save_annotated and recognition_results:
                self._save_annotated_image(image, recognition_results, image_path)
            
            self.statistics['processed_files'] += 1
            
        except Exception as e:
            result['error'] = str(e)
            self.statistics['failed_files'] += 1
            print(f"Error processing {image_path}: {e}")
        
        return result
    
    def _save_annotated_image(self, image: np.ndarray, 
                              results: List[RecognitionResult], 
                              original_path: str):
        """Save image with face annotations"""
        # Draw annotations
        for result in results:
            top, right, bottom, left = result.location
            
            # Choose color based on recognition
            if result.name == "Unknown":
                color = (0, 0, 255)  # Red for unknown
            else:
                color = (0, 255, 0)  # Green for recognized
            
            # Draw rectangle
            cv2.rectangle(image, (left, top), (right, bottom), color, 2)
            
            # Add label
            label = f"{result.name} ({result.confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            cv2.rectangle(image, (left, top - label_size[1] - 10),
                         (left + label_size[0], top), color, -1)
            cv2.putText(image, label, (left, top - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Save annotated image
        output_dir = Path(original_path).parent / "annotated"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"annotated_{Path(original_path).name}"
        cv2.imwrite(str(output_path), image)
    
    def process_directory(self, directory: str, 
                         pattern: str = "*.jpg",
                         recursive: bool = False,
                         save_annotated: bool = False,
                         max_workers: int = 1) -> List[Dict[str, Any]]:
        """
        Process all images in a directory
        
        Args:
            directory: Directory path
            pattern: File pattern to match
            recursive: Search recursively
            save_annotated: Save annotated images
            max_workers: Number of parallel workers (1 for sequential)
        
        Returns:
            List of processing results
        """
        # Get image files
        if recursive:
            image_files = list(Path(directory).rglob(pattern))
        else:
            image_files = list(Path(directory).glob(pattern))
        
        self.statistics['total_files'] = len(image_files)
        print(f"Found {len(image_files)} images to process")
        
        results = []
        start_time = time.time()
        
        if max_workers > 1:
            # Parallel processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self.process_image, str(img), save_annotated): img
                    for img in image_files
                }
                
                for future in tqdm(concurrent.futures.as_completed(futures), 
                                  total=len(image_files),
                                  desc="Processing images"):
                    result = future.result()
                    results.append(result)
                    self.results.append(result)
        else:
            # Sequential processing
            for image_file in tqdm(image_files, desc="Processing images"):
                result = self.process_image(str(image_file), save_annotated)
                results.append(result)
                self.results.append(result)
        
        self.statistics['processing_time'] = time.time() - start_time
        
        return results
    
    def process_video(self, video_path: str, 
                     output_path: Optional[str] = None,
                     frame_skip: int = 30,
                     max_frames: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Process a video file
        
        Args:
            video_path: Path to video file
            output_path: Path for output video with annotations
            frame_skip: Process every Nth frame
            max_frames: Maximum frames to process
        
        Returns:
            List of frame processing results
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video: {fps} FPS, {total_frames} frames, {width}x{height}")
        
        # Setup output video if requested
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        results = []
        frame_count = 0
        processed_count = 0
        
        # Progress bar
        pbar = tqdm(total=min(total_frames, max_frames) if max_frames else total_frames,
                   desc="Processing video")
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Check frame limit
                if max_frames and frame_count >= max_frames:
                    break
                
                # Process frame based on skip rate
                if frame_count % frame_skip == 0:
                    # Convert BGR to RGB for processing
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Process frame
                    recognition_results = self.api.process_frame(rgb_frame)
                    
                    # Store results
                    frame_result = {
                        'frame': frame_count,
                        'timestamp': frame_count / fps,
                        'faces': []
                    }
                    
                    for rec_result in recognition_results:
                        face_data = {
                            'name': rec_result.name,
                            'confidence': float(rec_result.confidence),
                            'location': rec_result.location
                        }
                        frame_result['faces'].append(face_data)
                        
                        # Draw on frame if outputting video
                        if out:
                            top, right, bottom, left = rec_result.location
                            color = (0, 255, 0) if rec_result.name != "Unknown" else (0, 0, 255)
                            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                            label = f"{rec_result.name}"
                            cv2.putText(frame, label, (left, top - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    
                    results.append(frame_result)
                    processed_count += 1
                
                # Write frame to output video
                if out:
                    out.write(frame)
                
                frame_count += 1
                pbar.update(1)
        
        finally:
            pbar.close()
            cap.release()
            if out:
                out.release()
        
        print(f"Processed {processed_count} frames out of {frame_count}")
        
        # Update statistics
        self.statistics['total_files'] += 1
        self.statistics['processed_files'] += 1
        
        return results
    
    def export_results_csv(self, output_path: str):
        """Export results to CSV file"""
        if not self.results:
            print("No results to export")
            return
        
        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = ['file', 'timestamp', 'person_name', 'confidence', 
                         'location_top', 'location_right', 'location_bottom', 'location_left']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in self.results:
                if 'faces' in result:
                    for face in result['faces']:
                        row = {
                            'file': result.get('file', result.get('frame', '')),
                            'timestamp': result['timestamp'],
                            'person_name': face['name'],
                            'confidence': face['confidence'],
                            'location_top': face['location'][0],
                            'location_right': face['location'][1],
                            'location_bottom': face['location'][2],
                            'location_left': face['location'][3]
                        }
                        writer.writerow(row)
        
        print(f"Results exported to {output_path}")
    
    def export_results_json(self, output_path: str):
        """Export results to JSON file"""
        if not self.results:
            print("No results to export")
            return
        
        export_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_results': len(self.results)
            },
            'statistics': dict(self.statistics),
            'results': self.results
        }
        
        # Convert Counter to dict for JSON serialization
        export_data['statistics']['people_found'] = dict(self.statistics['people_found'])
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Results exported to {output_path}")
    
    def generate_report(self) -> str:
        """Generate processing report"""
        report = []
        report.append("=" * 60)
        report.append("Batch Processing Report")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # File statistics
        report.append("File Statistics:")
        report.append(f"  Total files: {self.statistics['total_files']}")
        report.append(f"  Processed: {self.statistics['processed_files']}")
        report.append(f"  Failed: {self.statistics['failed_files']}")
        report.append("")
        
        # Face statistics
        report.append("Face Recognition Statistics:")
        report.append(f"  Total faces detected: {self.statistics['total_faces']}")
        report.append(f"  Recognized faces: {self.statistics['recognized_faces']}")
        report.append(f"  Unknown faces: {self.statistics['unknown_faces']}")
        
        if self.statistics['total_faces'] > 0:
            recognition_rate = (self.statistics['recognized_faces'] / 
                              self.statistics['total_faces']) * 100
            report.append(f"  Recognition rate: {recognition_rate:.1f}%")
        report.append("")
        
        # People found
        if self.statistics['people_found']:
            report.append("People Found:")
            for person, count in self.statistics['people_found'].most_common():
                report.append(f"  {person}: {count} times")
            report.append("")
        
        # Performance
        if self.statistics['processing_time'] > 0:
            report.append("Performance:")
            report.append(f"  Total processing time: {self.statistics['processing_time']:.2f} seconds")
            
            if self.statistics['processed_files'] > 0:
                avg_time = self.statistics['processing_time'] / self.statistics['processed_files']
                report.append(f"  Average time per file: {avg_time:.3f} seconds")
                report.append(f"  Processing rate: {1/avg_time:.1f} files/second")
        
        report.append("=" * 60)
        
        return "\n".join(report)


class DatasetAnalyzer:
    """
    Analyze face recognition datasets
    """
    
    def __init__(self, api: FaceRecognitionAPI):
        self.api = api
    
    def analyze_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """
        Analyze a dataset of face images
        
        Expected structure:
        dataset/
        ├── person1/
        │   ├── image1.jpg
        │   └── image2.jpg
        └── person2/
            ├── image1.jpg
            └── image2.jpg
        """
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
        
        analysis = {
            'dataset_path': str(dataset_path),
            'people': {},
            'total_images': 0,
            'total_people': 0,
            'quality_metrics': {},
            'recommendations': []
        }
        
        # Analyze each person's directory
        for person_dir in dataset_path.iterdir():
            if person_dir.is_dir():
                person_name = person_dir.name
                person_images = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))
                
                if person_images:
                    person_analysis = self._analyze_person_images(person_name, person_images)
                    analysis['people'][person_name] = person_analysis
                    analysis['total_images'] += len(person_images)
                    analysis['total_people'] += 1
        
        # Generate quality metrics
        analysis['quality_metrics'] = self._calculate_quality_metrics(analysis['people'])
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _analyze_person_images(self, person_name: str, 
                               image_paths: List[Path]) -> Dict[str, Any]:
        """Analyze images for a specific person"""
        person_analysis = {
            'image_count': len(image_paths),
            'faces_detected': 0,
            'multiple_faces': 0,
            'no_faces': 0,
            'encoding_variance': 0,
            'images': []
        }
        
        encodings = []
        
        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            
            # Detect faces
            detections = self.api.detect_faces(image)
            
            image_info = {
                'path': str(image_path),
                'faces_count': len(detections)
            }
            
            if len(detections) == 0:
                person_analysis['no_faces'] += 1
            elif len(detections) == 1:
                person_analysis['faces_detected'] += 1
                if detections[0].encoding is not None:
                    encodings.append(detections[0].encoding)
            else:
                person_analysis['multiple_faces'] += 1
            
            person_analysis['images'].append(image_info)
        
        # Calculate encoding variance
        if len(encodings) > 1:
            # Calculate pairwise distances
            distances = []
            for i in range(len(encodings)):
                for j in range(i + 1, len(encodings)):
                    dist = np.linalg.norm(encodings[i] - encodings[j])
                    distances.append(dist)
            
            person_analysis['encoding_variance'] = float(np.std(distances))
        
        return person_analysis
    
    def _calculate_quality_metrics(self, people_data: Dict) -> Dict[str, Any]:
        """Calculate overall quality metrics"""
        metrics = {
            'avg_images_per_person': 0,
            'detection_rate': 0,
            'single_face_rate': 0,
            'avg_encoding_variance': 0
        }
        
        if not people_data:
            return metrics
        
        total_images = sum(p['image_count'] for p in people_data.values())
        total_detected = sum(p['faces_detected'] for p in people_data.values())
        total_variances = [p['encoding_variance'] for p in people_data.values() if p['encoding_variance'] > 0]
        
        metrics['avg_images_per_person'] = total_images / len(people_data)
        
        if total_images > 0:
            metrics['detection_rate'] = total_detected / total_images
            metrics['single_face_rate'] = total_detected / total_images
        
        if total_variances:
            metrics['avg_encoding_variance'] = float(np.mean(total_variances))
        
        return metrics
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Check detection rate
        detection_rate = analysis['quality_metrics']['detection_rate']
        if detection_rate < 0.8:
            recommendations.append(
                f"Low face detection rate ({detection_rate:.1%}). "
                "Consider improving image quality or lighting."
            )
        
        # Check for people with few images
        for person, data in analysis['people'].items():
            if data['image_count'] < 3:
                recommendations.append(
                    f"{person} has only {data['image_count']} image(s). "
                    "Consider adding more for better recognition."
                )
            
            if data['multiple_faces'] > 0:
                recommendations.append(
                    f"{person} has {data['multiple_faces']} image(s) with multiple faces. "
                    "Consider cropping to single face."
                )
        
        # Check encoding variance
        avg_variance = analysis['quality_metrics']['avg_encoding_variance']
        if avg_variance > 0.5:
            recommendations.append(
                f"High encoding variance ({avg_variance:.3f}). "
                "Images may be too different or low quality."
            )
        
        if not recommendations:
            recommendations.append("Dataset quality looks good!")
        
        return recommendations


def demo_image_batch_processing():
    """Demonstrate batch processing of images"""
    print("=" * 60)
    print("Image Batch Processing Demo")
    print("=" * 60)
    
    # Create batch processor
    processor = BatchProcessor()
    
    # Create test directory with sample images
    test_dir = Path("batch_test_images")
    if not test_dir.exists():
        print(f"\nPlease create a directory '{test_dir}' with test images")
        print("Or specify a different directory")
        return
    
    # Process directory
    print(f"\nProcessing images in {test_dir}...")
    results = processor.process_directory(
        str(test_dir),
        pattern="*.jpg",
        save_annotated=True,
        max_workers=2  # Use 2 parallel workers
    )
    
    # Export results
    processor.export_results_csv("batch_results.csv")
    processor.export_results_json("batch_results.json")
    
    # Print report
    print("\n" + processor.generate_report())


def demo_video_processing():
    """Demonstrate video processing"""
    print("=" * 60)
    print("Video Processing Demo")
    print("=" * 60)
    
    # Create batch processor
    processor = BatchProcessor()
    
    video_path = "test_video.mp4"
    if not Path(video_path).exists():
        print(f"\nPlease provide a video file: {video_path}")
        return
    
    # Process video
    print(f"\nProcessing video: {video_path}")
    results = processor.process_video(
        video_path,
        output_path="annotated_video.mp4",
        frame_skip=30,  # Process every 30th frame
        max_frames=300   # Limit to 300 frames for demo
    )
    
    # Analyze results
    frames_with_faces = sum(1 for r in results if r['faces'])
    total_people = set()
    for result in results:
        for face in result['faces']:
            if face['name'] != "Unknown":
                total_people.add(face['name'])
    
    print(f"\nVideo Processing Results:")
    print(f"  Frames processed: {len(results)}")
    print(f"  Frames with faces: {frames_with_faces}")
    print(f"  Unique people found: {len(total_people)}")
    if total_people:
        print(f"  People: {', '.join(total_people)}")


def demo_dataset_analysis():
    """Demonstrate dataset analysis"""
    print("=" * 60)
    print("Dataset Analysis Demo")
    print("=" * 60)
    
    # Create API and analyzer
    api = create_face_recognition_api()
    analyzer = DatasetAnalyzer(api)
    
    dataset_path = "face_dataset"
    if not Path(dataset_path).exists():
        print(f"\nPlease create a dataset directory structure:")
        print(f"{dataset_path}/")
        print("├── person1/")
        print("│   ├── image1.jpg")
        print("│   └── image2.jpg")
        print("└── person2/")
        print("    └── image1.jpg")
        return
    
    # Analyze dataset
    print(f"\nAnalyzing dataset: {dataset_path}")
    analysis = analyzer.analyze_dataset(dataset_path)
    
    # Print analysis results
    print(f"\nDataset Analysis Results:")
    print(f"  Total people: {analysis['total_people']}")
    print(f"  Total images: {analysis['total_images']}")
    print(f"  Avg images per person: {analysis['quality_metrics']['avg_images_per_person']:.1f}")
    print(f"  Detection rate: {analysis['quality_metrics']['detection_rate']:.1%}")
    
    print(f"\nPeople in dataset:")
    for person, data in analysis['people'].items():
        print(f"  {person}: {data['image_count']} images, "
              f"{data['faces_detected']} with single face")
    
    print(f"\nRecommendations:")
    for rec in analysis['recommendations']:
        print(f"  - {rec}")
    
    # Save full analysis
    with open("dataset_analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\nFull analysis saved to dataset_analysis.json")


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Face Recognition Batch Processing')
    parser.add_argument('--mode', choices=['images', 'video', 'dataset', 'demo'],
                       default='demo', help='Processing mode')
    parser.add_argument('--input', help='Input directory or file')
    parser.add_argument('--output', help='Output file for results')
    parser.add_argument('--pattern', default='*.jpg', help='File pattern for images')
    parser.add_argument('--workers', type=int, default=1, help='Number of parallel workers')
    parser.add_argument('--annotate', action='store_true', help='Save annotated images')
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        # Interactive demo mode
        print("\n" + "=" * 60)
        print("Face Recognition API - Batch Processing")
        print("=" * 60)
        print("\nSelect a demo:")
        print("1. Image Batch Processing")
        print("2. Video Processing")
        print("3. Dataset Analysis")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-3): ").strip()
        
        if choice == '1':
            demo_image_batch_processing()
        elif choice == '2':
            demo_video_processing()
        elif choice == '3':
            demo_dataset_analysis()
        elif choice == '0':
            print("Exiting...")
        else:
            print("Invalid choice")
    
    elif args.mode == 'images':
        # Batch process images
        if not args.input:
            print("Error: --input directory required for image processing")
            return
        
        processor = BatchProcessor()
        results = processor.process_directory(
            args.input,
            pattern=args.pattern,
            save_annotated=args.annotate,
            max_workers=args.workers
        )
        
        # Export results
        if args.output:
            if args.output.endswith('.csv'):
                processor.export_results_csv(args.output)
            else:
                processor.export_results_json(args.output)
        
        print(processor.generate_report())
    
    elif args.mode == 'video':
        # Process video
        if not args.input:
            print("Error: --input video file required")
            return
        
        processor = BatchProcessor()
        results = processor.process_video(
            args.input,
            output_path=args.output
        )
        
        print(f"Processed {len(results)} frames")
    
    elif args.mode == 'dataset':
        # Analyze dataset
        if not args.input:
            print("Error: --input dataset directory required")
            return
        
        api = create_face_recognition_api()
        analyzer = DatasetAnalyzer(api)
        analysis = analyzer.analyze_dataset(args.input)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(analysis, f, indent=2)
            print(f"Analysis saved to {args.output}")
        else:
            print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()