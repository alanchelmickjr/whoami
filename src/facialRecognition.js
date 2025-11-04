/**
 * Facial Recognition Module
 * 
 * Handles real-time facial detection and recognition using face-api.js
 * Optimized for Jetson Nano's GPU acceleration
 * 
 * Features:
 * - Real-time face detection and recognition
 * - Face descriptor extraction and matching
 * - Support for multiple camera sources
 * - GPU-accelerated processing when available
 */

import * as faceapi from '@vladmandic/face-api';
import { Canvas, Image, ImageData } from 'canvas';
import { secureDatabase } from './secureDatabase.js';
import { createHash } from 'crypto';

// Patch face-api.js to work with node-canvas
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

class FacialRecognition {
  constructor() {
    this.modelsLoaded = false;
    this.minConfidence = 0.7;
    this.descriptorThreshold = 0.6;
  }

  /**
   * Load face-api.js models
   * Models should be downloaded to ./models directory
   */
  async loadModels(modelsPath = './models') {
    if (this.modelsLoaded) {
      return;
    }

    console.log('Loading facial recognition models...');

    try {
      // Load required models for face detection and recognition
      await faceapi.nets.ssdMobilenetv1.loadFromDisk(modelsPath);
      await faceapi.nets.faceLandmark68Net.loadFromDisk(modelsPath);
      await faceapi.nets.faceRecognitionNet.loadFromDisk(modelsPath);

      this.modelsLoaded = true;
      console.log('Models loaded successfully');
    } catch (error) {
      throw new Error('Failed to load models: ' + error.message);
    }
  }

  /**
   * Detect faces in an image and extract descriptors
   * Returns array of face detections with descriptors
   */
  async detectFaces(imageInput) {
    if (!this.modelsLoaded) {
      throw new Error('Models not loaded. Call loadModels() first.');
    }

    try {
      // Detect all faces with landmarks and descriptors
      const detections = await faceapi
        .detectAllFaces(imageInput, new faceapi.SsdMobilenetv1Options({ minConfidence: this.minConfidence }))
        .withFaceLandmarks()
        .withFaceDescriptors();

      return detections;
    } catch (error) {
      throw new Error('Face detection failed: ' + error.message);
    }
  }

  /**
   * Register a new face with a person's name
   * Extracts face descriptor and stores it securely
   */
  async registerFace(imageInput, personName) {
    const detections = await this.detectFaces(imageInput);

    if (detections.length === 0) {
      throw new Error('No face detected in image');
    }

    if (detections.length > 1) {
      throw new Error('Multiple faces detected. Please ensure only one face is in the image.');
    }

    const detection = detections[0];
    const descriptor = Array.from(detection.descriptor);

    // Create a unique face ID based on person name
    const faceId = createHash('sha256')
      .update(personName)
      .digest('hex');

    // Store face data securely
    const faceData = {
      personName,
      descriptor,
      registeredAt: new Date().toISOString(),
      confidence: detection.detection.score
    };

    await secureDatabase.storeFaceData(faceId, faceData);

    console.log(`Face registered for ${personName}`);
    return faceId;
  }

  /**
   * Recognize a face by comparing with stored descriptors
   * Returns the best match if found
   */
  async recognizeFace(imageInput) {
    const detections = await this.detectFaces(imageInput);

    if (detections.length === 0) {
      return { recognized: false, message: 'No face detected' };
    }

    // Get all stored faces
    const faceIds = await secureDatabase.listFaceIds();
    
    if (faceIds.length === 0) {
      return { recognized: false, message: 'No faces registered yet' };
    }

    const results = [];

    // Check each detected face
    for (const detection of detections) {
      const queryDescriptor = detection.descriptor;
      let bestMatch = null;
      let bestDistance = Infinity;

      // Compare with all stored faces
      for (const faceId of faceIds) {
        try {
          const storedFace = await secureDatabase.retrieveFaceData(faceId);
          
          if (!storedFace || !storedFace.descriptor) {
            continue;
          }

          // Calculate Euclidean distance between descriptors
          const storedDescriptor = new Float32Array(storedFace.descriptor);
          const distance = faceapi.euclideanDistance(queryDescriptor, storedDescriptor);

          if (distance < bestDistance) {
            bestDistance = distance;
            bestMatch = {
              personName: storedFace.personName,
              distance,
              confidence: 1 - distance,
              faceId
            };
          }
        } catch (error) {
          console.error(`Error comparing with face ${faceId}:`, error.message);
        }
      }

      // Check if match is good enough
      if (bestMatch && bestDistance < this.descriptorThreshold) {
        results.push({
          recognized: true,
          ...bestMatch,
          box: detection.detection.box
        });
      } else {
        results.push({
          recognized: false,
          message: 'No matching face found',
          box: detection.detection.box
        });
      }
    }

    return results.length === 1 ? results[0] : results;
  }

  /**
   * Process video frame for real-time recognition
   * Optimized for continuous processing
   */
  async processFrame(canvas, context, video) {
    if (!this.modelsLoaded) {
      return null;
    }

    try {
      // Draw video frame to canvas
      context.drawImage(video, 0, 0, canvas.width, canvas.height);

      // Detect and recognize faces
      const result = await this.recognizeFace(canvas);

      return result;
    } catch (error) {
      console.error('Frame processing error:', error.message);
      return null;
    }
  }

  /**
   * Set the minimum confidence threshold for face detection
   */
  setMinConfidence(confidence) {
    this.minConfidence = Math.max(0, Math.min(1, confidence));
  }

  /**
   * Set the descriptor matching threshold
   * Lower values = stricter matching
   */
  setDescriptorThreshold(threshold) {
    this.descriptorThreshold = Math.max(0, Math.min(1, threshold));
  }
}

// Export singleton instance
export const facialRecognition = new FacialRecognition();
