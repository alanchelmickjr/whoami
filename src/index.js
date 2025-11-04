/**
 * WhoAmI - Secure Facial Recognition System for Jetson Nano
 * 
 * Main application entry point that orchestrates:
 * - Hardware-backed security initialization
 * - Gun.js secure database setup
 * - Facial recognition system
 * - Real-time camera processing
 * 
 * Security Architecture:
 * 1. All encryption keys derived from hardware (CPU serial + MAC address)
 * 2. Double encryption: hardware-backed + Gun.js SEA
 * 3. No keys stored in code or plaintext
 * 4. Each robot has unique identity tied to hardware
 * 5. Cannot be reverse engineered - requires specific hardware to decrypt
 */

import { secureKeyManager } from './secureKeyManager.js';
import { secureDatabase } from './secureDatabase.js';
import { facialRecognition } from './facialRecognition.js';
import { readFileSync } from 'fs';
import { join } from 'path';

class WhoAmI {
  constructor() {
    this.initialized = false;
    this.config = null;
  }

  /**
   * Load configuration from file
   */
  loadConfig(configPath = './config/config.json') {
    try {
      const configData = readFileSync(configPath, 'utf8');
      this.config = JSON.parse(configData);
      return this.config;
    } catch (error) {
      // Use default configuration if file doesn't exist
      console.warn('Config file not found, using defaults');
      this.config = {
        modelsPath: './models',
        dataPath: './data/gun',
        minConfidence: 0.7,
        descriptorThreshold: 0.6,
        peers: []
      };
      return this.config;
    }
  }

  /**
   * Initialize all system components
   */
  async initialize(configPath) {
    if (this.initialized) {
      console.log('System already initialized');
      return;
    }

    console.log('Initializing WhoAmI Facial Recognition System...');
    console.log('='.repeat(50));

    try {
      // Load configuration
      this.loadConfig(configPath);

      // Step 1: Initialize hardware-backed security
      console.log('1. Initializing hardware-backed security...');
      await secureKeyManager.initialize();
      console.log('   ✓ Security keys derived from hardware');

      // Step 2: Initialize secure database
      console.log('2. Initializing secure Gun.js database...');
      await secureDatabase.initialize({
        dataPath: this.config.dataPath,
        peers: this.config.peers
      });
      console.log('   ✓ Database initialized and authenticated');

      // Step 3: Load facial recognition models
      console.log('3. Loading facial recognition models...');
      await facialRecognition.loadModels(this.config.modelsPath);
      
      // Set thresholds from config
      facialRecognition.setMinConfidence(this.config.minConfidence);
      facialRecognition.setDescriptorThreshold(this.config.descriptorThreshold);
      console.log('   ✓ Models loaded successfully');

      this.initialized = true;
      
      console.log('='.repeat(50));
      console.log('System initialized successfully!');
      console.log('Hardware-backed security: ACTIVE');
      console.log('Data encryption: DOUBLE-LAYER');
      console.log('Ready for facial recognition operations');
      console.log('='.repeat(50));

    } catch (error) {
      console.error('Initialization failed:', error.message);
      throw error;
    }
  }

  /**
   * Register a new person's face
   */
  async registerPerson(imageInput, personName) {
    if (!this.initialized) {
      throw new Error('System not initialized. Call initialize() first.');
    }

    console.log(`Registering face for: ${personName}`);
    const faceId = await facialRecognition.registerFace(imageInput, personName);
    console.log(`✓ Registration complete. Face ID: ${faceId}`);
    
    return faceId;
  }

  /**
   * Recognize a person from an image
   */
  async recognize(imageInput) {
    if (!this.initialized) {
      throw new Error('System not initialized. Call initialize() first.');
    }

    const result = await facialRecognition.recognizeFace(imageInput);
    
    if (Array.isArray(result)) {
      console.log(`Detected ${result.length} face(s)`);
      result.forEach((r, i) => {
        if (r.recognized) {
          console.log(`  Face ${i + 1}: ${r.personName} (confidence: ${(r.confidence * 100).toFixed(2)}%)`);
        } else {
          console.log(`  Face ${i + 1}: Unknown`);
        }
      });
    } else {
      if (result.recognized) {
        console.log(`Recognized: ${result.personName} (confidence: ${(result.confidence * 100).toFixed(2)}%)`);
      } else {
        console.log('No match found');
      }
    }

    return result;
  }

  /**
   * List all registered persons
   */
  async listRegistered() {
    if (!this.initialized) {
      throw new Error('System not initialized. Call initialize() first.');
    }

    const faceIds = await secureDatabase.listFaceIds();
    console.log(`Registered faces: ${faceIds.length}`);
    
    const persons = [];
    for (const faceId of faceIds) {
      try {
        const faceData = await secureDatabase.retrieveFaceData(faceId);
        if (faceData) {
          persons.push({
            name: faceData.personName,
            registeredAt: faceData.registeredAt,
            faceId: faceId.substring(0, 8) + '...'
          });
        }
      } catch (error) {
        console.error(`Error reading face ${faceId}:`, error.message);
      }
    }

    return persons;
  }

  /**
   * Remove a registered person
   */
  async removePerson(faceId) {
    if (!this.initialized) {
      throw new Error('System not initialized. Call initialize() first.');
    }

    await secureDatabase.deleteFaceData(faceId);
    console.log(`Removed face: ${faceId}`);
  }

  /**
   * Get system status and security information
   */
  getStatus() {
    return {
      initialized: this.initialized,
      hardwareSecurityActive: secureKeyManager.initialized,
      databaseInitialized: secureDatabase.initialized,
      modelsLoaded: facialRecognition.modelsLoaded,
      securityLevel: 'Hardware-backed double encryption',
      reversEngineeringResistance: 'Device-specific keys derived from hardware'
    };
  }
}

// Export singleton instance
export const whoami = new WhoAmI();

// If running as main module, provide CLI interface
if (import.meta.url === `file://${process.argv[1]}`) {
  console.log('WhoAmI - Secure Facial Recognition System');
  console.log('Starting initialization...');
  
  whoami.initialize()
    .then(() => {
      console.log('\nSystem ready for operations');
      console.log('Import this module to use the facial recognition API');
    })
    .catch(error => {
      console.error('Failed to start:', error.message);
      process.exit(1);
    });
}
