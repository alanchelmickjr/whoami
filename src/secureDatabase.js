/**
 * Secure Gun.js Database Module
 * 
 * Implements a secure, decentralized database for storing facial recognition data
 * with end-to-end encryption and hardware-backed security.
 * 
 * Security features:
 * - All data encrypted before storage using SEA (Security, Encryption, Authorization)
 * - User authentication tied to hardware
 * - No plaintext storage of sensitive data
 * - Peer-to-peer replication optional (can be isolated)
 */

import Gun from 'gun';
import 'gun/sea.js';
import { secureKeyManager } from './secureKeyManager.js';

class SecureDatabase {
  constructor() {
    this.gun = null;
    this.user = null;
    this.initialized = false;
  }

  /**
   * Initialize the Gun.js database with secure configuration
   */
  async initialize(config = {}) {
    if (this.initialized) {
      return;
    }

    // Ensure key manager is initialized
    await secureKeyManager.initialize();

    // Initialize Gun with security-focused configuration
    this.gun = Gun({
      // Store data locally only (no peers by default for maximum security)
      // Can be configured to replicate to trusted peers if needed
      peers: config.peers || [],
      
      // Local storage path
      file: config.dataPath || './data/gun',
      
      // Increase security with these options
      localStorage: false, // Don't use browser localStorage
      radisk: true, // Use RAD (Radix) storage for better performance
      
      // Optional: WebRTC for secure peer connections
      // webrtc: config.webrtc || null
    });

    // Create or authenticate user with hardware-derived credentials
    await this.authenticateUser();

    this.initialized = true;
  }

  /**
   * Authenticate user using hardware-derived credentials
   * This ensures only this specific robot can access the data
   */
  async authenticateUser() {
    return new Promise((resolve, reject) => {
      const userID = secureKeyManager.generateUserID();
      const password = secureKeyManager.deriveGunPassword();

      this.user = this.gun.user();

      // Try to authenticate
      this.user.auth(userID, password, (ack) => {
        if (ack.err) {
          // User doesn't exist, create it
          this.user.create(userID, password, (createAck) => {
            if (createAck.err) {
              reject(new Error('Failed to create user: ' + createAck.err));
              return;
            }
            
            // Now authenticate
            this.user.auth(userID, password, (authAck) => {
              if (authAck.err) {
                reject(new Error('Failed to authenticate: ' + authAck.err));
                return;
              }
              console.log('User authenticated successfully');
              resolve();
            });
          });
        } else {
          console.log('User authenticated successfully');
          resolve();
        }
      });
    });
  }

  /**
   * Store facial recognition data securely
   * Data is encrypted at the application layer before Gun's SEA encryption
   */
  async storeFaceData(faceId, faceData) {
    if (!this.initialized || !this.user) {
      throw new Error('Database not initialized');
    }

    // Double encryption: first our hardware-backed encryption
    const encryptedData = secureKeyManager.encrypt(faceData);
    
    // Then Gun's SEA encryption
    const sealedData = await Gun.SEA.encrypt(encryptedData, this.user._.sea);
    
    // Store with timestamp
    const record = {
      data: sealedData,
      timestamp: Date.now(),
      version: '1.0'
    };

    return new Promise((resolve, reject) => {
      this.user.get('faces').get(faceId).put(record, (ack) => {
        if (ack.err) {
          reject(new Error('Failed to store face data: ' + ack.err));
        } else {
          resolve(ack);
        }
      });
    });
  }

  /**
   * Retrieve facial recognition data securely
   * Decrypts both Gun's SEA encryption and our hardware-backed encryption
   */
  async retrieveFaceData(faceId) {
    if (!this.initialized || !this.user) {
      throw new Error('Database not initialized');
    }

    return new Promise((resolve, reject) => {
      this.user.get('faces').get(faceId).once(async (encryptedRecord) => {
        if (!encryptedRecord || !encryptedRecord.data) {
          resolve(null);
          return;
        }

        try {
          // First decrypt Gun's SEA encryption
          const encryptedData = await Gun.SEA.decrypt(encryptedRecord.data, this.user._.sea);
          
          if (!encryptedData) {
            reject(new Error('Failed to decrypt SEA data'));
            return;
          }

          // Then decrypt our hardware-backed encryption
          const faceData = secureKeyManager.decrypt(encryptedData);
          
          resolve({
            ...faceData,
            timestamp: encryptedRecord.timestamp
          });
        } catch (error) {
          reject(new Error('Failed to retrieve face data: ' + error.message));
        }
      });
    });
  }

  /**
   * List all stored face IDs
   */
  async listFaceIds() {
    if (!this.initialized || !this.user) {
      throw new Error('Database not initialized');
    }

    return new Promise((resolve) => {
      const faceIds = [];
      this.user.get('faces').map().once((data, id) => {
        if (id && id !== '_') {
          faceIds.push(id);
        }
      });

      // Give Gun time to retrieve all data
      setTimeout(() => resolve(faceIds), 1000);
    });
  }

  /**
   * Delete facial recognition data
   */
  async deleteFaceData(faceId) {
    if (!this.initialized || !this.user) {
      throw new Error('Database not initialized');
    }

    return new Promise((resolve, reject) => {
      this.user.get('faces').get(faceId).put(null, (ack) => {
        if (ack.err) {
          reject(new Error('Failed to delete face data: ' + ack.err));
        } else {
          resolve(ack);
        }
      });
    });
  }

  /**
   * Get database instance for advanced operations
   */
  getGun() {
    return this.gun;
  }

  /**
   * Get authenticated user for advanced operations
   */
  getUser() {
    return this.user;
  }
}

// Export singleton instance
export const secureDatabase = new SecureDatabase();
