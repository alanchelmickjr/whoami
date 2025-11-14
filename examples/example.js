/**
 * Example: Real-time facial recognition with camera
 * 
 * This example demonstrates how to use the WhoAmI system
 * for real-time facial recognition from a camera feed.
 * 
 * NOTE: This requires a camera and the face-api.js models
 */

import { whoami } from '../src/index.js';
import { createCanvas, loadImage } from 'canvas';

async function main() {
  try {
    // Initialize the system
    console.log('Initializing WhoAmI system...\n');
    await whoami.initialize('./config/config.json');

    // Example 1: Register a person from an image file
    console.log('\n--- Example 1: Register Person ---');
    try {
      // Load an image (replace with actual image path)
      // const image = await loadImage('./path/to/person1.jpg');
      // const faceId = await whoami.registerPerson(image, 'John Doe');
      console.log('To register: const faceId = await whoami.registerPerson(imageElement, "Person Name");');
    } catch (error) {
      console.log('Image registration example (requires image file)');
    }

    // Example 2: Recognize a person from an image
    console.log('\n--- Example 2: Recognize Person ---');
    try {
      // const image = await loadImage('./path/to/test.jpg');
      // const result = await whoami.recognize(image);
      console.log('To recognize: const result = await whoami.recognize(imageElement);');
    } catch (error) {
      console.log('Recognition example (requires image file)');
    }

    // Example 3: List all registered persons
    console.log('\n--- Example 3: List Registered Persons ---');
    const persons = await whoami.listRegistered();
    console.log('Registered persons:', persons);

    // Example 4: Get system status
    console.log('\n--- Example 4: System Status ---');
    const status = whoami.getStatus();
    console.log('Status:', JSON.stringify(status, null, 2));

    console.log('\n='.repeat(50));
    console.log('Examples completed!');
    console.log('='.repeat(50));

  } catch (error) {
    console.error('Error:', error.message);
    process.exit(1);
  }
}

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}

export { main };
