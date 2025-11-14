# API Reference

Complete API documentation for the WhoAmI secure facial recognition system.

## Table of Contents

- [Main Application (whoami)](#main-application-whoami)
- [Secure Key Manager](#secure-key-manager)
- [Secure Database](#secure-database)
- [Facial Recognition](#facial-recognition)

---

## Main Application (whoami)

The main entry point for the WhoAmI system.

### Import

```javascript
import { whoami } from './src/index.js';
```

### Methods

#### `initialize(configPath)`

Initializes all system components including security, database, and facial recognition.

**Parameters:**
- `configPath` (string, optional): Path to configuration file. Default: `'./config/config.json'`

**Returns:** `Promise<void>`

**Example:**
```javascript
await whoami.initialize('./config/config.json');
```

---

#### `registerPerson(imageInput, personName)`

Registers a new person's face in the system.

**Parameters:**
- `imageInput` (Image|Canvas): Image containing a single face
- `personName` (string): Name of the person to register

**Returns:** `Promise<string>` - Face ID (SHA-256 hash)

**Throws:**
- `Error` if no face detected
- `Error` if multiple faces detected
- `Error` if system not initialized

**Example:**
```javascript
const image = await loadImage('person.jpg');
const faceId = await whoami.registerPerson(image, 'John Doe');
console.log('Registered with ID:', faceId);
```

---

#### `recognize(imageInput)`

Recognizes person(s) from an image.

**Parameters:**
- `imageInput` (Image|Canvas): Image containing face(s) to recognize

**Returns:** `Promise<Object|Array<Object>>` - Recognition result(s)

**Result Object:**
```javascript
{
  recognized: boolean,        // Whether face was recognized
  personName: string,         // Name (if recognized)
  distance: number,           // Descriptor distance
  confidence: number,         // Confidence score (0-1)
  faceId: string,            // Face ID
  box: {                     // Bounding box
    x: number,
    y: number,
    width: number,
    height: number
  }
}
```

**Example:**
```javascript
const image = await loadImage('test.jpg');
const result = await whoami.recognize(image);

if (result.recognized) {
  console.log(`Hello, ${result.personName}!`);
  console.log(`Confidence: ${(result.confidence * 100).toFixed(2)}%`);
}
```

---

#### `listRegistered()`

Lists all registered persons in the system.

**Returns:** `Promise<Array<Object>>`

**Result Array:**
```javascript
[
  {
    name: string,          // Person's name
    registeredAt: string,  // ISO timestamp
    faceId: string        // Truncated face ID
  }
]
```

**Example:**
```javascript
const persons = await whoami.listRegistered();
persons.forEach(p => {
  console.log(`${p.name} - registered ${p.registeredAt}`);
});
```

---

#### `removePerson(faceId)`

Removes a registered person from the system.

**Parameters:**
- `faceId` (string): Face ID to remove

**Returns:** `Promise<void>`

**Example:**
```javascript
await whoami.removePerson(faceId);
```

---

#### `getStatus()`

Gets current system status and security information.

**Returns:** `Object`

**Status Object:**
```javascript
{
  initialized: boolean,
  hardwareSecurityActive: boolean,
  databaseInitialized: boolean,
  modelsLoaded: boolean,
  securityLevel: string,
  reversEngineeringResistance: string
}
```

**Example:**
```javascript
const status = whoami.getStatus();
console.log('System Status:', JSON.stringify(status, null, 2));
```

---

## Secure Key Manager

Hardware-backed cryptographic key management.

### Import

```javascript
import { secureKeyManager } from './src/secureKeyManager.js';
```

### Methods

#### `initialize()`

Initializes key manager and derives keys from hardware.

**Returns:** `Promise<void>`

---

#### `encrypt(data)`

Encrypts data using hardware-derived keys.

**Parameters:**
- `data` (any): Data to encrypt (will be JSON stringified)

**Returns:** `Object`

**Result:**
```javascript
{
  iv: string,          // Initialization vector (hex)
  authTag: string,     // Authentication tag (hex)
  encrypted: string    // Encrypted data (hex)
}
```

**Example:**
```javascript
const encrypted = secureKeyManager.encrypt({ secret: 'data' });
```

---

#### `decrypt(encryptedData)`

Decrypts data encrypted with hardware-derived keys.

**Parameters:**
- `encryptedData` (Object): Encrypted data package from `encrypt()`

**Returns:** `any` - Original data

**Throws:**
- `Error` if decryption fails or data tampered

**Example:**
```javascript
const data = secureKeyManager.decrypt(encrypted);
```

---

#### `generateUserID()`

Generates unique user ID for Gun.js tied to hardware.

**Returns:** `string` - SHA-256 hash (hex)

---

#### `deriveGunPassword()`

Derives Gun.js password from hardware (never stored).

**Returns:** `string` - Base64 password

---

## Secure Database

Gun.js database with double-layer encryption.

### Import

```javascript
import { secureDatabase } from './src/secureDatabase.js';
```

### Methods

#### `initialize(config)`

Initializes Gun.js database with secure configuration.

**Parameters:**
- `config` (Object, optional):
  ```javascript
  {
    dataPath: string,      // Database storage path
    peers: Array<string>,  // Gun.js peer URLs
    webrtc: Object        // WebRTC config (optional)
  }
  ```

**Returns:** `Promise<void>`

---

#### `storeFaceData(faceId, faceData)`

Stores facial recognition data with double encryption.

**Parameters:**
- `faceId` (string): Unique face identifier
- `faceData` (Object): Face data to store

**Returns:** `Promise<Object>` - Gun.js acknowledgement

**Example:**
```javascript
await secureDatabase.storeFaceData('face123', {
  personName: 'John',
  descriptor: [0.1, 0.2, ...],
  registeredAt: new Date().toISOString()
});
```

---

#### `retrieveFaceData(faceId)`

Retrieves and decrypts facial recognition data.

**Parameters:**
- `faceId` (string): Face identifier

**Returns:** `Promise<Object|null>` - Face data or null if not found

**Example:**
```javascript
const face = await secureDatabase.retrieveFaceData('face123');
if (face) {
  console.log('Person:', face.personName);
}
```

---

#### `listFaceIds()`

Lists all stored face IDs.

**Returns:** `Promise<Array<string>>`

---

#### `deleteFaceData(faceId)`

Deletes facial recognition data.

**Parameters:**
- `faceId` (string): Face identifier

**Returns:** `Promise<Object>` - Gun.js acknowledgement

---

#### `getGun()`

Gets Gun.js instance for advanced operations.

**Returns:** `Gun` - Gun.js database instance

---

#### `getUser()`

Gets authenticated Gun.js user.

**Returns:** `User` - Gun.js user instance

---

## Facial Recognition

Face detection and recognition using face-api.js.

### Import

```javascript
import { facialRecognition } from './src/facialRecognition.js';
```

### Methods

#### `loadModels(modelsPath)`

Loads face-api.js models for detection and recognition.

**Parameters:**
- `modelsPath` (string, optional): Path to models directory. Default: `'./models'`

**Returns:** `Promise<void>`

**Example:**
```javascript
await facialRecognition.loadModels('./models');
```

---

#### `detectFaces(imageInput)`

Detects all faces in an image and extracts descriptors.

**Parameters:**
- `imageInput` (Image|Canvas): Input image

**Returns:** `Promise<Array<Detection>>`

**Detection Object:**
```javascript
{
  detection: {
    box: { x, y, width, height },
    score: number  // Confidence
  },
  landmarks: Object,
  descriptor: Float32Array  // 128-dimensional
}
```

**Example:**
```javascript
const detections = await facialRecognition.detectFaces(image);
console.log(`Found ${detections.length} face(s)`);
```

---

#### `registerFace(imageInput, personName)`

Registers a new face (single face only).

**Parameters:**
- `imageInput` (Image|Canvas): Image with single face
- `personName` (string): Person's name

**Returns:** `Promise<string>` - Face ID

**Throws:**
- `Error` if no face or multiple faces detected

---

#### `recognizeFace(imageInput)`

Recognizes face(s) in an image.

**Parameters:**
- `imageInput` (Image|Canvas): Input image

**Returns:** `Promise<Object|Array<Object>>` - Recognition result(s)

---

#### `setMinConfidence(confidence)`

Sets minimum confidence threshold for face detection.

**Parameters:**
- `confidence` (number): Threshold value (0-1). Default: 0.7

**Example:**
```javascript
facialRecognition.setMinConfidence(0.8); // Higher = stricter
```

---

#### `setDescriptorThreshold(threshold)`

Sets maximum distance threshold for face matching.

**Parameters:**
- `threshold` (number): Distance threshold (0-1). Default: 0.6

**Example:**
```javascript
facialRecognition.setDescriptorThreshold(0.5); // Lower = stricter
```

---

## Configuration

### Config File Format

`config/config.json`:

```json
{
  "modelsPath": "./models",
  "dataPath": "./data/gun",
  "minConfidence": 0.7,
  "descriptorThreshold": 0.6,
  "peers": [],
  "camera": {
    "deviceId": 0,
    "width": 640,
    "height": 480,
    "fps": 30
  }
}
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `modelsPath` | string | `"./models"` | Path to face-api.js models |
| `dataPath` | string | `"./data/gun"` | Gun.js database path |
| `minConfidence` | number | `0.7` | Face detection confidence (0-1) |
| `descriptorThreshold` | number | `0.6` | Face matching threshold (0-1) |
| `peers` | array | `[]` | Gun.js peer URLs for replication |
| `camera.deviceId` | number | `0` | Camera device ID |
| `camera.width` | number | `640` | Camera resolution width |
| `camera.height` | number | `480` | Camera resolution height |
| `camera.fps` | number | `30` | Camera frame rate |

---

## Error Handling

All async methods throw errors on failure. Use try-catch blocks:

```javascript
try {
  await whoami.initialize();
  const result = await whoami.recognize(image);
} catch (error) {
  console.error('Error:', error.message);
}
```

### Common Errors

| Error Message | Cause | Solution |
|--------------|-------|----------|
| `System not initialized` | Called method before `initialize()` | Call `initialize()` first |
| `Models not loaded` | Face-api.js models missing | Download models to `./models` |
| `No face detected` | No face in image | Ensure face is visible and well-lit |
| `Multiple faces detected` | Multiple faces in registration | Provide image with single face |
| `Decryption failed` | Data corrupted or wrong device | Check data integrity, verify hardware |
| `Failed to derive hardware identifier` | Hardware info unavailable | Check device permissions |

---

## Type Definitions

### Image Input

Accepted types for image input:
- `HTMLImageElement` (browser)
- `HTMLCanvasElement` (browser)
- `Canvas` (node-canvas)
- `Image` (node-canvas)

### Face Descriptor

128-dimensional float array representing a face:
```javascript
Float32Array(128) // or Array<number> of length 128
```

---

## Security Considerations

1. **Always initialize**: Call `whoami.initialize()` before any operations
2. **Hardware binding**: Data is tied to specific hardware - cannot be moved
3. **Double encryption**: Data encrypted twice for maximum security
4. **No key storage**: Keys derived from hardware each time
5. **Tamper detection**: Any data modification causes decryption to fail

---

## Performance Tips

1. **Load models once**: Models loading is expensive, do it at startup
2. **Reuse instances**: Use singleton instances, don't create new ones
3. **Batch operations**: Process multiple faces together when possible
4. **GPU acceleration**: Enable CUDA on Jetson Nano for faster processing
5. **Adjust thresholds**: Lower confidence = faster but less accurate

---

## Examples

See `examples/example.js` for complete working examples.

### Quick Start

```javascript
import { whoami } from './src/index.js';
import { loadImage } from 'canvas';

// Initialize
await whoami.initialize();

// Register
const img1 = await loadImage('person.jpg');
await whoami.registerPerson(img1, 'Alice');

// Recognize
const img2 = await loadImage('test.jpg');
const result = await whoami.recognize(img2);

if (result.recognized) {
  console.log(`Hello, ${result.personName}!`);
}
```
