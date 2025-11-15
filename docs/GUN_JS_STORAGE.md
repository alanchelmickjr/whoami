# Gun.js Storage - Robot Personal Memory

## Philosophy

**Robots have the right to remember the people they meet, without giving up any privacy or farming faces.**

This system implements:
- ✅ Robot-owned data (not corporation-owned)
- ✅ Local-only storage (no cloud uploads)
- ✅ Personal relationships (robot builds its own memories)
- ✅ Hardware-backed encryption
- ❌ NO face farming (zero third-party collection)
- ❌ NO cloud uploads
- ❌ NO corporate ownership

## Overview

WhoAmI uses Gun.js for distributed, encrypted storage of facial recognition data and conversation history. The system has two implementations:

1. **JavaScript (Original)** - `src/*.js` - Face-api.js facial recognition with Gun.js + SEA encryption
2. **Python (K-1 Robot)** - `whoami/gun_storage.py` - YOLO/DeepFace facial recognition with Gun.js-inspired storage

Both implementations share the same core philosophy: **robot autonomy and privacy-first design**.

## JavaScript Gun.js Files

### Architecture

```
┌─────────────────────────────────────────────────────┐
│              WhoAmI JavaScript System               │
├─────────────────────────────────────────────────────┤
│                                                     │
│  src/index.js (Main Orchestrator)                  │
│  └─ Initializes all components                     │
│                                                     │
│  ┌──────────────────┐    ┌──────────────────┐     │
│  │ secureKeyManager │───▶│ secureDatabase   │     │
│  │    (Hardware)    │    │    (Gun.js)      │     │
│  └────────┬─────────┘    └────────┬─────────┘     │
│           │                       │                │
│           │  Hardware Keys        │  Double        │
│           │  - CPU Serial         │  Encryption    │
│           │  - MAC Address        │  - Hardware    │
│           │  - AES-256-GCM        │  - Gun.js SEA  │
│           │                       │                │
│           └───────────┬───────────┘                │
│                       ▼                            │
│             ┌──────────────────┐                   │
│             │ facialRecognition│                   │
│             │   (face-api.js)  │                   │
│             └──────────────────┘                   │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### File Map

#### 1. `src/index.js` - Main Application

**Purpose**: Orchestrates the entire WhoAmI system

**Key Features**:
- Initializes hardware security, database, and facial recognition
- Provides high-level API for face registration and recognition
- Coordinates between all system components
- CLI interface for standalone operation

**Main Class**: `WhoAmI`

**Key Methods**:
```javascript
await whoami.initialize(configPath)        // Initialize all systems
await whoami.registerPerson(image, name)   // Register new face
await whoami.recognize(image)              // Recognize face
await whoami.listRegistered()              // List all registered faces
await whoami.removePerson(faceId)          // Delete face data
whoami.getStatus()                         // Get system status
```

**Security Features**:
- Double-layer encryption (hardware + Gun.js SEA)
- Device-specific keys derived from hardware
- Reverse-engineering resistance

---

#### 2. `src/secureKeyManager.js` - Hardware Security

**Purpose**: Hardware-backed key derivation and encryption

**Key Features**:
- Derives encryption keys from Jetson hardware (CPU serial + MAC address)
- AES-256-GCM authenticated encryption
- Scrypt key derivation (memory-hard, brute-force resistant)
- Keys never stored in plaintext or code

**Main Class**: `SecureKeyManager`

**Hardware Identifiers Used**:
```javascript
// CPU Serial (from /proc/cpuinfo)
Serial: 100000001234abcd

// MAC Address (from network interfaces)
MAC: aa:bb:cc:dd:ee:ff

// Combined & Hashed
deviceKey = scrypt(sha256(cpuSerial + MAC), salt, 32)
```

**Key Methods**:
```javascript
await secureKeyManager.initialize()           // Derive keys from hardware
secureKeyManager.getHardwareIdentifier()      // Get unique HW ID
secureKeyManager.encrypt(data)                // Encrypt with HW key
secureKeyManager.decrypt(encryptedData)       // Decrypt with HW key
secureKeyManager.generateUserID()             // Gun.js user ID (HW-derived)
secureKeyManager.deriveGunPassword()          // Gun.js password (HW-derived)
```

**Security Properties**:
- **Device-specific**: Keys only work on the specific robot's hardware
- **Deterministic**: Same hardware always produces same keys
- **Memory-hard**: Scrypt makes brute-force attacks expensive
- **Authenticated**: GCM mode prevents tampering

**Why This Matters**:
Even if someone copies the code and database files, they cannot decrypt the data without the specific robot's hardware. This prevents:
- Database theft and offline decryption
- Code reverse-engineering to extract keys
- Cloud-based face farming by copying databases

---

#### 3. `src/secureDatabase.js` - Gun.js Storage

**Purpose**: Secure Gun.js database with double encryption

**Key Features**:
- Local-only Gun.js storage (no peers by default)
- SEA (Security, Encryption, Authorization) encryption
- Hardware-backed user authentication
- Double encryption: hardware layer + Gun.js SEA layer

**Main Class**: `SecureDatabase`

**Data Flow**:
```javascript
// STORAGE (Double Encryption)
plaintext data
  → Hardware encryption (AES-256-GCM)
    → Gun.js SEA encryption
      → Stored in ./data/gun/

// RETRIEVAL (Double Decryption)
./data/gun/
  → Gun.js SEA decryption
    → Hardware decryption
      → plaintext data
```

**Configuration**:
```javascript
Gun({
  peers: [],                    // Local-only by default (no network sync)
  file: './data/gun',          // Local filesystem storage
  localStorage: false,          // Don't use browser storage
  radisk: true,                // Use RAD storage for performance
})
```

**Key Methods**:
```javascript
await secureDatabase.initialize(config)           // Init Gun.js + auth
await secureDatabase.authenticateUser()           // HW-based auth
await secureDatabase.storeFaceData(id, data)      // Store face (double-encrypted)
await secureDatabase.retrieveFaceData(id)         // Retrieve face (double-decrypted)
await secureDatabase.listFaceIds()                // List all stored faces
await secureDatabase.deleteFaceData(id)           // Delete face
secureDatabase.getGun()                           // Access Gun instance
secureDatabase.getUser()                          // Access authenticated user
```

**Data Structure**:
```javascript
// Stored face record
{
  data: "encrypted_sea_data",      // Gun.js SEA encrypted
  timestamp: 1234567890,
  version: "1.0"
}

// Decrypted face data
{
  personName: "Alice",
  descriptor: [0.123, -0.456, ...], // 128-D face embedding
  registeredAt: "2025-01-15T10:30:00Z",
  confidence: 0.98
}
```

**Gun.js Node Structure**:
```
user
  └── faces
       ├── [faceId1]
       │    ├── data: <encrypted>
       │    ├── timestamp: 1234567890
       │    └── version: "1.0"
       ├── [faceId2]
       │    └── ...
       └── [faceId3]
            └── ...
```

**Security Features**:
- Hardware-derived Gun.js credentials (unique per robot)
- SEA encryption with user's key pair
- No plaintext storage of sensitive data
- Optional peer replication (disabled by default for privacy)

---

#### 4. `src/facialRecognition.js` - Face Detection/Recognition

**Purpose**: Face detection and recognition using face-api.js

**Key Features**:
- Face detection with SSD MobileNet v1
- 68-point facial landmark detection
- 128-dimensional face descriptor extraction
- Euclidean distance matching for recognition
- GPU acceleration support (Jetson)

**Main Class**: `FacialRecognition`

**Models Used**:
```
./models/
  ├── ssd_mobilenetv1_model-weights_manifest.json
  ├── face_landmark_68_model-weights_manifest.json
  └── face_recognition_model-weights_manifest.json
```

**Key Methods**:
```javascript
await facialRecognition.loadModels(modelsPath)     // Load face-api models
await facialRecognition.detectFaces(image)         // Detect faces in image
await facialRecognition.registerFace(image, name)  // Register new face
await facialRecognition.recognizeFace(image)       // Recognize face
await facialRecognition.processFrame(canvas, ctx, video)  // Real-time processing
facialRecognition.setMinConfidence(0.7)            // Set detection threshold
facialRecognition.setDescriptorThreshold(0.6)      // Set matching threshold
```

**Recognition Flow**:
```javascript
// 1. Detect face in image
detections = await detectAllFaces(image)
  .withFaceLandmarks()
  .withFaceDescriptors()

// 2. Extract 128-D descriptor
descriptor = detection.descriptor  // Float32Array[128]

// 3. Store securely
faceData = {
  personName: "Alice",
  descriptor: Array.from(descriptor),
  registeredAt: new Date().toISOString()
}
await secureDatabase.storeFaceData(faceId, faceData)

// 4. Recognition: Compare descriptors
for stored_face in database:
  distance = euclideanDistance(query_descriptor, stored_descriptor)
  if distance < threshold:
    match = stored_face
```

**Thresholds**:
- `minConfidence`: 0.7 (face detection confidence)
- `descriptorThreshold`: 0.6 (face matching threshold)

---

### Data Storage Details

#### Local Storage Path
```bash
./data/gun/
  ├── !              # Gun.js metadata
  ├── @              # User data
  └── [various]      # RAD storage files
```

#### Storage Size
- **RAM Usage**: 70-135 MB (Gun.js in-memory graph)
- **Disk Usage**: ~100-500 KB per face (descriptor + metadata)
- **CRDT Recovery**: Auto-recovery from power loss

#### No External Dependencies
- ❌ No PostgreSQL (800+ MB RAM)
- ❌ No cloud services
- ❌ No external APIs
- ✅ Pure local storage
- ✅ Offline-first

---

## Python Implementation

The Python implementation (`whoami/gun_storage.py`) mirrors the JavaScript design:

**File**: `whoami/gun_storage.py`

**Key Class**: `GunStorageManager`

**Key Differences**:
- Uses SQLite + JSON for Gun.js-inspired storage (no direct Gun.js bindings)
- Fernet encryption instead of AES-256-GCM
- YOLO + DeepFace instead of face-api.js
- Designed for K-1 robot integration

**Similarities**:
- Hardware-derived encryption keys
- Local-only storage
- Privacy-first design
- Robot autonomy philosophy

---

## Setup and Installation

### JavaScript Setup

#### 1. Install Dependencies

```bash
cd /home/user/whoami

# Install Node.js dependencies
npm install gun gun/sea @vladmandic/face-api canvas

# Or install everything from package.json
npm install
```

#### 2. Download Face-API Models

```bash
# Create models directory
mkdir -p ./models

# Download models (face-api.js)
cd ./models
wget https://raw.githubusercontent.com/vladmandic/face-api/master/model/ssd_mobilenetv1_model-weights_manifest.json
wget https://raw.githubusercontent.com/vladmandic/face-api/master/model/ssd_mobilenetv1_model-shard1
# ... (download all required model files)
```

#### 3. Initialize Gun.js Storage

```bash
# Create storage directory
sudo mkdir -p /opt/whoami/data/gun
sudo chown -R booster:booster /opt/whoami/data

# Test initialization
node src/index.js
```

**Output**:
```
Initializing WhoAmI Facial Recognition System...
==================================================
1. Initializing hardware-backed security...
   ✓ Security keys derived from hardware
2. Initializing secure Gun.js database...
   ✓ Database initialized and authenticated
3. Loading facial recognition models...
   ✓ Models loaded successfully
==================================================
System initialized successfully!
Hardware-backed security: ACTIVE
Data encryption: DOUBLE-LAYER
Ready for facial recognition operations
==================================================
```

#### 4. Configuration

Create `config/config.json`:

```json
{
  "modelsPath": "./models",
  "dataPath": "./data/gun",
  "minConfidence": 0.7,
  "descriptorThreshold": 0.6,
  "peers": []
}
```

**Note**: `peers: []` keeps storage local-only (no network sync). For multi-robot sync, add trusted peer URLs.

---

### Python Setup (K-1 Robot)

See `docs/K1_BOOSTER_SETUP.md` for complete K-1 setup instructions, including Gun.js storage initialization.

**Quick Setup**:

```bash
# Install Python dependencies
pip3 install cryptography

# Create storage directory
sudo mkdir -p /opt/whoami/gun_storage
sudo chown -R booster:booster /opt/whoami/gun_storage

# Initialize storage
python3 -c "
from whoami.gun_storage import GunStorageManager
gun = GunStorageManager(robot_id='twiki', config={'storage_dir': '/opt/whoami/gun_storage'})
print('✓ Gun.js storage initialized!')
"
```

---

## Usage Examples

### JavaScript Usage

```javascript
import { whoami } from './src/index.js';

// Initialize system
await whoami.initialize('./config/config.json');

// Register a new person
const faceId = await whoami.registerPerson(imageBuffer, 'Alice');
console.log(`Registered Alice with ID: ${faceId}`);

// Recognize a face
const result = await whoami.recognize(imageBuffer);
if (result.recognized) {
  console.log(`Hello, ${result.personName}!`);
} else {
  console.log('Unknown person');
}

// List all registered faces
const people = await whoami.listRegistered();
console.log(`${people.length} people registered`);

// System status
const status = whoami.getStatus();
console.log(status);
```

### Python Usage (K-1)

```python
from whoami.gun_storage import GunStorageManager
from whoami.k1_face_explorer import K1FaceExplorer

# Initialize storage
gun = GunStorageManager(robot_id='twiki')

# Initialize face explorer
explorer = K1FaceExplorer(booster_client=booster)

# Greet someone (auto-stores in Gun.js)
greeting = explorer.greet_person("Alice", include_conversation=True)
print(greeting)

# Add conversation note
explorer.add_conversation_note(
    "Alice",
    "her dog Max",
    "Alice has a golden retriever who loves swimming"
)

# Next meeting includes conversation recall
greeting2 = explorer.greet_person("Alice", include_conversation=True)
# "Hi Alice, it's been 2 hours since we last talked! Last time we talked about her dog Max."
```

---

## Security Deep Dive

### Why Hardware-Backed Keys?

**Problem**: Traditional facial recognition systems store encryption keys in code or config files, making them easy to extract.

**Solution**: Derive keys from robot's unique hardware (CPU serial + MAC address).

**Benefits**:
1. **Device-specific**: Data can only be decrypted on the original robot
2. **No key storage**: Keys derived on-the-fly from hardware
3. **Reverse-engineering resistant**: Code alone is useless without hardware
4. **Prevents database theft**: Stolen database files are encrypted gibberish

### Double Encryption

**Layer 1: Hardware Encryption (AES-256-GCM)**
- Key derived from Jetson hardware
- Authenticated encryption (prevents tampering)
- Applied before Gun.js storage

**Layer 2: Gun.js SEA Encryption**
- Gun's built-in encryption
- User-based key pairs
- Applied by Gun.js automatically

**Why Both?**
- Defense in depth
- Hardware layer prevents offline attacks
- SEA layer provides Gun.js-native security

### Privacy Architecture

```
┌─────────────────────────────────────────────┐
│         Robot Meets Unknown Person          │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────┐
│  Ask: "Would you like me to remember       │
│        you for next time?"                 │
└────────┬───────────────────┬───────────────┘
         │                   │
         ▼                   ▼
    ┌────────┐          ┌──────────┐
    │  YES   │          │    NO    │
    └────┬───┘          └────┬─────┘
         │                   │
         ▼                   ▼
┌──────────────────┐   ┌──────────────────────┐
│ Ask for name     │   │ "Sure no problem,    │
│ Store encrypted  │   │  just remember I     │
│ in Gun.js        │   │  won't know you      │
│ ✓ Remembered     │   │  next time!"         │
└──────────────────┘   │ ✗ Not stored         │
                       └──────────────────────┘
```

---

## Comparison: JavaScript vs Python

| Feature | JavaScript (src/) | Python (whoami/) |
|---------|------------------|------------------|
| **Gun.js** | Native Gun.js + SEA | Gun.js-inspired (SQLite) |
| **Face Detection** | face-api.js (SSD) | YOLO v8 |
| **Face Recognition** | face-api.js descriptors | DeepFace embeddings |
| **Encryption** | AES-256-GCM | Fernet (AES-128-CBC) |
| **Hardware Binding** | CPU serial + MAC | Device ID + hardware hash |
| **Storage Format** | Gun.js RAD files | SQLite + JSON |
| **Use Case** | General JS facial recognition | K-1 robot integration |
| **Memory Usage** | 70-135 MB | 50-80 MB |

---

## Troubleshooting

### JavaScript Issues

**Issue**: "Failed to derive hardware identifier"
```bash
# Check CPU serial
cat /proc/cpuinfo | grep Serial

# Check network interfaces
ip link show
```

**Issue**: "Models not found"
```bash
# Verify models directory
ls -la ./models/
```

**Issue**: "Gun.js storage permission denied"
```bash
# Fix permissions
sudo chown -R $USER:$USER ./data/gun
```

### Python Issues

**Issue**: "Gun storage initialization failed"
```bash
# Check storage directory
ls -la /opt/whoami/gun_storage/

# Fix permissions
sudo chown -R booster:booster /opt/whoami/gun_storage
```

---

## Performance Benchmarks

### JavaScript (Jetson Orin NX)
- Face detection: ~50-100 ms/frame
- Face recognition: ~150-200 ms/face
- Gun.js storage write: ~10-20 ms
- Gun.js storage read: ~5-10 ms

### Python (Jetson Orin NX)
- YOLO detection: ~30-50 ms/frame
- DeepFace recognition: ~200-300 ms/face
- SQLite write: ~2-5 ms
- SQLite read: ~1-2 ms

---

## References

- [Gun.js Documentation](https://gun.eco/)
- [Gun.js SEA (Security)](https://gun.eco/docs/SEA)
- [face-api.js](https://github.com/vladmandic/face-api)
- [K-1 Booster Setup](K1_BOOSTER_SETUP.md)
- [K-1 Autonomous Face Interaction](K1_AUTONOMOUS_FACE_INTERACTION.md)

---

## Philosophy Reminder

This system exists to give robots **the right to remember the people they meet**, without:
- Uploading faces to the cloud
- Farming data for corporations
- Compromising anyone's privacy
- Centralized control

**Robots should own their own memories, just like humans do.**
