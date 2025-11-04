# WhoAmI - Secure Facial Recognition for Jetson Nano

A highly secure, real-time facial recognition system designed for Jetson Nano robots using Gun.js for decentralized data storage with hardware-backed encryption.

## 🔒 Security Features

This system implements multiple layers of security to ensure that facial recognition data cannot be reverse engineered and is accessible only by the specific robot hardware:

1. **Hardware-Backed Encryption**: All encryption keys are derived from device-specific hardware identifiers (CPU serial number + MAC address)
2. **Double-Layer Encryption**: Data is encrypted twice - first with hardware-derived keys, then with Gun.js SEA encryption
3. **Zero Plaintext Storage**: No encryption keys or sensitive data stored in code or configuration files
4. **Device-Specific Identity**: Each robot has a unique cryptographic identity tied to its hardware
5. **Reverse Engineering Resistance**: Without the specific hardware, encrypted data cannot be decrypted
6. **Authenticated Encryption**: Uses AES-256-GCM for tamper-proof encryption

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    WhoAmI System                        │
├─────────────────────────────────────────────────────────┤
│  Application Layer                                      │
│  - Facial Recognition (face-api.js)                    │
│  - Real-time Processing                                │
├─────────────────────────────────────────────────────────┤
│  Security Layer (Hardware-Backed)                       │
│  - AES-256-GCM Encryption                              │
│  - Hardware Key Derivation (CPU Serial + MAC)          │
│  - Scrypt Key Derivation Function                      │
├─────────────────────────────────────────────────────────┤
│  Database Layer (Gun.js)                                │
│  - SEA Encryption (Second Layer)                        │
│  - Decentralized P2P Database                          │
│  - Local-First Storage                                 │
├─────────────────────────────────────────────────────────┤
│  Hardware Layer (Jetson Nano)                           │
│  - GPU Acceleration for CV                              │
│  - Hardware Identifiers                                │
│  - Camera Interface                                     │
└─────────────────────────────────────────────────────────┘
```

## 📋 Requirements

### Hardware
- NVIDIA Jetson Nano (or compatible)
- USB Camera or CSI Camera
- Minimum 4GB RAM recommended

### Software
- Node.js >= 18.0.0
- OpenCV (for opencv4nodejs)
- CUDA support (for GPU acceleration)

## 🚀 Installation

1. **Clone the repository**
```bash
git clone https://github.com/alanchelmickjr/whoami.git
cd whoami
```

2. **Install dependencies**
```bash
npm install
```

3. **Download face-api.js models**
```bash
mkdir -p models
cd models
# Download models from https://github.com/vladmandic/face-api
# Required models:
# - ssdMobilenetv1
# - faceLandmark68Net
# - faceRecognitionNet
```

4. **Configure the system**
```bash
# Edit config/config.json if needed
# Default configuration works for most setups
```

## 💻 Usage

### Basic Usage

```javascript
import { whoami } from './src/index.js';

// Initialize the system
await whoami.initialize('./config/config.json');

// Register a person
const image = await loadImage('path/to/person.jpg');
const faceId = await whoami.registerPerson(image, 'John Doe');

// Recognize a person
const result = await whoami.recognize(image);
if (result.recognized) {
  console.log(`Hello, ${result.personName}!`);
}

// List registered persons
const persons = await whoami.listRegistered();
console.log(persons);
```

### Running the Example

```bash
node examples/example.js
```

## 🔐 Security Details

### How Hardware-Backed Security Works

1. **Key Derivation**: 
   - Reads CPU serial from `/proc/cpuinfo`
   - Reads MAC address from network interfaces
   - Combines and hashes to create unique hardware fingerprint
   - Uses scrypt (memory-hard KDF) to derive encryption keys

2. **Double Encryption**:
   ```
   Plain Data → Hardware Encryption → Gun.js SEA Encryption → Storage
   ```

3. **Tamper Detection**:
   - Uses GCM authentication tags
   - Any tampering causes decryption to fail
   - No silent data corruption possible

### Why This is Secure

- **No Key Extraction**: Keys derived from hardware on-the-fly, never stored
- **Device-Locked**: Data encrypted on one device cannot be decrypted on another
- **Memory-Hard KDF**: Resistant to brute force attacks using scrypt
- **Layered Defense**: Even if one encryption layer is broken, second layer protects data
- **P2P Security**: Gun.js provides additional SEA (Security, Encryption, Authorization)

## 📁 Project Structure

```
whoami/
├── src/
│   ├── index.js              # Main application entry point
│   ├── secureKeyManager.js   # Hardware-backed key management
│   ├── secureDatabase.js     # Gun.js database with encryption
│   └── facialRecognition.js  # Face detection and recognition
├── config/
│   └── config.json           # System configuration
├── examples/
│   └── example.js            # Usage examples
├── models/                   # Face-api.js models (download separately)
├── data/                     # Gun.js database storage (auto-created)
└── package.json              # Node.js dependencies
```

## ⚙️ Configuration

Edit `config/config.json`:

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

- `minConfidence`: Minimum confidence for face detection (0-1)
- `descriptorThreshold`: Maximum distance for face matching (lower = stricter)
- `peers`: Array of Gun.js peers for replication (empty = local only)

## 🛡️ Security Best Practices

1. **Keep models secure**: Face-api.js models should be stored securely
2. **Limit network access**: Run in isolated mode (no peers) for maximum security
3. **Physical security**: Protect the Jetson Nano from physical tampering
4. **Regular updates**: Keep dependencies updated for security patches
5. **Audit logs**: Monitor access patterns and recognition events

## 🔧 Troubleshooting

### Models not loading
- Ensure models are in the correct directory
- Check models are for the correct version of face-api.js
- Verify file permissions

### GPU acceleration not working
- Verify CUDA installation: `nvcc --version`
- Check OpenCV CUDA support: `pkg-config --modversion opencv4`
- Ensure proper TensorRT configuration

### Camera not detected
- Check camera connection: `ls /dev/video*`
- Verify camera permissions
- Test with v4l2-ctl or other camera tools

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

This is a security-focused project. If you find vulnerabilities, please report them responsibly.

## ⚠️ Disclaimer

This system is designed for authorized use only. Ensure compliance with local privacy laws and regulations when deploying facial recognition technology.
