# Security Architecture

This document details the security mechanisms implemented in the WhoAmI facial recognition system to prevent reverse engineering and ensure data is only accessible by the specific robot hardware.

## Threat Model

The system is designed to protect against:

1. **Data Extraction**: Attacker gains physical or remote access to storage
2. **Code Reverse Engineering**: Attacker analyzes source code to extract keys
3. **Database Cloning**: Attacker copies database files to another device
4. **Memory Dumping**: Attacker dumps process memory to extract keys
5. **Side-Channel Attacks**: Timing attacks, power analysis, etc.
6. **Man-in-the-Middle**: Network interception of data

## Defense Mechanisms

### 1. Hardware-Backed Key Derivation

**Implementation**: `src/secureKeyManager.js`

```
Hardware Identifiers → SHA-256 → Scrypt KDF → Device Key → Encryption Key
```

**Components**:
- **CPU Serial Number**: Read from `/proc/cpuinfo` (Jetson-specific)
- **MAC Address**: Network interface hardware address
- **Scrypt KDF**: Memory-hard key derivation (resistant to brute force)
- **Multiple Derivation Layers**: Device key → Encryption key → Gun.js password

**Security Properties**:
- Keys never stored in code or files
- Different on every device (hardware-unique)
- Cannot be extracted from source code
- Memory-hard derivation prevents brute force
- Keys regenerated from hardware each session

### 2. Double-Layer Encryption

**Layer 1: Hardware-Backed AES-256-GCM**
```javascript
Plain Data → AES-256-GCM Encryption → Encrypted Data Package
```

Properties:
- **Algorithm**: AES-256-GCM (authenticated encryption)
- **Key Size**: 256 bits
- **IV**: Random 16 bytes per encryption
- **Auth Tag**: 16 bytes for tamper detection
- **Mode**: GCM provides both confidentiality and authenticity

**Layer 2: Gun.js SEA Encryption**
```javascript
Encrypted Package → Gun.js SEA.encrypt → Double-Encrypted Storage
```

Properties:
- **SEA**: Security, Encryption, Authorization module
- **User Authentication**: Required for decryption
- **Per-User Encryption**: Different keys per robot

**Combined Flow**:
```
Face Data → Hardware Encrypt → SEA Encrypt → Gun.js Store
Gun.js Retrieve → SEA Decrypt → Hardware Decrypt → Face Data
```

### 3. Key Derivation Process

Detailed derivation chain:

```
1. Hardware Fingerprint
   CPU Serial + MAC Address → SHA-256 Hash

2. Device Key
   Hardware Fingerprint + Salt → Scrypt(N=16384, r=8, p=1) → 32 bytes

3. Encryption Key
   Device Key + Salt → Scrypt(N=16384, r=8, p=1) → 32 bytes

4. Gun.js Credentials
   Device Key + Context → SHA-256/SHA-512 → User ID + Password
```

**Why Scrypt?**
- Memory-hard: Requires 128MB RAM per derivation
- ASIC-resistant: Expensive to build custom hardware
- Time-hard: Slow by design (security vs. usability trade-off)
- Industry standard: Used by major password managers

### 4. Data Flow Security

#### Enrollment (Registration)
```
1. Camera → Face Image
2. Face Detection → Extract Descriptor (128D vector)
3. Generate Face ID = SHA-256(Person Name)
4. Create Face Data = {name, descriptor, timestamp}
5. Encrypt with Hardware Key → AES-256-GCM
6. Encrypt with Gun.js SEA → User's Private Space
7. Store in Gun.js Database
```

#### Recognition
```
1. Camera → Face Image
2. Face Detection → Extract Descriptor
3. Retrieve All Enrolled Faces → Gun.js SEA Decrypt → Hardware Decrypt
4. Compare Descriptors → Euclidean Distance
5. Find Best Match (if distance < threshold)
6. Return Recognition Result
```

### 5. Tamper Detection

**Authentication Tags**:
- Every encryption includes GCM authentication tag
- Any bit modification causes decryption to fail
- No silent corruption possible

**Database Integrity**:
- Gun.js provides built-in data verification
- SEA encryption includes signatures
- Invalid signatures rejected automatically

### 6. Reverse Engineering Resistance

**Why This System Cannot Be Reverse Engineered**:

1. **No Keys in Code**: All keys derived from hardware at runtime
2. **Hardware-Specific**: Data encrypted on Device A cannot decrypt on Device B
3. **No Extraction**: Reading source code reveals algorithm, but not keys
4. **Memory Dumps Ineffective**: Keys can be derived again from hardware
5. **Database Useless Alone**: Without hardware, database is encrypted gibberish

**Attack Scenarios**:

| Attack | Protection |
|--------|-----------|
| Copy database to another device | Decryption fails - wrong hardware fingerprint |
| Dump memory for keys | Keys can be re-derived from hardware anyway |
| Analyze source code | No hardcoded keys - algorithm visible but useless |
| Brute force encryption | AES-256 + Scrypt makes this computationally infeasible |
| Clone device hardware | Extremely difficult - needs exact CPU serial + MAC |

### 7. Additional Security Measures

**Isolation**:
- Default configuration: No peer-to-peer connections
- Local-only database storage
- No cloud dependencies

**Minimal Attack Surface**:
- No web interface by default
- No remote API without explicit configuration
- File system permissions: 700 for data directory

**Dependency Security**:
- All dependencies from npm (auditable)
- Minimal dependency tree
- Regular security updates recommended

## Security Guarantees

### What This System Guarantees

✅ **Data Confidentiality**: Face data encrypted with military-grade AES-256
✅ **Device Binding**: Data only accessible on original hardware
✅ **Tamper Detection**: Any modification detected via GCM auth tags
✅ **Authenticity**: Each encryption verified with cryptographic signatures
✅ **Reverse Engineering Resistance**: No keys extractable from code or database

### What This System Does NOT Guarantee

❌ **Physical Security**: If attacker has physical access and can clone hardware
❌ **Side-Channel Immunity**: Advanced attacks (power analysis) not addressed
❌ **Quantum Resistance**: AES-256 considered quantum-resistant, but not proven
❌ **Zero Trust**: System trusts the underlying OS and hardware

## Security Best Practices

### Deployment

1. **Physical Security**:
   - Secure the Jetson Nano in tamper-evident enclosure
   - Use tamper-detection switches
   - Monitor physical access

2. **Network Security**:
   - Firewall: Block all unnecessary ports
   - VPN: Use VPN for any remote access
   - Isolation: Keep on isolated network segment

3. **Updates**:
   - Regularly update Node.js dependencies
   - Monitor security advisories
   - Apply OS security patches

4. **Monitoring**:
   - Log all authentication attempts
   - Alert on unusual patterns
   - Regular security audits

### Operational Security

1. **Key Management**:
   - Never backup hardware keys (they're derived)
   - Don't share device serial numbers
   - Protect physical access to device

2. **Data Management**:
   - Encrypted backups only
   - Secure deletion of old data
   - Regular data integrity checks

3. **Access Control**:
   - Limit who can physically access device
   - Use strong OS passwords
   - Enable full disk encryption

## Cryptographic Specifications

| Component | Algorithm | Key Size | Notes |
|-----------|-----------|----------|-------|
| Key Derivation | Scrypt | 256 bits | N=16384, r=8, p=1 |
| Symmetric Encryption | AES-GCM | 256 bits | Random IV, 128-bit auth tag |
| Hashing | SHA-256 | 256 bits | For fingerprints and IDs |
| Password Derivation | SHA-512 | 512 bits | For Gun.js credentials |
| Face Descriptors | Euclidean | 128 floats | Distance-based matching |

## Compliance Considerations

This system can help meet requirements for:

- **GDPR**: Data minimization, encryption at rest
- **CCPA**: Secure storage of biometric data
- **BIPA**: Biometric data protection
- **ISO 27001**: Information security management

**Note**: Compliance requires additional measures beyond this system (policies, procedures, audits).

## Security Audit Checklist

- [ ] All dependencies up to date
- [ ] No hardcoded secrets in code
- [ ] File permissions properly set (700 for data/)
- [ ] Firewall configured correctly
- [ ] Physical security measures in place
- [ ] Monitoring and logging active
- [ ] Backup procedures established
- [ ] Incident response plan documented
- [ ] Regular security reviews scheduled

## Vulnerability Reporting

If you discover a security vulnerability:

1. **DO NOT** open a public issue
2. Contact the repository owner privately
3. Provide detailed reproduction steps
4. Allow reasonable time for fix before disclosure

## References

- [NIST SP 800-38D](https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-38d.pdf) - GCM Mode
- [Scrypt RFC 7914](https://tools.ietf.org/html/rfc7914) - Password-Based KDF
- [Gun.js SEA](https://gun.eco/docs/SEA) - Security Module
- [OWASP Cryptographic Storage](https://cheatsheetseries.owasp.org/cheatsheets/Cryptographic_Storage_Cheat_Sheet.html)
