# Deployment Guide for Jetson Nano

This guide covers deploying the WhoAmI facial recognition system on an NVIDIA Jetson Nano.

## Prerequisites

### 1. Jetson Nano Setup

1. Flash Jetson Nano with JetPack 4.6+ 
2. Complete initial setup and connect to network
3. Update system:
```bash
sudo apt update
sudo apt upgrade -y
```

### 2. Install Node.js

```bash
# Install Node.js 18.x
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Verify installation
node --version  # Should be >= 18.0.0
npm --version
```

### 3. Install OpenCV Dependencies

```bash
# Install OpenCV and dependencies for opencv4nodejs
sudo apt install -y \
  build-essential \
  cmake \
  git \
  libopencv-dev \
  python3-opencv

# Verify OpenCV installation
pkg-config --modversion opencv4
```

### 4. Install Canvas Dependencies

```bash
# Required for node-canvas
sudo apt install -y \
  libcairo2-dev \
  libpango1.0-dev \
  libjpeg-dev \
  libgif-dev \
  librsvg2-dev
```

## Installation

### 1. Clone Repository

```bash
cd ~
git clone https://github.com/alanchelmickjr/whoami.git
cd whoami
```

### 2. Install Node Dependencies

```bash
npm install
```

**Note**: Installation may take 10-20 minutes on Jetson Nano due to native module compilation.

### 3. Download Face-api.js Models

```bash
# Follow instructions in models/README.md
cd models
# Download models using wget or curl as described
```

### 4. Configure Camera

Check available cameras:
```bash
ls /dev/video*
```

Update `config/config.json` with correct camera device ID if needed.

### 5. Test Installation

```bash
node src/index.js
```

You should see:
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
```

## Running as a Service

### Create systemd Service

```bash
sudo nano /etc/systemd/system/whoami.service
```

Add the following:

```ini
[Unit]
Description=WhoAmI Facial Recognition Service
After=network.target

[Service]
Type=simple
User=jetson
WorkingDirectory=/home/jetson/whoami
ExecStart=/usr/bin/node src/index.js
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable whoami.service
sudo systemctl start whoami.service

# Check status
sudo systemctl status whoami.service

# View logs
sudo journalctl -u whoami.service -f
```

## Performance Optimization

### 1. Enable CUDA Acceleration

Verify CUDA is working:
```bash
nvcc --version
```

### 2. Set Power Mode

For maximum performance:
```bash
sudo nvpmodel -m 0  # Max performance mode
sudo jetson_clocks   # Lock clocks to maximum
```

### 3. Monitor Performance

```bash
# Install jtop for monitoring
sudo -H pip3 install -U jetson-stats

# Run monitoring
sudo jtop
```

### 4. Optimize Node.js Memory

If experiencing memory issues, adjust Node.js heap size:
```bash
node --max-old-space-size=2048 src/index.js
```

## Security Hardening

### 1. Firewall Configuration

```bash
# Allow only necessary ports
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw enable
```

### 2. Disable Unnecessary Services

```bash
sudo systemctl disable bluetooth
sudo systemctl disable cups
```

### 3. File Permissions

```bash
# Ensure proper permissions
chmod 700 data/
chmod 600 config/config.json
```

### 4. Physical Security

- Place Jetson Nano in a secure enclosure
- Consider tamper detection switches
- Use security cables if in public spaces

## Troubleshooting

### Camera Not Working

```bash
# Test camera with gstreamer
gst-launch-1.0 nvarguscamerasrc ! nvoverlaysink

# Or for USB cameras
gst-launch-1.0 v4l2src device=/dev/video0 ! videoconvert ! autovideosink
```

### Memory Issues

- Increase swap space
- Close unnecessary applications
- Use lower resolution camera feed

### Model Loading Errors

- Verify all model files are present
- Check file permissions
- Ensure models are from correct version of face-api.js

### GPU Not Being Used

- Verify CUDA installation
- Check TensorRT installation
- Review OpenCV CUDA support

## Backup and Recovery

### Backup Data

```bash
# Backup encrypted database
tar -czf whoami-backup-$(date +%Y%m%d).tar.gz data/

# Backup configuration
cp config/config.json config/config.json.backup
```

### Recovery

The system is designed to be resilient. If data is corrupted:

1. Stop the service: `sudo systemctl stop whoami.service`
2. Remove corrupted data: `rm -rf data/gun`
3. Restart the service: `sudo systemctl start whoami.service`

The system will re-initialize with hardware-backed security.

## Updates

To update the system:

```bash
cd ~/whoami
git pull
npm install  # Update dependencies if needed
sudo systemctl restart whoami.service
```

## Monitoring

### View Logs

```bash
# System logs
sudo journalctl -u whoami.service -f

# Application logs
tail -f /var/log/syslog | grep whoami
```

### Health Checks

Create a health check script:

```bash
#!/bin/bash
# health-check.sh

response=$(node -e "
  import('./src/index.js').then(m => {
    console.log(JSON.stringify(m.whoami.getStatus()));
  });
")

echo "$response"
```

## Support

For issues:
1. Check logs: `sudo journalctl -u whoami.service -n 100`
2. Verify hardware: `sudo jtop`
3. Test camera: `ls /dev/video*`
4. Review README.md for common issues
