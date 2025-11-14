# K-1 Booster Setup Guide

Complete setup guide for Jetson Orin NX on K-1 Booster carrier board with head/neck gimbal system and audio capabilities.

## Overview

The K-1 booster configuration provides:
- **Dual Gimbal System**: Separate head (tilt) and neck (tilt) control
- **Audio I/O**: Voice reporting, audio tracking, and speech recognition
- **Remote Access**: VNC and direct access for operator control
- **Expanded I/O**: Dual Ethernet, additional USB, M.2 slots

## Hardware Requirements

### Core Components
- Jetson Orin NX 16GB module
- K-1 Booster carrier board
- Power supply: 19V, 65W minimum
- OAK-D Series 3 camera with USB 3.0 cable

### Gimbal System
- Head Gimbal (1-axis):
  - 1x Feetech STS/SCS servo for tilt (up/down)
  - Serial connection to `/dev/ttyTHS1`
- Neck Gimbal (1-axis):
  - 1x Feetech STS/SCS servo for neck tilt (forward/back nod)
  - Serial connection to `/dev/ttyTHS2`
- Total: 2-axis gimbal system (head tilt, neck tilt)

### Audio System
- USB audio interface (recommended: USB Audio Class 2.0)
- Microphone for voice input and audio tracking
- Speaker or headphone output for voice reporting

### Network
- Ethernet cable (primary connection)
- Optional: Second Ethernet for redundancy
- Optional: WiFi module for backup connectivity

## Initial Setup

### 1. Flash JetPack

Flash JetPack 5.1.2 or later to the Orin NX:

```bash
# Use NVIDIA SDK Manager or command line
sudo ./flash.sh jetson-orin-nx-devkit mmcblk0p1
```

### 2. First Boot Configuration

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y \
    python3-pip \
    git \
    vim \
    htop \
    net-tools \
    usbutils \
    pulseaudio \
    pulseaudio-utils \
    alsa-utils

# Configure serial ports
sudo usermod -a -G dialout $USER
sudo chmod 666 /dev/ttyTHS0 /dev/ttyTHS1 /dev/ttyTHS2
```

### 3. Configure Hardware Detection

```bash
# Clone WhoAmI repository
cd ~
git clone https://github.com/yourusername/whoami.git
cd whoami

# Set hardware profile
export WHOAMI_HARDWARE_PROFILE="jetson_orin_nx_k1"
echo 'export WHOAMI_HARDWARE_PROFILE="jetson_orin_nx_k1"' >> ~/.bashrc

# Verify detection
python3 -m whoami.hardware_detector
```

Expected output:
```
=== Hardware Detection ===
Detected Hardware: Jetson Orin NX on K-1 Booster
Profile Name: jetson_orin_nx_k1
Module: orin_nx
Carrier: k1_booster

=== Peripherals ===
Serial Port: /dev/ttyTHS1
Available Serial Ports: ['/dev/ttyTHS0', '/dev/ttyTHS1', '/dev/ttyTHS2']
GPIO Chip: gpiochip1
I2C Buses: [0, 2, 8]

=== Gimbal Configuration ===
Type: head_neck_dual
Head Gimbal Port: /dev/ttyTHS1
Head Gimbal Axes: tilt
Neck Gimbal Port: /dev/ttyTHS2
Neck Gimbal Axes: neck_tilt

=== Audio Configuration ===
Input Device: hw:2,0
Output Device: hw:2,0
Sample Rate: 48000Hz
Supported Features: voice_reporting, audio_tracking, speech_synthesis, voice_commands

=== Remote Access ===
VNC Enabled: True
VNC Port: 5900
Direct HDMI: True
Resolution: 1920x1080
```

## Gimbal System Setup

### 1. Wire Head Gimbal to ttyTHS1

Connect Feetech servo:
- Servo 1 (Tilt): ID 1
- Baudrate: 1Mbps
- Serial: `/dev/ttyTHS1`

### 2. Wire Neck Gimbal to ttyTHS2

Connect Feetech servo:
- Servo 2 (Neck Tilt): ID 2
- Baudrate: 1Mbps
- Serial: `/dev/ttyTHS2`

### 3. Test Gimbal Systems

```bash
# Test head gimbal
python3 -c "
from whoami.feetech_sdk import FeetchController
head = FeetchController('/dev/ttyTHS1', 1000000)
head.ping(1)  # Ping tilt servo
print('Head gimbal OK')
"

# Test neck gimbal
python3 -c "
from whoami.feetech_sdk import FeetchController
neck = FeetchController('/dev/ttyTHS2', 1000000)
neck.ping(2)  # Ping neck tilt servo
print('Neck gimbal OK')
"
```

### 4. Calibrate Servos

```bash
# Use K-1 configuration
python3 -m whoami.calibrate_servos --config config/k1_booster_config.json
```

## Audio System Setup

### 1. Connect USB Audio Interface

```bash
# List audio devices
aplay -l
arecord -l

# Should show USB audio interface (e.g., hw:2,0)
```

### 2. Configure PulseAudio

```bash
# Edit PulseAudio config
nano ~/.config/pulse/default.pa

# Add:
load-module module-alsa-sink device=hw:2,0
load-module module-alsa-source device=hw:2,0

# Restart PulseAudio
pulseaudio -k
pulseaudio --start
```

### 3. Test Audio

```bash
# Test speaker output
speaker-test -D hw:2,0 -c 2

# Test microphone input
arecord -D hw:2,0 -f S16_LE -r 48000 -c 2 -d 5 test.wav
aplay test.wav
```

### 4. Install Voice Synthesis

```bash
# Install pyttsx3 for text-to-speech
pip3 install pyttsx3

# Test voice reporting
python3 -c "
import pyttsx3
engine = pyttsx3.init()
engine.say('K-1 booster audio system initialized')
engine.runAndWait()
"
```

### 5. Install Speech Recognition (Optional)

```bash
# Download Vosk model
mkdir -p /opt/whoami/models
cd /opt/whoami/models
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip

# Install Vosk
pip3 install vosk
```

## Remote Access Setup

### 1. Install VNC Server

```bash
# Install TigerVNC or x11vnc
sudo apt install -y tigervnc-standalone-server

# Configure VNC
vncserver :0 -geometry 1920x1080 -depth 24

# Set VNC password
vncpasswd
```

### 2. Create VNC Service

```bash
# Create systemd service
sudo nano /etc/systemd/system/vncserver@.service
```

Add:
```ini
[Unit]
Description=Remote desktop service (VNC)
After=syslog.target network.target

[Service]
Type=simple
User=your_username
PAMName=login
PIDFile=/home/your_username/.vnc/%H%i.pid
ExecStartPre=/bin/sh -c '/usr/bin/vncserver -kill %i > /dev/null 2>&1 || :'
ExecStart=/usr/bin/vncserver %i -geometry 1920x1080 -depth 24 -alwaysshared
ExecStop=/usr/bin/vncserver -kill %i

[Install]
WantedBy=multi-user.target
```

```bash
# Enable VNC service
sudo systemctl enable vncserver@0.service
sudo systemctl start vncserver@0.service
```

### 3. Configure Firewall

```bash
# Allow VNC port
sudo ufw allow 5900/tcp

# Allow SSH
sudo ufw allow 22/tcp

# Enable firewall
sudo ufw enable
```

## Network Configuration

### 1. Configure Dual Ethernet

```bash
# Edit netplan
sudo nano /etc/netplan/01-network-manager-all.yaml
```

Add:
```yaml
network:
  version: 2
  renderer: NetworkManager
  ethernets:
    eth0:
      dhcp4: true
      dhcp4-overrides:
        route-metric: 100
    eth1:
      dhcp4: true
      dhcp4-overrides:
        route-metric: 200
```

```bash
# Apply configuration
sudo netplan apply

# Test both interfaces
ping -I eth0 8.8.8.8
ping -I eth1 8.8.8.8
```

### 2. Set Static IP (Optional)

```yaml
network:
  version: 2
  ethernets:
    eth0:
      addresses:
        - 192.168.1.100/24
      gateway4: 192.168.1.1
      nameservers:
        addresses: [8.8.8.8, 8.8.4.4]
```

## WhoAmI System Installation

### 1. Run K-1 Setup Script

```bash
cd ~/whoami
./jetson_setup_v2.sh --full

# Or with K-1 specific flag (if added)
./jetson_setup_k1.sh
```

### 2. Install Python Dependencies

```bash
# Activate virtual environment
source ~/whoami_env/bin/activate

# Install WhoAmI
pip install -e .

# Install audio dependencies
pip install pyttsx3 vosk pyaudio sounddevice

# Install remote access dependencies
pip install flask flask-socketio
```

### 3. Copy K-1 Configuration

```bash
# Use K-1 specific config
cp config/k1_booster_config.json config/active_config.json

# Or set environment variable
export WHOAMI_CONFIG="config/k1_booster_config.json"
```

## Testing and Validation

### 1. Test Hardware Detection

```bash
python3 -m whoami.hardware_detector
```

### 2. Test Gimbal Systems

```bash
# Test head gimbal only
python3 -m whoami.gimbal_controller --gimbal head --test

# Test neck gimbal only
python3 -m whoami.gimbal_controller --gimbal neck --test

# Test coordinated movement
python3 -m whoami.gimbal_controller --gimbal both --test
```

### 3. Test Audio System

```bash
# Test voice reporting
python3 -c "
from whoami.audio import VoiceReporter
reporter = VoiceReporter()
reporter.say('K-1 booster system operational')
"

# Test audio tracking
python3 -m whoami.audio_tracker --test
```

### 4. Run Full System

```bash
# Start WhoAmI with K-1 config
python3 -m whoami.gui --config config/k1_booster_config.json
```

## Remote Operation

### 1. Connect via VNC

From remote computer:
```bash
# Using VNC client
vncviewer <jetson-ip>:5900

# Or using macOS Screen Sharing
open vnc://<jetson-ip>:5900
```

### 2. SSH Access

```bash
# Connect via SSH
ssh username@<jetson-ip>

# Run WhoAmI in headless mode
python3 -m whoami.headless --config config/k1_booster_config.json
```

### 3. Web Interface (Optional)

```bash
# Start web server
python3 -m whoami.web_interface --port 8080

# Access from browser
http://<jetson-ip>:8080
```

## Performance Optimization

### 1. Set Power Mode

```bash
# Set to maximum performance
sudo nvpmodel -m 0
sudo jetson_clocks

# Verify
sudo nvpmodel -q
```

### 2. Monitor Performance

```bash
# Monitor GPU/CPU/Memory
tegrastats

# Monitor temperature
cat /sys/class/thermal/thermal_zone*/temp
```

## Troubleshooting

### Gimbal Not Responding

```bash
# Check serial permissions
ls -l /dev/ttyTHS*

# Test serial port
sudo minicom -D /dev/ttyTHS1 -b 1000000

# Check servo power
# Ensure 6-8V power supply connected to servos
```

### Audio Not Working

```bash
# List audio devices
aplay -L
arecord -L

# Check PulseAudio
pulseaudio --check
pulseaudio --start

# Test with speaker-test
speaker-test -D hw:2,0
```

### VNC Connection Issues

```bash
# Check VNC server status
systemctl status vncserver@0

# Check firewall
sudo ufw status

# Check VNC logs
cat ~/.vnc/*.log
```

### Network Problems

```bash
# Check interface status
ip link show
ip addr show

# Restart NetworkManager
sudo systemctl restart NetworkManager

# Test connectivity
ping -c 4 8.8.8.8
```

## Operational Modes

The K-1 booster supports three operational modes:

### 1. Remote VNC Mode
- Operator controls via VNC
- Full GUI access
- Bidirectional audio
- Real-time video feedback

### 2. Direct Access Mode
- HDMI monitor, keyboard, mouse connected
- Local operation
- Full system access

### 3. Autonomous Mode
- Headless operation
- Audio feedback for status
- Automated behaviors
- Remote monitoring via web interface

Set mode in config:
```json
{
  "operational_mode": {
    "mode": "remote_vnc"
  }
}
```

## Maintenance

### Daily Checks
- Verify both Ethernet connections
- Check gimbal movement range
- Test audio I/O
- Monitor system temperature

### Weekly Checks
- Update system packages
- Check log files for errors
- Verify VNC accessibility
- Test failover between Ethernet interfaces

### Monthly Checks
- Recalibrate servos if needed
- Update WhoAmI system
- Backup configuration
- Clean dust from heatsinks

## See Also

- [Hardware Configuration Guide](HARDWARE_CONFIG_GUIDE.md)
- [Installation Guide](../INSTALLATION.md)
- [Gimbal 3DOF Guide](GIMBAL_3DOF_GUIDE.md)
- [API Reference](API_REFERENCE.md)

## Support

For issues specific to K-1 booster:
- Check hardware profiles: `config/hardware/hardware_profiles.json`
- Review K-1 config: `config/k1_booster_config.json`
- Run diagnostics: `python3 -m whoami.diagnostics --k1`
- Open GitHub issue with `k1-booster` tag
