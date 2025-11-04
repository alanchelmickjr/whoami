# Quick Start Guide - WhoAmI

Get up and running with WhoAmI facial recognition in minutes!

## Prerequisites

- Oak D Series 3 camera (or compatible DepthAI device)
- Python 3.8 or higher
- USB 3.0 port

## Installation

### Option 1: Quick Install

```bash
git clone https://github.com/alanchelmickjr/whoami.git
cd whoami
pip install -r requirements.txt
```

### Option 2: Install as Package

```bash
git clone https://github.com/alanchelmickjr/whoami.git
cd whoami
pip install -e .
```

## First Run - GUI

1. **Connect your Oak D camera** via USB 3.0

2. **Launch the GUI**:
   ```bash
   python run_gui.py
   # or if installed as package:
   whoami-gui
   ```

3. **Click "Start Camera"** to initialize the Oak D

4. **Add your first face**:
   - Click "Add Face"
   - Enter a name
   - Position face in camera view
   - Click "Capture"

5. **Start recognizing**! Your face will be automatically detected and labeled.

## First Run - CLI

1. **Connect your Oak D camera**

2. **List current faces**:
   ```bash
   python run_cli.py list
   ```

3. **Add a face**:
   ```bash
   python run_cli.py add "Your Name"
   # Position your face and press SPACE
   ```

4. **Run recognition**:
   ```bash
   python run_cli.py recognize
   # Press ESC to exit
   ```

## Common Issues

### Camera not detected
```bash
# Check if device is connected
lsusb | grep Movidius

# Verify DepthAI installation
python -c "import depthai; print('DepthAI version:', depthai.__version__)"
```

### Permission denied
On Linux, add yourself to the plugdev group:
```bash
sudo usermod -a -G plugdev $USER
# Log out and back in
```

### Import errors
Reinstall dependencies:
```bash
pip install --upgrade -r requirements.txt
```

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check out [examples/](examples/) for integration examples
- See [CONTRIBUTING.md](CONTRIBUTING.md) to contribute

## Jetson Orin Nano Specific Setup

For optimal performance on Jetson:

```bash
# Enable max performance mode
sudo nvpmodel -m 0
sudo jetson_clocks

# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3-pip python3-tk cmake libopencv-dev
sudo apt-get install -y build-essential cmake libopenblas-dev liblapack-dev

# Install WhoAmI
pip install -r requirements.txt
```

## Tips

- **Good Lighting**: Ensure adequate lighting for best face detection
- **Face Position**: Keep face centered and at comfortable distance (1-2 meters)
- **Multiple Faces**: System can handle multiple people at once
- **Database**: All faces stored locally in `face_database.pkl`
- **Privacy**: No internet required, all processing on-device

## Help

For more help:
- Open an issue on GitHub
- Check the full README.md
- Review example code in `examples/`

Happy face recognizing! 🎉
