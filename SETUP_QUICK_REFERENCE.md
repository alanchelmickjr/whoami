# WhoAmI - Quick Setup Reference

Quick reference guide for Jetson Orin Nano setup.

## 🚀 Quick Start

```bash
# Full installation (recommended)
./jetson_setup_v2.sh

# That's it! Script handles everything.
```

## 📋 Setup Options

```bash
# Full installation (all features)
./jetson_setup_v2.sh --full

# Minimal installation (no 3D scanning, no gimbal)
./jetson_setup_v2.sh --minimal

# Camera permissions only
./jetson_setup_v2.sh --camera-only

# Python environment only
./jetson_setup_v2.sh --python-only

# Skip performance configuration
./jetson_setup_v2.sh --skip-performance

# Verification only
./jetson_setup_v2.sh --verify-only

# Help
./jetson_setup_v2.sh --help
```

## ✅ Post-Installation Checklist

1. **Logout/Login** (apply group permissions)
   ```bash
   logout  # Or: sudo reboot
   ```

2. **Activate Environment**
   ```bash
   source ~/whoami_env/bin/activate
   ```

3. **Test Camera**
   ```bash
   python -c "import depthai as dai; print(f'{len(dai.Device.getAllAvailableDevices())} camera(s)')"
   ```

4. **Run Verification**
   ```bash
   python verify_install_v2.py
   ```

5. **Launch WhoAmI**
   ```bash
   python run_gui.py    # GUI mode
   python run_cli.py    # CLI mode
   ```

## 🔍 Verification

```bash
# Quick verification
source ~/whoami_env/bin/activate
python verify_install_v2.py

# Manual checks
lsusb | grep 03e7                    # Camera connected?
groups | grep -E "video|dialout"     # In correct groups?
python -c "import depthai"           # DepthAI installed?
```

## 🛠️ Common Commands

```bash
# Activate environment
source ~/whoami_env/bin/activate

# Check camera
lsusb | grep 03e7

# Check serial ports (gimbal)
ls -la /dev/ttyUSB* /dev/ttyTHS*

# Monitor Jetson
jtop

# Check performance mode
sudo nvpmodel -q
jetson_clocks --show

# Set max performance
sudo nvpmodel -m 0
sudo jetson_clocks

# View logs
cat ~/whoami_setup_*.log
```

## 🐛 Quick Troubleshooting

### Camera Not Detected

```bash
# Check USB
lsusb | grep 03e7

# Check permissions
groups | grep video

# Reload udev
sudo udevadm trigger

# Try different USB port (blue = USB 3.0)
```

### Import Errors

```bash
# Activate environment
source ~/whoami_env/bin/activate

# Verify activation
which python  # Should show ~/whoami_env/bin/python

# Reinstall if needed
pip install -r requirements.txt
```

### Gimbal Not Working

```bash
# Check serial port
ls /dev/ttyUSB* /dev/ttyTHS*

# Check permissions
groups | grep dialout

# Test serial
python -c "import serial; serial.Serial('/dev/ttyUSB0', 1000000)"
```

### Performance Issues

```bash
# Max performance
sudo nvpmodel -m 0
sudo jetson_clocks

# Monitor resources
jtop

# Reduce resolution in config.json
```

## 📁 File Locations

```
~/whoami/                    # Project
~/whoami_env/               # Virtual environment
~/activate_whoami.sh        # Quick activation
~/Desktop/WhoAmI-*.desktop  # Desktop shortcuts
~/whoami_setup_*.log        # Setup logs
```

## 📚 Documentation

- **Full Installation Guide:** `INSTALLATION.md`
- **API Reference:** `docs/API_REFERENCE.md`
- **Usage Guide:** `docs/USAGE_GUIDE.md`
- **Jetson Setup Details:** `SETUP_JETSON_M4.md`
- **Main README:** `README.md`

## 🔄 Script Comparison

| Feature | jetson_setup.sh (v1) | jetson_setup_v2.sh (v2) |
|---------|---------------------|------------------------|
| Modular options | ❌ | ✅ --full, --minimal |
| Error handling | Basic | ✅ Comprehensive |
| Logging | Limited | ✅ Full with timestamps |
| Verification | Basic | ✅ Comprehensive |
| 3D scanning | ✅ | ✅ Optional |
| Gimbal support | ❌ | ✅ Optional |
| Config init | ❌ | ✅ Automatic |
| Recovery | ❌ | ✅ Built-in |

**Recommendation:** Use `jetson_setup_v2.sh` for new installations.

## 🆘 Getting Help

1. Run verification: `python verify_install_v2.py`
2. Check logs: `cat ~/whoami_setup_*.log`
3. Read full guide: `INSTALLATION.md`
4. Check GitHub issues
5. Create issue with logs + verification output

---

**Version:** 2.0
**Last Updated:** 2025-11-06
**For:** Jetson Orin Nano + OAK-D Series 3
