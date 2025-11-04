#!/usr/bin/env python3
"""
Installation verification script for WhoAmI
Checks that all dependencies are properly installed
"""

import sys
import importlib

def check_module(module_name, package_name=None):
    """Check if a module can be imported"""
    try:
        importlib.import_module(module_name)
        print(f"✓ {package_name or module_name}")
        return True
    except ImportError as e:
        print(f"✗ {package_name or module_name} - {e}")
        return False

def main():
    """Run installation verification"""
    print("WhoAmI Installation Verification")
    print("=" * 50)
    
    print("\n1. Checking Python version...")
    version = sys.version_info
    if version >= (3, 8):
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro}")
        print("  Requires Python 3.8 or higher")
        return False
    
    print("\n2. Checking core dependencies...")
    dependencies = [
        ('depthai', 'depthai'),
        ('cv2', 'opencv-python'),
        ('numpy', 'numpy'),
        ('PIL', 'pillow'),
        ('face_recognition', 'face-recognition'),
    ]
    
    all_ok = True
    for module, package in dependencies:
        if not check_module(module, package):
            all_ok = False
    
    print("\n3. Checking WhoAmI package...")
    whoami_modules = [
        'whoami',
        'whoami.face_recognizer',
        'whoami.gui',
        'whoami.cli',
        'whoami.config',
    ]
    
    for module in whoami_modules:
        if not check_module(module):
            all_ok = False
    
    print("\n4. Checking optional GUI dependencies...")
    try:
        import tkinter
        print("✓ tkinter (GUI support)")
    except ImportError:
        print("✗ tkinter (GUI will not work)")
        print("  Install: sudo apt-get install python3-tk")
        all_ok = False
    
    print("\n" + "=" * 50)
    if all_ok:
        print("✅ All checks passed! WhoAmI is ready to use.")
        print("\nNext steps:")
        print("  - Connect your Oak D camera")
        print("  - Run: python run_gui.py")
        print("  - Or run: python run_cli.py --help")
    else:
        print("❌ Some dependencies are missing.")
        print("\nTo install missing dependencies:")
        print("  pip install -r requirements.txt")
    
    return all_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
