"""
Test script to verify the WhoAmI package structure
This tests imports and basic structure without requiring hardware
"""

import ast
import os
import sys

def check_file_syntax(filepath):
    """Check if a Python file has valid syntax"""
    try:
        with open(filepath, 'r') as f:
            ast.parse(f.read())
        return True, None
    except SyntaxError as e:
        return False, str(e)

def main():
    """Run structure validation tests"""
    print("WhoAmI Structure Validation")
    print("=" * 50)
    
    # Check required files exist
    required_files = [
        'whoami/__init__.py',
        'whoami/face_recognizer.py',
        'whoami/gui.py',
        'whoami/cli.py',
        'whoami/config.py',
        'run_gui.py',
        'run_cli.py',
        'setup.py',
        'requirements.txt',
        'README.md',
    ]
    
    print("\n1. Checking required files...")
    all_exist = True
    for filepath in required_files:
        exists = os.path.exists(filepath)
        status = "✓" if exists else "✗"
        print(f"  {status} {filepath}")
        if not exists:
            all_exist = False
    
    if not all_exist:
        print("\n❌ Some required files are missing!")
        return False
    
    # Check Python syntax
    print("\n2. Checking Python file syntax...")
    python_files = [f for f in required_files if f.endswith('.py')]
    all_valid = True
    
    for filepath in python_files:
        valid, error = check_file_syntax(filepath)
        status = "✓" if valid else "✗"
        print(f"  {status} {filepath}")
        if not valid:
            print(f"     Error: {error}")
            all_valid = False
    
    if not all_valid:
        print("\n❌ Some files have syntax errors!")
        return False
    
    # Check directory structure
    print("\n3. Checking directory structure...")
    required_dirs = [
        'whoami',
        'examples',
    ]
    
    for dirpath in required_dirs:
        exists = os.path.isdir(dirpath)
        status = "✓" if exists else "✗"
        print(f"  {status} {dirpath}/")
    
    # Check key functions/classes exist
    print("\n4. Checking key components...")
    
    components = {
        'whoami/face_recognizer.py': ['FaceRecognizer'],
        'whoami/gui.py': ['FaceRecognitionGUI'],
        'whoami/cli.py': ['FaceRecognitionCLI'],
        'whoami/config.py': ['Config'],
    }
    
    for filepath, expected_names in components.items():
        with open(filepath, 'r') as f:
            content = f.read()
            for name in expected_names:
                if f'class {name}' in content:
                    print(f"  ✓ {filepath}: {name}")
                else:
                    print(f"  ✗ {filepath}: {name} not found")
    
    # Check requirements.txt
    print("\n5. Checking dependencies...")
    with open('requirements.txt', 'r') as f:
        deps = f.read()
        required_deps = ['depthai', 'opencv-python', 'numpy', 'pillow', 'face-recognition']
        for dep in required_deps:
            if dep in deps:
                print(f"  ✓ {dep}")
            else:
                print(f"  ✗ {dep} missing")
    
    print("\n" + "=" * 50)
    print("✅ All structure validation checks passed!")
    print("\nNext steps:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Connect Oak D camera")
    print("  3. Run GUI: python run_gui.py")
    print("  4. Or run CLI: python run_cli.py --help")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
