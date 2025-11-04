"""
Setup script for WhoAmI facial recognition system
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="whoami-facial-recognition",
    version="1.0.0",
    author="whoami project",
    description="Facial recognition system for Oak D Series 3 and Jetson Orin Nano",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alanchelmickjr/whoami",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "depthai>=2.24.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "pillow>=10.0.0",
        "face-recognition>=1.3.0",
    ],
    entry_points={
        "console_scripts": [
            "whoami-gui=whoami.gui:main",
            "whoami-cli=whoami.cli:main",
        ],
    },
)
