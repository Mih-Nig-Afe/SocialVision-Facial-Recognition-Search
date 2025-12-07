"""
Setup configuration for SocialVision
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="socialvision",
    version="1.0.0",
    author="Mihretab N. Afework",
    author_email="mtabdevt@gmail.com",
    description="Advanced Facial Recognition Search Engine with auto-training",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Mih-Nig-Afe/SocialVision-Facial-Recognition-Search",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.9",
    install_requires=[
        "streamlit>=1.39.0",
        "opencv-python>=4.10.0",
        "face-recognition>=1.3.0",
        "numpy>=2.1.3",
        "Pillow>=11.0.0",
        "python-dotenv>=1.0.1",
    ],
    extras_require={
        "dev": [
            "pytest>=8.3.3",
            "pytest-cov>=6.0.0",
            "black>=24.10.0",
            "flake8>=7.1.1",
            "mypy>=1.13.0",
        ],
        "ml": [
            "tensorflow>=2.18.0",
            "torch>=2.5.1",
            "torchvision>=0.20.1",
            "mediapipe>=0.10.18",
        ],
    },
)
