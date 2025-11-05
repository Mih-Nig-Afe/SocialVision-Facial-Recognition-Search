"""
SocialVision: Advanced Facial Recognition Search Engine
Main package initialization
"""

__version__ = "1.0.0"
__author__ = "Mihretab N. Afework"
__email__ = "mtabdevt@gmail.com"

from src.config import Config
from src.logger import setup_logger

# Initialize logger
logger = setup_logger(__name__)

__all__ = ["Config", "logger"]

