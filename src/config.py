"""
Configuration management for SocialVision application
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Main configuration class for SocialVision"""

    # Application settings
    APP_NAME = "SocialVision"
    APP_VERSION = "1.0.0"
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"

    # Paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    UPLOADS_DIR = BASE_DIR / "uploads"
    MODELS_DIR = BASE_DIR / "models"
    LOGS_DIR = BASE_DIR / "logs"
    CONFIG_DIR = BASE_DIR / "config"

    # Create directories if they don't exist
    for directory in [DATA_DIR, UPLOADS_DIR, MODELS_DIR, LOGS_DIR, CONFIG_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    # Firebase configuration
    FIREBASE_CONFIG_PATH = os.getenv(
        "FIREBASE_CONFIG_PATH", str(CONFIG_DIR / "firebase_config.json")
    )
    FIREBASE_ENABLED = os.getenv("FIREBASE_ENABLED", "False").lower() == "true"

    # Face Recognition settings
    FACE_RECOGNITION_MODEL = "hog"  # "hog" or "cnn" (cnn is more accurate but slower)
    FACE_DETECTION_CONFIDENCE = float(os.getenv("FACE_DETECTION_CONFIDENCE", "0.5"))
    FACE_MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.6"))
    DEEPFACE_MODEL = os.getenv("DEEPFACE_MODEL", "Facenet512")
    DEEPFACE_DETECTOR_BACKEND = os.getenv("DEEPFACE_DETECTOR_BACKEND", "opencv")
    FACE_SIMILARITY_THRESHOLD = float(os.getenv("FACE_SIMILARITY_THRESHOLD", "0.35"))
    ENABLE_DUAL_EMBEDDINGS = (
        os.getenv("ENABLE_DUAL_EMBEDDINGS", "True").lower() == "true"
    )
    EMBEDDING_WEIGHTS = {
        "deepface": float(os.getenv("DEEPFACE_EMBEDDING_WEIGHT", "0.7")),
        "dlib": float(os.getenv("DLIB_EMBEDDING_WEIGHT", "0.3")),
    }
    DEFAULT_EMBEDDING_SOURCE = os.getenv("DEFAULT_EMBEDDING_SOURCE", "deepface")

    # Image processing settings
    MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "10485760"))  # 10MB
    ALLOWED_IMAGE_FORMATS = {"jpg", "jpeg", "png", "gif", "bmp", "webp"}
    IMAGE_QUALITY = int(os.getenv("IMAGE_QUALITY", "85"))

    # Search settings
    MAX_RESULTS = int(os.getenv("MAX_RESULTS", "50"))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.6"))

    # Database settings
    DB_TYPE = os.getenv("DB_TYPE", "local")  # "local" or "firebase"
    LOCAL_DB_PATH = str(DATA_DIR / "faces_database.json")

    # API settings
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    API_WORKERS = int(os.getenv("API_WORKERS", "4"))

    # Streamlit settings
    STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))
    STREAMLIT_THEME = os.getenv("STREAMLIT_THEME", "dark")

    # Logging settings
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Performance settings
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
    NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))
    CACHE_SIZE = int(os.getenv("CACHE_SIZE", "1000"))

    @classmethod
    def load_firebase_config(cls) -> Optional[Dict[str, Any]]:
        """Load Firebase configuration from JSON file"""
        try:
            if os.path.exists(cls.FIREBASE_CONFIG_PATH):
                with open(cls.FIREBASE_CONFIG_PATH, "r") as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading Firebase config: {e}")
        return None

    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            key: getattr(cls, key)
            for key in dir(cls)
            if not key.startswith("_") and key.isupper()
        }


class DevelopmentConfig(Config):
    """Development configuration"""

    DEBUG = True
    LOG_LEVEL = "DEBUG"


class ProductionConfig(Config):
    """Production configuration"""

    DEBUG = False
    LOG_LEVEL = "INFO"


class TestingConfig(Config):
    """Testing configuration"""

    DEBUG = True
    DB_TYPE = "local"
    LOCAL_DB_PATH = str(Config.DATA_DIR / "test_faces_database.json")


def get_config() -> Config:
    """Get appropriate configuration based on environment"""
    env = os.getenv("ENVIRONMENT", "development").lower()

    config_map = {
        "development": DevelopmentConfig,
        "production": ProductionConfig,
        "testing": TestingConfig,
    }

    return config_map.get(env, DevelopmentConfig)
