"""
Core facial recognition engine using deepface library
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from src.logger import setup_logger
from src.config import get_config

logger = setup_logger(__name__)
config = get_config()

# Supported DeepFace models / detector backends (per upstream docs)
SUPPORTED_DEEPFACE_MODELS = {
    "VGG-Face",
    "Facenet",
    "Facenet512",
    "OpenFace",
    "DeepFace",
    "DeepID",
    "ArcFace",
    "Dlib",
    "SFace",
    "GhostFaceNet",
}
SUPPORTED_DETECTOR_BACKENDS = {
    "opencv",
    "ssd",
    "mtcnn",
    "retinaface",
    "mediapipe",
    "yolov8",
    "yunet",
    "fastmtcnn",
    "dlib",
}

DEFAULT_DEEPFACE_MODEL = "Facenet512"
DEFAULT_DETECTOR_BACKEND = "opencv"


def _resolve_setting(
    value: Optional[str], allowed: set, default: str, setting_name: str
) -> str:
    if value and value in allowed:
        return value
    if value and value not in allowed:
        logger.warning(
            "%s '%s' not supported by DeepFace. Falling back to %s.",
            setting_name,
            value,
            default,
        )
    return default


DEEPFACE_MODEL_NAME = _resolve_setting(
    getattr(config, "DEEPFACE_MODEL", DEFAULT_DEEPFACE_MODEL),
    SUPPORTED_DEEPFACE_MODELS,
    DEFAULT_DEEPFACE_MODEL,
    "DEEPFACE_MODEL",
)
DEEPFACE_DETECTOR_BACKEND = _resolve_setting(
    getattr(config, "DEEPFACE_DETECTOR_BACKEND", DEFAULT_DETECTOR_BACKEND),
    SUPPORTED_DETECTOR_BACKENDS,
    DEFAULT_DETECTOR_BACKEND,
    "DEEPFACE_DETECTOR_BACKEND",
)

try:
    # DeepFace is the preferred backend for face detection and embeddings.
    # However, it has a heavy dependency stack (TensorFlow / Keras) that can be
    # fragile across Python and OS versions. To keep the rest of the
    # application testable even when DeepFace cannot be imported, we treat it
    # as optional and gracefully degrade behaviour when it is unavailable.
    from deepface import DeepFace  # type: ignore

    # Test that DeepFace actually works by doing a simple import check
    # This helps catch issues early
    import tensorflow as tf
    import keras

    logger.info(
        f"DeepFace imported successfully. TensorFlow: {tf.__version__}, Keras: {keras.__version__}"
    )

    HAS_DEEPFACE = True
except Exception as exc:  # pragma: no cover - environment dependent
    DeepFace = None  # type: ignore[assignment]
    HAS_DEEPFACE = False
    logger.error(
        "DeepFace could not be imported (%s). "
        "FaceRecognitionEngine will run in degraded mode (no face "
        "detection or embedding extraction). Please check TensorFlow/Keras compatibility.",
        exc,
        exc_info=True,
    )

# Optional: use face_recognition (dlib) as a fallback for embeddings when DeepFace
# or TensorFlow/Keras stack is not available. This library provides 128-dim
# dlib-based face encodings and works well as a lighter-weight fallback.
try:
    import face_recognition  # type: ignore

    HAS_FACE_RECOG = True
    logger.info("face_recognition (dlib) available as fallback for embeddings")
except Exception:
    HAS_FACE_RECOG = False

# Try to import cv2, but make it optional
try:
    import cv2

    HAS_CV2 = True
    # Try to load OpenCV DNN face detector (fallback when DeepFace fails)
    try:
        # Download DNN models if not present (we'll use OpenCV's built-in face detector)
        # For now, we'll use Haar cascades or DNN if available
        FACE_CASCADE = None
        try:
            FACE_CASCADE = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            if FACE_CASCADE.empty():
                FACE_CASCADE = None
        except Exception:
            FACE_CASCADE = None

        # Try to initialize DNN face detector
        DNN_FACE_DETECTOR = None
        try:
            # OpenCV DNN face detector (more accurate than Haar)
            prototxt_path = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
            model_path = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
            # We'll download these on first use if needed
            DNN_FACE_DETECTOR_AVAILABLE = False
        except Exception:
            DNN_FACE_DETECTOR_AVAILABLE = False
    except Exception:
        FACE_CASCADE = None
        DNN_FACE_DETECTOR_AVAILABLE = False
except ImportError:
    HAS_CV2 = False
    FACE_CASCADE = None
    DNN_FACE_DETECTOR_AVAILABLE = False


class FaceRecognitionEngine:
    """Main facial recognition engine"""

    def __init__(self, model: str = "hog"):
        """
        Initialize face recognition engine

        Args:
            model: "hog" (faster) or "cnn" (more accurate)
        """
        self.model = model
        logger.info(f"Initialized FaceRecognitionEngine with model: {model}")

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image using DeepFace (primary) or OpenCV (fallback)

        Args:
            image: Input image as numpy array (BGR format from OpenCV)

        Returns:
            List of face locations as (top, right, bottom, left) tuples
        """
        # Always try DeepFace first for best accuracy
        if not HAS_DEEPFACE:
            logger.warning(
                "DeepFace not available, falling back to OpenCV (lower accuracy)"
            )
            # If OpenCV isn't available but face_recognition is, use dlib-based detector
            if not HAS_CV2 and HAS_FACE_RECOG:
                try:
                    logger.info(
                        "OpenCV not available; using face_recognition for detection"
                    )
                    # face_recognition expects RGB
                    rgb_image = image[:, :, ::-1] if image is not None else image
                    locations = face_recognition.face_locations(
                        rgb_image, model=self.model
                    )
                    return locations
                except Exception as e:
                    logger.error(
                        f"face_recognition detection failed: {e}", exc_info=True
                    )
                    return []

            return self._detect_faces_opencv(image)

        try:
            logger.info("Using DeepFace for face detection (high accuracy)")
            # Convert BGR to RGB for deepface
            if HAS_CV2:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                # Manual BGR to RGB conversion
                rgb_image = image[:, :, ::-1]

            # Use DeepFace.represent() which reliably returns facial_area information
            # This is more reliable than extract_faces() which may return just arrays
            # Try opencv first (fastest), then fallback to others if needed
            objs = None
            detector_backends = ["opencv", "ssd"]  # Start with fastest backends

            for backend in detector_backends:
                try:
                    logger.info(f"Trying DeepFace with {backend} detector backend...")
                    objs = DeepFace.represent(
                        rgb_image, enforce_detection=False, detector_backend=backend
                    )
                    if objs and len(objs) > 0:
                        logger.info(
                            f"Successfully detected {len(objs)} face(s) using {backend} backend"
                        )
                        break
                except Exception as e:
                    logger.debug(f"{backend} backend failed: {e}")
                    continue

            # If fast backends failed, try default (may be slower but more accurate)
            if not objs:
                try:
                    logger.info("Trying DeepFace with default detector...")
                    objs = DeepFace.represent(rgb_image, enforce_detection=False)
                    if objs and len(objs) > 0:
                        logger.info(
                            f"Successfully detected {len(objs)} face(s) using default detector"
                        )
                except Exception as e2:
                    logger.warning(f"DeepFace detection failed. Last error: {e2}")
                    # Fallback to OpenCV
                    if HAS_CV2:
                        logger.info("Falling back to OpenCV face detection")
                        return self._detect_faces_opencv(image)
                    return []

            # Convert deepface format to (top, right, bottom, left)
            face_locations = []

            if not objs:
                logger.warning("No faces detected in image")
                return []

            for obj in objs:
                # Extract facial_area from the returned object
                if isinstance(obj, dict) and "facial_area" in obj:
                    facial_area = obj["facial_area"]
                    x = facial_area["x"]
                    y = facial_area["y"]
                    w = facial_area["w"]
                    h = facial_area["h"]
                    # Convert to (top, right, bottom, left) format
                    face_locations.append((y, x + w, y + h, x))
                elif isinstance(obj, dict) and "region" in obj:
                    # Alternative format with region key
                    region = obj["region"]
                    x = region["x"]
                    y = region["y"]
                    w = region["w"]
                    h = region["h"]
                    face_locations.append((y, x + w, y + h, x))

            logger.info(f"Detected {len(face_locations)} faces in image")
            return face_locations

        except Exception as e:
            logger.error(f"Error detecting faces: {e}", exc_info=True)
            # Fallback to OpenCV if DeepFace fails
            if HAS_CV2:
                logger.info("Falling back to OpenCV face detection")
                return self._detect_faces_opencv(image)
            return []

    def _detect_faces_opencv(
        self, image: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using OpenCV (fallback when DeepFace is unavailable)

        Args:
            image: Input image as numpy array (BGR format from OpenCV)

        Returns:
            List of face locations as (top, right, bottom, left) tuples
        """
        if not HAS_CV2:
            logger.warning("OpenCV not available for face detection")
            return []

        try:
            logger.info("Starting OpenCV face detection...")

            # Resize image if too large for faster processing
            height, width = image.shape[:2]
            max_dimension = 1000
            if width > max_dimension or height > max_dimension:
                scale = min(max_dimension / width, max_dimension / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(
                    image, (new_width, new_height), interpolation=cv2.INTER_AREA
                )
                logger.info(
                    f"Resized image to {new_width}x{new_height} for faster detection"
                )

            # Convert to grayscale for face detection
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            face_locations = []

            # Try using Haar Cascade first (built-in, always available)
            try:
                logger.info("Loading Haar Cascade classifier...")
                cascade_path = (
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                )
                face_cascade = cv2.CascadeClassifier(cascade_path)

                if face_cascade.empty():
                    logger.error(f"Failed to load Haar Cascade from {cascade_path}")
                    return []

                logger.info("Running face detection with Haar Cascade...")
                # Optimized parameters for faster detection
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.2,  # Slightly larger steps for speed
                    minNeighbors=4,  # Reduced for better detection
                    minSize=(50, 50),  # Minimum face size
                    flags=cv2.CASCADE_SCALE_IMAGE,
                    maxSize=(gray.shape[1], gray.shape[0]),  # Don't exceed image size
                )

                logger.info(f"Haar Cascade found {len(faces)} face(s)")

                for x, y, w, h in faces:
                    # Convert to (top, right, bottom, left) format
                    # Adjust coordinates back to original image size if we resized
                    if width != image.shape[1] or height != image.shape[0]:
                        scale_x = width / image.shape[1]
                        scale_y = height / image.shape[0]
                        x = int(x * scale_x)
                        y = int(y * scale_y)
                        w = int(w * scale_x)
                        h = int(h * scale_y)
                    face_locations.append((y, x + w, y + h, x))

                if face_locations:
                    logger.info(
                        f"Detected {len(face_locations)} faces using OpenCV Haar Cascade"
                    )
                    return face_locations
                else:
                    logger.warning("Haar Cascade found faces but conversion failed")

            except Exception as cascade_error:
                logger.error(
                    f"Haar Cascade detection failed: {cascade_error}", exc_info=True
                )

            # If Haar Cascade didn't work, log and return empty
            logger.warning("OpenCV Haar Cascade detected no faces")
            return []

        except Exception as e:
            logger.error(f"Error in OpenCV face detection: {e}", exc_info=True)
            return []

    def _extract_embeddings_with_face_recognition(
        self, image: np.ndarray, face_locations: List[Tuple[int, int, int, int]]
    ) -> np.ndarray:
        """Fallback embedding extraction using dlib-based face_recognition."""
        if not HAS_FACE_RECOG:
            logger.error(
                "face_recognition fallback requested but library is unavailable"
            )
            return np.array([])

        try:
            logger.info(
                "Using face_recognition (dlib) to extract embeddings as fallback"
            )
            if image is None:
                return np.array([])

            # Convert BGR to RGB if OpenCV is available, otherwise slice channels
            rgb_image = (
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if HAS_CV2 else image[:, :, ::-1]
            )

            # If DeepFace failed before producing locations, detect using dlib
            locations = (
                face_locations
                if face_locations
                else face_recognition.face_locations(rgb_image, model=self.model)
            )

            encodings = face_recognition.face_encodings(
                rgb_image, known_face_locations=locations
            )

            if not encodings:
                logger.warning("face_recognition extracted no embeddings")
                return np.array([])

            logger.info(f"Extracted {len(encodings)} embeddings via face_recognition")
            return np.array(encodings)

        except Exception as exc:  # pragma: no cover - env specific
            logger.error(
                f"face_recognition embedding extraction failed: {exc}",
                exc_info=True,
            )
            return np.array([])

    def extract_face_embeddings(
        self, image: np.ndarray, face_locations: List[Tuple[int, int, int, int]]
    ) -> np.ndarray:
        """
        Extract face embeddings using DeepFace (optimized for performance)

        Args:
            image: Input image as numpy array (BGR format)
            face_locations: List of face locations (not used with deepface, but can be used for optimization)

        Returns:
            Array of face embeddings (N x embedding_dim for selected model)
        """
        if not HAS_DEEPFACE:
            return self._extract_embeddings_with_face_recognition(image, face_locations)

        try:
            logger.info("Extracting face embeddings with DeepFace...")
            # Convert BGR to RGB
            if HAS_CV2:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                # Manual BGR to RGB conversion
                rgb_image = image[:, :, ::-1]

            # Use represent() which is more efficient - it detects faces and extracts embeddings in one call
            # This is faster than calling extract_faces() and then represent() separately
            embedding_objs = DeepFace.represent(
                rgb_image,
                enforce_detection=False,
                model_name=DEEPFACE_MODEL_NAME,
                detector_backend=DEEPFACE_DETECTOR_BACKEND,
                align=True,
            )

            if not embedding_objs:
                logger.warning("No embeddings extracted from image")
                return np.array([])

            # Extract embeddings from the results
            embeddings_list = []
            for obj in embedding_objs:
                if isinstance(obj, dict) and "embedding" in obj:
                    embeddings_list.append(obj["embedding"])

            logger.info(f"Extracted {len(embeddings_list)} face embeddings")
            return np.array(embeddings_list) if embeddings_list else np.array([])

        except Exception as e:
            logger.error(f"Error extracting embeddings: {e}", exc_info=True)
            return self._extract_embeddings_with_face_recognition(image, face_locations)

    def compare_faces(
        self,
        known_embeddings: np.ndarray,
        test_embedding: np.ndarray,
        tolerance: float = 0.6,
    ) -> np.ndarray:
        """
        Compare test embedding against known embeddings using Euclidean distance

        Args:
            known_embeddings: Array of known embeddings (N x 512)
            test_embedding: Single test embedding (512,)
            tolerance: Distance threshold for matching

        Returns:
            Boolean array indicating matches
        """
        try:
            if len(known_embeddings) == 0 or len(test_embedding) == 0:
                return np.array([])

            # Calculate Euclidean distances
            distances = np.linalg.norm(known_embeddings - test_embedding, axis=1)

            # Convert to boolean array based on tolerance
            matches = distances < tolerance
            return matches

        except Exception as e:
            logger.error(f"Error comparing faces: {e}")
            return np.array([])

    def face_distance(
        self, known_embeddings: np.ndarray, test_embedding: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Euclidean distance between test embedding and known embeddings

        Args:
            known_embeddings: Array of known embeddings (N x 512)
            test_embedding: Single test embedding (512,)

        Returns:
            Array of distances
        """
        try:
            if len(known_embeddings) == 0 or len(test_embedding) == 0:
                return np.array([])

            # Calculate Euclidean distances
            distances = np.linalg.norm(known_embeddings - test_embedding, axis=1)
            return distances

        except Exception as e:
            logger.error(f"Error calculating face distance: {e}")
            return np.array([])

    def process_image(self, image_path: str) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Process image and extract all face embeddings

        Args:
            image_path: Path to image file

        Returns:
            Tuple of (image, list of embeddings)
        """
        try:
            # Import here to avoid circular imports
            from src.image_utils import ImageProcessor

            # Read image
            image = ImageProcessor.load_image(image_path)
            if image is None:
                logger.error(f"Failed to read image: {image_path}")
                return None, []

            # Detect faces
            face_locations = self.detect_faces(image)
            if not face_locations:
                logger.warning(f"No faces detected in {image_path}")
                return image, []

            # Extract embeddings
            embeddings = self.extract_face_embeddings(image, face_locations)

            return image, embeddings.tolist()

        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return None, []

    def batch_process_images(
        self, image_paths: List[str]
    ) -> Dict[str, List[np.ndarray]]:
        """
        Process multiple images

        Args:
            image_paths: List of image file paths

        Returns:
            Dictionary mapping image paths to embeddings
        """
        results = {}
        for image_path in image_paths:
            image, embeddings = self.process_image(image_path)
            if embeddings:
                results[image_path] = embeddings

        logger.info(f"Batch processed {len(results)} images with faces")
        return results
