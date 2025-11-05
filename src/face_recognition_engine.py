"""
Core facial recognition engine using face_recognition library
"""

import cv2
import numpy as np
import face_recognition
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from src.logger import setup_logger
from src.config import get_config

logger = setup_logger(__name__)
config = get_config()


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
        Detect faces in an image
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV)
        
        Returns:
            List of face locations as (top, right, bottom, left) tuples
        """
        try:
            # Convert BGR to RGB for face_recognition
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            face_locations = face_recognition.face_locations(
                rgb_image,
                model=self.model
            )
            
            logger.info(f"Detected {len(face_locations)} faces in image")
            return face_locations
        
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []
    
    def extract_face_embeddings(
        self,
        image: np.ndarray,
        face_locations: List[Tuple[int, int, int, int]]
    ) -> np.ndarray:
        """
        Extract 128-dimensional face embeddings
        
        Args:
            image: Input image as numpy array (BGR format)
            face_locations: List of face locations
        
        Returns:
            Array of face embeddings (N x 128)
        """
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Extract embeddings
            embeddings = face_recognition.face_encodings(
                rgb_image,
                face_locations
            )
            
            logger.info(f"Extracted {len(embeddings)} face embeddings")
            return np.array(embeddings)
        
        except Exception as e:
            logger.error(f"Error extracting embeddings: {e}")
            return np.array([])
    
    def compare_faces(
        self,
        known_embeddings: np.ndarray,
        test_embedding: np.ndarray,
        tolerance: float = 0.6
    ) -> np.ndarray:
        """
        Compare test embedding against known embeddings
        
        Args:
            known_embeddings: Array of known embeddings (N x 128)
            test_embedding: Single test embedding (128,)
            tolerance: Distance threshold for matching
        
        Returns:
            Boolean array indicating matches
        """
        try:
            matches = face_recognition.compare_faces(
                known_embeddings,
                test_embedding,
                tolerance=tolerance
            )
            return np.array(matches)
        
        except Exception as e:
            logger.error(f"Error comparing faces: {e}")
            return np.array([])
    
    def face_distance(
        self,
        known_embeddings: np.ndarray,
        test_embedding: np.ndarray
    ) -> np.ndarray:
        """
        Calculate distance between test embedding and known embeddings
        
        Args:
            known_embeddings: Array of known embeddings (N x 128)
            test_embedding: Single test embedding (128,)
        
        Returns:
            Array of distances
        """
        try:
            distances = face_recognition.face_distance(
                known_embeddings,
                test_embedding
            )
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
            # Read image
            image = cv2.imread(image_path)
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
        self,
        image_paths: List[str]
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

