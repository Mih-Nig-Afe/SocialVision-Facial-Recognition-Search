"""
Tests for face recognition engine
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile
from src.face_recognition_engine import FaceRecognitionEngine
from src.image_utils import ImageProcessor


@pytest.fixture
def face_engine():
    """Create face recognition engine"""
    return FaceRecognitionEngine(model="hog")


@pytest.fixture
def sample_image():
    """Create a sample image for testing"""
    # Create a simple test image (100x100 RGB)
    image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def test_face_engine_initialization(face_engine):
    """Test face engine initialization"""
    assert face_engine is not None
    assert face_engine.model == "hog"


def test_detect_faces_empty_image(face_engine, sample_image):
    """Test face detection on empty image"""
    faces = face_engine.detect_faces(sample_image)
    assert isinstance(faces, list)
    assert len(faces) == 0


def test_extract_embeddings_empty(face_engine, sample_image):
    """Test embedding extraction with no faces"""
    embeddings = face_engine.extract_face_embeddings(sample_image, [])
    assert isinstance(embeddings, np.ndarray)
    assert len(embeddings) == 0


def test_compare_faces(face_engine):
    """Test face comparison"""
    # Create dummy embeddings
    known = np.random.rand(5, 128)
    test = known[0]
    
    matches = face_engine.compare_faces(known, test, tolerance=0.6)
    assert isinstance(matches, np.ndarray)
    assert len(matches) == 5


def test_face_distance(face_engine):
    """Test face distance calculation"""
    known = np.random.rand(5, 128)
    test = known[0]
    
    distances = face_engine.face_distance(known, test)
    assert isinstance(distances, np.ndarray)
    assert len(distances) == 5
    assert distances[0] < 0.1  # Should be very close to itself


def test_process_image_nonexistent(face_engine):
    """Test processing nonexistent image"""
    image, embeddings = face_engine.process_image("/nonexistent/path.jpg")
    assert image is None
    assert embeddings == []


def test_batch_process_images(face_engine):
    """Test batch processing"""
    results = face_engine.batch_process_images(["/nonexistent/1.jpg", "/nonexistent/2.jpg"])
    assert isinstance(results, dict)
    assert len(results) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

