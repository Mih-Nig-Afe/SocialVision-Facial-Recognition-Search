"""
Tests for database module
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from src.database import FaceDatabase


@pytest.fixture
def temp_db():
    """Create temporary database for testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        db_path = f.name
    
    db = FaceDatabase(db_path)
    yield db
    
    # Cleanup
    Path(db_path).unlink(missing_ok=True)


def test_database_initialization(temp_db):
    """Test database initialization"""
    assert temp_db is not None
    assert temp_db.db_path is not None


def test_add_face(temp_db):
    """Test adding a face to database"""
    embedding = np.random.rand(128).tolist()
    result = temp_db.add_face(
        embedding,
        "testuser",
        "profile_pic"
    )
    assert result is True
    assert len(temp_db.data["faces"]) == 1


def test_get_all_embeddings(temp_db):
    """Test getting all embeddings"""
    # Add some faces
    for i in range(3):
        embedding = np.random.rand(128).tolist()
        temp_db.add_face(embedding, f"user{i}", "post")
    
    embeddings = temp_db.get_all_embeddings()
    assert embeddings.shape == (3, 128)


def test_get_face_by_id(temp_db):
    """Test getting face by ID"""
    embedding = np.random.rand(128).tolist()
    temp_db.add_face(embedding, "testuser", "profile_pic")
    
    face = temp_db.get_face_by_id(0)
    assert face is not None
    assert face["username"] == "testuser"


def test_get_faces_by_username(temp_db):
    """Test getting faces by username"""
    # Add multiple faces for same user
    for i in range(3):
        embedding = np.random.rand(128).tolist()
        temp_db.add_face(embedding, "testuser", f"source{i}")
    
    faces = temp_db.get_faces_by_username("testuser")
    assert len(faces) == 3


def test_search_similar_faces(temp_db):
    """Test searching similar faces"""
    # Add some faces
    base_embedding = np.random.rand(128)
    for i in range(5):
        embedding = base_embedding + np.random.rand(128) * 0.1
        temp_db.add_face(embedding.tolist(), f"user{i}", "post")
    
    # Search
    results = temp_db.search_similar_faces(base_embedding, threshold=0.5, top_k=3)
    assert len(results) <= 3


def test_get_statistics(temp_db):
    """Test getting database statistics"""
    # Add faces
    for i in range(3):
        embedding = np.random.rand(128).tolist()
        temp_db.add_face(embedding, f"user{i}", "post")
    
    stats = temp_db.get_statistics()
    assert stats["total_faces"] == 3
    assert stats["unique_users"] == 3


def test_clear_database(temp_db):
    """Test clearing database"""
    # Add faces
    for i in range(3):
        embedding = np.random.rand(128).tolist()
        temp_db.add_face(embedding, f"user{i}", "post")
    
    assert len(temp_db.data["faces"]) == 3
    
    # Clear
    result = temp_db.clear_database()
    assert result is True
    assert len(temp_db.data["faces"]) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

