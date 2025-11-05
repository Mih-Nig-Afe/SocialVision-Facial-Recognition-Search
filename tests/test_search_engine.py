"""
Tests for search engine
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from src.database import FaceDatabase
from src.search_engine import SearchEngine


@pytest.fixture
def temp_db():
    """Create temporary database"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        db_path = f.name
    
    db = FaceDatabase(db_path)
    yield db
    
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def search_engine(temp_db):
    """Create search engine"""
    return SearchEngine(temp_db)


@pytest.fixture
def populated_db(temp_db):
    """Create database with sample data"""
    base_embedding = np.random.rand(128)
    
    for i in range(10):
        # Create similar embeddings
        embedding = base_embedding + np.random.rand(128) * 0.05
        temp_db.add_face(
            embedding.tolist(),
            f"user{i % 3}",
            ["profile_pic", "post", "story"][i % 3]
        )
    
    return temp_db


def test_search_engine_initialization(search_engine):
    """Test search engine initialization"""
    assert search_engine is not None


def test_search_by_embedding_empty(search_engine):
    """Test search with empty database"""
    embedding = np.random.rand(128)
    results = search_engine.search_by_embedding(embedding)
    assert isinstance(results, list)
    assert len(results) == 0


def test_search_by_embedding_with_data(search_engine, populated_db):
    """Test search with populated database"""
    search_engine.database = populated_db
    
    embedding = np.random.rand(128)
    results = search_engine.search_by_embedding(embedding, threshold=0.5, top_k=5)
    assert isinstance(results, list)


def test_get_unique_usernames(search_engine, populated_db):
    """Test extracting unique usernames"""
    search_engine.database = populated_db
    
    search_results = {
        "faces": [
            {
                "matches": [
                    {"username": "user1"},
                    {"username": "user2"}
                ]
            }
        ]
    }
    
    usernames = search_engine.get_unique_usernames(search_results)
    assert len(usernames) == 2
    assert "user1" in usernames


def test_get_results_by_username(search_engine):
    """Test grouping results by username"""
    search_results = {
        "faces": [
            {
                "matches": [
                    {"username": "user1", "similarity_score": 0.9},
                    {"username": "user2", "similarity_score": 0.8}
                ]
            }
        ]
    }
    
    results_by_user = search_engine.get_results_by_username(search_results)
    assert "user1" in results_by_user
    assert "user2" in results_by_user


def test_get_top_usernames(search_engine):
    """Test getting top usernames"""
    search_results = {
        "faces": [
            {
                "matches": [
                    {"username": "user1", "similarity_score": 0.95, "source": "profile_pic"},
                    {"username": "user1", "similarity_score": 0.92, "source": "post"},
                    {"username": "user2", "similarity_score": 0.85, "source": "story"}
                ]
            }
        ]
    }
    
    top_users = search_engine.get_top_usernames(search_results, top_k=5)
    assert len(top_users) > 0
    assert top_users[0]["username"] == "user1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

