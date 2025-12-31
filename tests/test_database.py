"""
Tests for database module
"""

import json
import pytest
import numpy as np
import tempfile
from datetime import datetime
from pathlib import Path
from src.database import FaceDatabase


@pytest.fixture
def temp_db():
    """Create temporary database for testing"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
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
    embedding = {
        "deepface": np.random.rand(128).tolist(),
        "dlib": np.random.rand(128).tolist(),
    }

    result = temp_db.add_face(embedding, "testuser", "profile_pic")
    assert result is True
    assert len(temp_db.data["faces"]) == 1
    stored = temp_db.data["faces"][0]
    assert "embeddings" in stored
    assert set(stored["embeddings"].keys()) == {"deepface", "dlib"}


def test_add_face_enriches_existing_record_with_new_dimensions_only(temp_db):
    """If a user exists and we add a NEW embedding key, add_face should patch it."""

    # First add a deepface-only face.
    assert temp_db.add_face({"deepface": [0.0, 1.0, 0.0, 0.0]}, "testuser", "profile")
    assert len(temp_db.data["faces"]) == 1
    assert "dlib" not in (temp_db.data["faces"][0].get("embeddings") or {})

    # Now add dlib-only for the same user (new dimension key).
    assert temp_db.add_face({"dlib": [1.0, 0.0, 0.0, 0.0]}, "testuser", "profile")
    # Should not create a second face record; should enrich existing.
    assert len(temp_db.data["faces"]) == 1
    assert "dlib" in (temp_db.data["faces"][0].get("embeddings") or {})


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


def test_search_similar_faces_with_bundles(temp_db):
    """Ensure searching works when both query and stored faces use bundles."""
    deepface_vec = np.random.rand(4).tolist()
    dlib_vec = np.random.rand(4).tolist()
    temp_db.add_face({"deepface": deepface_vec, "dlib": dlib_vec}, "combo", "profile")

    query = {"deepface": deepface_vec}
    results = temp_db.search_similar_faces(query, threshold=0.2, top_k=1)

    assert len(results) == 1
    assert results[0]["username"] == "combo"


def test_search_similar_faces_ignores_dimension_mismatch(temp_db):
    """Search should not error when embeddings have different dimensions."""
    # Store a 128-d face
    temp_db.add_face(np.random.rand(128).tolist(), "small", "profile")

    # Query with a 512-d vector (e.g., DeepFace Facenet512)
    results = temp_db.search_similar_faces(np.random.rand(512).tolist(), threshold=0.0)

    # No crash; may return 0 matches due to mismatch
    assert isinstance(results, list)


def test_append_embeddings_updates_profile_metadata(temp_db):
    """Ensure append_embeddings_to_username stores metadata and recomputes centroid."""
    base_embedding = {"deepface": np.ones(4).tolist()}
    temp_db.add_face(base_embedding, "tester", "profile")

    new_embeddings = [
        {"deepface": [0.8, 0.2, 0.0, 0.0], "dlib": [0.4, 0.6, 0.0, 0.0]},
        {"deepface": [0.7, 0.3, 0.0, 0.0]},
    ]

    summary = temp_db.append_embeddings_to_username(
        "tester", new_embeddings, source="post"
    )

    assert summary["added"] == 2
    assert summary["updated_profile_dim"] == 4

    metadata = temp_db.data["metadata"].get("tester")
    assert metadata["embeddings_count"] == 3
    assert "profile_embedding" in metadata


def test_append_embeddings_patch_only_does_not_create_new_face(temp_db):
    """Patch-only appends should enrich an existing record without creating new faces."""

    # Start with a deepface-only record for a user.
    temp_db.add_face({"deepface": np.ones(4).tolist()}, "tester", "profile")
    assert len(temp_db.data["faces"]) == 1
    before = temp_db.data["faces"][0]
    assert "dlib" not in (before.get("embeddings") or {})

    # Patch in a new embedding key.
    summary = temp_db.append_embeddings_to_username(
        "tester",
        [{"dlib": [0.25, 0.25, 0.25, 0.25]}],
        source="live_auto_enrich",
        allow_create=False,
    )

    assert summary["added"] == 0
    assert summary["patched"] == 1
    assert summary["patched_keys"] >= 1
    assert len(temp_db.data["faces"]) == 1
    after = temp_db.data["faces"][0]
    assert "dlib" in (after.get("embeddings") or {})


def test_search_identity_returns_sorted_matches(temp_db):
    """Identity search should return usernames ordered by similarity."""
    temp_db.add_face([1.0, 0.0, 0.0, 0.0], "alice", "profile")
    temp_db.add_face([0.0, 1.0, 0.0, 0.0], "bob", "profile")

    results = temp_db.search_identity([1.0, 0.0, 0.0, 0.0], threshold=0.1, top_k=2)

    assert len(results) >= 1
    assert results[0]["username"] == "alice"
    assert results[0]["similarity_score"] >= results[-1]["similarity_score"]


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


def test_restore_from_backup_when_primary_empty(tmp_path):
    """Ensure backup file is used if the primary local DB is unexpectedly empty."""

    db_path = tmp_path / "faces.json"
    backup_path = db_path.with_suffix(db_path.suffix + ".bak")

    primary_payload = {
        "version": "1.0",
        "created_at": datetime.now().isoformat(),
        "faces": [],
        "metadata": {},
    }
    backup_payload = {
        "version": "1.0",
        "created_at": datetime.now().isoformat(),
        "faces": [
            {
                "id": 0,
                "embedding": [1.0, 0.0],
                "embeddings": {"deepface": [1.0, 0.0]},
                "username": "backup_user",
                "source": "profile_pic",
                "image_url": None,
                "added_at": datetime.now().isoformat(),
                "metadata": {},
            }
        ],
        "metadata": {
            "backup_user": {
                "profile_embedding": [1.0, 0.0],
                "embeddings_count": 1,
            }
        },
    }

    db_path.write_text(json.dumps(primary_payload))
    backup_path.write_text(json.dumps(backup_payload))

    restored = FaceDatabase(str(db_path))

    assert restored.data["faces"], "Backup payload should repopulate faces"
    assert restored.data["faces"][0]["username"] == "backup_user"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
