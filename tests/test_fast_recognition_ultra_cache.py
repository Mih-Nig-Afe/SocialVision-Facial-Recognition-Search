import numpy as np

import src.fast_recognition as fast_recognition


def _clear_ultra_fast_cache() -> None:
    fast_recognition._embedding_cache.clear()
    fast_recognition._embedding_cache_time.clear()


class _FakeLocalDB:
    def __init__(self, faces):
        self.use_firestore = False
        self.use_realtime = False
        self.data = {"faces": faces}


def test_ultra_fast_search_uses_dlib_embeddings_and_does_not_dim_mismatch():
    _clear_ultra_fast_cache()
    faces = [
        {
            "id": 1,
            "username": "alice",
            "source": "test",
            # Primary embedding is 512 (default)
            "embedding": [0.1] * 512,
            # But dlib 128 is available for ultra-fast mode
            "embeddings": {"deepface": [0.1] * 512, "dlib": [0.2] * 128},
        },
        {
            "id": 2,
            "username": "bob",
            "source": "test",
            "embedding": [0.1] * 512,
            "embeddings": {"deepface": [0.1] * 512, "dlib": [0.0] * 128},
        },
    ]
    db = _FakeLocalDB(faces)

    query = np.array([0.2] * 128, dtype=np.float32)
    # With threshold=0 we should always get matches (cosine sim >= 0)
    results = fast_recognition.ultra_fast_search(query, db, threshold=0.0, limit=1)

    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0]["username"] in {"alice", "bob"}


def test_ultra_fast_search_returns_empty_when_no_dlib_embeddings_present():
    _clear_ultra_fast_cache()
    faces = [
        {
            "id": 1,
            "username": "alice",
            "source": "test",
            "embedding": [0.1] * 512,
            "embeddings": {"deepface": [0.1] * 512},
        }
    ]
    db = _FakeLocalDB(faces)

    query = np.array([0.2] * 128, dtype=np.float32)
    results = fast_recognition.ultra_fast_search(query, db, threshold=0.0, limit=1)

    assert results == []
