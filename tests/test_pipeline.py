"""Tests for the high-level FacePipeline module."""

import io
from datetime import datetime

import pytest
from PIL import Image

from src.database import FaceDatabase
from src.pipeline import FacePipeline, PipelineResult


@pytest.fixture()
def temp_db(tmp_path):
    db_path = tmp_path / "pipeline_db.json"
    db = FaceDatabase(str(db_path))
    yield db


def _sample_image_bytes(color=(255, 0, 0)) -> bytes:
    """Create a tiny RGB image and return as JPEG bytes."""
    image = Image.new("RGB", (4, 4), color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


class StubSearchEngine:
    def __init__(self, database, response):
        self.database = database
        self._response = response

    def enrich_face_from_image(self, **_kwargs):
        return self._response


def test_pipeline_successful_match(temp_db):
    """FacePipeline should normalize successful enrichment responses."""
    username = "unit_user"
    temp_db.add_face([1.0, 0.0, 0.0, 0.0], username, "profile")
    temp_db.data.setdefault("metadata", {})[username] = {
        "embeddings_count": 5,
        "last_updated": datetime.now().isoformat(),
    }

    stub_response = {
        "result": "Person found",
        "username": username,
        "match_confidence": 0.88,
        "summary": {"added": 1, "updated_profile_dim": 4},
        "last_added_face_id": 42,
    }

    pipeline = FacePipeline(
        database=temp_db, search_engine=StubSearchEngine(temp_db, stub_response)
    )

    bytes_image = _sample_image_bytes()
    result: PipelineResult = pipeline.process_image_bytes(bytes_image)

    assert result.is_match
    payload = result.response
    assert payload["person_id"] == username
    assert payload["data_updated"]["embeddings_added"] == 1
    assert payload["data_updated"]["last_added_face_id"] == 42


def test_pipeline_returns_not_found(temp_db):
    """When enrichment fails the pipeline should return the canonical message."""
    stub_response = {"result": "Person not found."}
    pipeline = FacePipeline(
        database=temp_db, search_engine=StubSearchEngine(temp_db, stub_response)
    )

    bytes_image = _sample_image_bytes(color=(0, 255, 0))
    result = pipeline.process_image_bytes(bytes_image)

    assert not result.is_match
    assert result.response == {"result": "Person not found."}
