"""High-level face search and enrichment pipeline"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np

from src.config import get_config
from src.database import FaceDatabase
from src.image_utils import ImageProcessor
from src.logger import setup_logger
from src.search_engine import SearchEngine

logger = setup_logger(__name__)
config = get_config()
DEFAULT_SIMILARITY_THRESHOLD = getattr(config, "FACE_SIMILARITY_THRESHOLD", 0.35)


def _normalize_threshold(value: Optional[float]) -> float:
    if value is None:
        return DEFAULT_SIMILARITY_THRESHOLD
    # Clamp threshold to [0, 1]
    return float(max(0.0, min(1.0, value)))


@dataclass
class PipelineResult:
    """Return metadata for pipeline executions."""

    payload: Dict[str, Any]

    @property
    def is_match(self) -> bool:
        return self.payload.get("result") == "Person found"

    @property
    def response(self) -> Dict[str, Any]:
        return self.payload


class FacePipeline:
    """Implements the full search-and-enrichment workflow for new faces."""

    def __init__(
        self,
        database: Optional[FaceDatabase] = None,
        search_engine: Optional[SearchEngine] = None,
    ) -> None:
        self.database = database or FaceDatabase()
        self.search_engine = search_engine or SearchEngine(self.database)
        logger.info("Initialized FacePipeline with local database")

    def _prepare_image(self, image: np.ndarray, enhance: bool = True) -> np.ndarray:
        """Resize/enhance images to maximize embedding quality."""
        image = ImageProcessor.resize_image(image)
        if enhance:
            image = ImageProcessor.enhance_image(image)
        return image

    def _run_enrichment(
        self, image: np.ndarray, source: str, threshold: Optional[float], top_k: int
    ) -> PipelineResult:
        """Execute the search/update loop and normalize the output schema."""
        similarity_threshold = _normalize_threshold(threshold)
        enrichment = self.search_engine.enrich_face_from_image(
            image=image,
            source=source,
            threshold=similarity_threshold,
            top_k=top_k,
        )

        if enrichment.get("result") != "Person found":
            return PipelineResult({"result": "Person not found."})

        username = enrichment.get("username")
        metadata = self.database.data.get("metadata", {}).get(username, {})
        summary = enrichment.get("summary", {}) or {}

        response = {
            "result": "Person found",
            "person_id": username,
            "person_name": username,
            "match_confidence": float(enrichment.get("match_confidence", 0.0)),
            "data_updated": {
                "embeddings_added": summary.get("added", 0),
                "profile_dimensions": summary.get("updated_profile_dim"),
                "total_embeddings": metadata.get("embeddings_count"),
                "last_updated": metadata.get("last_updated"),
                "last_added_face_id": enrichment.get("last_added_face_id"),
            },
        }

        return PipelineResult(response)

    def process_image_bytes(
        self,
        image_bytes: bytes,
        source: str = "unknown",
        threshold: Optional[float] = None,
        top_k: int = 10,
    ) -> PipelineResult:
        """Load image from bytes and run the enrichment pipeline."""
        image = ImageProcessor.load_image_from_bytes(image_bytes)
        if image is None:
            logger.warning("Unable to decode uploaded image bytes")
            return PipelineResult({"result": "Person not found."})

        prepared = self._prepare_image(image)
        return self._run_enrichment(prepared, source, threshold, top_k)

    def process_image_file(
        self,
        image_path: str,
        source: str = "unknown",
        threshold: Optional[float] = None,
        top_k: int = 10,
    ) -> PipelineResult:
        """Convenience helper for CLI/Batch usage."""
        image = ImageProcessor.load_image(image_path)
        if image is None:
            logger.warning("Unable to read image at %s", image_path)
            return PipelineResult({"result": "Person not found."})

        prepared = self._prepare_image(image)
        return self._run_enrichment(prepared, source, threshold, top_k)


__all__ = ["FacePipeline", "PipelineResult"]
