"""Multi-backend face extraction pipeline.

This module implements a robust extraction strategy that:
1. First extracts embeddings from the ORIGINAL image (no upscaling)
2. Then tries each upscaling backend and extracts from upscaled versions
3. Aggregates all successful embeddings for better matching
4. Gracefully handles failures - if one backend fails, tries next
5. If all backends fail, falls back to original image embeddings
"""

from __future__ import annotations

import gc
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from src.config import get_config
from src.logger import setup_logger

logger = setup_logger(__name__)
config = get_config()


class MultiBackendExtractor:
    """Extracts face embeddings from multiple image versions for robust matching."""

    def __init__(self, face_engine: Any, upscaler: Optional[Any] = None) -> None:
        """Initialize with face recognition engine and optional upscaler."""
        self.face_engine = face_engine
        self._upscaler = upscaler
        self._upscaler_initialized = False
        self.multi_backend_enabled = getattr(config, "MULTI_BACKEND_EXTRACTION", True)

    @property
    def upscaler(self):
        """Lazy-load the upscaler to avoid circular imports and memory issues."""
        if self._upscaler is None and not self._upscaler_initialized:
            try:
                from src.image_upscaler import get_image_upscaler

                self._upscaler = get_image_upscaler()
                self._upscaler_initialized = True
            except Exception as exc:
                logger.warning("Failed to initialize upscaler: %s", exc)
                self._upscaler_initialized = True
        return self._upscaler

    def extract_all_embeddings(
        self,
        image: np.ndarray,
        source: str = "unknown",
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Extract embeddings from original image and all available upscaled versions.

        Args:
            image: Input image as numpy array (BGR format)
            source: Source identifier for logging

        Returns:
            Tuple of (aggregated_embeddings, extraction_metadata)
        """
        metadata = {
            "source": source,
            "backends_tried": [],
            "backends_succeeded": [],
            "backends_failed": [],
            "total_embeddings": 0,
        }

        all_embeddings: List[Dict[str, Any]] = []

        # Step 1: Always extract from ORIGINAL image first
        logger.info("Extracting embeddings from original image...")
        original_embeddings = self._extract_from_image(image, "original")
        if original_embeddings:
            all_embeddings.extend(original_embeddings)
            metadata["backends_succeeded"].append("original")
            logger.info(
                "Extracted %d embedding(s) from original", len(original_embeddings)
            )
        else:
            metadata["backends_failed"].append("original")
            logger.warning("No embeddings from original image")

        # Step 2: If multi-backend enabled, try each upscaling backend
        if self.multi_backend_enabled and self.upscaler:
            upscale_backends = self._get_upscale_backends()

            for backend_name in upscale_backends:
                metadata["backends_tried"].append(backend_name)
                try:
                    # Try to upscale with this specific backend
                    upscaled = self._try_upscale_with_backend(image, backend_name)
                    if upscaled is None:
                        logger.debug("Backend %s returned None", backend_name)
                        continue

                    # Extract embeddings from upscaled image
                    upscaled_embeddings = self._extract_from_image(
                        upscaled, f"upscaled_{backend_name}"
                    )

                    if upscaled_embeddings:
                        all_embeddings.extend(upscaled_embeddings)
                        metadata["backends_succeeded"].append(backend_name)
                        logger.info(
                            "Extracted %d embedding(s) from %s upscaled",
                            len(upscaled_embeddings),
                            backend_name,
                        )

                    # Clean up memory after each backend
                    del upscaled
                    self._cleanup_memory()

                except MemoryError:
                    logger.warning("Backend %s ran out of memory", backend_name)
                    metadata["backends_failed"].append(backend_name)
                    self._cleanup_memory()
                except Exception as exc:
                    logger.warning("Backend %s failed: %s", backend_name, exc)
                    metadata["backends_failed"].append(backend_name)

        metadata["total_embeddings"] = len(all_embeddings)

        # Step 3: If we have no embeddings at all, return empty with metadata
        if not all_embeddings:
            logger.warning("No embeddings extracted from any backend")
            return [], metadata

        logger.info(
            "Multi-backend extraction complete: %d total embeddings from %s",
            len(all_embeddings),
            metadata["backends_succeeded"],
        )
        return all_embeddings, metadata

    def _extract_from_image(
        self, image: np.ndarray, source_tag: str
    ) -> List[Dict[str, Any]]:
        """Extract embeddings from a single image using the face engine."""
        try:
            # Detect faces
            face_locations = self.face_engine.detect_faces(image)
            if not face_locations:
                logger.debug("No faces detected in %s", source_tag)
                return []

            # Extract embeddings
            embeddings = self.face_engine.extract_face_embeddings(image, face_locations)
            if not embeddings:
                logger.debug("No embeddings extracted from %s", source_tag)
                return []

            # Tag each embedding with its source
            for emb in embeddings:
                emb["extraction_source"] = source_tag

            return embeddings
        except Exception as exc:
            logger.warning("Extraction failed for %s: %s", source_tag, exc)
            return []

    def _get_upscale_backends(self) -> List[str]:
        """Get the list of upscaling backends to try."""
        priority_raw = getattr(config, "IMAGE_UPSCALING_BACKEND_PRIORITY", "")
        if not priority_raw:
            return ["ibm_max", "realesrgan", "opencv", "lanczos"]

        return [b.strip().lower() for b in priority_raw.split(",") if b.strip()]

    def _try_upscale_with_backend(
        self, image: np.ndarray, backend_name: str
    ) -> Optional[np.ndarray]:
        """Try to upscale an image with a specific backend."""
        if not self.upscaler:
            return None

        try:
            # Get the handler for this backend
            handler = self.upscaler._get_backend_handler(backend_name)
            if handler is None:
                return None

            # Calculate target scale
            outscale = getattr(self.upscaler, "target_scale", 2.0)

            # Try the backend
            result = handler(image, outscale)
            return result
        except Exception as exc:
            logger.debug("Backend %s upscale failed: %s", backend_name, exc)
            return None

    def _cleanup_memory(self) -> None:
        """Clean up memory after processing."""
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def aggregate_embeddings(
        self, embeddings_list: List[Dict[str, Any]]
    ) -> Dict[str, np.ndarray]:
        """
        Aggregate embeddings from multiple sources into combined vectors.

        This averages embeddings from different upscaling backends to produce
        a more robust representation.
        """
        if not embeddings_list:
            return {}

        deepface_embeddings = []
        dlib_embeddings = []

        for emb_bundle in embeddings_list:
            if "deepface" in emb_bundle and emb_bundle["deepface"]:
                vec = emb_bundle["deepface"]
                if isinstance(vec, list):
                    vec = np.array(vec)
                if isinstance(vec, np.ndarray) and vec.size > 0:
                    deepface_embeddings.append(vec)

            if "dlib" in emb_bundle and emb_bundle["dlib"]:
                vec = emb_bundle["dlib"]
                if isinstance(vec, list):
                    vec = np.array(vec)
                if isinstance(vec, np.ndarray) and vec.size > 0:
                    dlib_embeddings.append(vec)

        result = {}

        if deepface_embeddings:
            # Average all deepface embeddings
            stacked = np.vstack(deepface_embeddings)
            result["deepface"] = np.mean(stacked, axis=0)

        if dlib_embeddings:
            # Average all dlib embeddings
            stacked = np.vstack(dlib_embeddings)
            result["dlib"] = np.mean(stacked, axis=0)

        return result


def get_multi_extractor(face_engine: Any) -> MultiBackendExtractor:
    """Factory function to create a MultiBackendExtractor."""
    return MultiBackendExtractor(face_engine)
