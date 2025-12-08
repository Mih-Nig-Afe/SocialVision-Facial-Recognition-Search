"""
Search engine for finding similar faces
"""

import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from src.logger import setup_logger
from src.database import FaceDatabase
from src.face_recognition_engine import FaceRecognitionEngine
from src.image_upscaler import get_image_upscaler
from src.config import get_config

logger = setup_logger(__name__)
config = get_config()
DEFAULT_SIMILARITY_THRESHOLD = getattr(config, "FACE_SIMILARITY_THRESHOLD", 0.35)
DEFAULT_EMBEDDING_SOURCE = getattr(config, "DEFAULT_EMBEDDING_SOURCE", "deepface")
UPSCALE_RETRY_ENABLED = bool(getattr(config, "UPSCALE_RETRY_ENABLED", True))
UPSCALE_RETRY_ON_ZERO_MATCH = bool(getattr(config, "UPSCALE_RETRY_ON_ZERO_MATCH", True))
UPSCALE_RETRY_MIN_OUTSCALE = float(
    max(1.0, getattr(config, "UPSCALE_RETRY_MIN_OUTSCALE", 2.0))
)
_DEFAULT_RETRY_BACKENDS = ("ibm_max", "realesrgan", "opencv", "lanczos")


def _parse_retry_backends(raw: Optional[str]) -> List[str]:
    if not raw:
        return list(_DEFAULT_RETRY_BACKENDS)
    parsed: List[str] = []
    for token in raw.split(","):
        name = token.strip().lower()
        if not name or name in parsed:
            continue
        parsed.append(name)
    return parsed or list(_DEFAULT_RETRY_BACKENDS)


UPSCALE_RETRY_BACKENDS = _parse_retry_backends(
    getattr(config, "UPSCALE_RETRY_BACKENDS", "ibm_max,realesrgan,opencv,lanczos")
)


class SearchEngine:
    """Facial recognition search engine with multi-backend extraction support."""

    def __init__(self, database: FaceDatabase):
        """
        Initialize search engine

        Args:
            database: FaceDatabase instance
        """
        self.database = database
        self.face_engine = FaceRecognitionEngine()
        self._multi_extractor = None
        self._multi_extractor_initialized = False
        self._upscaler = None
        self._upscaler_initialized = False
        logger.info("Initialized SearchEngine")

    @property
    def multi_extractor(self):
        """Lazy-load the multi-backend extractor."""
        if self._multi_extractor is None and not self._multi_extractor_initialized:
            try:
                from src.multi_extraction import get_multi_extractor

                self._multi_extractor = get_multi_extractor(self.face_engine)
                self._multi_extractor_initialized = True
            except Exception as exc:
                logger.warning("Failed to initialize multi-extractor: %s", exc)
                self._multi_extractor_initialized = True
        return self._multi_extractor

    @property
    def upscaler(self):
        """Lazy-load the upscaler for fallback passes."""
        if self._upscaler is None and not self._upscaler_initialized:
            try:
                self._upscaler = get_image_upscaler()
            except Exception as exc:
                logger.warning("Failed to initialize upscaler for retries: %s", exc)
                self._upscaler = None
            finally:
                self._upscaler_initialized = True
        return self._upscaler

    def _extract_with_multi_backend(
        self, image: np.ndarray, source: str = "unknown"
    ) -> List[Dict]:
        """
        Extract embeddings using multi-backend strategy.

        Tries original image first, then each upscaling backend,
        aggregating all successful extractions.
        """
        # Check if multi-backend extraction is enabled
        multi_enabled = getattr(config, "MULTI_BACKEND_EXTRACTION", True)

        if multi_enabled and self.multi_extractor:
            try:
                embeddings, metadata = self.multi_extractor.extract_all_embeddings(
                    image, source
                )
                if embeddings:
                    logger.info(
                        "Multi-backend extraction: %d embeddings from %s",
                        len(embeddings),
                        metadata.get("backends_succeeded", []),
                    )
                    return embeddings
            except Exception as exc:
                logger.warning("Multi-backend extraction failed: %s", exc)

        # Fallback to standard single extraction
        logger.info("Using standard single-backend extraction")
        face_locations = self.face_engine.detect_faces(image)
        if not face_locations:
            return []

        embeddings = self.face_engine.extract_face_embeddings(image, face_locations)
        return embeddings if embeddings else []

    @staticmethod
    def _serialize_embedding_bundle(embedding: Any) -> Optional[Dict[str, List[float]]]:
        """Convert numpy-backed embedding bundles into plain python lists."""

        if not embedding:
            return None

        def _to_list(value: Any) -> Optional[List[float]]:
            if value is None:
                return None
            if hasattr(value, "tolist"):
                value = value.tolist()
            elif isinstance(value, tuple):
                value = list(value)
            elif isinstance(value, list):
                value = list(value)
            else:
                # unsupported scalar value
                return None

            return value if len(value) > 0 else None

        if isinstance(embedding, dict):
            serialized: Dict[str, List[float]] = {}
            for key, value in embedding.items():
                converted = _to_list(value)
                if converted is not None:
                    serialized[key] = converted
            return serialized or None

        converted = _to_list(embedding)
        if converted is None:
            return None
        return {DEFAULT_EMBEDDING_SOURCE: converted}

    def _auto_enrich_identity(
        self,
        username: str,
        embedding_bundle: Optional[Dict[str, List[float]]],
        similarity: Optional[float],
    ) -> Optional[Dict[str, Any]]:
        """Append new embeddings to an identity profile when a confident match occurs."""

        if not username or not embedding_bundle:
            return None

        try:
            metadata = {
                "origin": "search_enrichment",
                "trigger_similarity": (
                    float(similarity) if similarity is not None else None
                ),
            }
            summary = self.database.append_embeddings_to_username(
                username,
                [embedding_bundle],
                source="search_auto_enrich",
                metadata=metadata,
            )
            summary["username"] = username
            summary["trigger_similarity"] = metadata["trigger_similarity"]
            return summary
        except Exception as exc:
            logger.error(
                "Auto-enrichment failed for %s: %s", username, exc, exc_info=True
            )
            return {"username": username, "error": str(exc)}

    def search_by_embedding(
        self,
        query_embedding: Any,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        top_k: int = 50,
    ) -> List[Dict]:
        """
        Search for similar faces by embedding

        Args:
            query_embedding: Query face embedding
            threshold: Similarity threshold
            top_k: Number of top results

        Returns:
            List of similar faces with metadata
        """
        try:
            results = self.database.search_similar_faces(
                query_embedding, threshold=threshold, top_k=top_k
            )

            logger.info(f"Search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error searching by embedding: {e}")
            return []

    def search_by_image(
        self,
        image: np.ndarray,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        top_k: int = 50,
    ) -> Dict:
        """Search for similar faces, retrying with an aggressively upscaled frame if necessary."""

        try:
            base_shape = self._shape_from_image(image)
            primary = self._execute_search_pass(
                image=image,
                threshold=threshold,
                top_k=top_k,
                context="original",
                report_shape=base_shape,
            )

            retry_reason = self._should_retry_with_upscale(primary)
            if retry_reason:
                fallback = self._try_upscaled_retry(
                    image=image,
                    threshold=threshold,
                    top_k=top_k,
                    report_shape=base_shape,
                    reason=retry_reason,
                )
                if fallback:
                    return fallback

            return primary
        except Exception as e:
            logger.error(f"Error searching by image: {e}", exc_info=True)
            return {"faces": [], "total_matches": 0}

    def _execute_search_pass(
        self,
        image: np.ndarray,
        threshold: float,
        top_k: int,
        context: str,
        report_shape: Optional[Tuple[int, int]],
    ) -> Dict:
        face_locations = self.face_engine.detect_faces(image)
        embeddings: List[Dict] = []
        if face_locations:
            embeddings = self.face_engine.extract_face_embeddings(image, face_locations)

        source_shape = self._shape_from_image(image)
        normalized_locations = self._normalize_locations(
            face_locations,
            source_shape,
            report_shape or source_shape,
        )

        results = self._build_results(
            normalized_locations,
            embeddings,
            threshold,
            top_k,
        )

        diagnostics = {
            "context": context,
            "faces_detected": len(face_locations),
            "embeddings_extracted": len(embeddings),
        }
        if source_shape:
            diagnostics["source_shape"] = self._shape_dict(source_shape)
        if report_shape and report_shape != source_shape:
            diagnostics["reported_shape"] = self._shape_dict(report_shape)

        results["diagnostics"] = diagnostics
        results["fallback_used"] = context != "original"
        results["suggest_add_face"] = (
            len(embeddings) > 0 and results.get("total_matches", 0) == 0
        )
        logger.info(
            "Image search (%s) completed with %s total matches",
            context,
            results.get("total_matches", 0),
        )
        return results

    def _build_results(
        self,
        face_locations: List[Tuple[int, int, int, int]],
        embeddings: List[Dict],
        threshold: float,
        top_k: int,
    ) -> Dict:
        results: Dict[str, Any] = {"faces": [], "total_matches": 0}

        if not face_locations:
            return results

        if len(embeddings) == 0:
            results["faces"] = [
                {"face_index": i, "location": loc, "matches": []}
                for i, loc in enumerate(face_locations)
            ]
            return results

        logger.info("Searching database for %d face(s)...", len(embeddings))

        for i, location in enumerate(face_locations):
            embedding = embeddings[i] if i < len(embeddings) else {}
            enrichment_summary = None
            if not embedding:
                logger.warning("No embeddings available for face %s", i)
                face_results = []
            else:
                face_results = self.search_by_embedding(
                    embedding, threshold=threshold, top_k=top_k
                )

                if face_results:
                    top_match = face_results[0]
                    username = top_match.get("username")
                    similarity = top_match.get("similarity_score")
                    serialized_bundle = self._serialize_embedding_bundle(embedding)
                    enrichment_summary = self._auto_enrich_identity(
                        username, serialized_bundle, similarity
                    )

            results["faces"].append(
                {
                    "face_index": i,
                    "location": location,
                    "matches": face_results,
                    "enrichment": enrichment_summary,
                }
            )
            results["total_matches"] += len(face_results)

        return results

    def _normalize_locations(
        self,
        locations: List[Tuple[int, int, int, int]],
        source_shape: Optional[Tuple[int, int]],
        target_shape: Optional[Tuple[int, int]],
    ) -> List[Tuple[int, int, int, int]]:
        if not locations:
            return []

        if not source_shape or not target_shape or source_shape == target_shape:
            return list(locations)

        source_h, source_w = source_shape
        target_h, target_w = target_shape
        if source_h <= 0 or source_w <= 0:
            return list(locations)

        scale_y = target_h / float(source_h)
        scale_x = target_w / float(source_w)
        normalized: List[Tuple[int, int, int, int]] = []
        for top, right, bottom, left in locations:
            normalized.append(
                (
                    int(round(top * scale_y)),
                    int(round(right * scale_x)),
                    int(round(bottom * scale_y)),
                    int(round(left * scale_x)),
                )
            )
        return normalized

    def _should_retry_with_upscale(self, result: Dict) -> Optional[str]:
        if not UPSCALE_RETRY_ENABLED:
            return None

        diagnostics = result.get("diagnostics") or {}
        faces = diagnostics.get("faces_detected", 0)
        embeddings = diagnostics.get("embeddings_extracted", 0)

        if faces == 0:
            return "no_faces_detected"
        if embeddings == 0:
            return "no_embeddings"
        if (
            UPSCALE_RETRY_ON_ZERO_MATCH
            and result.get("total_matches", 0) == 0
            and embeddings > 0
        ):
            return "no_matches"
        return None

    def _try_upscaled_retry(
        self,
        image: np.ndarray,
        threshold: float,
        top_k: int,
        report_shape: Optional[Tuple[int, int]],
        reason: str,
    ) -> Optional[Dict]:
        enhanced, backend_used = self._run_retry_backends(image)
        if enhanced is None:
            return None

        fallback = self._execute_search_pass(
            image=enhanced,
            threshold=threshold,
            top_k=top_k,
            context="upscaled_retry",
            report_shape=report_shape,
        )
        fallback.setdefault("diagnostics", {})["retry_reason"] = reason
        fallback["diagnostics"]["retry_minimum_outscale"] = UPSCALE_RETRY_MIN_OUTSCALE
        fallback["diagnostics"]["retry_backend"] = backend_used
        return fallback

    def _run_retry_backends(
        self, image: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        upscaler = self.upscaler
        if upscaler is None:
            return None, None

        target_scale = max(
            UPSCALE_RETRY_MIN_OUTSCALE,
            float(getattr(upscaler, "target_scale", UPSCALE_RETRY_MIN_OUTSCALE)),
        )

        for backend_name in UPSCALE_RETRY_BACKENDS:
            handler = getattr(upscaler, "_get_backend_handler", lambda name: None)(
                backend_name
            )
            if handler is None:
                continue
            try:
                logger.info(
                    "Retrying with %s backend at >=%.2fx", backend_name, target_scale
                )
                enhanced = handler(image, target_scale)
            except Exception as exc:
                logger.warning(
                    "Retry backend %s failed: %s", backend_name, exc, exc_info=True
                )
                continue

            if enhanced is not None:
                logger.info("Upscale retry succeeded via %s backend", backend_name)
                return enhanced, backend_name

        logger.warning("All retry backends failed; skipping upscale retry")
        return None, None

    @staticmethod
    def _shape_from_image(image: Optional[np.ndarray]) -> Optional[Tuple[int, int]]:
        if image is None or not hasattr(image, "shape") or len(image.shape) < 2:
            return None
        height, width = image.shape[:2]
        return (int(height), int(width))

    @staticmethod
    def _shape_dict(shape: Tuple[int, int]) -> Dict[str, int]:
        return {"height": int(shape[0]), "width": int(shape[1])}

    def enrich_face_from_image(
        self,
        image: np.ndarray,
        source: str = "unknown",
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        top_k: int = 10,
    ) -> Dict:
        """
        Given an input image, detect the first face, extract a full embedding,
        search the database at the identity (username) level, and if a matching
        identity is found, append the new embedding(s) to that identity's
        profile and return a structured summary. If no match found, returns
        {"result": "Person not found."}.

        This method now uses multi-backend extraction:
        1. Extract from original image first
        2. Try each upscaling backend and extract from each
        3. Aggregate all embeddings for better matching and enrichment
        """
        try:
            # Use multi-backend extraction for robust embedding extraction
            embeddings = self._extract_with_multi_backend(image, source)

            if not embeddings:
                logger.warning("No embeddings extracted from any backend")
                return {"result": "Person not found."}

            # Ensure embeddings is a 2D array/list; take first embedding for identification
            first_emb_bundle = embeddings[0] if embeddings else {}
            first_emb = None
            if isinstance(first_emb_bundle, dict):
                priority = [DEFAULT_EMBEDDING_SOURCE]
                for key in first_emb_bundle.keys():
                    if key not in priority:
                        priority.append(key)
                for key in priority:
                    candidate = first_emb_bundle.get(key)
                    if candidate:
                        first_emb = candidate
                        break
            else:
                first_emb = first_emb_bundle

            if not first_emb:
                logger.warning("No suitable primary embedding for enrichment")
                return {"result": "Person not found."}

            # Search for identity-level matches (uses aggregated profile embeddings)
            identity_matches = self.database.search_identity(
                first_emb, threshold=threshold, top_k=top_k
            )

            if not identity_matches:
                logger.info("No identity-level matches found")
                return {"result": "Person not found."}

            # Pick top identity match
            top = identity_matches[0]
            username = top["username"]
            similarity = top["similarity_score"]

            # Append all extracted embeddings for this image to the matched username
            # Convert numpy arrays to lists if necessary
            emb_lists = []
            for emb in embeddings:
                if isinstance(emb, dict):
                    emb_lists.append(emb)
                    continue
                try:
                    emb_list = emb.tolist() if hasattr(emb, "tolist") else list(emb)
                    emb_lists.append({DEFAULT_EMBEDDING_SOURCE: emb_list})
                except Exception:
                    logger.warning(
                        "Skipping a malformed embedding when preparing to append"
                    )

            summary = self.database.append_embeddings_to_username(
                username, emb_lists, source
            )

            # Find most recent face id(s) appended for returned detail
            faces_for_user = self.database.get_faces_by_username(username)
            last_added_id = None
            if faces_for_user:
                last_added_id = max(face.get("id", -1) for face in faces_for_user)

            result = {
                "result": "Person found",
                "username": username,
                "match_confidence": float(similarity),
                "summary": summary,
                "last_added_face_id": last_added_id,
            }

            logger.info(f"Enriched profile for {username}: {summary}")
            return result

        except Exception as e:
            logger.error(f"Error during enrichment: {e}", exc_info=True)
            return {"result": "Person not found."}

    def get_unique_usernames(self, search_results: Dict) -> List[str]:
        """
        Extract unique usernames from search results

        Args:
            search_results: Search results dictionary

        Returns:
            List of unique usernames
        """
        try:
            usernames = set()

            for face_result in search_results.get("faces", []):
                for match in face_result.get("matches", []):
                    usernames.add(match["username"])

            return sorted(list(usernames))

        except Exception as e:
            logger.error(f"Error extracting usernames: {e}")
            return []

    def get_results_by_username(self, search_results: Dict) -> Dict[str, List]:
        """
        Group search results by username

        Args:
            search_results: Search results dictionary

        Returns:
            Dictionary mapping usernames to their matches
        """
        try:
            results_by_user = {}

            for face_result in search_results.get("faces", []):
                for match in face_result.get("matches", []):
                    username = match["username"]
                    if username not in results_by_user:
                        results_by_user[username] = []
                    results_by_user[username].append(match)

            return results_by_user

        except Exception as e:
            logger.error(f"Error grouping results by username: {e}")
            return {}

    def get_top_usernames(self, search_results: Dict, top_k: int = 10) -> List[Dict]:
        """
        Get top usernames by match count and average similarity

        Args:
            search_results: Search results dictionary
            top_k: Number of top results

        Returns:
            List of top usernames with statistics
        """
        try:
            results_by_user = self.get_results_by_username(search_results)

            user_stats = []
            for username, matches in results_by_user.items():
                avg_similarity = np.mean([m["similarity_score"] for m in matches])
                user_stats.append(
                    {
                        "username": username,
                        "match_count": len(matches),
                        "avg_similarity": float(avg_similarity),
                        "sources": list(set(m["source"] for m in matches)),
                    }
                )

            # Sort by match count and similarity
            user_stats.sort(
                key=lambda x: (x["match_count"], x["avg_similarity"]), reverse=True
            )

            return user_stats[:top_k]

        except Exception as e:
            logger.error(f"Error getting top usernames: {e}")
            return []
