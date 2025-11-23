"""
Search engine for finding similar faces
"""

import numpy as np
from typing import List, Dict, Optional, Any
from src.logger import setup_logger
from src.database import FaceDatabase
from src.face_recognition_engine import FaceRecognitionEngine
from src.config import get_config

logger = setup_logger(__name__)
config = get_config()
DEFAULT_SIMILARITY_THRESHOLD = getattr(config, "FACE_SIMILARITY_THRESHOLD", 0.35)
DEFAULT_EMBEDDING_SOURCE = getattr(config, "DEFAULT_EMBEDDING_SOURCE", "deepface")


class SearchEngine:
    """Facial recognition search engine"""

    def __init__(self, database: FaceDatabase):
        """
        Initialize search engine

        Args:
            database: FaceDatabase instance
        """
        self.database = database
        self.face_engine = FaceRecognitionEngine()
        logger.info("Initialized SearchEngine")

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
        """
        Search for similar faces by image

        Args:
            image: Input image (BGR format)
            threshold: Similarity threshold
            top_k: Number of top results

        Returns:
            Dictionary with search results for each detected face
        """
        try:
            logger.info("Starting face detection in search_by_image...")
            # Detect faces
            face_locations = self.face_engine.detect_faces(image)
            logger.info(
                f"Face detection completed. Found {len(face_locations)} face(s)"
            )

            if not face_locations:
                logger.warning("No faces detected in query image")
                return {"faces": [], "total_matches": 0}

            logger.info("Extracting face embeddings...")
            # Extract embeddings
            embeddings = self.face_engine.extract_face_embeddings(image, face_locations)
            logger.info(f"Extracted {len(embeddings)} embedding bundle(s)")

            if len(embeddings) == 0:
                logger.warning(
                    "No embeddings extracted - DeepFace may not be available"
                )
                # Return empty results but indicate faces were detected
                return {
                    "faces": [
                        {"face_index": i, "location": loc, "matches": []}
                        for i, loc in enumerate(face_locations)
                    ],
                    "total_matches": 0,
                }

            # Search for each face
            results = {"faces": [], "total_matches": 0}

            logger.info(f"Searching database for {len(embeddings)} face(s)...")
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

            logger.info(
                f"Image search completed with {results['total_matches']} total matches"
            )
            return results

        except Exception as e:
            logger.error(f"Error searching by image: {e}", exc_info=True)
            return {"faces": [], "total_matches": 0}

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

        This method focuses on enrichment: it makes the matched user's stored
        embeddings broader and more representative over time.
        """
        try:
            # Detect faces
            face_locations = self.face_engine.detect_faces(image)
            if not face_locations:
                logger.warning("No faces detected for enrichment")
                return {"result": "Person not found."}

            # Extract embeddings for faces (we will use the first face for identity search)
            embeddings = self.face_engine.extract_face_embeddings(image, face_locations)
            if embeddings is None or len(embeddings) == 0:
                logger.warning("No embeddings extracted for enrichment")
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
