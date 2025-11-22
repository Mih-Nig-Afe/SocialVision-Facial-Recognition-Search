"""Database management for face embeddings and metadata"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
from src.logger import setup_logger
from src.config import get_config

logger = setup_logger(__name__)
config = get_config()
DEFAULT_SIMILARITY_THRESHOLD = getattr(config, "FACE_SIMILARITY_THRESHOLD", 0.35)
DEFAULT_EMBEDDING_SOURCE = getattr(config, "DEFAULT_EMBEDDING_SOURCE", "deepface")
EMBEDDING_WEIGHTS = getattr(
    config,
    "EMBEDDING_WEIGHTS",
    {
        "deepface": 0.7,
        "dlib": 0.3,
    },
)


class FaceDatabase:
    """Local JSON-based face database"""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path or config.LOCAL_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._load_database()
        logger.info(f"Initialized FaceDatabase at {self.db_path}")

    @staticmethod
    def _normalize_embedding_vector(embedding: Optional[List[float]]) -> np.ndarray:
        """Convert embedding to an L2-normalized numpy vector."""
        if embedding is None:
            raise ValueError("Embedding vector is missing")

        arr = np.asarray(embedding, dtype=np.float32).flatten()
        if arr.size == 0:
            raise ValueError("Embedding vector is empty")
        norm = np.linalg.norm(arr)
        if norm == 0:
            raise ValueError("Embedding vector has zero magnitude")
        return arr / norm

    def _normalize_embedding_bundle(
        self, bundle: Optional[Dict[str, List[float]]]
    ) -> Dict[str, List[float]]:
        """Normalize every embedding inside a bundle keyed by backend name."""
        normalized: Dict[str, List[float]] = {}
        if not bundle:
            return normalized

        for key, value in bundle.items():
            try:
                normalized[key] = self._normalize_embedding_vector(value).tolist()
            except ValueError:
                logger.warning("Skipping invalid %s embedding while normalizing", key)
        return normalized

    def _select_primary_embedding(
        self, bundle: Dict[str, List[float]]
    ) -> Optional[List[float]]:
        """Pick the preferred embedding vector from a bundle."""
        if not bundle:
            return None

        priority = [DEFAULT_EMBEDDING_SOURCE]
        for key in bundle.keys():
            if key not in priority:
                priority.append(key)

        for key in priority:
            candidate = bundle.get(key)
            if candidate:
                return candidate

        return None

    def _load_database(self) -> Dict[str, Any]:
        try:
            if self.db_path.exists():
                with open(self.db_path, "r") as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading database: {e}")

        return {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "faces": [],
            "metadata": {},
        }

    def _save_database(self) -> bool:
        try:
            with open(self.db_path, "w") as f:
                json.dump(self.data, f, indent=2)
            logger.info(f"Database saved successfully to {self.db_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving database: {e}", exc_info=True)
            return False

    def add_face(
        self,
        embedding: Any,
        username: str,
        source: str,
        image_url: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> bool:
        try:
            bundle: Dict[str, List[float]]
            if isinstance(embedding, dict):
                bundle = self._normalize_embedding_bundle(embedding)
            else:
                if embedding is None:
                    logger.error("Empty embedding provided")
                    return False
                if not isinstance(embedding, list):
                    try:
                        embedding = list(embedding)
                    except Exception:
                        logger.error(
                            "Embedding must be list-like; received %s", type(embedding)
                        )
                        return False

                normalized = self._normalize_embedding_vector(embedding)
                bundle = {DEFAULT_EMBEDDING_SOURCE: normalized.tolist()}

            if not bundle:
                logger.error("No valid embeddings provided")
                return False

            primary_embedding = self._select_primary_embedding(bundle)
            if not primary_embedding:
                logger.error("Unable to determine primary embedding for face record")
                return False

            embedding_dim = len(primary_embedding)
            logger.info(
                "Adding face: username=%s, source=%s, embedding_dim=%s",
                username,
                source,
                embedding_dim,
            )

            face_record = {
                "id": len(self.data["faces"]),
                "embedding": primary_embedding,
                "embeddings": bundle,
                "username": username,
                "source": source,
                "image_url": image_url,
                "added_at": datetime.now().isoformat(),
                "metadata": metadata or {},
            }

            self.data["faces"].append(face_record)

            if self._save_database():
                logger.info(
                    "Successfully added face for user %s from %s (embedding_dim=%s)",
                    username,
                    source,
                    embedding_dim,
                )
                return True

            logger.error("Failed to save database after adding face")
            self.data["faces"].pop()
            return False

        except Exception as e:
            logger.error(f"Error adding face: {e}", exc_info=True)
            return False

    def get_face_by_id(self, face_id: int) -> Optional[Dict]:
        try:
            for face in self.data["faces"]:
                if face["id"] == face_id:
                    return face
        except Exception as e:
            logger.error(f"Error getting face by ID: {e}")
        return None

    def get_faces_by_username(self, username: str) -> List[Dict]:
        try:
            return [face for face in self.data["faces"] if face["username"] == username]
        except Exception as e:
            logger.error(f"Error getting faces by username: {e}")
            return []

    def get_profile_embedding_for_username(self, username: str) -> Optional[np.ndarray]:
        """
        Compute or retrieve an aggregated (centroid) normalized embedding for a username.

        Returns:
            L2-normalized numpy vector representing the user's profile, or None.
        """
        try:
            # If metadata contains cached profile embedding, try to use it
            meta = self.data.get("metadata", {}).get(username, {})
            if meta and "profile_embedding" in meta:
                return self._normalize_embedding_vector(meta["profile_embedding"])

            # Otherwise compute from all face records for username
            faces = self.get_faces_by_username(username)
            if not faces:
                return None

            emb_list = []
            for face in faces:
                try:
                    vec = self._normalize_embedding_vector(face["embedding"])
                    emb_list.append(vec)
                except Exception:
                    continue

            if not emb_list:
                return None

            # Average then re-normalize to get centroid profile
            stacked = np.vstack(emb_list)
            centroid = np.mean(stacked, axis=0)
            norm = np.linalg.norm(centroid)
            if norm == 0:
                return None
            return centroid / norm
        except Exception as e:
            logger.error(f"Error computing profile embedding for {username}: {e}")
            return None

    def append_embeddings_to_username(
        self,
        username: str,
        embeddings: List[Any],
        source: str,
        image_url: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Append new embeddings to a username's profile by creating new face records
        and updating a cached profile embedding (centroid) in metadata.

        Returns a summary dict with counts and the updated profile similarity dimension.
        """
        summary = {"added": 0, "username": username, "updated_profile_dim": None}
        try:
            if not embeddings:
                return summary

            # Ensure metadata container exists
            self.data.setdefault("metadata", {})
            self.data["metadata"].setdefault(username, {})

            added = 0
            for emb in embeddings:
                try:
                    if isinstance(emb, dict):
                        bundle = self._normalize_embedding_bundle(emb)
                    else:
                        if not isinstance(emb, list):
                            emb = list(emb)
                        norm_vec = self._normalize_embedding_vector(emb)
                        bundle = {DEFAULT_EMBEDDING_SOURCE: norm_vec.tolist()}
                except Exception:
                    logger.warning("Skipping invalid embedding during append")
                    continue

                if not bundle:
                    continue

                primary_embedding = self._select_primary_embedding(bundle)
                if not primary_embedding:
                    continue

                face_record = {
                    "id": len(self.data["faces"]),
                    "embedding": primary_embedding,
                    "embeddings": bundle,
                    "username": username,
                    "source": source,
                    "image_url": image_url,
                    "added_at": datetime.now().isoformat(),
                    "metadata": metadata or {},
                }

                self.data["faces"].append(face_record)
                added += 1

            # Recompute and cache profile embedding
            profile = self.get_profile_embedding_for_username(username)
            if profile is not None:
                self.data["metadata"][username]["profile_embedding"] = profile.tolist()
                self.data["metadata"][username][
                    "last_updated"
                ] = datetime.now().isoformat()
                self.data["metadata"][username]["embeddings_count"] = len(
                    self.get_faces_by_username(username)
                )
                summary["updated_profile_dim"] = profile.shape[0]

            # Persist to disk
            if not self._save_database():
                logger.error("Failed to save database after appending embeddings")
            summary["added"] = added
            return summary

        except Exception as e:
            logger.error(f"Error appending embeddings for {username}: {e}")
            return summary

    def search_identity(
        self,
        query_embedding: List[float],
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        top_k: int = 10,
    ) -> List[Dict]:
        """
        Search identities (usernames) by comparing a query embedding to per-username
        profile embeddings (centroid of all stored embeddings for that username).

        Returns list of dicts: {username, similarity_score, embeddings_count}
        """
        try:
            query_vec = self._normalize_embedding_vector(query_embedding)
            candidates = []

            # Iterate over usernames present in metadata or faces
            usernames = set(face["username"] for face in self.data.get("faces", []))
            for username in usernames:
                profile = self.get_profile_embedding_for_username(username)
                if profile is None:
                    continue

                if profile.shape != query_vec.shape:
                    # Dimension mismatch - skip
                    continue

                similarity = float(np.dot(profile, query_vec))
                if similarity >= threshold:
                    candidates.append(
                        {
                            "username": username,
                            "similarity_score": similarity,
                            "embeddings_count": len(
                                self.get_faces_by_username(username)
                            ),
                        }
                    )

            candidates.sort(key=lambda item: item["similarity_score"], reverse=True)
            return candidates[:top_k]

        except Exception as e:
            logger.error(f"Error searching identities: {e}")
            return []

    def get_all_embeddings(self) -> np.ndarray:
        """Return all stored embeddings as a normalized numpy matrix."""
        try:
            embeddings = []
            target_dim: Optional[int] = None

            for face in self.data["faces"]:
                try:
                    vec = self._normalize_embedding_vector(face["embedding"])
                except ValueError:
                    continue

                if target_dim is None:
                    target_dim = vec.shape[0]

                if vec.shape[0] != target_dim:
                    # Skip embeddings with mismatched dimensions (different models)
                    continue

                embeddings.append(vec)

            if not embeddings:
                return np.empty((0, 0), dtype=np.float32)

            return np.vstack(embeddings)

        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            return np.empty((0, 0), dtype=np.float32)

    def search_similar_faces(
        self,
        query_embedding: Any,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        top_k: int = 50,
    ) -> List[Dict]:
        try:
            if not self.data["faces"]:
                return []

            query_bundle: Dict[str, np.ndarray] = {}
            if isinstance(query_embedding, dict):
                for key, value in query_embedding.items():
                    try:
                        query_bundle[key] = self._normalize_embedding_vector(value)
                    except ValueError:
                        continue
            else:
                try:
                    query_vec = self._normalize_embedding_vector(query_embedding)
                    query_bundle[DEFAULT_EMBEDDING_SOURCE] = query_vec
                except ValueError:
                    return []

            if not query_bundle:
                return []

            results: List[Dict[str, Any]] = []

            for face in self.data["faces"]:
                stored_bundle = face.get("embeddings") or {}
                if not stored_bundle and face.get("embedding"):
                    stored_bundle = {DEFAULT_EMBEDDING_SOURCE: face["embedding"]}

                weighted_score = 0.0
                weight_total = 0.0

                for key, q_vec in query_bundle.items():
                    raw_face_vec = stored_bundle.get(key)
                    if raw_face_vec is None:
                        continue
                    try:
                        face_vec = self._normalize_embedding_vector(raw_face_vec)
                    except ValueError:
                        continue

                    similarity = float(np.dot(face_vec, q_vec))
                    weight = EMBEDDING_WEIGHTS.get(key, 1.0)
                    if weight <= 0:
                        weight = 1.0
                    weighted_score += weight * similarity
                    weight_total += weight

                if weight_total == 0.0:
                    # Fallback to primary embedding comparison when bundle types differ
                    try:
                        fallback_face_vec = self._normalize_embedding_vector(
                            face.get("embedding")
                        )
                        fallback_query_vec = next(iter(query_bundle.values()))
                        similarity = float(
                            np.dot(fallback_face_vec, fallback_query_vec)
                        )
                        score = similarity
                    except (StopIteration, ValueError):
                        continue
                else:
                    score = weighted_score / weight_total

                if score >= threshold:
                    results.append({**face, "similarity_score": score})

            results.sort(key=lambda item: item["similarity_score"], reverse=True)
            top_results = results[:top_k]
            logger.info(f"Found {len(top_results)} similar faces")
            return top_results

        except Exception as e:
            logger.error(f"Error searching similar faces: {e}")
            return []

    def get_statistics(self) -> Dict[str, Any]:
        try:
            usernames = set(face["username"] for face in self.data["faces"])
            sources: Dict[str, int] = {}
            for face in self.data["faces"]:
                source = face["source"]
                sources[source] = sources.get(source, 0) + 1

            return {
                "total_faces": len(self.data["faces"]),
                "unique_users": len(usernames),
                "sources": sources,
                "created_at": self.data["created_at"],
            }
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}

    def clear_database(self) -> bool:
        try:
            self.data = {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "faces": [],
                "metadata": {},
            }
            self._save_database()
            logger.warning("Database cleared")
            return True
        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            return False
