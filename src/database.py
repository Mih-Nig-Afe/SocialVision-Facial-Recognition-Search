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


class FaceDatabase:
    """Local JSON-based face database"""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path or config.LOCAL_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._load_database()
        logger.info(f"Initialized FaceDatabase at {self.db_path}")

    @staticmethod
    def _normalize_embedding_vector(embedding: List[float]) -> np.ndarray:
        """Convert embedding to an L2-normalized numpy vector."""
        arr = np.asarray(embedding, dtype=np.float32).flatten()
        if arr.size == 0:
            raise ValueError("Embedding vector is empty")
        norm = np.linalg.norm(arr)
        if norm == 0:
            raise ValueError("Embedding vector has zero magnitude")
        return arr / norm

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
        embedding: List[float],
        username: str,
        source: str,
        image_url: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> bool:
        try:
            if not embedding:
                logger.error("Empty embedding provided")
                return False

            if not isinstance(embedding, list):
                logger.error(f"Embedding must be a list, got {type(embedding)}")
                return False

            normalized_embedding = self._normalize_embedding_vector(embedding)
            embedding_dim = normalized_embedding.shape[0]
            logger.info(
                f"Adding face: username={username}, source={source}, embedding_dim={embedding_dim}"
            )

            face_record = {
                "id": len(self.data["faces"]),
                "embedding": normalized_embedding.tolist(),
                "username": username,
                "source": source,
                "image_url": image_url,
                "added_at": datetime.now().isoformat(),
                "metadata": metadata or {},
            }

            self.data["faces"].append(face_record)

            if self._save_database():
                logger.info(
                    f"Successfully added face for user {username} from {source} (embedding_dim={embedding_dim})"
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
        query_embedding: np.ndarray,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        top_k: int = 50,
    ) -> List[Dict]:
        try:
            if not self.data["faces"]:
                return []

            query_vec = self._normalize_embedding_vector(query_embedding)
            results: List[Dict[str, Any]] = []

            for face in self.data["faces"]:
                try:
                    face_vec = self._normalize_embedding_vector(face["embedding"])
                except ValueError:
                    continue

                if face_vec.shape != query_vec.shape:
                    continue

                similarity = float(np.dot(face_vec, query_vec))
                if similarity >= threshold:
                    results.append({**face, "similarity_score": similarity})

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
