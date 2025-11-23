"""Database management for face embeddings and metadata"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any, Iterable, Tuple
from datetime import datetime
from src.logger import setup_logger
from src.config import get_config

try:  # Firestore Admin client (optional)
    from google.cloud import firestore_admin_v1  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    firestore_admin_v1 = None

try:  # Firestore data client (required in Firestore mode)
    from google.cloud import firestore as google_firestore  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    google_firestore = None

try:  # Google auth helpers for admin operations
    import google.auth  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    google = None

try:  # Service account credential helpers
    from google.oauth2 import service_account  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    service_account = None

try:  # Shared google api exceptions
    from google.api_core import exceptions as google_exceptions  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    google_exceptions = None

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
        self.use_firestore = getattr(config, "DB_TYPE", "local").lower() == "firestore"
        self.db_path = Path(db_path or config.LOCAL_DB_PATH)
        self._firestore_client = None
        self._faces_collection = None
        self._profiles_collection = None
        self._collection_prefix = getattr(
            config, "FIRESTORE_COLLECTION_PREFIX", "socialvision_"
        )
        self._database_id = (
            getattr(config, "FIRESTORE_DATABASE_ID", "(default)") or "(default)"
        ).strip()
        self._firestore_location = getattr(
            config, "FIRESTORE_LOCATION_ID", "us-central"
        )
        self._google_credentials = None

        if self.use_firestore:
            self._init_firestore()
            self.data = {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "faces": [],
                "metadata": {},
            }
            logger.info(
                "Initialized Firestore FaceDatabase for project %s",
                getattr(config, "FIREBASE_PROJECT_ID", "unknown"),
            )
        else:
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

    # ------------------------------------------------------------------
    # Firestore helpers
    # ------------------------------------------------------------------
    def _init_firestore(self) -> None:
        if google_firestore is None:
            raise RuntimeError(
                "google-cloud-firestore is required for Firestore mode. Install it and set DB_TYPE=firestore."
            )

        project_id = getattr(config, "FIREBASE_PROJECT_ID", None)
        credentials_obj, inferred_project = self._resolve_firestore_credentials()
        self._google_credentials = credentials_obj
        project_id = project_id or inferred_project

        if not project_id:
            raise RuntimeError(
                "FIREBASE_PROJECT_ID must be set (or derivable from credentials) to use Firestore"
            )

        try:
            self._firestore_client = google_firestore.Client(
                project=project_id,
                credentials=self._google_credentials,
                database=self._database_id,
            )
        except Exception as exc:
            not_found = google_exceptions is not None and isinstance(
                exc, google_exceptions.NotFound
            )
            if not_found and self._database_id == "(default)":
                logger.warning(
                    "Firestore database %s missing; attempting to create it",
                    self._database_id,
                )
                self._ensure_firestore_database_exists(project_id)
                self._firestore_client = google_firestore.Client(
                    project=project_id,
                    credentials=self._google_credentials,
                    database=self._database_id,
                )
            else:
                logger.error(
                    "Unable to initialize Firestore client for database %s: %s",
                    self._database_id,
                    exc,
                )
                raise
        prefix = self._collection_prefix
        self._faces_collection = self._firestore_client.collection(f"{prefix}faces")
        self._profiles_collection = self._firestore_client.collection(
            f"{prefix}profiles"
        )

    def _resolve_firestore_credentials(self) -> Tuple[Any, Optional[str]]:
        scopes = ["https://www.googleapis.com/auth/datastore"]

        firebase_config = config.load_firebase_config()
        if firebase_config and service_account is not None:
            creds = service_account.Credentials.from_service_account_info(
                firebase_config, scopes=scopes
            )
            return creds, firebase_config.get("project_id")

        config_path = getattr(config, "FIREBASE_CONFIG_PATH", None)
        if (
            service_account is not None
            and config_path
            and Path(config_path).expanduser().exists()
        ):
            creds = service_account.Credentials.from_service_account_file(
                str(Path(config_path).expanduser()), scopes=scopes
            )
            return creds, getattr(creds, "project_id", None)

        if google is None:
            raise RuntimeError(
                "google-auth is required to resolve Firestore credentials"
            )

        creds, inferred_project = google.auth.default(scopes=scopes)
        return creds, inferred_project

    def _ensure_firestore_database_exists(self, project_id: str) -> None:
        """Create the default Firestore database if it does not exist."""

        if firestore_admin_v1 is None:
            raise RuntimeError(
                "google-cloud-firestore-admin is required to auto-create Firestore databases"
            )
        if google is None:
            raise RuntimeError(
                "google-auth is required to auto-create Firestore databases"
            )
        if google_exceptions is None:
            raise RuntimeError(
                "google-api-core is required to detect Firestore provisioning errors"
            )

        logger.info(
            "Ensuring Firestore database %s (%s) exists", self._database_id, project_id
        )
        scopes = ["https://www.googleapis.com/auth/datastore"]
        if self._google_credentials is None:
            credentials_to_use, _ = google.auth.default(scopes=scopes)
        else:
            credentials_to_use = self._google_credentials

        client = firestore_admin_v1.FirestoreAdminClient(credentials=credentials_to_use)
        parent = f"projects/{project_id}"
        database_path = f"{parent}/databases/{self._database_id}"

        try:
            existing = client.get_database(name=database_path)
            if existing:
                logger.info("Firestore database %s already exists", database_path)
                return
        except google_exceptions.NotFound:
            pass

        database = firestore_admin_v1.Database(
            name=database_path,
            location_id=self._firestore_location,
            type_=firestore_admin_v1.Database.DatabaseType.FIRESTORE_NATIVE,
        )
        operation = client.create_database(
            parent=parent, database=database, database_id=self._database_id
        )
        logger.info("Waiting for Firestore database creation operation to finish...")
        operation.result(timeout=120)
        logger.info("Firestore database %s created", database_path)

    def _firestore_face_from_snapshot(self, snapshot) -> Dict[str, Any]:
        data = snapshot.to_dict() or {}
        data.setdefault("id", snapshot.id)
        data.setdefault("added_at", datetime.now().isoformat())
        return data

    def _firestore_faces_iter(
        self, username: Optional[str] = None
    ) -> Iterable[Dict[str, Any]]:
        if not self._faces_collection:
            return []
        query = self._faces_collection
        if username:
            query = query.where("username", "==", username)
        return (self._firestore_face_from_snapshot(doc) for doc in query.stream())

    def _write_profile_metadata(
        self, username: str, profile_vector: Optional[np.ndarray], count: int
    ) -> None:
        if self.use_firestore:
            payload = {
                "last_updated": datetime.now().isoformat(),
                "embeddings_count": count,
            }
            if profile_vector is not None:
                payload["profile_embedding"] = profile_vector.tolist()
            self._profiles_collection.document(username).set(payload, merge=True)
        else:
            self.data.setdefault("metadata", {})
            meta = self.data["metadata"].setdefault(username, {})
            if profile_vector is not None:
                meta["profile_embedding"] = profile_vector.tolist()
            meta["last_updated"] = datetime.now().isoformat()
            meta["embeddings_count"] = count

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
        if self.use_firestore:
            # Firestore writes happen per-operation; nothing to persist locally
            return True

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
                "embedding": primary_embedding,
                "embeddings": bundle,
                "username": username,
                "source": source,
                "image_url": image_url,
                "added_at": datetime.now().isoformat(),
                "metadata": metadata or {},
            }

            if self.use_firestore:
                doc_ref = self._faces_collection.document()
                face_record["id"] = doc_ref.id
                doc_ref.set(face_record)
                # Refresh centroid/profile cache for this user
                self.get_profile_embedding_for_username(username, force_refresh=True)
                logger.info(
                    "Stored face for %s in Firestore collection %s",
                    username,
                    self._faces_collection.id,
                )
                return True

            face_record["id"] = len(self.data["faces"])
            self.data["faces"].append(face_record)

            if self._save_database():
                self.get_profile_embedding_for_username(username, force_refresh=True)
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
            if self.use_firestore:
                doc = self._faces_collection.document(str(face_id)).get()
                if doc.exists:
                    return self._firestore_face_from_snapshot(doc)
                # Fall back to querying on the stored id field
                matches = self._faces_collection.where("id", "==", str(face_id)).limit(
                    1
                )
                for snapshot in matches.stream():
                    return self._firestore_face_from_snapshot(snapshot)
                return None

            for face in self.data["faces"]:
                if face["id"] == face_id:
                    return face
        except Exception as e:
            logger.error(f"Error getting face by ID: {e}")
        return None

    def get_faces_by_username(self, username: str) -> List[Dict]:
        try:
            if self.use_firestore:
                return list(self._firestore_faces_iter(username))
            return [face for face in self.data["faces"] if face["username"] == username]
        except Exception as e:
            logger.error(f"Error getting faces by username: {e}")
            return []

    def get_profile_embedding_for_username(
        self, username: str, force_refresh: bool = False
    ) -> Optional[np.ndarray]:
        """
        Compute or retrieve an aggregated (centroid) normalized embedding for a username.

        Returns:
            L2-normalized numpy vector representing the user's profile, or None.
        """
        try:
            if self.use_firestore:
                meta_doc = self._profiles_collection.document(username).get()
                if meta_doc.exists and not force_refresh:
                    payload = meta_doc.to_dict() or {}
                    if "profile_embedding" in payload:
                        return self._normalize_embedding_vector(
                            payload["profile_embedding"]
                        )
            else:
                meta = self.data.get("metadata", {}).get(username, {})
                if meta and "profile_embedding" in meta and not force_refresh:
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
            profile_vec = centroid / norm
            self._write_profile_metadata(username, profile_vec, len(faces))
            return profile_vec
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

            added = 0
            face_records: List[Dict[str, Any]] = []
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
                    "embedding": primary_embedding,
                    "embeddings": bundle,
                    "username": username,
                    "source": source,
                    "image_url": image_url,
                    "added_at": datetime.now().isoformat(),
                    "metadata": metadata or {},
                }
                face_records.append(face_record)
                added += 1

            if added == 0:
                return summary

            if self.use_firestore:
                if not self._firestore_client:
                    raise RuntimeError("Firestore client not initialized")
                batch = self._firestore_client.batch()
                for record in face_records:
                    doc_ref = self._faces_collection.document()
                    record["id"] = doc_ref.id
                    batch.set(doc_ref, record)
                batch.commit()
            else:
                for record in face_records:
                    record["id"] = len(self.data["faces"])
                    self.data["faces"].append(record)
                if not self._save_database():
                    logger.error("Failed to save database after appending embeddings")

            profile = self.get_profile_embedding_for_username(
                username, force_refresh=True
            )
            if profile is not None:
                summary["updated_profile_dim"] = profile.shape[0]
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
            if self.use_firestore:
                profile_docs = self._profiles_collection.stream()
                for doc in profile_docs:
                    username = doc.id
                    payload = doc.to_dict() or {}
                    profile_raw = payload.get("profile_embedding")
                    if not profile_raw:
                        continue
                    try:
                        profile = self._normalize_embedding_vector(profile_raw)
                    except ValueError:
                        continue
                    if profile.shape != query_vec.shape:
                        continue
                    similarity = float(np.dot(profile, query_vec))
                    if similarity >= threshold:
                        candidates.append(
                            {
                                "username": username,
                                "similarity_score": similarity,
                                "embeddings_count": payload.get("embeddings_count", 0),
                            }
                        )
            else:
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

            faces_iter: Iterable[Dict[str, Any]]
            if self.use_firestore:
                faces_iter = self._firestore_faces_iter()
            else:
                faces_iter = self.data["faces"]

            for face in faces_iter:
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

            faces_iter: Iterable[Dict[str, Any]]
            if self.use_firestore:
                faces_iter = list(self._firestore_faces_iter())
                if not faces_iter:
                    return []
            else:
                faces_iter = self.data["faces"]
                if not faces_iter:
                    return []

            for face in faces_iter:
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
            if self.use_firestore:
                usernames = set()
                sources: Dict[str, int] = {}
                total = 0
                for face in self._firestore_faces_iter():
                    total += 1
                    usernames.add(face.get("username"))
                    src = face.get("source", "unknown")
                    sources[src] = sources.get(src, 0) + 1
                return {
                    "total_faces": total,
                    "unique_users": len([u for u in usernames if u]),
                    "sources": sources,
                    "created_at": self.data.get("created_at"),
                }

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
            if self.use_firestore:
                for doc in self._faces_collection.stream():
                    doc.reference.delete()
                for doc in self._profiles_collection.stream():
                    doc.reference.delete()
                logger.warning("Firestore face database cleared")
                return True

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
