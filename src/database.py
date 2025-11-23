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
    from google.cloud.firestore_admin_v1 import types as firestore_admin_types  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    firestore_admin_v1 = None
    firestore_admin_types = None

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

#
# Additional imports required for REST-based Firestore fallback
import time
import requests

try:
    # For refreshing service account tokens (optional)
    from google.auth.transport.requests import Request as _GoogleAuthRequest  # type: ignore
except Exception:
    _GoogleAuthRequest = None


def _get_bearer_token_from_credentials(creds) -> str:
    """Ensure credentials are fresh and return an OAuth2 bearer token string.

    This uses google-auth service account credentials when available. If the
    credentials object already contains a valid token, it will be used.
    """
    if creds is None:
        raise RuntimeError("No credentials available for Firestore REST client")

    # If credentials already have a token and it's still valid, use it
    token = getattr(creds, "token", None)
    expiry = getattr(creds, "expiry", None)
    if token and expiry:
        try:
            # expiry may be a datetime; if far in future we can reuse
            if expiry.timestamp() - time.time() > 60:
                return token
        except Exception:
            pass

    # Otherwise try to refresh using google.auth.transport.requests.Request
    if _GoogleAuthRequest is None:
        # As a last resort, try to return existing token if present
        if token:
            return token
        raise RuntimeError(
            "Cannot refresh credentials: google.auth.transport.requests.Request is not available"
        )

    request = _GoogleAuthRequest()
    try:
        creds.refresh(request)
    except Exception as exc:
        raise RuntimeError("Failed to refresh service account credentials: %s" % exc)

    token = getattr(creds, "token", None)
    if not token:
        raise RuntimeError("Unable to obtain access token from credentials")
    return token


class RestDocumentSnapshot:
    """Minimal snapshot wrapper for a document returned by Firestore REST API."""

    def __init__(self, name: str, fields: dict):
        # name is full resource name: projects/{p}/databases/(default)/documents/{col}/{doc}
        self._name = name
        self.id = name.split("/")[-1]
        self._fields = fields or {}

    @staticmethod
    def _decode_firestore_value(value: Dict[str, Any]):
        if value is None:
            return None
        if "nullValue" in value:
            return None
        if "booleanValue" in value:
            return bool(value["booleanValue"])
        if "integerValue" in value:
            try:
                return int(value["integerValue"])
            except Exception:
                return value["integerValue"]
        if "doubleValue" in value:
            return float(value["doubleValue"])
        if "timestampValue" in value:
            return value["timestampValue"]
        if "stringValue" in value:
            return value["stringValue"]
        if "referenceValue" in value:
            return value["referenceValue"]
        if "arrayValue" in value:
            arr = value.get("arrayValue", {}).get("values", [])
            return [RestDocumentSnapshot._decode_firestore_value(item) for item in arr]
        if "mapValue" in value:
            nested_fields = value.get("mapValue", {}).get("fields", {})
            return {
                key: RestDocumentSnapshot._decode_firestore_value(val)
                for key, val in nested_fields.items()
            }
        # Unknown type; return as-is for debugging
        return value

    def to_dict(self) -> dict:
        result = {}
        for k, v in self._fields.items():
            result[k] = self._decode_firestore_value(v)
        return result


class RestCollection:
    def __init__(self, client, collection_path: str):
        self._client = client
        # collection_path is the tail e.g. socialvision_faces
        self._collection_path = collection_path

    def document(self, doc_id: Optional[str] = None):
        # Create a pseudo-document reference object
        return RestDocument(self._client, self._collection_path, doc_id)

    def stream(self):
        # List documents in collection
        return self._client.list_documents(self._collection_path)

    def where(self, field_name, op, value):
        # For simple equality queries only; return an object with stream()
        return self._client.query_collection(self._collection_path, field_name, value)

    @property
    def id(self) -> str:
        # Match attribute exposed by native Firestore collection references
        return self._collection_path


class RestDocument:
    def __init__(self, client, collection_path: str, doc_id: Optional[str]):
        self._client = client
        self._collection_path = collection_path
        self.id = doc_id or self._client.make_id()

    def set(self, payload: dict, merge: bool = False):
        return self._client.set_document(
            self._collection_path, self.id, payload, merge=merge
        )

    def get(self):
        doc = self._client.get_document(self._collection_path, self.id)
        if doc is None:

            class _Fake:
                exists = False

            return _Fake()
        snapshot = RestDocumentSnapshot(doc["name"], doc.get("fields", {}))

        # emulate .exists attribute and .to_dict()
        class _Snap:
            def __init__(self, snap):
                self._snap = snap
                self.exists = True

            def to_dict(self):
                return snap.to_dict()

        snap = snapshot
        wrapper = type(
            "_SnapWrapper",
            (),
            {"exists": True, "to_dict": lambda self=None: snap.to_dict()},
        )()
        return wrapper

    @property
    def reference(self):
        # Used for deletion in clear_database
        return self

    def delete(self):
        return self._client.delete_document(self._collection_path, self.id)


class RestBatch:
    def __init__(self, client):
        self._client = client
        self._writes = []

    def set(self, doc_ref: RestDocument, record: dict):
        self._writes.append((doc_ref._collection_path, doc_ref.id, record))

    def commit(self):
        for col, doc_id, record in self._writes:
            self._client.set_document(col, doc_id, record)
        return True


class RestFirestoreClient:
    def __init__(
        self,
        project: str,
        database: str = "(default)",
        location_id: Optional[str] = None,
        credentials=None,
    ):
        self.project = project
        self._database = database or "(default)"
        self._location_id = location_id or "us-central"
        self._creds = credentials
        self._api_root = "https://firestore.googleapis.com/v1"
        self._database_path = f"projects/{project}/databases/{self._database}"
        self._documents_base = f"{self._api_root}/{self._database_path}/documents"

    def collection(self, name: str):
        return RestCollection(self, name)

    def make_id(self) -> str:
        # fallback ID generator
        return str(int(time.time() * 1000))

    def _auth_headers(self):
        token = _get_bearer_token_from_credentials(self._creds)
        return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    def database_exists(self) -> bool:
        url = f"{self._api_root}/{self._database_path}"
        resp = requests.get(url, headers=self._auth_headers())
        if resp.status_code == 404:
            return False
        if resp.status_code == 403:
            logger.warning(
                "Insufficient permission to verify Firestore database %s; assuming it exists.",
                self._database,
            )
            return True
        resp.raise_for_status()
        return True

    def ensure_database(self) -> bool:
        if self.database_exists():
            return False
        self._create_database()
        return True

    def _create_database(self) -> None:
        url = f"{self._api_root}/projects/{self.project}/databases"
        params = {"databaseId": self._database}
        body = {
            "type": "FIRESTORE_NATIVE",
            "locationId": self._location_id,
        }
        resp = requests.post(
            url, headers=self._auth_headers(), params=params, json=body
        )
        if resp.status_code == 409:
            return
        if resp.status_code == 403:
            raise PermissionError(
                "Permission denied while attempting to create Firestore database."
            )
        resp.raise_for_status()
        data = resp.json()
        operation_name = data.get("name")
        if operation_name:
            self._wait_for_operation(operation_name)

    def _wait_for_operation(self, operation_name: str, timeout: int = 120) -> None:
        deadline = time.time() + timeout
        url = f"{self._api_root}/{operation_name}"
        while time.time() < deadline:
            resp = requests.get(url, headers=self._auth_headers())
            resp.raise_for_status()
            payload = resp.json()
            if payload.get("done"):
                if "error" in payload:
                    raise RuntimeError(
                        f"Firestore admin operation failed: {payload['error']}"
                    )
                return
            time.sleep(2)
        raise TimeoutError("Timed out waiting for Firestore admin operation to finish")

    def list_documents(self, collection_path: str):
        url = f"{self._documents_base}/{collection_path}"
        resp = requests.get(url, headers=self._auth_headers())
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        data = resp.json()
        docs = data.get("documents", [])
        for d in docs:
            name = d.get("name")
            fields = d.get("fields", {})
            yield RestDocumentSnapshot(name, fields)

    def get_document(self, collection_path: str, doc_id: str):
        url = f"{self._documents_base}/{collection_path}/{doc_id}"
        resp = requests.get(url, headers=self._auth_headers())
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    def set_document(
        self, collection_path: str, doc_id: str, payload: dict, merge: bool = False
    ):
        # Convert simple python dict to Firestore REST typed fields for arrays and scalars
        def _to_fields(obj):
            fields = {}
            for k, v in obj.items():
                if v is None:
                    continue
                if isinstance(v, str):
                    fields[k] = {"stringValue": v}
                elif isinstance(v, bool):
                    fields[k] = {"booleanValue": v}
                elif isinstance(v, int):
                    fields[k] = {"integerValue": str(v)}
                elif isinstance(v, float):
                    fields[k] = {"doubleValue": v}
                elif isinstance(v, list):
                    arr = []
                    for item in v:
                        if isinstance(item, str):
                            arr.append({"stringValue": item})
                        elif isinstance(item, int):
                            arr.append({"integerValue": str(item)})
                        elif isinstance(item, float):
                            arr.append({"doubleValue": item})
                        else:
                            # fallback to string
                            arr.append({"stringValue": str(item)})
                    fields[k] = {"arrayValue": {"values": arr}}
                elif isinstance(v, dict):
                    fields[k] = {"mapValue": {"fields": _to_fields(v)}}
                else:
                    fields[k] = {"stringValue": str(v)}
            return fields

        body = {"fields": _to_fields(payload)}

        if merge:
            # Merges must target an existing document
            url = f"{self._documents_base}/{collection_path}/{doc_id}"
            resp = requests.patch(url, headers=self._auth_headers(), json=body)
        else:
            # Firestore REST API requires POST for creating documents with custom IDs
            url = f"{self._documents_base}/{collection_path}"
            params = {"documentId": doc_id}
            resp = requests.post(
                url, headers=self._auth_headers(), params=params, json=body
            )
            if resp.status_code == 409:
                # Document already exists, fall back to patch/overwrite
                url = f"{self._documents_base}/{collection_path}/{doc_id}"
                resp = requests.patch(url, headers=self._auth_headers(), json=body)

        if resp.status_code in (200, 201):
            return True

        resp.raise_for_status()
        return True

    def delete_document(self, collection_path: str, doc_id: str):
        url = f"{self._documents_base}/{collection_path}/{doc_id}"
        resp = requests.delete(url, headers=self._auth_headers())
        if resp.status_code in (200, 204):
            return True
        if resp.status_code == 404:
            return False
        resp.raise_for_status()

    def batch(self):
        return RestBatch(self)

    def query_collection(self, collection_path: str, field_name: str, value):
        # Very small helper that lists documents and filters by field equality
        docs = list(self.list_documents(collection_path))
        filtered = []
        for d in docs:
            payload = d.to_dict()
            if payload.get(field_name) == value:
                filtered.append(d)

        class _Q:
            def __init__(self, docs):
                self._docs = docs

            def stream(self):
                for d in self._docs:
                    yield d

        return _Q(filtered)


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
    """Local JSON or Firestore-backed face database."""

    def __init__(
        self,
        db_path: Optional[str] = None,
        project_id: Optional[str] = None,
        db_type: Optional[str] = None,
        collection_prefix: Optional[str] = None,
    ):
        db_mode = (db_type or getattr(config, "DB_TYPE", "local")).lower()
        if db_path is not None and db_type is None:
            # Explicit local path implies local mode unless overridden
            db_mode = "local"
        self.use_firestore = db_mode == "firestore"
        self.db_path = Path(db_path or config.LOCAL_DB_PATH)
        self._firestore_client = None
        self._faces_collection = None
        self._profiles_collection = None
        self._collection_prefix = collection_prefix or getattr(
            config, "FIRESTORE_COLLECTION_PREFIX", "socialvision_"
        )
        self._project_id_override = project_id
        self._database_id = (
            getattr(config, "FIRESTORE_DATABASE_ID", "(default)") or "(default)"
        ).strip()
        self._firestore_location = getattr(
            config, "FIRESTORE_LOCATION_ID", "us-central"
        ).strip()
        self._ensure_firestore_database = getattr(
            config, "FIRESTORE_ENSURE_DATABASE", False
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
        project_id = self._project_id_override or getattr(
            config, "FIREBASE_PROJECT_ID", None
        )
        credentials_obj, inferred_project = self._resolve_firestore_credentials()
        self._google_credentials = credentials_obj
        project_id = project_id or inferred_project

        if not project_id:
            raise RuntimeError(
                "FIREBASE_PROJECT_ID must be set (or derivable from credentials) to use Firestore"
            )

        self._database_id, inferred_location = self._select_firestore_database(
            project_id, self._database_id, self._firestore_location
        )
        if inferred_location:
            self._firestore_location = inferred_location

        try:
            self._firestore_client = RestFirestoreClient(
                project_id,
                database=self._database_id,
                location_id=self._firestore_location,
                credentials=self._google_credentials,
            )
        except Exception as exc:
            logger.error(
                "Unable to initialize Firestore REST client: %s",
                exc,
            )
            raise
        if self._ensure_firestore_database:
            try:
                created = self._firestore_client.ensure_database()
                if created:
                    logger.info(
                        "Provisioned Firestore database %s for project %s",
                        self._database_id,
                        project_id,
                    )
            except PermissionError as exc:
                raise RuntimeError(
                    "Firestore database %s does not exist and could not be created automatically. "
                    "Visit https://console.cloud.google.com/datastore/setup?project=%s to create a Firestore database (Native mode) before running in Firestore mode, or set FIRESTORE_ENSURE_DATABASE=false to skip auto-provisioning."
                    % (self._database_id, project_id)
                ) from exc
        prefix = self._collection_prefix
        self._faces_collection = self._firestore_client.collection(f"{prefix}faces")
        self._profiles_collection = self._firestore_client.collection(
            f"{prefix}profiles"
        )

    def _resolve_firestore_credentials(self) -> Tuple[Any, Optional[str]]:
        scopes = [
            "https://www.googleapis.com/auth/datastore",
            "https://www.googleapis.com/auth/cloud-platform",
        ]

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

    def _select_firestore_database(
        self, project_id: str, desired_db: str, default_location: str
    ) -> Tuple[str, Optional[str]]:
        desired = (desired_db or "(default)").strip()
        try:
            databases = self._list_firestore_databases(project_id)
        except PermissionError:
            logger.warning(
                "Firestore credentials lack permission to list databases; defaulting to %s",
                desired,
            )
            return desired, default_location
        except Exception as exc:
            logger.warning(
                "Unable to inspect Firestore databases for project %s: %s",
                project_id,
                exc,
            )
            return desired, default_location

        if not databases:
            return desired, default_location

        def _parse(entry: Dict[str, Any]) -> Tuple[str, Optional[str]]:
            name = entry.get("name", "")
            db_id = name.split("/")[-1] if name else ""
            return db_id, entry.get("locationId")

        parsed = [_parse(entry) for entry in databases]

        # First try exact match on desired id
        for db_id, location in parsed:
            if db_id == desired:
                return db_id, location or default_location

        # Fall back to known defaults
        for fallback in ("default", "(default)"):
            for db_id, location in parsed:
                if db_id == fallback:
                    return db_id, location or default_location

        # Otherwise pick the first available database
        db_id, location = parsed[0]
        logger.warning(
            "Using Firestore database %s for project %s (requested %s not found)",
            db_id,
            project_id,
            desired,
        )
        return db_id or desired, location or default_location

    def _list_firestore_databases(self, project_id: str) -> List[Dict[str, Any]]:
        token = _get_bearer_token_from_credentials(self._google_credentials)
        url = f"https://firestore.googleapis.com/v1/projects/{project_id}/databases"
        resp = requests.get(url, headers={"Authorization": f"Bearer {token}"})
        if resp.status_code == 403:
            raise PermissionError("Insufficient permission to list Firestore databases")
        resp.raise_for_status()
        payload = resp.json()
        return payload.get("databases", [])

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

    def _write_document_with_retry(
        self,
        doc_ref,
        payload: Dict[str, Any],
        merge: bool = False,
        retries: int = 3,
        delay: float = 1.0,
    ) -> bool:
        """Best-effort helper to persist a document with simple backoff."""
        last_error: Optional[Exception] = None
        for attempt in range(1, retries + 1):
            try:
                doc_ref.set(payload, merge=merge)
                return True
            except Exception as exc:  # pragma: no cover - network dependent
                last_error = exc
                logger.warning(
                    "Attempt %s/%s failed writing Firestore document %s: %s",
                    attempt,
                    retries,
                    getattr(doc_ref, "id", "<unknown>"),
                    exc,
                )
                if attempt < retries:
                    time.sleep(delay)

        logger.error(
            "Failed writing Firestore document %s after %s attempts: %s",
            getattr(doc_ref, "id", "<unknown>"),
            retries,
            last_error,
        )
        return False

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
            doc_ref = self._profiles_collection.document(username)
            if not self._write_document_with_retry(doc_ref, payload, merge=True):
                logger.error("Failed to update profile metadata for %s", username)
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
                if not self._write_document_with_retry(doc_ref, face_record):
                    return False
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

    def add_or_update_face(
        self,
        embedding: Any,
        username: str,
        source: str = "profile_pic",
        image_url: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """Append embeddings for a username, creating or updating as needed."""

        metadata_payload = dict(metadata) if metadata else {}
        metadata_payload.setdefault("operation", "add_or_update")

        try:
            summary = self.append_embeddings_to_username(
                username,
                [embedding],
                source,
                image_url=image_url,
                metadata=metadata_payload,
            )
            added = summary.get("added", 0)
            logger.info(
                "add_or_update_face stored %s embedding(s) for %s",
                added,
                username,
            )
            return added > 0
        except Exception as exc:
            logger.error(
                "add_or_update_face failed for %s: %s", username, exc, exc_info=True
            )
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
