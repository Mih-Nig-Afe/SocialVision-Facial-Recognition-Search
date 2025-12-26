"""
Fast face recognition module optimized for live camera streaming.
Supports both ultra-fast mode and detailed mode with proper embedding matching.
"""

import numpy as np
import cv2
import time
import os
import threading
import weakref
from typing import List, Tuple, Optional, Dict, Any
from src.logger import setup_logger

logger = setup_logger(__name__)

# Cache for face cascade (loaded once)
_face_cascade = None

# In-memory cache for embeddings (ultra-fast search)
_embedding_cache: Dict[str, Tuple[np.ndarray, List[Dict[str, Any]]]] = {}
_embedding_cache_time: Dict[str, float] = {}
_CACHE_TTL_LOCAL = int(os.getenv("EMBEDDING_CACHE_TTL_LOCAL", "30"))
_CACHE_TTL_FIRESTORE = int(os.getenv("EMBEDDING_CACHE_TTL_FIRESTORE", "600"))

# Ultra-fast mode is designed around dlib/face_recognition 128-d encodings.
_ULTRA_FAST_EMBEDDING_KEY = (
    os.getenv("ULTRA_FAST_EMBEDDING_KEY", "dlib").strip() or "dlib"
)

# DeepFace model cache for detailed mode
_deepface_model = None

# Live auto-enrichment (append embeddings when we confidently recognize a face).
_LIVE_AUTO_ENRICH_ENABLED = os.getenv("LIVE_AUTO_ENRICH_ENABLED", "1").strip() not in (
    "0",
    "false",
    "False",
    "no",
    "NO",
)
_LIVE_AUTO_ENRICH_COOLDOWN_SECONDS = float(
    os.getenv("LIVE_AUTO_ENRICH_COOLDOWN_SECONDS", "0").strip() or "0"
)
_last_live_enrich_ts: Dict[str, float] = {}

# Batch enrichment to keep live recognition responsive, especially when using
# Firebase Realtime DB (full-database PUT on each write) or when DeepFace is heavy.
_LIVE_AUTO_ENRICH_FLUSH_SECONDS = float(
    os.getenv("LIVE_AUTO_ENRICH_FLUSH_SECONDS", "2").strip() or "2"
)
_LIVE_AUTO_ENRICH_MAX_PENDING = int(
    os.getenv("LIVE_AUTO_ENRICH_MAX_PENDING", "25").strip() or "25"
)
_pending_live_enrich: Dict[str, List[Dict[str, Any]]] = {}
_last_live_enrich_flush_ts: float = 0.0

_live_enrich_lock = threading.Lock()
_live_enrich_event = threading.Event()
_live_enrich_worker_started = False
_live_enrich_db_ref: Optional[weakref.ReferenceType] = None


def _serialize_embedding_bundle_for_db(
    bundle: Dict[str, Any],
) -> Optional[Dict[str, List[float]]]:
    if not bundle:
        return None
    serialized: Dict[str, List[float]] = {}
    for key, value in bundle.items():
        if value is None:
            continue
        if hasattr(value, "tolist"):
            vec = value.tolist()
        elif isinstance(value, (list, tuple)):
            vec = list(value)
        else:
            continue
        if vec:
            serialized[key] = vec
    return serialized or None


def _extract_deepface_embedding_only(
    image: np.ndarray, face_location: Tuple[int, int, int, int]
) -> Optional[Dict[str, np.ndarray]]:
    """Extract a DeepFace embedding for enrichment purposes.

    Important: this must stay consistent with the DB's deepface embedding space.
    No dlib fallback here (mixing 128-d/512-d primaries can break profile centroids).
    """

    try:
        from deepface import DeepFace
        from src.face_recognition_engine import DEEPFACE_MODEL_NAME

        top, right, bottom, left = face_location

        h, w = image.shape[:2]
        pad = int((bottom - top) * 0.2)
        y1 = max(0, top - pad)
        y2 = min(h, bottom + pad)
        x1 = max(0, left - pad)
        x2 = min(w, right + pad)
        face_img = image[y1:y2, x1:x2]
        if face_img.size == 0:
            return None

        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        embedding = DeepFace.represent(
            face_rgb,
            model_name=DEEPFACE_MODEL_NAME,
            enforce_detection=False,
            detector_backend="skip",
            align=True,
        )
        if embedding and len(embedding) > 0 and "embedding" in embedding[0]:
            vec = np.array(embedding[0]["embedding"], dtype=np.float32)
            if vec.size:
                return {"deepface": vec}
        return None
    except Exception as exc:
        logger.debug("DeepFace enrichment embedding failed: %s", exc)
        return None


def _extract_deepface_embedding_from_face_crop(
    face_bgr: np.ndarray,
) -> Optional[Dict[str, np.ndarray]]:
    """Extract a DeepFace embedding from an already-cropped face chip (BGR).

    This is used to keep fast-mode live recognition responsive: we do the cheap
    dlib match in the live loop, enqueue a small crop, then compute the deepface
    embedding during batch flush.
    """

    try:
        from deepface import DeepFace
        from src.face_recognition_engine import DEEPFACE_MODEL_NAME

        if face_bgr is None or not hasattr(face_bgr, "shape"):
            return None
        if face_bgr.size == 0:
            return None

        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        embedding = DeepFace.represent(
            face_rgb,
            model_name=DEEPFACE_MODEL_NAME,
            enforce_detection=False,
            detector_backend="skip",
            align=True,
        )
        if embedding and len(embedding) > 0 and "embedding" in embedding[0]:
            vec = np.array(embedding[0]["embedding"], dtype=np.float32)
            if vec.size:
                return {"deepface": vec}
        return None
    except Exception as exc:
        logger.debug("DeepFace crop enrichment failed: %s", exc)
        return None


def _maybe_live_auto_enrich(
    *,
    database: Any,
    username: Optional[str],
    embedding_bundle: Optional[Dict[str, Any]],
    similarity: float,
    fast_mode: bool,
    threshold: float,
) -> None:
    """Best-effort enrichment for live camera recognition.

    We append in batches. If we only have a dlib (128-d) vector (fast mode),
    the batch flush will try to attach a DeepFace vector from a queued face crop
    so primary embedding dimensions stay consistent.
    """

    if not _LIVE_AUTO_ENRICH_ENABLED:
        return
    if not username or not embedding_bundle:
        return
    if similarity < float(threshold):
        return

    now = time.time()
    key = str(username)
    last = _last_live_enrich_ts.get(key, 0.0)
    cooldown = max(0.0, float(_LIVE_AUTO_ENRICH_COOLDOWN_SECONDS))
    if cooldown and (now - last) < cooldown:
        return
    payload = _serialize_embedding_bundle_for_db(embedding_bundle)
    if not payload:
        return

    _ensure_live_enrich_worker(database)

    # Enqueue; flushing happens asynchronously to keep recognition fast.
    with _live_enrich_lock:
        bucket = _pending_live_enrich.setdefault(key, [])
        bucket.append(
            {
                "bundle": payload,
                "trigger_similarity": float(similarity),
                "fast_mode": bool(fast_mode),
                "face_crop": embedding_bundle.get("_face_crop"),
            }
        )
        _last_live_enrich_ts[key] = now

    _live_enrich_event.set()


def _ensure_live_enrich_worker(database: Any) -> None:
    global _live_enrich_worker_started, _live_enrich_db_ref

    try:
        _live_enrich_db_ref = weakref.ref(database)
    except Exception:
        _live_enrich_db_ref = None

    if _live_enrich_worker_started:
        return

    _live_enrich_worker_started = True
    thread = threading.Thread(
        target=_live_enrich_worker,
        name="live-auto-enrich-worker",
        daemon=True,
    )
    thread.start()


def _live_enrich_worker() -> None:
    """Background worker that periodically flushes queued enrichments."""

    global _last_live_enrich_flush_ts, _pending_live_enrich

    while True:
        flush_after = max(0.05, float(_LIVE_AUTO_ENRICH_FLUSH_SECONDS))
        _live_enrich_event.wait(timeout=flush_after)
        _live_enrich_event.clear()

        db = None
        if _live_enrich_db_ref is not None:
            try:
                db = _live_enrich_db_ref()
            except Exception:
                db = None
        if db is None:
            time.sleep(0.1)
            continue

        now = time.time()
        with _live_enrich_lock:
            total_pending = sum(len(v) for v in _pending_live_enrich.values())
            if total_pending == 0:
                continue

            # Flush when we have enough backlog, or enough time elapsed.
            max_pending = max(1, int(_LIVE_AUTO_ENRICH_MAX_PENDING))
            if (
                total_pending < max_pending
                and (now - _last_live_enrich_flush_ts) < flush_after
            ):
                continue

            pending = _pending_live_enrich
            _pending_live_enrich = {}
            _last_live_enrich_flush_ts = now

        # Flush outside the lock.
        _flush_live_auto_enrich(db, pending)


def _flush_live_auto_enrich(
    database: Any, pending: Dict[str, List[Dict[str, Any]]]
) -> None:
    """Flush queued live enrichments in batches.

    In fast mode we may only have dlib vectors. To keep DB profile dimensions
    consistent, we attach the user's existing DEFAULT_EMBEDDING_SOURCE profile
    vector (usually deepface) before appending.
    """

    append_fn = getattr(database, "append_embeddings_to_username", None)
    if append_fn is None:
        return

    from src.config import get_config

    default_key = getattr(get_config(), "DEFAULT_EMBEDDING_SOURCE", "deepface")
    now = time.time()

    for username, items in pending.items():
        if not items:
            continue

        bundles_to_append: List[Dict[str, List[float]]] = []
        trigger_max = 0.0
        any_fast = False

        profile_vec: Optional[np.ndarray] = None

        for item in items:
            bundle = item.get("bundle")
            if not isinstance(bundle, dict) or not bundle:
                continue

            trigger = float(item.get("trigger_similarity", 0.0) or 0.0)
            if trigger > trigger_max:
                trigger_max = trigger
            any_fast = any_fast or bool(item.get("fast_mode"))

            if default_key in bundle:
                bundles_to_append.append(bundle)
                continue

            # Prefer attaching the user's current profile vector (cheap).
            if profile_vec is None or profile_vec.size == 0:
                get_profile = getattr(
                    database, "get_profile_embedding_for_username", None
                )
                if callable(get_profile):
                    try:
                        profile_vec = get_profile(username)
                    except Exception:
                        profile_vec = None

            if profile_vec is not None and getattr(profile_vec, "size", 0) > 0:
                adjusted = dict(bundle)
                adjusted[default_key] = profile_vec.astype(np.float32).tolist()
                bundles_to_append.append(adjusted)
                continue

            # If no profile exists yet, try generating a deepface embedding from the crop once.
            face_crop = item.get("face_crop")
            if isinstance(face_crop, np.ndarray) and face_crop.size:
                deepface_bundle = _extract_deepface_embedding_from_face_crop(face_crop)
                serialized_deepface = (
                    _serialize_embedding_bundle_for_db(deepface_bundle)
                    if deepface_bundle
                    else None
                )
                if serialized_deepface and default_key in serialized_deepface:
                    adjusted = dict(bundle)
                    adjusted.update(serialized_deepface)
                    bundles_to_append.append(adjusted)
                    continue

            # Last resort: attach the user's current profile vector.
            if profile_vec is None or profile_vec.size == 0:
                get_profile = getattr(
                    database, "get_profile_embedding_for_username", None
                )
                if callable(get_profile):
                    try:
                        profile_vec = get_profile(username)
                    except Exception:
                        profile_vec = None

            if profile_vec is None or profile_vec.size == 0:
                continue
            adjusted = dict(bundle)
            adjusted[default_key] = profile_vec.astype(np.float32).tolist()
            bundles_to_append.append(adjusted)

        if not bundles_to_append:
            continue

        metadata = {
            "origin": "live_camera",
            "fast_mode": bool(any_fast),
            "trigger_similarity": trigger_max,
            "batch_size": len(bundles_to_append),
        }
        try:
            append_fn(
                username,
                bundles_to_append,
                source="live_auto_enrich",
                metadata=metadata,
            )
        except Exception as exc:
            logger.debug("Live auto-enrich flush failed for %s: %s", username, exc)


def get_face_cascade():
    """Get or create cached Haar cascade for fast detection."""
    global _face_cascade
    if _face_cascade is None:
        try:
            _face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
        except Exception as e:
            logger.error(f"Failed to load Haar cascade: {e}")
    return _face_cascade


def fast_detect_faces(
    image: np.ndarray, min_size: int = 80, scale_factor: float = 1.3
) -> List[Tuple[int, int, int, int]]:
    """
    Fast face detection using OpenCV Haar cascade.
    Optimized for speed over accuracy.

    Returns: List of (top, right, bottom, left) tuples
    """
    cascade = get_face_cascade()
    if cascade is None or cascade.empty():
        return []

    # Resize for faster processing
    max_dim = 400
    h, w = image.shape[:2]
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image = cv2.resize(image, None, fx=scale, fy=scale)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=3,
        minSize=(min_size, min_size),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    # Convert to (top, right, bottom, left) and scale back
    result = []
    for x, y, w, h in faces:
        if scale != 1.0:
            x = int(x / scale)
            y = int(y / scale)
            w = int(w / scale)
            h = int(h / scale)
        result.append((y, x + w, y + h, x))

    return result


def fast_extract_embedding(
    image: np.ndarray, face_location: Tuple[int, int, int, int]
) -> Optional[np.ndarray]:
    """
    Fast embedding extraction using face_recognition (dlib) with large model.
    Uses full model for better accuracy while maintaining speed.
    """
    try:
        import face_recognition

        top, right, bottom, left = face_location

        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Extract single encoding with large model for better matching
        encodings = face_recognition.face_encodings(
            rgb_image,
            known_face_locations=[face_location],
            num_jitters=1,  # Slight augmentation for better accuracy
            model="large",  # Full model for better matching
        )

        return encodings[0] if encodings else None
    except Exception as e:
        logger.debug(f"Fast embedding extraction failed: {e}")
        return None


def detailed_extract_embedding(
    image: np.ndarray, face_location: Tuple[int, int, int, int]
) -> Optional[Dict[str, np.ndarray]]:
    """
    Detailed embedding extraction using DeepFace (same as database).
    Returns embedding bundle compatible with database search.
    """
    global _deepface_model

    try:
        from deepface import DeepFace

        # Keep the live "detailed" embedding compatible with what we store in the DB.
        # FaceRecognitionEngine uses DEEPFACE_MODEL_NAME (default Facenet512).
        from src.face_recognition_engine import DEEPFACE_MODEL_NAME

        top, right, bottom, left = face_location

        # Crop face region with padding
        h, w = image.shape[:2]
        pad = int((bottom - top) * 0.2)  # 20% padding
        y1 = max(0, top - pad)
        y2 = min(h, bottom + pad)
        x1 = max(0, left - pad)
        x2 = min(w, right + pad)
        face_img = image[y1:y2, x1:x2]

        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        # Extract embedding using the same DeepFace model used by FaceRecognitionEngine.
        embedding = DeepFace.represent(
            face_rgb,
            model_name=DEEPFACE_MODEL_NAME,
            enforce_detection=False,
            detector_backend="skip",  # Already detected
            align=True,
        )

        if embedding and len(embedding) > 0:
            emb_vector = np.array(embedding[0]["embedding"], dtype=np.float32)
            return {"deepface": emb_vector}

        return None
    except Exception as e:
        logger.debug(f"Detailed embedding extraction failed: {e}")
        # Fallback to dlib
        emb = fast_extract_embedding(image, face_location)
        if emb is not None:
            return {"dlib": emb}
        return None


def fast_search_face(
    embedding, database, threshold: float = 0.6, limit: int = 1
) -> List[Dict[str, Any]]:
    """
    Fast database search for face embedding.
    Handles both numpy arrays and embedding bundles (dict).
    Uses search_similar_faces for proper matching.
    """
    try:
        # Handle embedding bundle (from detailed_extract_embedding)
        if isinstance(embedding, dict):
            query_embedding = embedding
        elif isinstance(embedding, np.ndarray):
            # Single numpy array - wrap in bundle with dlib source
            query_embedding = {"dlib": embedding.tolist()}
        else:
            query_embedding = list(embedding)

        # Use the correct database method
        results = database.search_similar_faces(
            query_embedding, threshold=threshold, top_k=limit
        )
        return results
    except Exception as e:
        logger.debug(f"Fast search failed: {e}")
        return []


def _get_cached_embeddings(
    database,
    *,
    embedding_key: str = _ULTRA_FAST_EMBEDDING_KEY,
    default_dim: int = 128,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Get embeddings from cache or refresh if stale.
    Returns (embedding_matrix, face_records) for ultra-fast numpy search.
    Works with both local and Firestore databases.
    """
    global _embedding_cache, _embedding_cache_time

    embedding_key = (embedding_key or _ULTRA_FAST_EMBEDDING_KEY).strip() or "dlib"

    current_time = time.time()
    ttl = _CACHE_TTL_LOCAL
    if hasattr(database, "use_firestore") and getattr(database, "use_firestore"):
        ttl = _CACHE_TTL_FIRESTORE
    if hasattr(database, "use_realtime") and getattr(database, "use_realtime"):
        ttl = _CACHE_TTL_FIRESTORE

    backend = "local"
    if hasattr(database, "use_firestore") and getattr(database, "use_firestore"):
        backend = "firestore"
    elif hasattr(database, "use_realtime") and getattr(database, "use_realtime"):
        backend = "realtime"

    cache_key = f"{backend}:{embedding_key}"

    last_refresh = _embedding_cache_time.get(cache_key, 0.0)
    if cache_key not in _embedding_cache or (current_time - last_refresh) > ttl:
        try:
            # Rebuild cache from database
            faces_list = []
            embeddings = []

            target_dim: int = int(default_dim)
            discovered_dim: Optional[int] = None

            # Get all faces - works with Firestore or local
            faces_iter = None
            if hasattr(database, "use_firestore") and database.use_firestore:
                # Firestore mode - use the iterator
                if hasattr(database, "_firestore_faces_iter"):
                    faces_iter = database._firestore_faces_iter()
            elif hasattr(database, "data") and "faces" in database.data:
                # Local mode
                faces_iter = database.data.get("faces", [])

            if faces_iter is not None:
                for face in faces_iter:
                    try:
                        stored_bundle = face.get("embeddings") or {}
                        emb = stored_bundle.get(embedding_key)

                        # Back-compat: some records may only have primary embedding.
                        # Only use it if it matches the expected dimension.
                        if emb is None:
                            primary = face.get("embedding")
                            if (
                                isinstance(primary, (list, tuple))
                                and len(primary) == target_dim
                            ):
                                emb = primary

                        if emb is None:
                            continue

                        emb_array = np.array(emb, dtype=np.float32).reshape(-1)
                        if emb_array.size == 0:
                            continue

                        if discovered_dim is None:
                            discovered_dim = int(emb_array.shape[0])
                        # Enforce consistent dimensionality inside the cache.
                        if int(emb_array.shape[0]) != int(discovered_dim):
                            continue

                        embeddings.append(emb_array)
                        faces_list.append(
                            {
                                "username": face.get("username", "Unknown"),
                                "id": face.get("id"),
                                "source": face.get("source", ""),
                            }
                        )
                    except Exception:
                        continue

            if embeddings:
                emb_matrix = np.vstack(embeddings)
                # Normalize for cosine similarity
                norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
                emb_matrix = emb_matrix / np.clip(norms, 1e-10, None)
                _embedding_cache[cache_key] = (emb_matrix, faces_list)
                logger.info(
                    "Loaded %s faces into cache (key=%s, dim=%s)",
                    len(faces_list),
                    embedding_key,
                    emb_matrix.shape[1],
                )
            else:
                _embedding_cache[cache_key] = (
                    np.empty((0, int(default_dim)), dtype=np.float32),
                    [],
                )
                logger.warning(
                    "No usable embeddings found for cache (key=%s, expected_dim=%s)",
                    embedding_key,
                    int(default_dim),
                )

            _embedding_cache_time[cache_key] = current_time

        except Exception as e:
            logger.error(f"Failed to refresh embedding cache: {e}")
            # If we already have a cache, keep serving it and back off before retrying.
            if cache_key in _embedding_cache:
                _embedding_cache_time[cache_key] = current_time
            else:
                _embedding_cache[cache_key] = (
                    np.empty((0, int(default_dim)), dtype=np.float32),
                    [],
                )
                _embedding_cache_time[cache_key] = current_time

    return _embedding_cache[cache_key]


def ultra_fast_search(
    embedding: np.ndarray, database, threshold: float = 0.6, limit: int = 1
) -> List[Dict[str, Any]]:
    """
    Ultra-fast search using cached numpy operations.
    ~100x faster than database search for real-time recognition.
    """
    try:
        emb_matrix, faces = _get_cached_embeddings(
            database,
            embedding_key=_ULTRA_FAST_EMBEDDING_KEY,
            default_dim=128,
        )

        if emb_matrix.shape[0] == 0:
            return []

        # Normalize query embedding
        query = np.array(embedding, dtype=np.float32).flatten()
        query_norm = np.linalg.norm(query)
        if query_norm > 0:
            query = query / query_norm

        # Guard against unexpected dimension mismatches (e.g., stale cache).
        if emb_matrix.shape[1] != query.shape[0]:
            logger.warning(
                "Ultra-fast cache dim mismatch (cache=%s query=%s). Clearing cache.",
                emb_matrix.shape[1],
                query.shape[0],
            )
            # Clear only this cache key so we rebuild on next call.
            backend = "local"
            if hasattr(database, "use_firestore") and getattr(
                database, "use_firestore"
            ):
                backend = "firestore"
            elif hasattr(database, "use_realtime") and getattr(
                database, "use_realtime"
            ):
                backend = "realtime"
            cache_key = f"{backend}:{_ULTRA_FAST_EMBEDDING_KEY}"
            _embedding_cache.pop(cache_key, None)
            _embedding_cache_time.pop(cache_key, None)
            return []

        # Compute all similarities at once (vectorized)
        similarities = np.dot(emb_matrix, query)

        # Find top matches above threshold
        mask = similarities >= threshold
        if not np.any(mask):
            return []

        # Get indices and scores
        indices = np.where(mask)[0]
        scores = similarities[indices]

        # Sort by score descending
        sorted_order = np.argsort(scores)[::-1][:limit]

        results = []
        for idx in sorted_order:
            face_idx = indices[idx]
            face = faces[face_idx]
            results.append(
                {
                    "username": face["username"],
                    "similarity": float(scores[idx]),
                    "id": face.get("id"),
                    "source": face.get("source", ""),
                }
            )

        return results

    except Exception as e:
        logger.debug(f"Ultra fast search failed: {e}")
        return []


def draw_face_boxes(
    image: np.ndarray,
    faces: List[Tuple[int, int, int, int]],
    labels: List[Optional[str]] = None,
    scores: List[float] = None,
) -> np.ndarray:
    """
    Draw bounding boxes on detected faces with labels.
    Green = recognized, Orange = unknown.
    """
    result = image.copy()

    if labels is None:
        labels = [None] * len(faces)
    if scores is None:
        scores = [0.0] * len(faces)

    for i, (top, right, bottom, left) in enumerate(faces):
        label = labels[i] if i < len(labels) else None
        score = scores[i] if i < len(scores) else 0.0

        if label:
            color = (0, 255, 0)  # Green for recognized
            text = f"{label} ({score:.0%})"
        else:
            color = (0, 165, 255)  # Orange for unknown
            text = "Unknown"

        # Draw box
        cv2.rectangle(result, (left, top), (right, bottom), color, 2)

        # Draw label background
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(
            result,
            (left, bottom),
            (left + text_size[0] + 4, bottom + text_size[1] + 8),
            color,
            -1,
        )

        # Draw text
        cv2.putText(
            result,
            text,
            (left + 2, bottom + text_size[1] + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    return result


def recognize_faces(
    image: np.ndarray, database, threshold: float = 0.6, fast_mode: bool = False
) -> Tuple[List[Tuple[int, int, int, int]], List[Optional[str]], List[float], float]:
    """
    Complete face recognition pipeline.

    Args:
        image: BGR image
        database: Face database instance
        threshold: Similarity threshold for matching
        fast_mode: If True, use ultra-fast detection and cached search
                   If False, use detailed DeepFace embeddings

    Returns:
        (face_locations, labels, scores, processing_time_ms)
    """
    start_time = time.time()

    # Detect faces
    if fast_mode:
        faces = fast_detect_faces(image, min_size=60, scale_factor=1.2)
    else:
        faces = fast_detect_faces(image, min_size=40, scale_factor=1.1)

    if not faces:
        return [], [], [], (time.time() - start_time) * 1000

    labels = []
    scores = []

    for face_loc in faces:
        if fast_mode:
            # Ultra-fast: dlib embedding + cached numpy search
            embedding = fast_extract_embedding(image, face_loc)
            if embedding is not None:
                results = ultra_fast_search(
                    embedding, database, threshold=threshold, limit=1
                )
                if not results:
                    # Fallback: use database search (still 128-d dlib query), avoids hard failure.
                    results = fast_search_face(
                        embedding, database, threshold=threshold, limit=1
                    )
                if results:
                    username = results[0].get("username")
                    score = float(
                        results[0].get(
                            "similarity_score", results[0].get("similarity", 0.0)
                        )
                        or 0.0
                    )
                    labels.append(username)
                    scores.append(score)

                    # Auto-enrich on every match while keeping fast mode fast.
                    # Queue the dlib embedding; flush will attach the user's existing
                    # deepface embedding from a queued face crop (or fall back to profile).
                    top, right, bottom, left = face_loc
                    h, w = image.shape[:2]
                    pad = int((bottom - top) * 0.2)
                    y1 = max(0, top - pad)
                    y2 = min(h, bottom + pad)
                    x1 = max(0, left - pad)
                    x2 = min(w, right + pad)
                    face_crop = image[y1:y2, x1:x2]
                    if face_crop is not None and getattr(face_crop, "size", 0) > 0:
                        # Keep crop small to limit memory/cpu on flush.
                        max_dim = 160
                        ch, cw = face_crop.shape[:2]
                        scale = min(1.0, float(max_dim) / float(max(ch, cw, 1)))
                        if scale < 1.0:
                            face_crop = cv2.resize(face_crop, None, fx=scale, fy=scale)
                    _maybe_live_auto_enrich(
                        database=database,
                        username=username,
                        embedding_bundle={"dlib": embedding, "_face_crop": face_crop},
                        similarity=score,
                        fast_mode=fast_mode,
                        threshold=threshold,
                    )
                else:
                    labels.append(None)
                    scores.append(0)
            else:
                labels.append(None)
                scores.append(0)
        else:
            # Detailed: DeepFace embedding + full database search
            embedding = detailed_extract_embedding(image, face_loc)
            if embedding is not None:
                results = fast_search_face(
                    embedding, database, threshold=threshold, limit=1
                )
                if results:
                    username = results[0].get("username")
                    score = float(results[0].get("similarity_score", 0) or 0.0)
                    labels.append(username)
                    scores.append(score)

                    _maybe_live_auto_enrich(
                        database=database,
                        username=username,
                        embedding_bundle=embedding,
                        similarity=score,
                        fast_mode=fast_mode,
                        threshold=threshold,
                    )
                else:
                    labels.append(None)
                    scores.append(0)
            else:
                labels.append(None)
                scores.append(0)

    processing_time_ms = (time.time() - start_time) * 1000
    return faces, labels, scores, processing_time_ms
