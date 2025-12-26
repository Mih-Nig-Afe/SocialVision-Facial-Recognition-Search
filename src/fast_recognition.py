"""
Fast face recognition module optimized for live camera streaming.
Supports both ultra-fast mode and detailed mode with proper embedding matching.
"""

import numpy as np
import cv2
import time
import os
from typing import List, Tuple, Optional, Dict, Any
from src.logger import setup_logger

logger = setup_logger(__name__)

# Cache for face cascade (loaded once)
_face_cascade = None

# In-memory cache for embeddings (ultra-fast search)
_embedding_cache = None
_embedding_cache_time = 0
_CACHE_TTL_LOCAL = int(os.getenv("EMBEDDING_CACHE_TTL_LOCAL", "30"))
_CACHE_TTL_FIRESTORE = int(os.getenv("EMBEDDING_CACHE_TTL_FIRESTORE", "600"))

# DeepFace model cache for detailed mode
_deepface_model = None


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

        # Extract embedding using DeepFace (same as database stores)
        embedding = DeepFace.represent(
            face_rgb,
            model_name="VGG-Face",
            enforce_detection=False,
            detector_backend="skip",  # Already detected
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


def _get_cached_embeddings(database) -> Tuple[np.ndarray, List[Dict]]:
    """
    Get embeddings from cache or refresh if stale.
    Returns (embedding_matrix, face_records) for ultra-fast numpy search.
    Works with both local and Firestore databases.
    """
    global _embedding_cache, _embedding_cache_time

    current_time = time.time()
    ttl = _CACHE_TTL_LOCAL
    if hasattr(database, "use_firestore") and getattr(database, "use_firestore"):
        ttl = _CACHE_TTL_FIRESTORE

    if _embedding_cache is None or (current_time - _embedding_cache_time) > ttl:
        try:
            # Rebuild cache from database
            faces_list = []
            embeddings = []

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
                    emb = face.get("embedding")
                    if emb is not None:
                        try:
                            emb_array = np.array(emb, dtype=np.float32)
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
                _embedding_cache = (emb_matrix, faces_list)
                logger.info(
                    f"Loaded {len(faces_list)} faces into cache (dim={emb_matrix.shape[1]})"
                )
            else:
                _embedding_cache = (np.empty((0, 128), dtype=np.float32), [])
                logger.warning("No embeddings found in database for cache")

            _embedding_cache_time = current_time

        except Exception as e:
            logger.error(f"Failed to refresh embedding cache: {e}")
            # If we already have a cache, keep serving it and back off before retrying.
            if _embedding_cache is not None:
                _embedding_cache_time = current_time
            else:
                _embedding_cache = (np.empty((0, 128), dtype=np.float32), [])
                _embedding_cache_time = current_time

    return _embedding_cache


def ultra_fast_search(
    embedding: np.ndarray, database, threshold: float = 0.6, limit: int = 1
) -> List[Dict[str, Any]]:
    """
    Ultra-fast search using cached numpy operations.
    ~100x faster than database search for real-time recognition.
    """
    try:
        emb_matrix, faces = _get_cached_embeddings(database)

        if emb_matrix.shape[0] == 0:
            return []

        # Normalize query embedding
        query = np.array(embedding, dtype=np.float32).flatten()
        query_norm = np.linalg.norm(query)
        if query_norm > 0:
            query = query / query_norm

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
                if results:
                    labels.append(results[0].get("username"))
                    scores.append(results[0].get("similarity", 0))
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
                    labels.append(results[0].get("username"))
                    scores.append(results[0].get("similarity_score", 0))
                else:
                    labels.append(None)
                    scores.append(0)
            else:
                labels.append(None)
                scores.append(0)

    processing_time_ms = (time.time() - start_time) * 1000
    return faces, labels, scores, processing_time_ms
