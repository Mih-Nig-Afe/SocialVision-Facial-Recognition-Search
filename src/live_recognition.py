"""
Real-time live camera face recognition with overlay.
Uses streamlit-webrtc for continuous video streaming.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Type
from dataclasses import dataclass
import threading
import time

try:
    from streamlit_webrtc import VideoProcessorBase
    import av

    HAS_WEBRTC = True
except ImportError:
    VideoProcessorBase = object  # Fallback for type hints
    av = None
    HAS_WEBRTC = False

from src.logger import setup_logger
from src.config import get_config

logger = setup_logger(__name__)
config = get_config()

# Recognition cache to avoid processing every frame
RECOGNITION_INTERVAL = 0.5  # Recognize faces every 0.5 seconds
CACHE_TTL = 2.0  # Cache results for 2 seconds


@dataclass
class FaceResult:
    """Stores face detection and recognition results."""

    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    username: Optional[str] = None
    similarity: float = 0.0
    timestamp: float = 0.0


class LiveRecognitionProcessor:
    """Processes video frames for real-time face recognition."""

    def __init__(self, search_engine, face_engine, threshold: float = 0.6):
        self.search_engine = search_engine
        self.face_engine = face_engine
        self.threshold = threshold
        self._results_cache: List[FaceResult] = []
        self._last_recognition_time = 0.0
        self._lock = threading.Lock()
        self._frame_count = 0
        logger.info("LiveRecognitionProcessor initialized")

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame: detect faces, recognize, and draw overlays."""
        if frame is None:
            return frame

        self._frame_count += 1
        current_time = time.time()

        # Run recognition periodically (not every frame for performance)
        if current_time - self._last_recognition_time >= RECOGNITION_INTERVAL:
            self._run_recognition(frame, current_time)
            self._last_recognition_time = current_time

        # Draw overlays from cached results
        annotated_frame = self._draw_overlays(frame.copy())
        return annotated_frame

    def _run_recognition(self, frame: np.ndarray, current_time: float) -> None:
        """Run face detection and recognition on the frame."""
        try:
            # Detect faces (no upscaling for speed)
            face_locations = self.face_engine.detect_faces(frame)

            if not face_locations:
                with self._lock:
                    self._results_cache = []
                return

            # Extract embeddings
            embeddings = self.face_engine.extract_face_embeddings(frame, face_locations)

            new_results: List[FaceResult] = []

            for i, location in enumerate(face_locations):
                # Convert face_recognition format (top, right, bottom, left) to (x, y, w, h)
                top, right, bottom, left = location
                bbox = (left, top, right - left, bottom - top)

                username = None
                similarity = 0.0

                if i < len(embeddings) and embeddings[i]:
                    # Search database for matches (no upscaling for speed)
                    matches = self.search_engine.search_by_embedding(
                        embeddings[i], threshold=self.threshold, top_k=1
                    )
                    if matches:
                        username = matches[0].get("username")
                        similarity = matches[0].get("similarity_score", 0.0)

                new_results.append(
                    FaceResult(
                        bbox=bbox,
                        username=username,
                        similarity=similarity,
                        timestamp=current_time,
                    )
                )

            with self._lock:
                self._results_cache = new_results

        except Exception as e:
            logger.error(f"Recognition error: {e}")

    def _draw_overlays(self, frame: np.ndarray) -> np.ndarray:
        """Draw bounding boxes and labels on the frame."""
        with self._lock:
            results = self._results_cache.copy()

        for result in results:
            x, y, w, h = result.bbox

            # Choose color based on recognition status
            if result.username:
                color = (0, 255, 0)  # Green for recognized
                label = f"@{result.username} ({result.similarity:.0%})"
            else:
                color = (0, 165, 255)  # Orange for unknown
                label = "Unknown"

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Draw label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, y - 25), (x + label_size[0] + 10, y), color, -1)

            # Draw label text
            cv2.putText(
                frame,
                label,
                (x + 5, y - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        return frame

    def update_threshold(self, threshold: float) -> None:
        """Update the similarity threshold."""
        self.threshold = threshold


def create_video_processor_factory(search_engine, face_engine, threshold: float):
    """Create a factory function that returns a VideoProcessor class.

    This is needed because streamlit-webrtc requires a class factory,
    and we need to pass the engines into the processor.
    """

    class FaceRecognitionVideoProcessor(VideoProcessorBase):
        """Video processor for real-time face recognition."""

        def __init__(self):
            self._processor = LiveRecognitionProcessor(
                search_engine=search_engine,
                face_engine=face_engine,
                threshold=threshold,
            )
            logger.info("FaceRecognitionVideoProcessor initialized")

        def recv(self, frame):
            """Process each video frame."""
            try:
                # Convert frame to numpy array
                img = frame.to_ndarray(format="bgr24")

                # Process frame with face recognition overlays
                processed = self._processor.process_frame(img)

                # Return processed frame
                return av.VideoFrame.from_ndarray(processed, format="bgr24")
            except Exception as e:
                logger.error(f"Error processing video frame: {e}")
                # Return original frame on error
                return frame

    return FaceRecognitionVideoProcessor
