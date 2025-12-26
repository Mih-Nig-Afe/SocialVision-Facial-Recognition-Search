"""
SocialVision: Advanced Facial Recognition Search Engine
Main Streamlit Application
"""

from __future__ import annotations
import streamlit as st
from pathlib import Path
import numpy as np
from typing import Optional
from src.config import get_config
from src.logger import setup_logger

# Try to import cv2, but make it optional
try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Try to import streamlit-webrtc for live camera (works outside Docker)
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode
    import av

    HAS_WEBRTC = True
except ImportError:
    HAS_WEBRTC = False
    webrtc_streamer = None
    WebRtcMode = None
    av = None

# Configuration (lightweight)
config = get_config()
DEFAULT_EMBEDDING_SOURCE = getattr(config, "DEFAULT_EMBEDDING_SOURCE", "deepface")
logger = setup_logger(__name__)
UPLOAD_EXTENSIONS = sorted({ext.lower() for ext in config.ALLOWED_IMAGE_FORMATS})
VIDEO_EXTENSIONS = sorted({ext.lower() for ext in config.ALLOWED_VIDEO_FORMATS})
DEFAULT_FRAME_STRIDE = getattr(config, "VIDEO_FRAME_STRIDE", 5)
DEFAULT_MAX_FRAMES = getattr(config, "VIDEO_MAX_FRAMES", 90)


# Page configuration
st.set_page_config(
    page_title="SocialVision - Facial Recognition Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 1.1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


DISPLAY_FRAME = (520, 520)
FACE_TILE_FRAME = (240, 240)
BACKEND_LABELS = {
    "ibm_max": "IBM MAX SRGAN",
    "ncnn": "Real-ESRGAN NCNN",
    "realesrgan": "Real-ESRGAN (PyTorch)",
    "opencv": "OpenCV EDSR",
    "lanczos": "Lanczos Interpolation",
    "resize": "Bicubic Resize",
    "size_guard": "Size Guard (no-op)",
    "no_scale": "No Scaling Needed",
    "disabled": "Upscaling Disabled",
    "uninitialized": "Upscaler Initializing",
}


def _describe_backend(code: str) -> str:
    return BACKEND_LABELS.get(code, code or "unknown backend")


def _infer_temp_suffix(filename: str | None) -> str:
    suffix = Path(filename or "").suffix
    return suffix if suffix else ".jpg"


@st.cache_resource
def initialize_components():
    """Initialize and cache components - lazy loaded for faster startup"""
    # Import heavy modules only when needed
    from src.face_recognition_engine import FaceRecognitionEngine
    from src.database import FaceDatabase
    from src.search_engine import SearchEngine
    from src.face_quality import AutoFaceImprover, FaceQualityAssessor

    db = FaceDatabase()
    search_engine = SearchEngine(db)
    face_engine = FaceRecognitionEngine(database=db)
    auto_improver = AutoFaceImprover(face_engine, db)
    quality_assessor = FaceQualityAssessor()
    return db, search_engine, face_engine, auto_improver, quality_assessor


@st.cache_resource
def get_image_processor():
    """Lazy load image processor"""
    from src.image_utils import ImageProcessor

    return ImageProcessor


@st.cache_resource
def get_video_processor():
    """Lazy load video processor"""
    from src.image_utils import VideoProcessor

    return VideoProcessor


@st.cache_resource
def get_upscaler():
    """Lazy load image upscaler"""
    from src.image_upscaler import get_image_upscaler

    return get_image_upscaler()


def main():
    """Main application"""

    def _safe_rerun() -> None:
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()

    # Header
    st.title("🔍 SocialVision")
    st.markdown("### Advanced Facial Recognition Search Engine with Auto-Enrichment")
    st.markdown("---")

    # Initialize components
    db, search_engine, face_engine, auto_improver, quality_assessor = (
        initialize_components()
    )

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # Similarity threshold
        similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=config.FACE_MATCH_THRESHOLD,
            step=0.05,
            help="Lower values = more matches (less strict)",
        )

        # Top K results
        top_k = st.slider("Top K Results", min_value=5, max_value=100, value=50, step=5)

        # Auto-improvement settings
        st.markdown("---")
        st.subheader("🔧 Auto-Improvement")
        enable_auto_improve = st.checkbox(
            "Enable Auto Face Quality Improvement",
            value=True,
            help="Automatically enhance face quality before processing",
        )
        min_quality_score = st.slider(
            "Minimum Quality Score",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="Faces below this quality score will be rejected",
            disabled=not enable_auto_improve,
        )

        # Database info
        st.markdown("---")
        st.subheader("📊 Database Info")
        stats = db.get_statistics()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Faces", stats.get("total_faces", 0))
        with col2:
            st.metric("Unique Users", stats.get("unique_users", 0))

        if stats.get("sources"):
            st.write("**Sources:**")
            for source, count in stats["sources"].items():
                st.write(f"- {source}: {count}")

    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🔎 Search", "📤 Add Faces", "📈 Analytics"])

    # Tab 1: Search
    with tab1:
        st.header("Search for Similar Faces")
        input_mode = st.radio(
            "Input type",
            ["Image upload", "Video upload", "Live camera"],
            horizontal=True,
        )

        # Get lazy-loaded processors
        ImageProcessor = get_image_processor()

        def _load_image_from_upload(uploaded) -> np.ndarray | None:
            try:
                return ImageProcessor.load_image_from_bytes(uploaded.getbuffer())
            except Exception:
                return None

        def _run_image_search(raw_image: np.ndarray, label: str) -> None:
            processed_image = ImageProcessor.prepare_input_image(raw_image)
            backend_code = get_upscaler().last_backend
            framed_preview = ImageProcessor.frame_image_for_display(
                processed_image, frame_size=DISPLAY_FRAME
            )

            st.subheader(label)
            st.image(
                framed_preview,
                channels="BGR",
                use_column_width=False,
                caption=f"Enhanced preview via {_describe_backend(backend_code)}",
            )

            st.subheader("Search Results")
            try:
                results = search_engine.search_by_image(
                    processed_image,
                    threshold=similarity_threshold,
                    top_k=top_k,
                )
            except Exception as search_error:
                st.error(f"Error during search: {search_error}")
                logger.error(f"Search error: {search_error}", exc_info=True)
                results = {"faces": [], "total_matches": 0}

            if results["total_matches"] == 0:
                if results.get("faces") and len(results["faces"]) > 0:
                    st.info(
                        f"Detected {len(results['faces'])} face(s) but no matches found in database. Try adding faces to the database first."
                    )
                else:
                    st.warning("No matching faces found in database")
                return

            top_users = search_engine.get_top_usernames(results, top_k=10)
            st.success(
                f"Found {results['total_matches']} matches across {len(top_users)} users"
            )
            st.markdown("### 👥 Top Matching Accounts")

            for i, user_info in enumerate(top_users, 1):
                with st.expander(
                    f"{i}. @{user_info['username']} "
                    f"({user_info['match_count']} matches, "
                    f"{user_info['avg_similarity']:.2%} avg similarity)"
                ):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("Match Count", user_info["match_count"])
                    with c2:
                        st.metric(
                            "Avg Similarity", f"{user_info['avg_similarity']:.2%}"
                        )

                    st.write("**Sources:**")
                    for source in user_info["sources"]:
                        st.write(f"- {source}")

        if input_mode == "Image upload":
            col1, col2 = st.columns([2, 1])
            with col1:
                uploaded_file = st.file_uploader(
                    "Upload an image", type=UPLOAD_EXTENSIONS
                )
            with col2:
                search_button = st.button("🔍 Search", use_container_width=True)

            if uploaded_file and search_button:
                with st.spinner("Processing image..."):
                    raw_image = _load_image_from_upload(uploaded_file)
                    if raw_image is None:
                        st.error("Failed to load image")
                    else:
                        _run_image_search(raw_image, "Uploaded Image")

        elif input_mode == "Live camera":
            # Check if running in Docker (WebRTC won't work properly)
            import os

            in_docker = os.path.exists("/.dockerenv") or os.environ.get(
                "DOCKER_CONTAINER", False
            )
            webrtc_active = False

            if HAS_WEBRTC and not in_docker:
                # Real-time WebRTC mode (works outside Docker)
                st.info(
                    "📹 **Real-time face recognition** - Live video with face detection"
                )
                from src.live_recognition import create_video_processor_factory

                st.caption(
                    "Faces will be detected and matched in real-time. "
                    "Green = recognized, Orange = unknown."
                )

                try:
                    webrtc_ctx = webrtc_streamer(
                        key="live-face-recognition",
                        mode=WebRtcMode.SENDRECV,
                        video_processor_factory=create_video_processor_factory(
                            search_engine, face_engine, similarity_threshold
                        ),
                        media_stream_constraints={
                            "video": {"width": 640, "height": 480},
                            "audio": False,
                        },
                        async_processing=True,
                        rtc_configuration={
                            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                        },
                    )

                    if webrtc_ctx.state.playing:
                        st.success("Live recognition active")
                    else:
                        st.warning("Click START to begin")
                    webrtc_active = True
                except Exception as e:
                    st.warning(f"WebRTC unavailable: {e}. Using capture mode.")

            if not webrtc_active:
                # Continuous live video with auto face detection
                import time

                st.info("📹 **Live Camera Recognition** - Capture and detect faces")

                # Initialize session state for live mode
                if "live_running" not in st.session_state:
                    st.session_state.live_running = False
                if "frame_count" not in st.session_state:
                    st.session_state.frame_count = 0
                if "last_results" not in st.session_state:
                    st.session_state.last_results = []
                if "fast_mode" not in st.session_state:
                    st.session_state.fast_mode = (
                        True  # Default to fast mode for responsiveness
                    )
                if "auto_capture" not in st.session_state:
                    st.session_state.auto_capture = False

                # Controls in sidebar-style layout
                st.markdown("#### ⚙️ Recognition Settings")
                col1, col2 = st.columns(2)
                with col1:
                    st.session_state.fast_mode = st.checkbox(
                        "⚡ Fast mode (recommended)",
                        value=st.session_state.fast_mode,
                        help="Fast: ~100-200ms, uses dlib. Unchecked: ~500-1000ms, uses DeepFace",
                    )
                with col2:
                    st.session_state.auto_capture = st.checkbox(
                        "🔄 Auto-capture",
                        value=st.session_state.auto_capture,
                        help="Automatically process frames continuously",
                    )

                mode_text = (
                    "⚡ Fast (dlib)"
                    if st.session_state.fast_mode
                    else "🎯 Detailed (DeepFace)"
                )
                st.caption(
                    f"Mode: {mode_text} | 🟢 Green = Recognized | 🟠 Orange = Unknown"
                )

                st.markdown("---")

                # Camera capture - using a stable key
                camera_capture = st.camera_input(
                    "📷 Click to capture a frame for face detection",
                    key="live_camera_capture",
                )

                # Process captured frame
                if camera_capture:
                    from src.fast_recognition import recognize_faces, draw_face_boxes

                    raw_image = _load_image_from_upload(camera_capture)
                    if raw_image is not None:
                        st.session_state.frame_count += 1

                        with st.spinner(
                            f"Processing frame #{st.session_state.frame_count}..."
                        ):
                            # Use unified recognize_faces function
                            faces, labels, scores, proc_time = recognize_faces(
                                raw_image,
                                db,
                                threshold=similarity_threshold,
                                fast_mode=st.session_state.fast_mode,
                            )

                        # Display results
                        if faces:
                            result_image = draw_face_boxes(
                                raw_image, faces, labels, scores
                            )
                            st.image(
                                result_image,
                                channels="BGR",
                                caption=f"Frame #{st.session_state.frame_count} | {len(faces)} face(s) detected | {proc_time:.0f}ms",
                            )

                            # Show match details
                            st.markdown("### 🎯 Detection Results")
                            cols = st.columns(min(len(faces), 4))
                            for i, (label, score) in enumerate(zip(labels, scores)):
                                with cols[i % len(cols)]:
                                    if label:
                                        st.success(f"✅ **{label}**\n{score:.1%}")
                                    else:
                                        st.warning(f"❓ Unknown #{i+1}")

                            st.session_state.last_results = list(zip(labels, scores))
                        else:
                            st.image(
                                raw_image,
                                channels="BGR",
                                caption=f"Frame #{st.session_state.frame_count} - No faces detected ({proc_time:.0f}ms)",
                            )
                            st.info(
                                "No faces detected in this frame. Try adjusting your position or lighting."
                            )

                        # Auto-capture: automatically trigger new capture
                        if st.session_state.auto_capture:
                            interval = 0.5 if st.session_state.fast_mode else 1.0
                            time.sleep(interval)
                            _safe_rerun()
                else:
                    st.markdown(
                        """
                    ### 📸 How to use:
                    1. **Allow camera access** when prompted by your browser
                    2. **Click the camera button** above to capture a frame
                    3. Faces will be detected and matched against the database
                    4. Enable **Auto-capture** for continuous processing

                    💡 **Tip**: Use **Fast mode** for quicker results (~200ms per frame)
                    """
                    )

                    if st.session_state.last_results:
                        st.markdown("---")
                        st.markdown("### 📋 Last Detection Results:")
                        for label, score in st.session_state.last_results:
                            if label:
                                st.success(f"✅ **{label}** ({score:.1%})")

        else:  # Video upload
            st.write("Upload a short clip; frames are sampled for matches.")
            video_file = st.file_uploader(
                "Upload a video", type=VIDEO_EXTENSIONS, key="search_video"
            )
            frame_stride = st.slider(
                "Frame stride (every Nth frame)",
                min_value=1,
                max_value=15,
                value=DEFAULT_FRAME_STRIDE,
            )
            max_frames = st.slider(
                "Max frames to scan",
                min_value=5,
                max_value=180,
                value=min(DEFAULT_MAX_FRAMES, 90),
            )
            search_video_button = st.button("🔍 Search Video", use_container_width=True)

            if video_file and search_video_button:
                VideoProcessor = get_video_processor()
                with st.spinner("Sampling frames and searching..."):
                    suffix = _infer_temp_suffix(video_file.name)
                    frames = VideoProcessor.sample_frames_from_bytes(
                        video_file.getbuffer(),
                        suffix=suffix,
                        frame_stride=frame_stride,
                        max_frames=max_frames,
                    )

                    if not frames:
                        st.error("No frames could be read from the video")
                    else:
                        results = search_engine.search_video_frames(
                            frames,
                            threshold=similarity_threshold,
                            top_k=top_k,
                        )

                        st.metric(
                            "Frames processed", results.get("frames_processed", 0)
                        )
                        st.metric(
                            "Frames with faces",
                            results.get("frames_with_faces", 0),
                        )

                        if results.get("matches_by_user"):
                            st.success(
                                f"Found matches for {len(results['matches_by_user'])} user(s)"
                            )
                            preview = frames[0]
                            framed_preview = ImageProcessor.frame_image_for_display(
                                preview, frame_size=DISPLAY_FRAME
                            )
                            st.image(
                                framed_preview,
                                channels="BGR",
                                caption="First sampled frame",
                            )

                            st.markdown("### 👥 Top Matching Accounts (video)")
                            for i, user_info in enumerate(
                                results["matches_by_user"], 1
                            ):
                                with st.expander(
                                    f"{i}. @{user_info['username']} "
                                    f"({user_info['match_count']} matches, "
                                    f"avg sim: {user_info.get('avg_similarity') or 0:.2%})"
                                ):
                                    st.write(
                                        f"Frames: {user_info.get('frames', [])} | Sources: {user_info.get('sources', [])}"
                                    )
                                    max_sim = user_info.get("max_similarity")
                                    if max_sim is not None:
                                        st.metric("Best similarity", f"{max_sim:.2%}")
                        else:
                            st.warning("No matches detected in sampled frames")

    # Tab 2: Add Faces
    with tab2:
        st.header("Add Faces to Database")
        add_mode = st.radio(
            "Input type",
            ["Image upload", "Video upload", "Live camera"],
            horizontal=True,
            key="add_mode",
        )

        username = st.text_input(
            "Profile Identifier", help="Unique identifier for the person"
        )
        source = st.selectbox(
            "Source", ["profile_pic", "post", "story", "reel", "video", "camera"]
        )

        # Auto-improvement settings for adding faces
        enable_auto_improve_add = st.checkbox(
            "Enable Auto Face Quality Improvement",
            value=True,
            key="add_auto_improve",
            help="Automatically enhance face quality before adding to database",
        )
        min_quality_score_add = st.slider(
            "Minimum Quality Score",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="Faces below this quality score will be rejected",
            key="add_min_quality",
            disabled=not enable_auto_improve_add,
        )

        uploaded_file = None
        video_file = None
        camera_capture = None
        frame_stride = DEFAULT_FRAME_STRIDE
        max_frames = min(DEFAULT_MAX_FRAMES, 60)

        if add_mode == "Image upload":
            uploaded_file = st.file_uploader(
                "Upload image with faces", type=UPLOAD_EXTENSIONS, key="add_faces"
            )
        elif add_mode == "Live camera":
            st.info(
                "💡 Tip: Ensure good lighting and face the camera directly for best results"
            )
            st.caption(
                "If the browser doesn’t prompt: open the app at http://localhost:8501 (not 0.0.0.0), "
                "and ensure camera permission is allowed in your browser site settings."
            )

            if "add_cam_version" not in st.session_state:
                st.session_state.add_cam_version = 0

            if st.button("Request camera access", key="add_request_camera"):
                st.session_state.add_cam_version += 1
                _safe_rerun()

            camera_capture = st.camera_input(
                "Capture from your camera",
                key=f"add_camera_{st.session_state.add_cam_version}",
                help="Click the camera button to capture an image",
            )
        else:
            video_file = st.file_uploader(
                "Upload video with faces", type=VIDEO_EXTENSIONS, key="add_video"
            )
            frame_stride = st.slider(
                "Frame stride (every Nth frame)",
                min_value=1,
                max_value=15,
                value=DEFAULT_FRAME_STRIDE,
                key="add_frame_stride",
            )
            max_frames = st.slider(
                "Max frames to scan",
                min_value=5,
                max_value=180,
                value=min(DEFAULT_MAX_FRAMES, 60),
                key="add_max_frames",
            )

        def _load_image(uploaded) -> np.ndarray | None:
            try:
                return ImageProcessor.load_image_from_bytes(uploaded.getbuffer())
            except Exception:
                return None

        add_button = st.button("➕ Add / Update Face", use_container_width=True)

        if add_button:
            if not username:
                st.error("Please enter a username")
            else:

                if add_mode == "Image upload":
                    if not uploaded_file:
                        st.error("Please upload an image")
                    else:
                        with st.spinner("Processing image..."):
                            raw_image = _load_image(uploaded_file)
                            if raw_image is None:
                                st.error("Failed to load image")
                            else:
                                processed_image = ImageProcessor.prepare_input_image(
                                    raw_image
                                )
                                backend_code = get_upscaler().last_backend
                                framed_preview = ImageProcessor.frame_image_for_display(
                                    processed_image, frame_size=DISPLAY_FRAME
                                )

                                st.subheader("Enhanced Preview")
                                st.image(
                                    framed_preview,
                                    channels="BGR",
                                    use_column_width=False,
                                    caption=(
                                        f"Auto-enhanced input via {_describe_backend(backend_code)}"
                                    ),
                                )

                                # Detect faces for preview
                                face_locations = face_engine.detect_faces(
                                    processed_image
                                )

                                if not face_locations:
                                    st.warning("No faces detected in image")
                                else:
                                    st.info(
                                        f"Detected {len(face_locations)} face(s). Processing..."
                                    )
                                    face_chips = face_engine.extract_face_chips(
                                        processed_image, face_locations
                                    )

                                    if face_chips:
                                        st.subheader("Detected Faces")
                                        cols = st.columns(min(3, len(face_chips)))
                                        for idx, chip in enumerate(face_chips):
                                            display_chip = chip
                                            if chip.ndim == 2:
                                                if HAS_CV2:
                                                    display_chip = cv2.cvtColor(
                                                        chip, cv2.COLOR_GRAY2BGR
                                                    )
                                                else:
                                                    display_chip = np.stack(
                                                        [chip] * 3, axis=-1
                                                    )
                                            framed_chip = (
                                                ImageProcessor.frame_image_for_display(
                                                    display_chip,
                                                    frame_size=FACE_TILE_FRAME,
                                                    padding=12,
                                                )
                                            )
                                            cols[idx % len(cols)].image(
                                                framed_chip,
                                                caption=f"Face {idx + 1}",
                                                channels="BGR",
                                                use_column_width=False,
                                            )

                                # Use auto-improver if enabled
                                if enable_auto_improve_add:
                                    summary = auto_improver.improve_and_add_face(
                                        username=username,
                                        image=processed_image,
                                        source=source,
                                        min_quality_score=min_quality_score_add,
                                        auto_improve=True,
                                        metadata={
                                            "origin": "streamlit_add_faces",
                                            "uploader": "app_session",
                                        },
                                    )
                                else:
                                    summary = face_engine.process_and_add_face(
                                        username=username,
                                        image=processed_image,
                                        source=source,
                                        metadata={
                                            "origin": "streamlit_add_faces",
                                            "uploader": "app_session",
                                        },
                                        return_summary=True,
                                    )

                                    if summary.get("success"):
                                        st.success(
                                            f"✅ Added {summary.get('faces_added', 0)} face(s) for @{username}"
                                        )
                                        if enable_auto_improve_add and summary.get(
                                            "quality_scores"
                                        ):
                                            quality_scores = summary.get(
                                                "quality_scores", []
                                            )
                                            avg_quality = sum(
                                                q.get("overall_score", 0)
                                                for q in quality_scores
                                            ) / max(len(quality_scores), 1)
                                            st.info(
                                                f"Average quality score: {avg_quality:.2f}"
                                            )
                                    else:
                                        st.error(
                                            "Failed to add or update faces. Check details below."
                                        )

                                    st.caption(
                                        f"Detected {summary.get('faces_detected', 0)} face(s); "
                                        f"stored {summary.get('faces_added', 0)} record(s)."
                                    )

                                if summary.get("faces_rejected", 0) > 0:
                                    st.warning(
                                        f"Rejected {summary.get('faces_rejected', 0)} low-quality face(s)"
                                    )

                                    if summary.get("errors"):
                                        with st.expander(
                                            "Processing details",
                                            expanded=not summary.get("success"),
                                        ):
                                            for msg in summary["errors"]:
                                                st.write(f"- {msg}")

                elif add_mode == "Live camera":
                    if camera_capture is None:
                        st.error("Please capture an image using the camera above")
                    else:
                        with st.spinner("Processing camera capture..."):
                            raw_image = _load_image(camera_capture)
                            if raw_image is None:
                                st.error("Failed to decode camera capture")
                            else:
                                if enable_auto_improve_add:
                                    quality_metrics = quality_assessor.assess_quality(
                                        raw_image
                                    )
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric(
                                            "Quality Score",
                                            f"{quality_metrics['overall_score']:.2f}",
                                        )
                                    with col2:
                                        if (
                                            quality_metrics["overall_score"]
                                            < min_quality_score_add
                                        ):
                                            st.warning(
                                                "⚠️ Quality below minimum threshold"
                                            )
                                        else:
                                            st.success("✓ Quality acceptable")

                                if enable_auto_improve_add:
                                    summary = auto_improver.improve_and_add_face(
                                        username=username,
                                        image=raw_image,
                                        source=source,
                                        min_quality_score=min_quality_score_add,
                                        auto_improve=True,
                                        metadata={
                                            "origin": "streamlit_camera",
                                            "uploader": "app_session",
                                        },
                                    )
                                else:
                                    summary = face_engine.process_and_add_face(
                                        username=username,
                                        image=raw_image,
                                        source=source,
                                        metadata={
                                            "origin": "streamlit_camera",
                                            "uploader": "app_session",
                                        },
                                        return_summary=True,
                                    )

                                if summary.get("success"):
                                    st.success(
                                        f"✅ Added {summary.get('faces_added', 0)} face(s) for @{username}"
                                    )
                                    if enable_auto_improve_add and summary.get(
                                        "quality_scores"
                                    ):
                                        quality_scores = summary.get(
                                            "quality_scores", []
                                        )
                                        avg_quality = sum(
                                            q.get("overall_score", 0)
                                            for q in quality_scores
                                        ) / max(len(quality_scores), 1)
                                        st.info(
                                            f"Average quality score: {avg_quality:.2f}"
                                        )
                                else:
                                    st.error("No faces added from camera capture")

                                if summary.get("faces_rejected", 0) > 0:
                                    st.warning(
                                        f"Rejected {summary.get('faces_rejected', 0)} low-quality face(s)"
                                    )

                                if summary.get("errors"):
                                    with st.expander(
                                        "Processing details",
                                        expanded=not summary.get("success"),
                                    ):
                                        for msg in summary["errors"]:
                                            st.write(f"- {msg}")

                else:  # Video upload
                    if not video_file:
                        st.error("Please upload a video")
                    else:
                        VideoProcessor = get_video_processor()
                        with st.spinner("Sampling frames and adding faces..."):
                            suffix = _infer_temp_suffix(video_file.name)
                            frames = VideoProcessor.sample_frames_from_bytes(
                                video_file.getbuffer(),
                                suffix=suffix,
                                frame_stride=frame_stride,
                                max_frames=max_frames,
                            )
                            if not frames:
                                st.error("No frames could be read from the video")
                            else:
                                total_added = 0
                                total_detected = 0
                                total_rejected = 0
                                errors: list[str] = []

                                progress_bar = st.progress(0)
                                for frame_idx, frame in enumerate(frames):
                                    if enable_auto_improve_add:
                                        summary = auto_improver.improve_and_add_face(
                                            username=username,
                                            image=frame,
                                            source=source,
                                            min_quality_score=min_quality_score_add,
                                            auto_improve=True,
                                            metadata={
                                                "origin": "streamlit_video",
                                                "frame_index": frame_idx,
                                                "uploader": "app_session",
                                            },
                                        )
                                    else:
                                        summary = face_engine.process_and_add_face(
                                            username=username,
                                            image=frame,
                                            source=source,
                                            metadata={
                                                "origin": "streamlit_video",
                                                "frame_index": frame_idx,
                                                "uploader": "app_session",
                                            },
                                            return_summary=True,
                                        )
                                    total_added += summary.get("faces_added", 0)
                                    total_detected += summary.get("faces_detected", 0)
                                    total_rejected += summary.get("faces_rejected", 0)
                                    errors.extend(summary.get("errors", []))
                                    progress_bar.progress((frame_idx + 1) / len(frames))

                                progress_bar.empty()

                                if total_added > 0:
                                    st.success(
                                        f"✅ Added {total_added} face(s) across {len(frames)} sampled frame(s)"
                                    )
                                else:
                                    st.error("No faces added from sampled video frames")

                                st.caption(
                                    f"Detected {total_detected} face(s) across sampled frames."
                                )

                                if enable_auto_improve_add and total_rejected > 0:
                                    st.warning(
                                        f"Rejected {total_rejected} low-quality face(s) from video"
                                    )

                                if errors:
                                    with st.expander("Processing details"):
                                        for msg in errors:
                                            st.write(f"- {msg}")

    # Tab 3: Analytics
    with tab3:
        st.header("Database Analytics")

        stats = db.get_statistics()

        if stats.get("error"):
            st.error(f"Database stats error: {stats['error']}")

        with st.expander("Database Connection"):
            try:
                status = db.get_backend_status()
            except Exception as exc:
                status = {"ok": False, "error": str(exc)}

            if status.get("ok"):
                st.success(f"Connected ({status.get('backend', 'unknown')})")
            else:
                st.error(
                    f"Not connected ({status.get('backend', 'unknown')}): {status.get('error', 'unknown error')}"
                )
            st.json(status)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Faces", stats.get("total_faces", 0))
        with col2:
            st.metric("Unique Users", stats.get("unique_users", 0))
        with col3:
            created_at = stats.get("created_at")
            if isinstance(created_at, str) and created_at:
                created_display = created_at[:10]
            elif created_at is not None:
                created_display = str(created_at)[:10]
            else:
                created_display = "N/A"
            st.metric("Database Created", created_display)

        if stats.get("sources"):
            st.subheader("Faces by Source")
            sources = stats["sources"]
            st.bar_chart(sources)


if __name__ == "__main__":
    main()
