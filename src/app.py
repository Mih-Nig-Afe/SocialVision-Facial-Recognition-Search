"""
SocialVision: Advanced Facial Recognition Search Engine
Main Streamlit Application
"""

import streamlit as st
from pathlib import Path
import tempfile
import numpy as np
from src.config import get_config
from src.logger import setup_logger
from src.face_recognition_engine import FaceRecognitionEngine
from src.database import FaceDatabase
from src.search_engine import SearchEngine
from src.image_utils import ImageProcessor
from src.image_upscaler import get_image_upscaler

# Try to import cv2, but make it optional
try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Configuration
config = get_config()
DEFAULT_EMBEDDING_SOURCE = getattr(config, "DEFAULT_EMBEDDING_SOURCE", "deepface")
logger = setup_logger(__name__)
UPLOAD_EXTENSIONS = sorted({ext.lower() for ext in config.ALLOWED_IMAGE_FORMATS})


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
    """Initialize and cache components"""
    db = FaceDatabase()
    search_engine = SearchEngine(db)
    return db, search_engine


def main():
    """Main application"""

    # Header
    st.title("🔍 SocialVision")
    st.markdown("### Advanced Facial Recognition Search Engine for Instagram")
    st.markdown("---")

    # Initialize components
    db, search_engine = initialize_components()

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

        col1, col2 = st.columns([2, 1])

        with col1:
            uploaded_file = st.file_uploader("Upload an image", type=UPLOAD_EXTENSIONS)

        with col2:
            search_button = st.button("🔍 Search", use_container_width=True)

        if uploaded_file and search_button:
            with st.spinner("Processing image..."):
                # Save uploaded file temporarily
                suffix = _infer_temp_suffix(uploaded_file.name)
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name

                try:
                    # Load and process image
                    raw_image = ImageProcessor.load_image(tmp_path)
                    if raw_image is None:
                        st.error("Failed to load image")
                    else:
                        processed_image = ImageProcessor.prepare_input_image(raw_image)
                        backend_code = get_image_upscaler().last_backend
                        framed_preview = ImageProcessor.frame_image_for_display(
                            processed_image, frame_size=DISPLAY_FRAME
                        )

                        # Display uploaded image
                        st.subheader("Uploaded Image")
                        st.image(
                            framed_preview,
                            channels="BGR",
                            use_column_width=False,
                            caption=(
                                f"Enhanced preview via {_describe_backend(backend_code)}"
                            ),
                        )

                        # Perform search
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
                            # Check if faces were detected but no matches found
                            if results.get("faces") and len(results["faces"]) > 0:
                                st.info(
                                    f"Detected {len(results['faces'])} face(s) but no matches found in database. Try adding faces to the database first."
                                )
                            else:
                                st.warning("No matching faces found in database")
                        else:
                            # Get top usernames
                            top_users = search_engine.get_top_usernames(
                                results, top_k=10
                            )

                            st.success(
                                f"Found {results['total_matches']} matches across {len(top_users)} users"
                            )

                            # Display results
                            st.markdown("### 👥 Top Matching Accounts")

                            for i, user_info in enumerate(top_users, 1):
                                with st.expander(
                                    f"{i}. @{user_info['username']} "
                                    f"({user_info['match_count']} matches, "
                                    f"{user_info['avg_similarity']:.2%} avg similarity)"
                                ):
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric(
                                            "Match Count", user_info["match_count"]
                                        )
                                    with col2:
                                        st.metric(
                                            "Avg Similarity",
                                            f"{user_info['avg_similarity']:.2%}",
                                        )

                                    st.write("**Sources:**")
                                    for source in user_info["sources"]:
                                        st.write(f"- {source}")

                finally:
                    # Clean up
                    Path(tmp_path).unlink(missing_ok=True)

    # Tab 2: Add Faces
    with tab2:
        st.header("Add Faces to Database")

        col1, col2 = st.columns(2)

        with col1:
            uploaded_file = st.file_uploader(
                "Upload image with faces", type=UPLOAD_EXTENSIONS, key="add_faces"
            )

        with col2:
            username = st.text_input("Instagram Username")
            source = st.selectbox("Source", ["profile_pic", "post", "story", "reel"])

        if st.button("➕ Add / Update Face", use_container_width=True):
            if not uploaded_file or not username:
                st.error("Please upload an image and enter a username")
            else:
                with st.spinner("Processing..."):
                    # Save temporarily
                    suffix = _infer_temp_suffix(uploaded_file.name)
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=suffix
                    ) as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        tmp_path = tmp.name

                    try:
                        # Load image
                        raw_image = ImageProcessor.load_image(tmp_path)
                        if raw_image is None:
                            st.error("Failed to load image")
                        else:
                            processed_image = ImageProcessor.prepare_input_image(
                                raw_image
                            )
                            backend_code = get_image_upscaler().last_backend
                            framed_preview = ImageProcessor.frame_image_for_display(
                                processed_image, frame_size=DISPLAY_FRAME
                            )

                            face_engine = FaceRecognitionEngine(database=db)
                            face_locations = face_engine.detect_faces(processed_image)

                            st.subheader("Enhanced Preview")
                            st.image(
                                framed_preview,
                                channels="BGR",
                                use_column_width=False,
                                caption=(
                                    f"Auto-enhanced input via {_describe_backend(backend_code)}"
                                ),
                            )

                            if not face_locations:
                                st.warning("No faces detected in image")
                            else:
                                st.info(
                                    f"Detected {len(face_locations)} face(s). Extracting embeddings..."
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
                                        f"✅ Added {summary['faces_added']} face(s) for @{username}"
                                    )
                                else:
                                    st.error(
                                        "Failed to add or update faces. Check details below."
                                    )

                                st.caption(
                                    f"Detected {summary['faces_detected']} face(s); "
                                    f"stored {summary['faces_added']} record(s)."
                                )

                                if summary.get("errors"):
                                    with st.expander(
                                        "Processing details",
                                        expanded=not summary.get("success"),
                                    ):
                                        for msg in summary["errors"]:
                                            st.write(f"- {msg}")

                    finally:
                        Path(tmp_path).unlink(missing_ok=True)

    # Tab 3: Analytics
    with tab3:
        st.header("Database Analytics")

        stats = db.get_statistics()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Faces", stats.get("total_faces", 0))
        with col2:
            st.metric("Unique Users", stats.get("unique_users", 0))
        with col3:
            st.metric("Database Created", stats.get("created_at", "N/A")[:10])

        if stats.get("sources"):
            st.subheader("Faces by Source")
            sources = stats["sources"]
            st.bar_chart(sources)


if __name__ == "__main__":
    main()
