"""
SocialVision: Advanced Facial Recognition Search Engine
Main Streamlit Application
"""

import streamlit as st
from pathlib import Path
import tempfile
from src.config import get_config
from src.logger import setup_logger
from src.face_recognition_engine import FaceRecognitionEngine
from src.database import FaceDatabase
from src.search_engine import SearchEngine
from src.image_utils import ImageProcessor

try:  # numpy is optional but helps normalize DeepFace payloads
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a declared dependency
    np = None

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


def _normalize_embedding_value(value):
    """Convert numpy arrays to vanilla lists and drop empty payloads."""

    if value is None:
        return None

    # Prefer numpy-specific handling when available.
    if np is not None and isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        return value.tolist()

    if hasattr(value, "tolist"):
        normalized = value.tolist()
        if isinstance(normalized, (list, tuple)) and not normalized:
            return None
        return normalized

    if isinstance(value, (list, tuple)):
        return list(value) if value else None

    return value


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
            uploaded_file = st.file_uploader(
                "Upload an image", type=["jpg", "jpeg", "png", "gif", "bmp", "webp"]
            )

        with col2:
            search_button = st.button("🔍 Search", use_container_width=True)

        if uploaded_file and search_button:
            with st.spinner("Processing image..."):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name

                try:
                    # Load and process image
                    image = ImageProcessor.load_image(tmp_path)
                    if image is None:
                        st.error("Failed to load image")
                    else:
                        # Display uploaded image
                        st.subheader("Uploaded Image")
                        st.image(image, channels="BGR", use_column_width=True)

                        # Perform search
                        st.subheader("Search Results")
                        try:
                            results = search_engine.search_by_image(
                                image, threshold=similarity_threshold, top_k=top_k
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
                "Upload image with faces", type=["jpg", "jpeg", "png"], key="add_faces"
            )

        with col2:
            username = st.text_input("Instagram Username")
            source = st.selectbox("Source", ["profile_pic", "post", "story", "reel"])

        if st.button("➕ Add to Database", use_container_width=True):
            if not uploaded_file or not username:
                st.error("Please upload an image and enter a username")
            else:
                with st.spinner("Processing..."):
                    # Save temporarily
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".jpg"
                    ) as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        tmp_path = tmp.name

                    try:
                        # Load image
                        image = ImageProcessor.load_image(tmp_path)
                        if image is None:
                            st.error("Failed to load image")
                        else:
                            # Detect and extract faces
                            face_engine = FaceRecognitionEngine()
                            face_locations = face_engine.detect_faces(image)

                            if not face_locations:
                                st.warning("No faces detected in image")
                            else:
                                st.info(
                                    f"Detected {len(face_locations)} face(s). Extracting embeddings..."
                                )
                                embeddings = face_engine.extract_face_embeddings(
                                    image, face_locations
                                )

                                if not embeddings:
                                    st.error(
                                        "Failed to extract face embeddings. DeepFace and dlib backends "
                                        "may both be unavailable. Please check the logs for details."
                                    )
                                    logger.error(
                                        "No embeddings extracted from available backends"
                                    )
                                elif not isinstance(embeddings, list):
                                    st.error(
                                        f"Unexpected embedding payload type: {type(embeddings)}. "
                                        "Please check the logs."
                                    )
                                    logger.error(
                                        "FaceRecognitionEngine returned unexpected type %s",
                                        type(embeddings),
                                    )
                                else:
                                    added_count = 0
                                    for i, embedding in enumerate(embeddings):
                                        try:
                                            if isinstance(embedding, dict):
                                                bundle = {}
                                                for key, value in embedding.items():
                                                    normalized = (
                                                        _normalize_embedding_value(
                                                            value
                                                        )
                                                    )
                                                    if normalized is not None:
                                                        bundle[key] = normalized
                                            else:
                                                normalized = _normalize_embedding_value(
                                                    embedding
                                                )
                                                if normalized is not None:
                                                    bundle = {
                                                        DEFAULT_EMBEDDING_SOURCE: normalized
                                                    }
                                                else:
                                                    bundle = {}
                                            if not bundle:
                                                logger.warning(
                                                    "Empty embedding bundle at index %s, skipping",
                                                    i,
                                                )
                                                continue

                                            if db.add_face(bundle, username, source):
                                                added_count += 1
                                                logger.info(
                                                    "Successfully added face %s/%s for %s",
                                                    i + 1,
                                                    len(embeddings),
                                                    username,
                                                )
                                            else:
                                                logger.error(
                                                    "Failed to add face %s/%s to database",
                                                    i + 1,
                                                    len(embeddings),
                                                )
                                        except Exception as e:
                                            logger.error(
                                                "Error adding face %s: %s",
                                                i + 1,
                                                e,
                                                exc_info=True,
                                            )
                                            st.error(f"Error adding face {i+1}: {e}")

                                    if added_count > 0:
                                        st.success(
                                            f"✅ Added {added_count} face(s) to database for @{username}"
                                        )
                                    else:
                                        st.error(
                                            "Failed to add any faces to database. "
                                            "Please check the logs for error details."
                                        )

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
