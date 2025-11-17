# Simplified single-stage build for SocialVision Facial Recognition Search Engine
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH="/app" \
    LIBGL_ALWAYS_INDIRECT=1 \
    DISPLAY="" \
    QT_QPA_PLATFORM=offscreen \
    LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/lib/aarch64-linux-gnu:/usr/local/lib \
    DEEPFACE_HOME=/root \
    DEEPFACE_MODEL=Facenet512 \
    DEEPFACE_DETECTOR_BACKEND=opencv

# Install minimal system dependencies
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libglvnd0 \
    libglvnd-dev \
    libglx0 \
    libglx-dev \
    libgl1 \
    libgl1-mesa-dri \
    libglx-mesa0 \
    libopenblas-dev \
    liblapack-dev \
    curl \
    cmake \
    build-essential \
    git \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/* || true

# Create app directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
# Use an extended timeout when upgrading pip/setuptools to avoid network read timeouts,
# then install project dependencies. Ensure standard OpenCV (cv2) is available for DeepFace.
# Install TensorFlow and Keras first to ensure compatibility, then DeepFace
RUN pip install --default-timeout=3600 --upgrade pip setuptools wheel && \
    pip install --default-timeout=3600 --retries 10 -r requirements.txt && \
    python -c "from deepface import DeepFace; import tensorflow as tf; import keras; print(f'DeepFace OK. TF: {tf.__version__}, Keras: {keras.__version__}')" || echo "DeepFace import check failed"

# Pre-fetch DeepFace weights so runtime workload doesn't redownload models
RUN python - <<'PY'
import os
from pathlib import Path
from deepface import DeepFace

models = [os.environ.get("DEEPFACE_MODEL", "Facenet512")]
weights_dir = Path(os.environ.get("DEEPFACE_HOME", "/root")) / ".deepface" / "weights"
weights_dir.mkdir(parents=True, exist_ok=True)

for model in dict.fromkeys(models):  # preserve order, avoid duplicates
    try:
        DeepFace.build_model(model)
        print(f"Cached DeepFace model: {model}")
    except Exception as exc:  # pragma: no cover - build-time diagnostic
        print(f"Warning: could not cache {model}: {exc}")
PY

# Copy application code
# Force rebuild by adding a unique comment
# Rebuild 2025-11-12 15:50
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/models /app/uploads

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Expose ports
EXPOSE 8501 8000

# Default command - run Streamlit app
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

