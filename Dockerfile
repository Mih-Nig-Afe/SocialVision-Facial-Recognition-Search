# syntax=docker/dockerfile:1.6
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

# Fail fast if cmake is still missing; dlib build requires it
RUN cmake --version

# Create app directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
# Install dependencies in optimized stages to handle network timeouts
# Stage 1: Upgrade pip and install core build tools
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=600 --retries=5 --upgrade pip setuptools wheel

# Stage 2: Install large packages separately to avoid JSON decode errors
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=1800 --retries=10 --no-cache-dir \
    numpy==1.26.4 scipy==1.14.1 || \
    pip install --default-timeout=1800 --retries=10 \
    numpy==1.26.4 scipy==1.14.1

# Stage 3: Install PyTorch separately (large package, prone to timeouts)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=1800 --retries=10 --no-cache-dir \
    torch==2.4.1 torchvision==0.19.1 || \
    pip install --default-timeout=1800 --retries=10 \
    torch==2.4.1 torchvision==0.19.1

# Stage 4: Install TensorFlow and Keras (quote version constraints to avoid shell redirection)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=1800 --retries=10 \
    tensorflow==2.16.1 "keras>=3.0.0,<4.0.0" tf-keras==2.16.0

# Stage 5: Install remaining requirements
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=1800 --retries=10 -r requirements.txt || \
    (pip install --default-timeout=1800 --retries=10 --no-cache-dir -r requirements.txt)

# Stage 6: Verify critical imports
RUN python - <<'PY'
try:
    from deepface import DeepFace
    import tensorflow as tf
    import keras
    import torch
    print(f"✓ DeepFace OK. TF: {tf.__version__}, Keras: {keras.__version__}, PyTorch: {torch.__version__}")
except Exception as e:
    print(f"✗ Import error: {e}")
    raise
PY

# Pre-fetch DeepFace weights so runtime workload doesn't redownload models
RUN python - <<'PY'
import os
from pathlib import Path

try:
    from deepface import DeepFace
except ModuleNotFoundError:
    print("DeepFace not installed; skipping model cache stage")
else:
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

# Default command - run Streamlit app via python -m to avoid PATH issues
CMD ["python", "-m", "streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

