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
    QT_QPA_PLATFORM=offscreen

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
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglx-mesa0 \
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
RUN pip install --upgrade pip setuptools wheel && \
    pip install --default-timeout=3600 --retries 10 -r requirements.txt

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

