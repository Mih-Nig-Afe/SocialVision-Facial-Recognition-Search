# SocialVision: Advanced Facial Recognition Search Engine

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.39+-red.svg)](https://streamlit.io)
[![DeepFace](https://img.shields.io/badge/DeepFace-Facenet512-purple.svg)](https://github.com/serengil/deepface)
[![face_recognition](https://img.shields.io/badge/dlib-face_recognition-green.svg)](https://github.com/ageitgey/face_recognition)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ed.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

SocialVision is an academic research project that builds an end‑to‑end facial recognition search engine for Instagram-style content. The stack combines **DeepFace (TensorFlow/Keras)** embeddings with **dlib/face_recognition** encodings, fuses both vectors per face, and exposes the experience through a Streamlit UI, local JSON database, and automated tests that keep the pipeline reproducible.

> **Maintainer**: Mihretab N. Afework ([@Mih-Nig-Afe](https://github.com/Mih-Nig-Afe)) · <mtabdevt@gmail.com>

---

## Contents

1. [Capabilities](#capabilities)
2. [Architecture](#architecture)
3. [Getting Started](#getting-started)
4. [Usage](#usage)
5. [Configuration](#configuration)
6. [Documentation Map](#documentation-map)
7. [Testing](#testing)
8. [Roadmap & Contributing](#roadmap--contributing)

---

## Capabilities

| Area | Highlights |
|------|------------|
| **Dual Embedding Pipeline** | DeepFace (Facenet512) + dlib encodings stored side-by-side, weighted similarity scoring, automatic fallbacks if TensorFlow is unavailable. |
| **Face Search Engine** | Detection, embedding, cosine similarity search, identity aggregation, configurable thresholds, enrichment workflows that continuously learn from matches. |
| **Streamlit Command Center** | Tabs for Search, Add Faces, Analytics; live metrics, threshold sliders, and enrichment summaries meant for operator demos. |
| **Data Layer** | Versioned JSON database, per-face metadata, cached profile centroids per username, normalized bundles for deterministic math. |
| **Operations** | Docker image with BuildKit pip caching (no repeated TensorFlow wheel downloads), DeepFace weight prefetch, CLI demo script, logging + health checks. |
| **Quality & Docs** | Pytest coverage for engine/database/search, reproducible fixtures, comprehensive docs mirroring professional OSS projects. |

---

## Architecture

```text
        +---------------------------+
        |        Streamlit UI       |
        |  Search / Add / Analytics |
        +-------------+-------------+
                      |
                      v
        +-------------+-------------+
        |     SearchEngine API      |
        | - Enrichment workflows    |
        | - Threshold controls      |
        +------+------+-------------+
               |      |
     +---------+      +---------+
     v                          v
FaceRecognitionEngine     FaceDatabase
(DeepFace + dlib)         (JSON + metadata cache)
     |                          |
     v                          v
 DeepFace cache        data/faces_database.json
 dlib (face_recognition)
```

- **Detection/Embedding**: `FaceRecognitionEngine` first attempts DeepFace (Facenet512) and, based on config, also runs dlib encoders. Embeddings are normalized, bundled, and tagged per backend (`{"deepface": [...], "dlib": [...]}`).
- **Storage/Search**: `FaceDatabase` stores both the bundle and a primary embedding for backward compatibility, computes weighted cosine similarities, and maintains username centroids for quick identity queries.
- **Presentation / Ops**: Streamlit orchestrates searches and enrichment, while Docker provides an isolated runtime with cached pip layers and pre-fetched DeepFace weights.

---

## Getting Started

### Prerequisites

- Python 3.9+
- pip
- Git
- (Optional) Docker 24+ with BuildKit enabled

### Local Setup

```bash
git clone https://github.com/Mih-Nig-Afe/SocialVision-Facial-Recognition-Search.git
cd SocialVision-Facial-Recognition-Search
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Launch Streamlit:

```bash
streamlit run src/app.py
```

### Docker Workflow (recommended for demos)

```bash
export DOCKER_BUILDKIT=1
docker compose build
docker compose up -d
# or
./docker-demo.sh
```

Features of the container image:

- BuildKit cache mount for pip (`/root/.cache/pip`) so large wheels (TensorFlow, DeepFace) download once.
- Pre-fetch of DeepFace weights during build, reducing cold-start latency.
- Health check hitting `/_stcore/health` to signal readiness.

Access the UI at `http://localhost:8501`.

---

## Usage

### Streamlit Tabs

1. **🔍 Search** – Upload an image, system detects faces, extracts dual embeddings, and surfaces matches with similarity scores.
2. **📤 Add Faces** – Upload faces for specific usernames; the UI now uploads full embedding bundles so the database can blend DeepFace+dlib vectors.
3. **📈 Analytics** – Watch total faces, unique users, and per-source charts sourced directly from the JSON database.

### Programmatic Example

```python
from src.database import FaceDatabase
from src.face_recognition_engine import FaceRecognitionEngine

db = FaceDatabase()
engine = FaceRecognitionEngine()

image, bundles = engine.process_image("path/to/photo.jpg")
for bundle in bundles:
    db.add_face(bundle, username="research_subject", source="post")

query_results = db.search_similar_faces(bundle, threshold=0.35, top_k=5)
```

---

## Configuration

Key environment variables (see `src/config.py` for defaults):

| Variable | Description |
|----------|-------------|
| `DEEPFACE_MODEL` | DeepFace backbone (default `Facenet512`). |
| `DEEPFACE_DETECTOR_BACKEND` | Detector backend (default `opencv`). |
| `ENABLE_DUAL_EMBEDDINGS` | `true` to run DeepFace + dlib together (default `true`). |
| `DEEPFACE_EMBEDDING_WEIGHT` / `DLIB_EMBEDDING_WEIGHT` | Similarity weights applied during search. |
| `FACE_SIMILARITY_THRESHOLD` | Global cosine similarity cut-off. |
| `LOCAL_DB_PATH` | Path to JSON database (default `data/faces_database.json`). |

Add optional secrets (Firebase, etc.) via `.env` or environment-specific config classes.

---

## Documentation Map

- **[docs/README.md](docs/README.md)** – navigation hub.
- **[docs/CURRENT_CAPABILITIES.md](docs/CURRENT_CAPABILITIES.md)** – quick reference for what works today.
- **[docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md)** – phase progress, KPIs, and blocking issues.
- **[docs/TESTING_GUIDE.md](docs/TESTING_GUIDE.md)** – manual + automated test instructions.
- **[docs/DEVELOPMENT_ROADMAP.md](docs/DEVELOPMENT_ROADMAP.md)** – upcoming milestones.
- **[docs/DOCKER_TESTING_GUIDE.md](docs/DOCKER_TESTING_GUIDE.md)** – container-focused workflows.

---

## Testing

```bash
pytest tests/ -v

# Focused runs
pytest tests/test_face_recognition.py -v
pytest tests/test_database.py -v
pytest tests/test_search_engine.py -v
```

All three suites are CI-friendly and cover the dual-embedding engine, bundle-aware database, and enrichment logic.

---

## Roadmap & Contributing

**Current focus:**

- Expanding dataset ingestion (Instagram automation, Firebase sync)
- Hardening API surface (FastAPI service layer)
- Enhancing scalability (vector index, embeddings rebalancing)

Contributions are welcome:

1. Fork the repo.
2. Create a feature branch.
3. Add/modify code + docs + tests.
4. Open a pull request describing the change and verification steps.

Please review the [Development Roadmap](docs/DEVELOPMENT_ROADMAP.md) and [Testing Guide](docs/TESTING_GUIDE.md) before submitting changes.

---

## Contact

- Email: <mtabdevt@gmail.com>
- GitHub: [@Mih-Nig-Afe](https://github.com/Mih-Nig-Afe)
- LinkedIn: [Mihretab N. Afework](https://linkedin.com/in/mihretab-afework)

---

*SocialVision is built for academic and ethical research demonstrations. Use responsibly and respect privacy regulations in your jurisdiction.*
