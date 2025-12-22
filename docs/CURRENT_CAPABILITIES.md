# SocialVision Current Capabilities

**Version:** 1.3.0  
**Last Updated:** December 2025 (Real-ESRGAN tiling + Firestore)  
**Audience:** Engineers, QA, demo facilitators

---

## Snapshot

- **Dual Embedding Bundles:** Every detected face stores DeepFace (Facenet512) + dlib encodings, normalized, weighted, and persisted for deterministic scoring.
- **High-Detail Preprocessing:** Native Real-ESRGAN is now the default super-resolution backend with configurable pass counts, minimum trigger scale, and per-frame tile targeting (e.g., force ~25 tiles). When IBM MAX or the NCNN CLI are available they slot ahead of OpenCV/Lanczos, but CPU-only Docker automatically clamps Real-ESRGAN to a single 4× pass to stay responsive.
- **Search Pathways:** Rank results per face, aggregate matches by username, enrich identities by appending fresh embeddings post-match.
- **Self-Training Profiles:** When a search discovers a confident match, the embedding bundle from that query is written back to the person’s profile with provenance metadata, so the system keeps learning dimensional stats (embeddings count, last added face, similarity history) automatically.
- **Operational Tooling:** Streamlit tri-tab UI, Docker build with pip cache mount, DeepFace weight prefetch, JSON database auto-versioning.
- **Quality Baseline:** Pytest suites cover engine/database/search; Streamlit workflows rely on the same API contracts.

---

## Feature Matrix

| Capability | Status | Notes |
|------------|--------|-------|
| Face detection | ✅ | DeepFace detector primary; face_recognition fallback. |
| Embedding extraction | ✅ | Dual embeddings per face; configurable weights. |
| Local database | ✅ | Stores bundles + primary vectors + metadata. |
| Similarity search | ✅ | Weighted cosine similarity; profile centroids per username. |
| Streamlit UI | ✅ | Search / Add / Analytics tabs with live metrics. |
| Auto-training enrichment | ✅ | Search matches append embeddings + metadata back into each identity to grow their profile dimensions without manual labeling. |
| Image upscaling | ✅ | Real-ESRGAN-first pipeline with configurable minimum trigger scale, max passes, target tile count, and CPU-aware clamps; IBM MAX and the NCNN CLI remain optional accelerators ahead of OpenCV/Lanczos. |
| Firestore integration | ✅ | Firestore-backed `FaceDatabase` provisions collections automatically, keeps provenance metadata, and mirrors writes with the enrichment pipeline; JSON store remains for offline-only demos. |
| Batch processing | ✅ | `FaceRecognitionEngine.batch_process_images` for offline ingestion. |
| Dockerized runtime | ✅ | BuildKit cache for TensorFlow, DeepFace weight caching, health checks. |
| Testing | ✅ | `tests/` suites covering engine, DB, search flows. |
| Automated ingestion | 🚧 Planned | Manual uploads only today. |
| API endpoints | 🚧 Planned | Streamlit UI currently drives core features. |

---

## Current Workflows

### Search in the UI

1. Open **Search** tab → upload photo.
2. Engine detects faces, generates dual embeddings, and runs weighted similarity search.
3. Results show per-face matches, usernames, and similarity scores above the configured threshold.

### Add Faces

1. Open **Add Faces** tab → upload image → provide username + source (profile_pic/post/story/reel).
2. For each detected face the UI now posts the full embedding bundle to the database.
3. Database normalizes, stores metadata, and recomputes the username centroid cache.

### Programmatic Snippet

```python
from src.face_recognition_engine import FaceRecognitionEngine
from src.database import FaceDatabase

engine = FaceRecognitionEngine()
db = FaceDatabase()

image, bundles = engine.process_image("samples/group.jpg")
for bundle in bundles:
    db.add_face(bundle, username="subject", source="post")

query = bundles[0]
matches = db.search_similar_faces(query, threshold=0.35, top_k=5)
```

---

## Limitations & Risks

| Area | Details | Mitigation |
|------|---------|------------|
| Data store | JSON file scales linearly; no vector index. | Keep dataset <10k faces or migrate to vector DB (planned). |
| Cloud sync | Firestore is default but still single-region. | Enable backup tooling / multi-region replication via Firebase console. |
| Automation | External ingestion and API endpoints not implemented. | Streamlit UI/manual uploads only for now. |
| GPU utilization | Pipelines run on CPU by default; containers assume CPU. | Evaluate GPU-enabled base image when moving beyond research demos. |

---

## Performance Notes (CPU reference laptop)

| Operation | Typical Time |
|-----------|--------------|
| Face detection (per image) | 1–3 s |
| Dual embedding extraction (per face) | 3–5 s |
| DB similarity search (<1k faces) | <0.5 s |
| Streamlit search request | <6 s end-to-end |

Search scales roughly linearly with stored faces because no ANN index is used yet.

---

## Testing

Automated coverage:

- `tests/test_face_recognition.py` – detection + bundling edge cases.
- `tests/test_database.py` – bundle storage, weighted search, metadata caches.
- `tests/test_search_engine.py` – enrichment flows, grouping helpers.

Commands:

```bash
pytest tests/ -v
pytest tests/test_database.py -vv
```

Manual smoke tests:

1. Run Streamlit, add a face, confirm database entry includes both `deepface` and `dlib` vectors.
2. Search for that face; check similarity score ≥ threshold and enrichment summary.
3. Rebuild Docker image with `DOCKER_BUILDKIT=1 docker compose build` to verify pip caching stage.

---

## Configuration Highlights

- `ENABLE_DUAL_EMBEDDINGS=true` keeps DeepFace + dlib active simultaneously.
- `DEEPFACE_EMBEDDING_WEIGHT` / `DLIB_EMBEDDING_WEIGHT` tune how similarity scores blend.
- `LOCAL_DB_PATH` switches JSON storage (default `data/faces_database.json`).
- `FACE_SIMILARITY_THRESHOLD` globally affects Search + Enrichment.
- `IMAGE_UPSCALING_ENABLED` keeps the multi-backend super-resolution stage active (default `true`).
- `IMAGE_UPSCALING_MIN_REALESRGAN_SCALE` + `IMAGE_UPSCALING_TARGET_TILES` tune when Real-ESRGAN fires and how many tiles it should consume per frame.
- `IBM_MAX_ENABLED`, `IBM_MAX_URL`, `IBM_MAX_TIMEOUT` control the optional IBM MAX microservice client.
- `NCNN_UPSCALING_ENABLED`, `NCNN_EXEC_PATH`, `NCNN_MODEL_NAME` configure the Real-ESRGAN NCNN Vulkan fallback.
- `DB_TYPE=firestore` switches the database driver from JSON to Firestore.
- `DB_TYPE=realtime` switches the database driver from JSON to Firebase Realtime Database.
- `DB_TYPE=firebase` prefers Firestore and falls back to Realtime Database.

See `src/config.py` for the full catalog.

---

## Related Docs

- **[PROJECT_STATUS.md](PROJECT_STATUS.md)** – macro progress + open work.
- **[TESTING_GUIDE.md](TESTING_GUIDE.md)** – step-by-step QA plans.
- **[DEMONSTRATION_GUIDE.md](DEMONSTRATION_GUIDE.md)** – narrative for live walkthroughs.

---

## Support

Questions go to Mihretab N. Afework · <mtabdevt@gmail.com>.

---

Last updated: December 2025

