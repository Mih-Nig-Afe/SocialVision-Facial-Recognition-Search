# SocialVision Current Capabilities

**Version:** 1.0.0  
**Last Updated:** January 2026 (video/live inputs + fast mode + multi-backend DB)  
**Audience:** Engineers, QA, demo facilitators

---

## Snapshot

- **Dual Embedding Bundles:** Faces can store DeepFace (Facenet512) and/or dlib encodings side-by-side.
- **Mode-Agnostic Matching:** Searches compare against the full DB regardless of extraction mode, using only compatible embedding keys and dimensions (prevents 128 vs 512 crashes).
- **Delta-Only Enrichment:** When a face is recognized confidently, the system enriches that identity by **adding only missing embedding keys (“dimensions”)** instead of re-uploading existing vectors.
- **Video + Live Camera Inputs:** Search and Add workflows support image uploads, sampled video frames, and live camera capture (WebRTC when available; capture-based mode otherwise).
- **Fast / Ultra-Fast Recognition:** A dlib (128-d) fast mode exists for responsiveness, with optional in-memory caching for ultra-fast searches (cache TTLs are environment-controlled).
- **Face Quality Gate + Auto-Improve:** Optional quality scoring (blur/brightness/contrast/sharpness) can reject poor frames and automatically apply improvements before embedding.
- **High-Detail Preprocessing:** Native Real-ESRGAN is now the default super-resolution backend with configurable pass counts, minimum trigger scale, and per-frame tile targeting (e.g., force ~25 tiles). When IBM MAX or the NCNN CLI are available they slot ahead of OpenCV/Lanczos, but CPU-only Docker automatically clamps Real-ESRGAN to a single 4× pass to stay responsive.
- **Search Pathways:** Rank results per face, aggregate matches by username, enrich identities by appending fresh embeddings post-match.
- **Self-Training Profiles:** Confident matches write back enrichment metadata (origin, trigger similarity, batch size) so identities improve over time.
- **Operational Tooling:** Streamlit operator console, Docker build designed for reproducible installs (staged dependency installs + layer caching), DeepFace weight prefetch, JSON database auto-versioning.
- **Programmatic Access:** A FastAPI service exposes search/add/enrich endpoints for images, videos, and camera frames.
- **Quality Baseline:** Pytest suites cover engine/database/search; Streamlit workflows rely on the same API contracts.

---

## Feature Matrix

| Capability | Status | Notes |
|------------|--------|-------|
| Face detection | ✅ | DeepFace detector primary; face_recognition fallback. |
| Embedding extraction | ✅ | Dual embeddings per face; configurable weights. |
| Local database | ✅ | Stores bundles + primary vectors + metadata. |
| Similarity search | ✅ | Weighted cosine similarity; dimension-safe + bundle-aware scoring. |
| Streamlit UI | ✅ | Search / Add / Analytics tabs with live metrics. |
| Auto-training enrichment | ✅ | After a confident match, adds only missing embedding keys (delta enrichment) and avoids rewriting full records. |
| Image upscaling | ✅ | Real-ESRGAN-first pipeline with configurable minimum trigger scale, max passes, target tile count, and CPU-aware clamps; IBM MAX and the NCNN CLI remain optional accelerators ahead of OpenCV/Lanczos. |
| Firebase Realtime DB integration | ✅ | Incremental writes + delta embedding patches to avoid oversized payloads; JSON store remains for offline-only demos. |
| Firestore integration | ✅ | Available as an alternative backend; JSON store remains for offline-only demos. |
| Video upload search | ✅ | Samples frames (`VIDEO_FRAME_STRIDE`, `VIDEO_MAX_FRAMES`) and aggregates matches per username. |
| Live camera recognition | ✅ | WebRTC when available outside Docker; capture-based mode as fallback. |
| Fast mode search | ✅ | dlib 128-d search path with optional upscale retry and detailed (512-d) fallback when needed. |
| Ultra-fast embedding cache | ✅ | In-memory embedding cache for fast-mode searches; TTLs controlled by `EMBEDDING_CACHE_TTL_LOCAL` / `EMBEDDING_CACHE_TTL_FIRESTORE`. |
| Face quality scoring | ✅ | Blur/brightness/contrast/sharpness scoring available; can filter low-quality faces before adding. |
| Auto face improvement | ✅ | Optional CLAHE/denoise/sharpen pipeline to improve add/enrich inputs before embedding. |
| Batch processing | ✅ | `FaceRecognitionEngine.batch_process_images` for offline ingestion. |
| Dockerized runtime | ✅ | BuildKit cache for TensorFlow, DeepFace weight caching, health checks. |
| Testing | ✅ | `tests/` suites covering engine, DB, search flows. |
| Automated ingestion | 🚧 Planned | Manual uploads only today. |
| API endpoints (FastAPI) | ✅ | REST endpoints for enrich/search/add over image/video + base64 camera frames; includes `/health`. |

---

## Current Workflows

### Search in the UI

1. Open **Search** tab → upload photo.
2. Engine detects faces, generates dual embeddings, and runs weighted similarity search.
3. Results show per-face matches, usernames, and similarity scores above the configured threshold.

### Add Faces

1. Open **Add Faces** tab → upload image → provide username + source (profile_pic/post/story/reel).
2. For each detected face the UI now posts the full embedding bundle to the database.
3. If the username already exists, the database will prefer **delta-only updates** (only missing embedding keys) rather than duplicating vectors.

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
| Cloud sync | Firebase backends are still single-region by default. | Enable backup tooling / multi-region replication via Firebase console. |
| Automation | External ingestion from third-party platforms is not implemented. | Streamlit UI + API exist today, but ingestion remains manual. |
| API hardening | API is MVP-level (no auth/rate limits by default). | Add authentication, rate limiting, and request validation for hosted deployments. |
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

1. Run Streamlit, add a face, confirm database entry includes expected embedding keys (`deepface` and/or `dlib`).
2. Search for that face in the other mode; confirm match succeeds and enrichment reports delta updates when needed.
3. Rebuild Docker image with `docker compose build` and verify cached layers reduce rebuild time.

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
- `DB_TYPE=firebase` prefers Firebase Realtime Database and falls back to Firestore (then local JSON).
- `EMBEDDING_CACHE_TTL_LOCAL` / `EMBEDDING_CACHE_TTL_FIRESTORE` tune ultra-fast cache lifetimes.
- `LIVE_AUTO_ENRICH_ENABLED` and related `LIVE_AUTO_ENRICH_*` knobs control live-camera enrichment batching.

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

Last updated: January 2026

