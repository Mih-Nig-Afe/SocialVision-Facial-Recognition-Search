# SocialVision Project Status

**Last updated:** December 2025 (Real-ESRGAN tiling + Firestore)  
**Current phase:** Phase 5 – Cloud Persistence & Operational Hardening  
**Overall completion:** ~74%

---

## Executive Summary

SocialVision now delivers a working demo stack that extracts dual embeddings (DeepFace + dlib), stores bundle-aware vectors, defaults to Firestore for persistence, and exposes a Streamlit operator console. The remaining roadmap focuses on hardening the hosted Firestore deployment, exposing public APIs, and automating data ingestion.

### Highlights This Iteration

- Dual-embedding pipeline with weighted similarity search is live end-to-end.
- Docker build uses BuildKit pip caching and pre-fetches DeepFace weights, shrinking rebuilds by ~60%.
- Real-ESRGAN is now the default super-resolution backend with configurable minimum trigger scale, pass count, and per-frame tile targeting (`IMAGE_UPSCALING_TARGET_TILES` keeps ~25 tiles even on CPU Docker). IBM MAX and the NCNN CLI remain optional accelerators.
- Firestore-backed `FaceDatabase` ships as the default runtime (auto-provisions collections, tracks provenance metadata, username centroid cache) while the JSON store is retained for offline demos.
- Documentation overhaul (README + capabilities + status) brings parity with established OSS projects.
- Auto-enrichment loop takes every confident search match and appends its embeddings back into the person’s profile, so dimensional metrics stay fresh without manual curation.

---

## Delivery Snapshot

| Workstream | Status | Notes |
|------------|--------|-------|
| Face recognition engine | ✅ Complete | Dual embeddings, fallbacks, batch helpers. |
| Local database layer | ✅ Complete | Bundle storage, username centroids, metadata. |
| Search / enrichment engine | ✅ Complete | Weighted cosine search, aggregation, enrichment. |
| Streamlit UI | ✅ Complete (90%) | Search/Add/Analytics tabs; advanced filters pending. |
| Docker & DevOps | ✅ Complete | BuildKit cache, DeepFace weight caching, health checks. |
| Testing | ✅ Complete (unit) | Pytest coverage for engine, DB, search. |
| Firestore/cloud storage | ✅ Complete | Firestore driver live w/ auto-provision + JSON fallback. |
| Automated ingestion | 🚧 Not started | Manual uploads only today. |
| Public API (FastAPI) | 🚧 Not started | Streamlit doubles as controller for now. |

---

## Completed Capabilities

### Dual Embedding Engine

- DeepFace (Facenet512) + dlib encodings extracted per detected face.
- Automatic fallback to dlib when TensorFlow stack is unavailable.
- Bundles serialized for downstream consumers (`{"deepface": [...], "dlib": [...]}`).

### Bundle-Aware Database

- Stores embedding bundles plus a primary vector for backward compatibility.
- Weighted cosine similarity search with configurable backend weights.
- Username centroids (profile embeddings) recomputed after every append.

### Search & Enrichment

- `SearchEngine` consumes bundles, aggregates matches by username, and enriches identities by appending newly captured embeddings.
- `_auto_enrich_identity` records the similarity that triggered the update plus provenance metadata so profiles track their own dimensional growth (total embeddings, last added face ID, confidence history).
- Thresholds and top-k settings surfaced in the Streamlit UI, along with per-face match breakdowns.

### Operations & Tooling

- Dockerfile now mounts `/root/.cache/pip` during build so TensorFlow/DeepFace wheels download once.
- DeepFace model weights cached at build time to reduce runtime downloads.
- Health check monitors Streamlit readiness; `docker-demo.sh` streamlines demos.
- Image preprocessing routes through IBM MAX when available, with NCNN/native Real-ESRGAN and OpenCV providing redundant safety nets—all three sourced from actively maintained GitHub projects so upgrades remain traceable.

### Quality & Docs

- Pytest suites updated to assert bundle semantics.
- Root README, capabilities, and status docs refreshed to describe architecture, workflows, and limitations.

---

## In Progress / Planned

| Item | Target phase | Notes |
|------|--------------|-------|
| Firestore hardening (rules/backups) | Phase 5 | Collections live; need backup automation + IAM tightening. |
| Automated ingestion pipeline | Phase 5 | Depends on external data-source credentials + compliance review. |
| Public API (FastAPI) | Phase 6 | Reuse `SearchEngine`/`FaceDatabase` for REST endpoints. |
| Advanced UI tooling | Phase 4 | Export, filtering, image gallery, role-based controls. |
| Vector index (FAISS/Annoy) | Phase 6 | Required for datasets >10k faces. |

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Firestore backups/security | Single-region backups still manual, IAM needs hardening | Script scheduled exports, document least-privilege service accounts, keep JSON fallback for emergencies. |
| Manual ingestion workflows | Limits authenticity of research demos | Build ingestion pipeline once legal review passes. |
| No public API | Integrations must screen-scrape UI | Prioritize FastAPI service once Firebase groundwork is done. |
| Lack of automated deployment | Hard to share hosted demo | Container image is ready; need hosting plan once cloud storage exists. |

---

## Quality Metrics

- **Automated tests:** `pytest tests/ -v` (26 tests) – green on Python 3.9–3.11.
- **Manual smoke:** Streamlit Search/Add/Analytics, enrichment workflow, Docker BuildKit build.
- **Logging:** Structured logs across engine, DB, search; Streamlit surfaces user-facing alerts.

Gaps: no load, fuzz, or security testing yet.

---

## Next 30-Day Objectives

1. Harden Firestore deployment (security rules, automated exports, documentation) while keeping the JSON fallback healthy.
2. Draft FastAPI skeleton exposing search/add endpoints and shared validators.
3. Extend documentation with API draft + data retention policy.
4. Evaluate FAISS/Annoy for similarity search to inform vector DB migration plan (still required for >10k faces).

---

## References

- [`README.md`](../README.md)
- [`CURRENT_CAPABILITIES.md`](CURRENT_CAPABILITIES.md)
- [`DEVELOPMENT_ROADMAP.md`](DEVELOPMENT_ROADMAP.md)
- [`TESTING_GUIDE.md`](TESTING_GUIDE.md)

