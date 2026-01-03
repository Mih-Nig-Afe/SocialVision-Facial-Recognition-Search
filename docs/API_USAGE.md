# SocialVision FastAPI Usage

**Audience:** Integrators, demo facilitators, QA

This repo includes a FastAPI service in `src/api.py` that exposes the same core workflows as the Streamlit UI (search, add, enrich) plus video and camera-frame support.

---

## Run the API

Local:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Interactive docs:

- Swagger UI: `http://localhost:8000/docs`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

Health:

```bash
curl -s http://localhost:8000/health
```

---

## Common Parameters

Most endpoints accept these controls:

- `threshold`: cosine similarity cutoff (float, default from `FACE_SIMILARITY_THRESHOLD`)
- `top_k`: max matches to return (int)
- `frame_stride`: sample every N frames for video (int, default from `VIDEO_FRAME_STRIDE`)
- `max_frames`: cap sampled frames (int, default from `VIDEO_MAX_FRAMES`)

Notes:

- For image/video uploads, the API uses `multipart/form-data`.
- For camera frames, the API expects base64 bytes in JSON.

---

## Search Endpoints

### `POST /api/search-face`

Search the database using an uploaded image.

```bash
curl -s -X POST \
  -F "image=@/path/to/photo.jpg" \
  -F "threshold=0.35" \
  -F "top_k=10" \
  http://localhost:8000/api/search-face | jq
```

Response: JSON list/structure of matches (same scoring logic as `SearchEngine.search_by_image`).

### `POST /api/search-video`

Search faces in a video by sampling frames.

```bash
curl -s -X POST \
  -F "video=@/path/to/clip.mp4" \
  -F "threshold=0.35" \
  -F "top_k=10" \
  -F "frame_stride=5" \
  -F "max_frames=90" \
  http://localhost:8000/api/search-video | jq
```

---

## Enrichment Endpoints (Search + Auto-Update)

These endpoints run the “enrichment pipeline”: try to find a person, and if found, append embeddings back into that identity profile.

### `POST /api/enrich-face`

```bash
curl -s -X POST \
  -F "image=@/path/to/photo.jpg" \
  -F "source=unknown" \
  -F "threshold=0.35" \
  -F "top_k=10" \
  http://localhost:8000/api/enrich-face | jq
```

If no confident match is found, the API returns a `404` with the text `Person not found.`

### `POST /api/enrich-video`

```bash
curl -s -X POST \
  -F "video=@/path/to/clip.mp4" \
  -F "source=unknown" \
  -F "threshold=0.35" \
  -F "top_k=10" \
  -F "frame_stride=5" \
  -F "max_frames=90" \
  http://localhost:8000/api/enrich-video | jq
```

If no match is found, the API returns a `404` with the text `Person not found in video.`

---

## Add / Ingest Endpoints

### `POST /api/add-face`

Add faces under a username (optionally applying quality scoring + auto-improvement).

```bash
curl -s -X POST \
  -F "image=@/path/to/photo.jpg" \
  -F "username=some_user" \
  -F "source=profile_pic" \
  -F "auto_improve=true" \
  -F "min_quality_score=0.3" \
  http://localhost:8000/api/add-face | jq
```

### `POST /api/add-video`

Add faces from sampled frames in a video.

```bash
curl -s -X POST \
  -F "video=@/path/to/clip.mp4" \
  -F "username=some_user" \
  -F "source=video" \
  -F "frame_stride=5" \
  -F "max_frames=90" \
  -F "auto_improve=true" \
  -F "min_quality_score=0.3" \
  http://localhost:8000/api/add-video | jq
```

---

## Camera Frame Search

### `POST /api/search-camera`

Send a base64-encoded image (raw bytes) and get matches.

Example (macOS/Linux) reading a JPEG and base64-encoding it:

```bash
IMG_B64=$(python - << 'PY'
import base64
from pathlib import Path
p = Path('/path/to/photo.jpg')
print(base64.b64encode(p.read_bytes()).decode('utf-8'))
PY
)

curl -s -X POST http://localhost:8000/api/search-camera \
  -H 'Content-Type: application/json' \
  -d '{
    "image_data": '"'"'$IMG_B64'"'"',
    "source": "camera",
    "threshold": 0.35,
    "top_k": 10,
    "auto_improve": true
  }' | jq
```

---

## Notes for Hosted Deployments

- This API is an MVP: it does **not** ship with authentication or rate limiting enabled by default.
- If you expose it publicly, add auth, rate limiting, and stricter request size limits.
