# SocialVision Architecture

**Version:** 1.0.0  
**Last Updated:** January 2026  
**Audience:** Engineers, QA, demo facilitators

---

## Overview

SocialVision is an end-to-end facial recognition search system built around a Streamlit UI and a pipeline that:

1. Ingests inputs (image upload, video upload, or live camera)
2. Optionally upscales frames/images
3. Detects faces and extracts embeddings (DeepFace 512-d when enabled, plus optional dlib 128-d)
4. Searches against a multi-backend database (local JSON, Firebase Realtime Database, Firestore)
5. Aggregates results and optionally enriches identities with delta-only embedding updates

This document focuses on **runtime architecture**, **data flow**, and **how processing happens**.

---

## System Context (High-Level)

If your Markdown viewer does not render Mermaid diagrams, scroll to **Always-Visible Diagrams (ASCII)** below.

```mermaid
flowchart LR
  User((Operator)) -->|Browser| StreamlitUI[Streamlit UI\nSearch / Add Faces / Analytics]
  StreamlitUI --> Pipeline[Pipeline + SearchEngine\n(face detection, embedding, search, enrichment)]

  Pipeline --> Upscaler[Image Upscaler\nReal-ESRGAN / Optional IBM MAX / Fallbacks]
  Pipeline --> FaceEngine[FaceRecognitionEngine\nDeepFace + dlib]
  Pipeline --> DB[FaceDatabase\nlocal JSON / Realtime DB / Firestore]

  DB --> LocalJSON[(data/faces_database.json)]
  DB --> RTDB[(Firebase Realtime DB)]
  DB --> Firestore[(Google Firestore)]

  Pipeline --> Logs[(logs/)]
```

---

## Always-Visible Diagrams (ASCII)

### A) System Context (what talks to what)

```text
  +-----------+        +-------------------+        +-----------------------------+
  | Operator  | -----> | Streamlit UI      | -----> | Pipeline / SearchEngine     |
  | (browser) |        | (Search/Add/Stat) |        | (orchestration + logic)     |
  +-----------+        +-------------------+        +-----------------------------+
                                                      |      |           |
                                                      |      |           |
                                                      v      v           v
                                            +--------------+  +---------------------+  +------------------+
                                            | ImageUpscaler|  | FaceRecognitionEngine|  | FaceDatabase      |
                                            | (Real-ESRGAN |  | (DeepFace + dlib)    |  | (local/RTDB/FS)   |
                                            |  optional MAX|  +---------------------+  +------------------+
                                            |  fallbacks)  |                           |   |     |     |
                                            +--------------+                           v   v     v     v
                                                                                      JSON RTDB Firestore Logs
```

### B) Database selection (DB_TYPE)

```text
DB_TYPE=local      -> local JSON
DB_TYPE=realtime   -> Firebase Realtime Database
DB_TYPE=firestore  -> Firestore

DB_TYPE=firebase auto-mode:

  [Try Realtime DB] --ok--> use RTDB
        |
        no
        v
  [Try Firestore]   --ok--> use Firestore
        |
        no
        v
  [Fallback]              use local JSON
```

### C) Sequence: image search (happy path)

```text
Operator -> UI : upload image + Search
UI -> Pipeline/SearchEngine : search_by_image
Pipeline -> Upscaler : maybe_upscale
Pipeline -> FaceRecognitionEngine : detect_faces + extract_embeddings
Pipeline -> FaceDatabase : search_similar_faces
Pipeline -> Pipeline : aggregate results by username
Pipeline -> FaceDatabase : optional delta-only enrich on confident match
Pipeline -> UI : results + telemetry
```

---

## Component Diagram (UML-style)

```mermaid
classDiagram
  class StreamlitApp {
    +Search tab
    +Add Faces tab
    +Analytics tab
  }

  class Pipeline {
    +process_image(...)
    +process_video(...)
    +enrich_identity_if_confident(...)
  }

  class SearchEngine {
    +search_by_image(...)
    +search_by_embedding(...)
    +aggregate_by_username(...)
  }

  class FaceRecognitionEngine {
    +detect_faces(image)
    +extract_embeddings(image, faces)
    +process_image(path_or_image)
  }

  class ImageUpscaler {
    +upscale(image)
    +retry_strategy(...)
  }

  class FaceDatabase {
    +add_face(bundle, username, source)
    +search_similar_faces(bundle, ...)
    +get_statistics()
  }

  class LocalJsonStore
  class FirebaseRealtimeStore
  class FirestoreStore

  StreamlitApp --> Pipeline
  Pipeline --> SearchEngine
  Pipeline --> FaceRecognitionEngine
  Pipeline --> ImageUpscaler
  SearchEngine --> FaceDatabase

  FaceDatabase <|-- LocalJsonStore
  FaceDatabase <|-- FirebaseRealtimeStore
  FaceDatabase <|-- FirestoreStore
```

---

## Processing Flows (Sequence Diagrams)

### 1) Search: Image Upload

```mermaid
sequenceDiagram
  participant U as Operator
  participant UI as Streamlit UI
  participant P as Pipeline/SearchEngine
  participant UP as Upscaler
  participant FE as FaceRecognitionEngine
  participant DB as FaceDatabase

  U->>UI: Upload image + click Search
  UI->>P: search_by_image(image, threshold, top_k)

  P->>UP: maybe_upscale(image)
  UP-->>P: upscaled_or_original

  P->>FE: detect_faces(frame)
  FE-->>P: face_locations

  P->>FE: extract_embeddings(frame, faces)
  FE-->>P: embedding bundles (deepface/dlib)

  P->>DB: search_similar_faces(bundles)
  DB-->>P: matches

  P->>P: aggregate results by username
  P->>DB: enrich_identity_if_confident(delta-only)
  DB-->>P: enrich status

  P-->>UI: results + telemetry
  UI-->>U: Render results
```

### 2) Search: Video Upload (Frame Sampling)

```mermaid
sequenceDiagram
  participant U as Operator
  participant UI as Streamlit UI
  participant VP as Video Processor
  participant P as Pipeline/SearchEngine
  participant FE as FaceRecognitionEngine
  participant DB as FaceDatabase

  U->>UI: Upload video + click Search
  UI->>VP: decode video
  VP-->>UI: sampled frames (stride/max frames)

  loop for each sampled frame
    UI->>P: search_by_image(frame)
    P->>FE: detect + embed
    FE-->>P: embedding bundles
    P->>DB: search_similar_faces(bundles)
    DB-->>P: matches
    P-->>UI: per-frame results
  end

  UI->>UI: aggregate per-username across frames
  UI-->>U: render top usernames + evidence frames
```

### 3) Live Camera Recognition

```mermaid
sequenceDiagram
  participant U as Operator
  participant UI as Streamlit UI
  participant Live as Live Video Component
  participant P as LiveRecognition/Pipeline
  participant FE as FaceRecognitionEngine
  participant DB as FaceDatabase

  U->>UI: Start live camera
  UI->>Live: initialize stream (WebRTC when available)

  loop frames
    Live->>P: frame
    P->>FE: detect + embed (fast path if enabled)
    FE-->>P: embedding bundle
    P->>DB: search_similar_faces(bundle)
    DB-->>P: match candidates
    P-->>Live: overlay result + similarity
  end

  Note over P,DB: enrichment may batch/flush to reduce write frequency
```

---

## Database Backend Selection (DB_TYPE)

SocialVision’s `FaceDatabase` supports multiple backends.

- `DB_TYPE=local`: local JSON file
- `DB_TYPE=realtime`: Firebase Realtime Database
- `DB_TYPE=firestore`: Google Firestore
- `DB_TYPE=firebase`: auto-mode with fallback order:

```mermaid
flowchart TD
  A[DB_TYPE=firebase] --> B{Realtime DB available?}
  B -- yes --> RT[Use Firebase Realtime DB]
  B -- no --> C{Firestore available?}
  C -- yes --> FS[Use Firestore]
  C -- no --> L[Fallback to local JSON]
```

---

## Data Model (Conceptual)

At a high level, each stored face record includes:

- `username`
- `source` (profile_pic/post/story/reel/etc.)
- `embeddings` bundle:
  - `deepface`: 512-d vector (when enabled/available)
  - `dlib`: 128-d vector (fast path)
- metadata (timestamps, provenance)

Enrichment behavior:
- When a confident match is found, the system **adds only missing embedding keys** for that identity (delta-only update) rather than rewriting existing vectors.

---

## Notes for Diagram Rendering

- Mermaid diagrams render on GitHub and most Markdown viewers that support Mermaid.
- If you view this in an environment without Mermaid rendering, you’ll still see the diagram source blocks.

### Viewing Mermaid diagrams locally (VS Code)

If you only see the Mermaid blocks as code in VS Code:

1. Open the Markdown Preview (`Cmd+Shift+V`).
2. Install a Mermaid-capable preview extension (example: “Markdown Preview Mermaid Support”).

If you want a no-setup option, open this file on GitHub — GitHub renders Mermaid directly in the browser.
