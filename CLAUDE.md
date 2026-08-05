# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

`wcm-facerec` is a FastAPI service for large-scale face recognition. It wraps the [DeepFace](https://github.com/serengil/deepface) library (running as a separate GPU-backed API container) and persists embeddings in PostgreSQL with the `pgvector` extension. The whole stack ships via `docker compose` and is fronted by a Vue 3 webui for browsing/curating the registered face database.

Two distinct deployments exist in this repo:

- **`compose.yaml`** — main API (`api/`), pgvector, and Nginx static-file serving. This is the one in active development.
- **`compose.cuda.yaml`** — overlay for an NVIDIA host: installs `tensorflow[and-cuda]`, exposes GPUs, sets `TF_FORCE_GPU_ALLOW_GROWTH=true`. Used in conjunction with `compose.yaml`.
- **`docker-compose-deepface.yml`** + **`Dockerfile.deepface.gpu`** — a *separate* deployment at `/opt/binarii/DeepFace` (per `README_deepface.md`): the raw DeepFace API with two replicas behind Nginx load balancing and NVIDIA MPS for VRAM isolation. Not built from this repo's source; it's referenced as a downstream dependency.

## Run / build commands

### Local (host Python, no Docker)
```bash
uv sync --extra dev          # install deps from pyproject.toml
uv run pytest -x            # run tests
uv run python main.py /path/to/image.jpg   # CLI face-detection visualizer
```

### Docker stack (the standard way)
```bash
docker compose up --build                          # CPU image
docker compose -f compose.yaml -f compose.cuda.yaml up --build   # CUDA overlay
docker compose exec api python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"   # verify GPU
docker compose logs -f api                         # tail API logs
docker compose restart api db                     # restart
```

API health check: `curl http://localhost:8000/api/v1/health`.

The first boot downloads DeepFace model weights to the volume mounted at `/home/aigc/.deepface/weights` (see `compose.yaml`). The api container needs a sibling DeepFace API reachable at `WCM_DEEPFACE_API_URL` (default `http://127.0.0.1:5000`) for embedding generation.

### Webui
```bash
cd webui && pnpm install && pnpm dev
```
The webui (Vue 3 + Vite, in `webui/`) talks to the API at the same origin and serves from the static files mounted into Nginx.

## Architecture (big picture)

Two-process split — the API is **not** where face recognition runs:

```
  ┌────────────┐    HTTP POST /represent, /search    ┌────────────────────┐
  │  api/      │ ──────────────────────────────────▶ │  DeepFace API     │
  │  (FastAPI) │ ◀──────────────────────────────────  │  (GPU container,  │
  │  port 8000 │    JSON: embeddings + matches        │   port 5000)      │
  └─────┬──────┘                                       └────────────────────┘
        │ SQLAlchemy
        ▼
  ┌──────────────────────────────┐
  │  Postgres + pgvector         │
  │  - persons (UUID PK)         │
  │  - face_records (UUID PK,    │
  │    embedding VECTOR(512),    │
  │    person_id FK)             │
  └──────────────────────────────┘
```

### Process boundaries (do not collapse them)
- **`api/`** (this repo, FastAPI): orchestrates requests, does I/O, talks to Postgres, and forwards image bytes to the DeepFace API for actual inference. Embeddings are *received*, never *computed* here.
- **DeepFace API** (separate container, port 5000): loads TensorFlow, runs face detection (`fastmtcnn`) and embedding (`Facenet512` by default). All GPU work happens here.
- **`/tmp/wcm/<category>/<name>_<md5>.<ext>`**: where uploaded images are persisted by `_persist_image()` in `face_engine.py`. The `/images/...` route served by Nginx fronts this directory. This path is hard-coded; don't assume `/data/wcm` (the `data_root` setting) is actually written to — face uploads go to `/tmp/wcm`.

### Key modules (`src/wcm_facerec/`)
- **`config.py`** — `Settings` (pydantic-settings, env prefix `WCM_`, reads `.env`). Owns `database_url`, `embedding_dim` (looked up from model name), `verify_distance_threshold` (default `0.3`, tighter than DeepFace's built-in). All env vars and `.env.example` document the contract.
- **`database.py`** — SQLAlchemy `Base`, `Person` and `FaceRecord` models (UUID PKs, `embedding VECTOR(512)`). `FaceRecord.person_id` is a nullable FK; a face can exist without a person profile. `get_session()` returns a scoped session.
- **`face_engine.py`** — `FaceEngine` class: `detect_faces`, `generate_embedding`, `search`, `register_from_image`, `_search_video_frames`, `_persist_image`. **All DeepFace calls go out over `httpx.AsyncClient`** to `settings.deepface_api_url` — there is no in-process TensorFlow. Image bytes are base64-wrapped into a `data:image/jpeg;base64,...` data URL before posting.

### Key modules (`api/`)
- **`main.py`** — FastAPI app entrypoint, gunicorn target (`api.main:app`). Mounts `/api/v1` prefix via the router. Serves the webui static files from `/www` and proxies `/images/...` to `/tmp/wcm`.
- **`routes.py`** — All HTTP/WS routes. See "Endpoints" below.
- **`handlers.py`** — Heavy lifters for media processing: `_search_video_frames`, `_process_detect_sensitive` (OCR + WasuGuard), `_process_detect_nsfw`, `_process_analyze_media`.
- **`utils.py`** — `_download_url_safe` (URL fetcher with size cap), `VIDEO_EXTENSIONS`.

### Scripts (`scripts/`) — batch jobs
- **`batch_register.py`** — bulk-register faces from a directory tree.
- **`extract_faces_from_directory.py`** — uses local DeepFace to extract face crops, no API.
- **`import_extracted_faces.py`** — imports pre-extracted crops into the DB.
- **`update_names_from_excel.py`** — joins DB rows with name spreadsheets (`libface_*.xls`).
- **`bulk_register_txt.py`**, **`import_sensitive_words.py`** — name-list and sensitive-word imports.

## API endpoints (`/api/v1/...`)

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness + model/embedding_dim/version |
| POST | `/detect` | Detect faces in an image (file or URL); returns bounding boxes only, no embedding |
| POST | `/register` | Generate embedding + persist one face (no Person attached) |
| POST | `/search` | Image/video → top-K similar faces by distance |
| WS | `/ws/search` | Async search with `taskId` callback pattern |
| POST | `/detect_sensitive` | OCR → WasuGuard sensitive-word check on image/video |
| WS | `/ws/detect_sensitive` | Async version |
| POST | `/detect_nsfw` | NSFW classification on image/video |
| WS | `/ws/detect_nsfw` | Async version |
| POST | `/analyze_media` | Combined: faces + sensitive + NSFW in one call |
| WS | `/ws/analyze_media` | Async version |
| GET | `/face_records` | List with pagination, `search`, `type` (Chinese categories: 劣迹艺人 / 时政敏感 / 落马官员) |
| GET | `/face_records/stats` | Counts by `Person.type_` |
| POST | `/face_records` | Create face + Person atomically; **requires exactly 1 detected face** |
| PUT | `/face_records/{record_id}` | Update face name and Person profile |
| DELETE | `/face_records/{record_id}` | Delete face record, Person, and image file |

## Conventions / gotchas

- **Person types are Chinese strings** (`劣迹艺人`, `时政敏感`, `落马官员`). The `Person.type_` column is mapped to the SQL keyword `type` — Python attribute is `type_` (trailing underscore), DB column is `type`. Don't rename either.
- **`FaceRecord.embedding` is `VECTOR(512)`** — sized for `Facenet512`. If you switch the DeepFace model, run a migration to resize the column AND update the `embedding_dim` lookup in `config.py` (`_EMBEDDING_DIMS`).
- **`verify_distance_threshold = 0.3`** (config.py:40) — tighter than DeepFace's default ~0.30 for Facenet512+cosine to reject borderline look-alikes. Don't relax it without testing.
- **Image upload path is `/tmp/wcm/<category>/<name>_<md5>.<ext>`**, not `/data/wcm`. The `data_root` setting is unused by the upload path. Nginx serves `/images/...` from `/tmp/wcm`.
- **`POST /face_records` runs face detection locally** (in-process via `DeepFace.extract_faces`) and rejects with 400 if the image contains 0 or ≥2 faces — it enforces single-face registration by re-running detection here, separate from the embedding call. Don't bypass it.
- **`/search` and `/ws/search` accept a video URL** (any extension in `VIDEO_EXTENSIONS` from `api/utils.py`) and use `sample_interval` (seconds). For image URLs, response is `{results, query_embedding_dim}`. For video URLs, response also includes `frames_processed`.
- **The DeepFace API base URL is `http://127.0.0.1:5000` by default** — fine on the host where the DeepFace container runs, but inside the api container you must set `WCM_DEEPFACE_API_URL=http://host.docker.internal:5000` (or similar) so the API container can reach the DeepFace service.
- **No covering tests** (`codegraph_explore` flags every FaceEngine/handler call site as untested). Tests under `dev` extras are configured but not authored yet.

## Where to look when changing X

- Adding a new DeepFace model → `src/wcm_facerec/config.py:_EMBEDDING_DIMS` + DB migration to resize `face_records.embedding`. The `VECTOR(512)` is the default model dimension.
- Adding a new endpoint → `api/routes.py` (HTTP) or `_process_*` handler in `api/handlers.py` + wire from routes.
- Adding a new Person category → just insert strings; the `type` filter and stats endpoints query `Person.type_` directly with the new string. Update `webui/src/views/FaceDashboard.vue` `filterType` options.
- Changing the face crop filter → `MIN_FACE_PIXELS` (`main.py`, `scripts/batch_register.py`) and the `min(w, h) < 80` checks in `api/routes.py` and `src/wcm_facerec/face_engine.py` — three places to keep aligned.