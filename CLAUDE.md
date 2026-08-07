# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

`wcm-facerec` is a FastAPI service for large-scale face recognition. It wraps the [InsightFace Server](https://github.com/deepinsight/insightface/tree/master/server) (`buffalo_m` model — SCRFD detector + ArcFace R50, 512-dim embeddings) over HTTP and persists embeddings in PostgreSQL with the `pgvector` extension. The whole stack ships via `docker compose` and is fronted by a Vue 3 webui for browsing/curating the registered face database.

The service was migrated from DeepFace on branch `feat/insightface-engine`. The DeepFace + TensorFlow + tf-keras dependencies are gone; no GPU work happens in this repo. Face detection and embedding live on the InsightFace Server (a separate GPU host or container).

## Run / build commands

### Local (host Python, no Docker)
```bash
uv sync --extra dev                              # install deps from pyproject.toml
uv run pytest                                    # unit suite only (RUN_LIVE gates live tests)
uv run pytest tests/test_ifs_adapter.py -v       # single file
uv run pytest -k 'search_multi_face'             # by keyword
RUN_LIVE=1 uv run pytest -m live tests/test_smoke_live.py -v   # live smoke test
uv run python main.py /path/to/image.jpg         # CLI face-detection visualizer
```

### Docker stack
```bash
docker compose up --build                        # build + start
docker compose logs -f api                       # tail API logs
docker compose restart api db                    # restart
```

API health check: `curl http://localhost:8000/api/v1/health` (returns `model`, `embedding_dim: 512`, `version`). The api container reaches InsightFace Server via `WCM_INSIGHTFACE_BASE_URL` (default `http://10.252.25.251:18097`).

The `compose.cuda.yaml` overlay from the DeepFace era is no longer used — there is no in-process TensorFlow in this repo.

### Lint / format
```bash
uv run ruff check src api main.py tests          # lint
uv run ruff format src api main.py tests         # format
pre-commit run --all-files                       # ruff + import-sort via pre-commit
```

Ruff line length is 100 (`pyproject.toml` `[tool.ruff]`); the vendored SDK at `src/wcm_facerec/vendor/` is excluded from linting.

### Webui
```bash
cd webui && pnpm install && pnpm dev
```
Vue 3 + Vite; talks to the API at the same origin. Nginx serves the static build from `/www` and proxies `/images/...` to `/tmp/wcm`.

## Architecture (big picture)

Three-process split — the API is **not** where face recognition runs:

```
  ┌────────────┐    HTTP /v1/collections/{id}/...    ┌────────────────────────┐
  │  api/      │ ──────────────────────────────────▶ │  InsightFace Server    │
  │  (FastAPI) │ ◀──────────────────────────────────  │  (GPU host,            │
  │  port 8000 │    JSON: detections + matches       │   buffalo_m, port 18097)│
  └─────┬──────┘                                     └────────────────────────┘
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

The Python SDK at `src/wcm_facerec/vendor/insightface_server/` is vendored (commit pinned in `_VENDORED.md`). The adapter at `src/wcm_facerec/ifs_adapter.py` wraps the vendored `Client` and owns every conversion between the InsightFace Server's JSON shapes and the legacy DeepFace-era dict shapes the rest of the code expects.

### InsightFace Server collections

Four collections are pre-provisioned on the server:

| Collection | Purpose |
|---|---|
| `all-persons` | Aggregated write target — every register lands here |
| `bad-artists` | Per-category mirror (`Person.type_ == "劣迹艺人"`) |
| `political` | Per-category mirror (`Person.type_ == "时政敏感"`) |
| `corrupt-officials` | Per-category mirror (`Person.type_ == "落马官员"`) |

`settings.insightface_category_collections` maps a Chinese `Person.type_` to its per-category collection id. `register_from_image` performs **two** `create_person` calls — one into `all-persons`, one into the mapped category collection. `search` reads from `all-persons` only. The mapping is also exposed in `webui/src/views/FaceDashboard.vue` (`filterType`).

### The `external_id` join

Each `FaceRecord` row carries a UUID. The matching InsightFace `Person.external_id` is set to that UUID at registration. When a search result comes back, the adapter reads `external_id` and joins back to the local `face_records` table to recover `id`, `created_at`, and `file_path`. Without this join, search results would lose the local DB pointer.

### Similarity ↔ distance duality

InsightFace Server's `/compare` and search return **similarity** in `[0,1]` (higher = better). Legacy callers — including webui code — expect **distance** (lower = better). The conversion happens **only** in `ifs_adapter.py`:

```python
match["distance"] = 1.0 - similarity
```

`search()` accepts a legacy cosine-distance threshold and converts internally to `min_similarity = max(0, 1 - threshold)`. `verify_faces` uses `settings.insightface_verify_similarity_threshold = 0.55` directly.

### Multi-face search fan-out

`POST /search` (and `/ws/search`) on an image with multiple faces: the adapter runs detection first, crops each face, fans out one `/search` per crop, and merges. Each match dict is tagged with `face_index` and `query_face_bbox` so the UI can show which query face produced it. `engine.search()` returns the flat sorted list (legacy shape); `engine.search_multi_face()` returns the structured `{faces, all_results}` shape.

## Key modules

### `src/wcm_facerec/`
- **`config.py`** — `Settings` (pydantic-settings, env prefix `WCM_`). New: `insightface_base_url`, `insightface_collection_id`, `insightface_timeout_s`, `insightface_verify_similarity_threshold`, `insightface_api_key`, `insightface_category_collections`. `embedding_dim` is a property returning the constant `512` (InsightFace Server ships a single fixed model). `warn_deprecated()` logs once if `WCM_DEEPFACE_API_URL` is still set.
- **`database.py`** — SQLAlchemy `Base`, `Person` and `FaceRecord` (UUID PKs, `embedding VECTOR(512)`). `FaceRecord.person_id` is a nullable FK.
- **`face_engine.py`** — `FaceEngine` class. Public API preserved from the DeepFace era (`detect_faces`, `generate_embedding`, `search`, `register_from_image`, `verify_faces`, `register_face`); new method `search_multi_face`. **Bytes-only at the engine boundary** — URLs are rejected by `_to_bytes` and must be downloaded upstream via `api.utils._download_url_safe`. The legacy `img_source` overload (path | bytes | np.ndarray) is normalized to bytes at the entry point. `_persist_image` saves under `/tmp/wcm/<category>/<name>_<md5>.<ext>` (hard-coded; `data_root` setting is unused for uploads).
- **`ifs_adapter.py`** — `InsightFaceAdapter` wraps the vendored `Client`. All conversions (similarity → distance, JSON bbox → `source_x/y/w/h`, IFS `Person.metadata` → flat match fields, fetch `matched_face_bbox` per match) happen here. Also owns the `min(w, h) ≥ 80` face-size filter and the top-3-by-area cap.
- **`vendor/insightface_server/`** — vendored Python SDK. Do not edit; upgrade by replacing contents and bumping `_VENDORED.md`.

### `api/`
- **`main.py`** — FastAPI entrypoint, gunicorn target (`api.main:app`). Mounts `/api/v1` prefix. Serves the webui static files from `/www` and proxies `/images/...` to `/tmp/wcm`.
- **`routes.py`** — All HTTP/WS routes.
- **`handlers.py`** — `_search_video_frames`, `_process_detect_sensitive`, `_process_detect_nsfw`, `_process_analyze_media`.
- **`utils.py`** — `_download_url_safe` (URL fetcher with size cap), `VIDEO_EXTENSIONS`.

### `scripts/` (batch jobs)
- `batch_register.py`, `extract_faces_from_directory.py`, `import_extracted_faces.py`, `update_names_from_excel.py` — bulk registration, name imports.
- `bulk_register_txt.py`, `import_sensitive_words.py` — name-list and sensitive-word imports.

### `tests/`
- `conftest.py` — `pytest_collection_modifyitems` gates `pytest.mark.live` tests on `RUN_LIVE=1`; provides `sample_image_bytes` / `face_like_image_bytes` fixtures and `FakeTransport` (httpx mock) so adapter/SDK code can exercise real serialization paths without hitting the server.
- `test_ifs_adapter.py`, `test_face_engine.py` — unit tests using `FakeTransport`.
- `test_smoke_live.py` — live round-trip against the real server, gated on `RUN_LIVE=1`.

## API endpoints (`/api/v1/...`)

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness + `model` + `embedding_dim` + `version` |
| POST | `/detect` | Detect faces in an image (file or URL); bounding boxes + embeddings |
| POST | `/register` | **410 Gone** — retired. Use `POST /face_records`. |
| POST | `/search` | Image/video → top-K similar faces by distance (multi-face fan-out for images) |
| WS | `/ws/search` | Async search with `taskId` callback pattern |
| POST | `/detect_sensitive` | OCR → WasuGuard sensitive-word check on image/video |
| WS | `/ws/detect_sensitive` | Async version |
| POST | `/detect_nsfw` | NSFW classification on image/video |
| WS | `/ws/detect_nsfw` | Async version |
| POST | `/analyze_media` | Combined: faces + sensitive + NSFW in one call |
| WS | `/ws/analyze_media` | Async version |
| GET | `/face_records` | List with pagination, `search`, `type` (Chinese categories) |
| GET | `/face_records/stats` | Counts by `Person.type_` |
| POST | `/face_records` | Create face + Person atomically; **requires exactly 1 detected face** (re-runs detection here, 400 on 0 or ≥2) |
| PUT | `/face_records/{record_id}` | Update face name and Person profile |
| DELETE | `/face_records/{record_id}` | Delete face record, Person, and image file |

## Conventions / gotchas

- **Person types are Chinese strings** (`劣迹艺人`, `时政敏感`, `落马官员`, plus free-form types like `普通人物`). `Person.type_` is mapped to the SQL keyword `type` — Python attribute is `type_` (trailing underscore), DB column is `type`. Don't rename either.
- **`FaceRecord.embedding` is `VECTOR(512)`** — fixed for buffalo_m/ArcFace R50. There is no per-model dim lookup; `config.embedding_dim` is a constant.
- **Two threshold scales are in play.** `insightface_verify_similarity_threshold = 0.55` (similarity scale, used by `verify_faces`). `verify_distance_threshold = 0.3` is the legacy cosine-distance setting — kept on the settings object as a no-op with a `DeprecationWarning`. `search()` callers still pass a distance threshold; conversion happens inside `engine.search_multi_face`. Don't mix the two scales when reading code.
- **Image upload path is `/tmp/wcm/<category>/<name>_<md5>.<ext>`**, not `/data/wcm`. The `data_root` setting is unused by the upload path. Nginx serves `/images/...` from `/tmp/wcm`.
- **`FaceEngine` accepts bytes only.** URLs raise `ValueError` from `_to_bytes`; the route layer must call `_download_url_safe` first. CLI tools (`main.py`, scripts that pass an `np.ndarray`) JPEG-encode before calling `detect_faces`.
- **`POST /face_records` enforces single-face registration** by calling `engine.detect_faces(image_bytes)` and returning 400 on 0 or ≥2 faces. The check happens in `routes.py`, separate from the actual register call.
- **Vendored SDK is pinned.** Don't `pip install insightface-server`; the SDK lives at `src/wcm_facerec/vendor/insightface_server/`. Upgrade by replacing the directory and bumping `_VENDORED.md` (upstream commit SHA recorded there).
- **Tests are gated.** `pytest` alone runs unit tests only. `RUN_LIVE=1 uv run pytest -m live` is required for `tests/test_smoke_live.py`. Unit tests use `tests/conftest.py::FakeTransport` to mock httpx — register it on the adapter's client when writing new tests.
- **No GPU stack in this repo.** `compose.cuda.yaml` and `Dockerfile.deepface.gpu` are DeepFace-era leftovers; the API container is CPU-only. All inference happens on the separate InsightFace Server host.

## Where to look when changing X

- **InsightFace base URL / auth / collection** → `src/wcm_facerec/config.py` (`insightface_*` settings) and `src/wcm_facerec/ifs_adapter.py` (`InsightFaceAdapter.__init__`). The vendored `Client` is constructed once per `FaceEngine`.
- **Adding a new per-category collection** → add an entry to `settings.insightface_category_collections` (config.py). The `register_from_image` fan-out picks it up automatically; nothing else changes.
- **Adding a new endpoint** → `api/routes.py` (HTTP) or a `_process_*` handler in `api/handlers.py` wired from routes. Heavy work should run via `await engine.<method>(...)`.
- **Adding a new Person category** → just insert strings; the `type` filter and stats endpoints query `Person.type_` directly. Update `webui/src/views/FaceDashboard.vue` `filterType` options if it should appear in the dashboard filter.
- **Match dict shape (legacy callers depend on these keys)** → `ifs_adapter.py:search_multi_face` and the per-match bbox enrichment loop in `face_engine.py:search_multi_face`. Keys: `id`, `name`, `person_id`, `matched_face_id`, `distance`, `person_name`, `occupation`, `type`, `category`, `remarks`, `file_path`, `source_x/y/w/h`, `face_index`, `query_face_bbox`. `distance = 1 - similarity`.
- **Face-size filter (`min(w, h) ≥ 80`)** → `ifs_adapter.py` (`detect`, `search_multi_face`); the `min_face_pixels` argument on the engine surfaces it.
- **Threshold scale changes** → `face_engine.py:search_multi_face` does `1 - threshold` → `min_similarity`; `verify_faces` compares against `insightface_verify_similarity_threshold` directly.
- **Vendored SDK upgrade** → `src/wcm_facerec/vendor/insightface_server/_VENDORED.md` records the upstream commit. Replace the directory contents, update the SHA, run `RUN_LIVE=1 uv run pytest -m live` to validate.