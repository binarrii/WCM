# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

`wcm-facerec` is a FastAPI service for large-scale face recognition. It wraps the [InsightFace Server](https://github.com/deepinsight/insightface/tree/master/server) (`buffalo_m` model — SCRFD detector + ArcFace R50, 512-dim embeddings) over HTTP. **InsightFace Server is the single source of truth** for persons, faces, and embeddings; the service ships via `docker compose` and is fronted by a Vue 3 webui for browsing/curating the registered face database.

The service was migrated from DeepFace on branch `feat/insightface-engine`. The DeepFace + TensorFlow + tf-keras dependencies are gone, and the former `FaceRecord` / `Person` SQL tables were dropped — there is no Postgres in this repo anymore. Face detection, embedding, person records, and search all live on the InsightFace Server.

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
docker compose up --build                        # build + start the API
docker compose logs -f api                       # tail API logs
docker compose restart api                       # restart
```

API health check: `curl http://localhost:8000/api/v1/health` (returns `model`, `embedding_dim: 512`, `version`). The api container reaches InsightFace Server via `WCM_INSIGHTFACE_BASE_URL` (default `http://10.252.25.251:18097`).

There is **no Postgres container** in `compose.yaml` anymore — Postgres was dropped along with the `FaceRecord` / `Person` tables. The `compose.cuda.yaml` overlay from the DeepFace era is also gone.

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

Two-process split — the API is **not** where face recognition runs, and there is no SQL store at runtime:

```
  ┌────────────┐    HTTP /v1/collections/{id}/...    ┌────────────────────────┐
  │  api/      │ ──────────────────────────────────▶ │  InsightFace Server    │
  │  (FastAPI) │ ◀──────────────────────────────────  │  (GPU host,            │
  │  port 8000 │    JSON: persons, faces, matches    │   buffalo_m, port 18097)│
  └────────────┘                                     └────────────────────────┘

  No SQLAlchemy / pgvector at runtime. The api container is stateless
  apart from /tmp/wcm (uploaded image files served via /images).
```

The Python SDK at `src/wcm_facerec/vendor/insightface_server/` is vendored (commit pinned in `_VENDORED.md`). The adapter at `src/wcm_facerec/ifs_adapter.py` wraps the vendored `Client` and owns every conversion between the InsightFace Server's JSON shapes and the legacy DeepFace-era dict shapes the rest of the code expects.

### InsightFace Server collections

Four collections are pre-provisioned on the server:

| Collection | Purpose |
|---|---|
| `all-persons` | Aggregated write target — every register lands here. All list/stats/update/delete operate on this collection. |
| `bad-artists` | Per-category mirror (`metadata.type == "劣迹艺人"`) |
| `political` | Per-category mirror (`metadata.type == "时政敏感"`) |
| `corrupt-officials` | Per-category mirror (`metadata.type == "落马官员"`) |

`settings.insightface_category_collections` maps a Chinese `category` to its per-category collection id. `register_from_image` performs **two** `create_person` calls — one into `all-persons`, one into the mapped category collection. The per-category duplicate carries `external_id=<aggregate Person.id>` so an admin backfill can correlate them later. `search` reads from `all-persons` only. The mapping is also exposed in `webui/src/views/FaceDashboard.vue` (`filterType`).

The webui's record `id` is the **aggregate IFS Person.id** (from `all-persons`), not the per-category Person.id. Treat the id as an opaque string — the webui does not parse it.

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
- **`config.py`** — `Settings` (pydantic-settings, env prefix `WCM_`). Owns `insightface_base_url`, `insightface_collection_id`, `insightface_timeout_s`, `insightface_verify_similarity_threshold`, `insightface_api_key`, `insightface_category_collections`, plus the legacy `verify_distance_threshold = 0.3` no-op. `embedding_dim` is a property returning the constant `512` (InsightFace Server ships a single fixed model). `warn_deprecated()` logs once if `WCM_DEEPFACE_API_URL` is still set.
- **`database.py`** — Reduced to `SensitiveWord` + engine/session plumbing only (used by `scripts/import_sensitive_words.py`). The former `Person` and `FaceRecord` SQLAlchemy models are gone.
- **`face_engine.py`** — `FaceEngine` class. Public API: `detect_faces`, `generate_embedding`, `search`, `search_multi_face`, `register_from_image`, `verify_faces`. **Bytes-only at the engine boundary** — URLs are rejected by `_to_bytes` and must be downloaded upstream via `api.utils._download_url_safe`. The legacy `img_source` overload (path | bytes | np.ndarray) is normalized to bytes at the entry point. `register_from_image` returns a flat record-item dict from IFS (no SQL row is written). `_persist_image` saves uploaded images under `/tmp/wcm/<category>/<name>_<md5>.<ext>` (hard-coded; `data_root` setting is unused for uploads).
- **`ifs_adapter.py`** — `InsightFaceAdapter` wraps the vendored `Client`. All conversions (similarity → distance, JSON bbox → `source_x/y/w/h`, IFS `Person.metadata` → flat record-item dict, fetch `matched_face_bbox` per match) happen here. Owns the `min(w, h) ≥ 80` face-size filter and the top-3-by-area cap. CRUD methods: `register_person`, `delete_person`, `list_persons` (cursor pagination), `get_person`, `update_person`. The shared `_person_to_item()` helper flattens an IFS Person into the legacy record-item shape (`id`, `name`, `category`, `occupation`, `type`, `remarks`, `file_path`, `created_at`, `face_count`).
- **`vendor/insightface_server/`** — vendored Python SDK. Do not edit; upgrade by replacing contents and bumping `_VENDORED.md`.

### `api/`
- **`main.py`** — FastAPI entrypoint, gunicorn target (`api.main:app`). Empty lifespan (no DB to init). Mounts `/api/v1` prefix. Serves the webui static files from `/www` and serves `/images/...` from `/tmp/wcm`.
- **`routes.py`** — All HTTP/WS routes. The five `/face_records*` endpoints are IFS-only — no SQLAlchemy imports remain. `_path_to_image_url` and `_item_with_person` are the shared item-shape helpers.
- **`handlers.py`** — `_search_video_frames`, `_process_detect_sensitive`, `_process_detect_nsfw`, `_process_analyze_media`.
- **`utils.py`** — `_download_url_safe` (URL fetcher with size cap), `VIDEO_EXTENSIONS`.

### `scripts/` (standalone offline tooling — not on the API path)
- `batch_register.py` — defines its own legacy `Base`/`Person`/`FaceRecord` (with `VECTOR(512)`) and writes only to a local Postgres. Standalone; not used by `api/`.
- `bulk_register_txt.py` — uses `_persist_image` only (filesystem).
- `import_extracted_faces.py`, `import_sensitive_words.py`, `extract_faces_from_directory.py`, `update_names_from_excel.py` — bulk tooling, all standalone.

### `tests/`
- `conftest.py` — `pytest_collection_modifyitems` gates `pytest.mark.live` tests on `RUN_LIVE=1`; provides `sample_image_bytes` / `face_like_image_bytes` fixtures and `FakeTransport` (httpx mock) so adapter/SDK code can exercise real serialization paths without hitting the server.
- `test_ifs_adapter.py` — adapter-level unit tests; covers `detect`, `embed`, `search`, `search_multi_face`, `compare`, `register_person`, `delete_person`, `get_face_bbox`, plus the new CRUD trio `list_persons` / `get_person` / `update_person`.
- `test_face_engine.py` — engine-level unit tests with `FakeTransport` mocked at the SDK level. No DB required.
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
| GET | `/face_records` | List with pagination + `search=` (name) + `type=` (Chinese category filter). All from IFS `all-persons`. |
| GET | `/face_records/stats` | Counts of `metadata.type` from IFS `all-persons` (`total`, `bad_artists`, `political`, `officials`) |
| POST | `/face_records` | Enroll a face into IFS; **requires exactly 1 detected face** (re-runs detection here, 400 on 0 or ≥2). Dual-writes to per-category collection when applicable. |
| PUT | `/face_records/{record_id}` | PATCH IFS Person name + metadata (occupation/type/remarks). Per-category mirror is **not** re-keyed on type change. |
| DELETE | `/face_records/{record_id}` | Delete aggregate IFS Person + its `/tmp/wcm` image file. Per-category mirrors are left stale (acceptable leak). |

## Conventions / gotchas

- **Person types are Chinese strings** (`劣迹艺人`, `时政敏感`, `落马官员`, plus free-form types like `普通人物`). Stored on IFS Person as `metadata.type` — there's no `Person.type_` Python attribute or DB column anymore.
- **Identifier `id` is the IFS aggregate `Person.id`** (from `all-persons`). The webui treats it as an opaque string — do not introduce UUID parsing.
- **Embedding dim is fixed at 512** (buffalo_m / ArcFace R50). `config.embedding_dim` is a constant property.
- **Two threshold scales are in play.** `insightface_verify_similarity_threshold = 0.55` (similarity scale, used by `verify_faces`). `verify_distance_threshold = 0.3` is the legacy cosine-distance setting — kept on the settings object as a no-op with a `DeprecationWarning`. `search()` callers still pass a distance threshold; conversion happens inside `engine.search_multi_face`. Don't mix the two scales when reading code.
- **List endpoint `search=` matches names server-side only.** IFS `/persons` does not support metadata-based filtering, so `type=` is post-filtered in Python. The legacy free-text search over `occupation`/`remarks` is gone — webui's free-text input now matches names only.
- **`total` on the list response is the page count, not a global count.** The dashboard pagination uses `has_more` exclusively (`FaceDashboard.vue:117`). ~5k records fit in a few IFS pages.
- **Image upload path is `/tmp/wcm/<category>/<name>_<md5>.<ext>`**, not `/data/wcm`. The `data_root` setting is unused by the upload path. Nginx serves `/images/...` from `/tmp/wcm`.
- **`FaceEngine` accepts bytes only.** URLs raise `ValueError` from `_to_bytes`; the route layer must call `_download_url_safe` first. CLI tools (`main.py`, scripts that pass an `np.ndarray`) JPEG-encode before calling `detect_faces`.
- **`POST /face_records` enforces single-face registration** by calling `engine.detect_faces(image_bytes)` and returning 400 on 0 or ≥2 faces. The check happens in `routes.py`, separate from the actual register call.
- **`PUT /face_records/{id}` does not propagate to per-category mirrors.** To recategorize a record, delete + re-register. (Documented as an acceptable trade-off — fixes require a per-collection PATCH keyed by `external_id`.)
- **`DELETE /face_records/{id}` does not clean up per-category mirrors.** Stale per-category duplicates from legacy batch imports remain on the server until an admin backfill runs. (Acceptable leak.)
- **Vendored SDK is pinned.** Don't `pip install insightface-server`; the SDK lives at `src/wcm_facerec/vendor/insightface_server/`. Upgrade by replacing the directory and bumping `_VENDORED.md` (upstream commit SHA recorded there).
- **Tests are gated.** `pytest` alone runs unit tests only — no Postgres needed. `RUN_LIVE=1 uv run pytest -m live` is required for `tests/test_smoke_live.py`. Unit tests use `tests/conftest.py::FakeTransport` to mock httpx.
- **No GPU stack in this repo.** `compose.cuda.yaml` and `Dockerfile.deepface.gpu` are DeepFace-era leftovers; the API container is CPU-only. All inference happens on the separate InsightFace Server host.
- **Standalone scripts (`scripts/*.py`) may still import `database.Person` / `database.FaceRecord`** for their own offline use. Those are legacy batch jobs and not on the runtime API path — they are not affected by the IFS-as-source-of-truth refactor and may break separately.

## Where to look when changing X

- **InsightFace base URL / auth / collection** → `src/wcm_facerec/config.py` (`insightface_*` settings) and `src/wcm_facerec/ifs_adapter.py` (`InsightFaceAdapter.__init__`). The vendored `Client` is constructed once per `FaceEngine`.
- **Adding a new per-category collection** → add an entry to `settings.insightface_category_collections` (config.py). The `register_from_image` fan-out picks it up automatically.
- **Adding a new endpoint** → `api/routes.py` (HTTP) or a `_process_*` handler in `api/handlers.py` wired from routes. Heavy work should run via `await engine.<method>(...)`.
- **Adding a new Person category** → just insert strings; the `type` filter and stats endpoints group by `metadata.type`. Update `webui/src/views/FaceDashboard.vue` `filterType` options if it should appear in the dashboard filter and add a stats tile in `FaceDashboard.vue` if it should be counted in `/stats`.
- **Record-item shape (legacy callers depend on these keys)** → `ifs_adapter.py:_person_to_item` and the CRUD methods (`list_persons`, `get_person`, `update_person`). Keys: `id` (IFS Person.id), `name`, `external_id`, `face_count`, `created_at`, `updated_at`, `category`, `occupation`, `type`, `remarks`, `file_path`. `id` is **never** `external_id` — the local-SQL join is gone.
- **Match dict shape (search results)** → `ifs_adapter.py:search_multi_face` and the per-match bbox enrichment loop in `face_engine.py:search_multi_face`. Keys: `id` (IFS Person.id), `name`, `person_id`, `matched_face_id`, `distance`, `person_name`, `occupation`, `type`, `category`, `remarks`, `file_path`, `source_x/y/w/h`, `face_index`, `query_face_bbox`. `distance = 1 - similarity`.
- **Face-size filter (`min(w, h) ≥ 80`)** → `ifs_adapter.py` (`detect`, `search_multi_face`); the `min_face_pixels` argument on the engine surfaces it.
- **Threshold scale changes** → `face_engine.py:search_multi_face` does `1 - threshold` → `min_similarity`; `verify_faces` compares against `insightface_verify_similarity_threshold` directly.
- **Vendored SDK upgrade** → `src/wcm_facerec/vendor/insightface_server/_VENDORED.md` records the upstream commit. Replace the directory contents, update the SHA, run `RUN_LIVE=1 uv run pytest -m live` to validate.