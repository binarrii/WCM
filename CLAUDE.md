# Repository guide

## Purpose

`wcm-facerec` is a FastAPI service with a Vue 3 dashboard for face-library
management and media moderation. InsightFace Server is the only store for
persons, faces and 512-dimensional embeddings. Source images are persisted at
`/tmp/wcm`, backed by the `wcm-images` Docker volume.

## Commands

```bash
uv sync --extra dev
uv run pytest
uv run ruff check src api main.py tests
RUN_LIVE=1 uv run pytest -m live tests/test_smoke_live.py -v

cd webui
pnpm install
pnpm dev
pnpm build

docker compose up --build
```

The default pytest collection is restricted to `tests/`; live tests are skipped
unless `RUN_LIVE=1`.

## Runtime architecture

- `api/main.py`: app factory, CORS, image mount and optional built Vue SPA.
- `api/routes.py`: detection, search, WebSocket and media-analysis routes.
- `api/face_records.py`: IFS-backed record CRUD, stats and cursor pagination.
- `api/handlers.py`: video sampling and moderation-model orchestration.
- `src/wcm_facerec/face_engine.py`: async public engine and cross-collection
  consistency/compensation logic.
- `src/wcm_facerec/ifs_adapter.py`: synchronous vendored-SDK adapter and all
  IFS/legacy response-shape conversions.
- `webui/src/views/FaceDashboard.vue`: dashboard.
- `webui/src/services/`: same-origin API client.

The API process performs no local face inference. It calls the external
InsightFace Server configured by `WCM_INSIGHTFACE_BASE_URL`.

## Collections and identifiers

`all-persons` is the aggregate collection. Classified records are mirrored to:

| Type | Collection |
|---|---|
| 劣迹艺人 | bad-artists |
| 时政敏感 | political |
| 落马官员 | corrupt-officials |

`metadata.type` is canonical; `metadata.category` mirrors it for compatibility.
New category mirrors reuse the aggregate Person id and also set
`external_id=<aggregate id>`. Legacy mirrors may have a different id, so CRUD
falls back to finding them by `external_id`. API responses always expose the
aggregate id.

Registration compensates the aggregate write when mirror creation fails. Updates
prepare/migrate the category mirror and roll back on failure. Deletes remove
category mirrors before the aggregate and attempt to restore deleted mirrors if
the aggregate delete fails.

## Pagination

`GET /api/v1/face_records` accepts `cursor`, `limit`, `search` and `type`.
The returned `next_cursor` is opaque and may contain an offset within an IFS page
for metadata-filtered views. Clients must reset it when search/type changes and
must not parse it.

## Important contracts

- InsightFace similarity is higher-is-better. Public search results also expose
  legacy `distance = 1 - similarity`.
- `FaceEngine` accepts paths, bytes and ndarrays; URL downloading belongs to the
  route layer.
- Registration requires exactly one detected face.
- The readiness endpoint contacts InsightFace and returns 503 when it is down.
- The vendored SDK under `src/wcm_facerec/vendor/` is pinned; do not edit it
  directly.

## Tests

Adapter and engine tests use `tests/conftest.py::FakeTransport`. API tests cover
readiness, CRUD delegation, category identifiers, cursor continuation and video
search. Add route tests for behavior changes; do not make unit tests depend on the
live model services.
