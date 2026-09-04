# WCM Face Recognition

WCM is a FastAPI + Vue 3 service for face-library management and media moderation.
Face detection, embeddings, persons and similarity search are provided by an
external InsightFace Server (buffalo_m, 512-dimensional embeddings).

The bundled WebUI provides three work areas:

- **People management** — maintain categorized person profiles, enroll multiple
  face images under one person id, search by name, and search by an uploaded face.
- **Video review** — submit a remote video for analysis, review findings on an
  interactive timeline, filter by category, and download the JSON result. The
  parameter panel collapses automatically after a completed result loads and can
  always be expanded manually.
- **Review tasks** — search persisted analysis jobs, reopen a completed task with
  its parameters and results preloaded, download one JSON result or a ZIP of
  multiple selected results, and delete tasks individually or in batches.

## Architecture

```text
Browser ──► FastAPI/Vue container ──► InsightFace Server
                    │
                    ├──► moderation model gateway (OCR / guard / caption)
                    ├──► image library (/tmp/wcm on the host)
                    └──► MySQL (review-task parameters, status and results)
```

InsightFace Server is the source of truth for persons and faces. Every classified
record is written to `all-persons` and mirrored to its category collection using
the same Person id. Uploaded source images are kept in the persistent
image directory. Review tasks are stored separately in MySQL and survive API
container recreation.

## Docker

Copy the example configuration and set the external service addresses:

```bash
cp .env.example .env
docker compose up --build
```

Open <http://localhost:8000>. Readiness is available at
<http://localhost:8000/api/v1/health>; it returns HTTP 503 when InsightFace is
unavailable. The health response also reports review-task storage status.

Compose starts both the API/WebUI container and MySQL 8.4. MySQL is published on
loopback port `13306` by default to avoid conflicting with a host installation;
override it with `WCM_MYSQL_PORT`. Review-task data is retained in the named
volume `review-tasks-mysql-data`. Change the example database passwords before a
production deployment.

The face-image library is mounted from `WCM_IMAGE_DIR` (default `/tmp/wcm`) and
therefore also survives container recreation:

```bash
docker compose ps
docker volume inspect wcm_review-tasks-mysql-data
```

If Docker builds on a host whose bridge network cannot resolve package registries,
set `WCM_BUILD_NETWORK=host` before running `docker compose build`.

## WebUI workflow

Open <http://localhost:8000> and use the left sidebar to switch between people,
video review and review tasks. Review-task rows are intentionally compact so more
history fits on one page. Search matches task ids, video URLs and recorded failure
messages; the status selector further filters processing, completed and failed
tasks.

Clicking a task row opens the video-review page and restores its video URL,
sampling interval, candidate count, similarity threshold and stored findings.
Only completed tasks have downloadable results. A row download produces
`analysis-<task-id>.json`; selecting completed rows enables a single ZIP download
containing one JSON file per task. Deletion and other destructive actions use the
theme-aware confirmation dialog rather than the browser's native prompt.

## Local development

```bash
uv sync --extra dev
uv run uvicorn api.main:app --reload
```

In another terminal:

```bash
cd webui
pnpm install
pnpm dev
```

Vite proxies `/api` and `/images` to the local API on port 8000.

## Verification

```bash
uv run pytest
uv run ruff check src api main.py tests
cd webui && pnpm build
```

The default suite does not contact external services. To run the live InsightFace
smoke tests:

```bash
RUN_LIVE=1 uv run pytest -m live tests/test_smoke_live.py -v
```

## API notes

- API prefix: `/api/v1`
- `POST /analyze_media` persists the submitted video URL and parameters before
  analysis, then records the completed result or failure state in MySQL. The
  legacy response body remains the analysis result, and `X-Review-Task-ID`
  identifies the persisted task.
- Review-task management endpoints are:
  - `GET /review_tasks` for paginated search and status filtering;
  - `GET /review_tasks/{task_id}` for parameters and full stored results;
  - `GET /review_tasks/{task_id}/results/download` for one JSON attachment;
  - `POST /review_tasks/results/download` with `{"ids": [...]}` for a ZIP;
  - `DELETE /review_tasks/{task_id}` and `DELETE /review_tasks` with
    `{"ids": [...]}` for single and batch deletion.
- Face-record listing uses opaque cursor pagination. Pass the returned
  `next_cursor` unchanged on the next request.
- `/face_records/stats` retains the person counts (`total`, `bad_artists`,
  `political`, `officials`) and adds corresponding `*_images` fields from each
  collection's registered face-sample count. Registration accepts one face per
  image. The dashboard shows these image totals below the person counts;
  the aggregate count is read directly, without adding category mirrors again.
- `type` is the canonical classification field and also selects the category
  collection.
- Search accepts legacy distance thresholds (lower is better); the adapter
  converts InsightFace similarity to `distance = 1 - similarity`.
  HTTP and WebSocket search/media analysis, engine search, adapter search,
  and face verification default to `0.5` (50% minimum similarity / 0.5 maximum
  distance). Explicit search thresholds and the verification setting
  `WCM_INSIGHTFACE_VERIFY_SIMILARITY_THRESHOLD` still override their defaults.
  Detection settings and the import script's exact-image deduplication threshold
  are separate and unchanged.
- Search collapses multiple enrolled-sample hits for the same person and query
  face (and video frame) into the highest-scoring hit. `matched_face_id`, the
  score and source bounding box describe that hit. `image_url` is the person's
  gallery cover, not necessarily the matched sample. `image_urls` contains all
  available, distinct gallery URLs, just as in `/face_records`; it does not mean
  every gallery image met the threshold. WebUI shows one card per person with
  the complete gallery, including full-size navigation.
- Image `/search` also returns `image_similarities`, keyed by gallery URL. These
  are individual, unweighted comparisons against the result's query face (using
  the same crop as search), not the person's highest score. WebUI updates the
  displayed similarity when switching images. An unreadable image or failed
  comparison gets `null` (shown as “暂无评分”), never another image's score.
  Gallery scores below the threshold remain visible; person ranking/filtering
  still uses the original best search score. Video/stream responses do not
  recompute gallery scores for every frame.
- For a minimum similarity of 30%, send `threshold=0.7`. `top_k` still limits
  upstream sample candidates, so the deduplicated person count may be smaller.

## Script tools

See [scripts/README.md](scripts/README.md) for detailed tool usage.
