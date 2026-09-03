# WCM Face Recognition

WCM is a FastAPI + Vue 3 service for face-library management and media moderation.
Face detection, embeddings, persons and similarity search are provided by an
external InsightFace Server (buffalo_m, 512-dimensional embeddings).

## Architecture

```text
Browser ──► FastAPI/Vue container ──► InsightFace Server
                    │
                    ├──► moderation model gateway (OCR / guard / caption)
                    └──► wcm-images Docker volume (/tmp/wcm)
```

InsightFace Server is the source of truth for persons and faces. Every classified
record is written to `all-persons` and mirrored to its category collection using
the same Person id. Uploaded source images are kept in the persistent
`wcm-images` volume.

## Docker

Copy the example configuration and set the external service addresses:

```bash
cp .env.example .env
docker compose up --build
```

Open <http://localhost:8000>. Readiness is available at
<http://localhost:8000/api/v1/health>; it returns HTTP 503 when InsightFace is
unavailable.

The image volume survives container recreation:

```bash
docker volume inspect wcm_wcm-images
```

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
- Face-record listing uses opaque cursor pagination. Pass the returned
  `next_cursor` unchanged on the next request.
- `type` is the canonical classification field and also selects the category
  collection.
- Search accepts legacy distance thresholds (lower is better); the adapter
  converts InsightFace similarity to `distance = 1 - similarity`.
- Search collapses multiple enrolled-sample hits for the same person and query
  face (and video frame) into the highest-scoring hit. `matched_face_id`, the
  score and source bounding box describe that hit. `image_url` is the person's
  gallery cover, not necessarily the matched sample. `image_urls` contains all
  available, distinct gallery URLs, just as in `/face_records`; it does not mean
  every gallery image met the threshold. WebUI shows one card per person with
  their highest similarity and the complete gallery, including full-size navigation.
- For a minimum similarity of 30%, send `threshold=0.7`. `top_k` still limits
  upstream sample candidates, so the deduplicated person count may be smaller.
