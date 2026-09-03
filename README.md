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

## Video review timeline

`scripts/mark_video_timeline.py` calls `/api/v1/analyze_media` and exports a
new MP4 with chapter marks plus a self-contained HTML review timeline. Requires
`ffmpeg` / `ffprobe` on PATH and the project's Python dependencies. These tools
are also included in the Docker image.

```bash
uv run python scripts/mark_video_timeline.py \
  'http://10.252.25.251:18080/videos/test_pieces-1.mp4' \
  --output-dir data/media_reviews/test_pieces-1
```

Defaults: API URL `http://10.252.25.251:8000/api/v1/analyze_media`, 1-second
sampling, `top_k=10`, maximum distance `threshold=0.5`, 1200-second API read
timeout, and 2048 MiB download limit. Override with `--api-url`,
`--sample-interval`, `--top-k`, `--threshold`, `--timeout`, `--max-download-mb`.
The threshold is **distance**, not similarity; `0` means no filtering, matching
the existing API. Long analyses are not automatically retried.

To reuse an existing API response (also supports a local input video):

```bash
uv run python scripts/mark_video_timeline.py /path/to/video.mp4 \
  --results /path/to/analysis.json --output-dir data/media_reviews/review-2
```

Each output directory contains:

- `review.html`: open in a browser; click timeline points or list entries to
  seek, use previous/next marker buttons, or filter by category. Keep it beside
  `marked.mp4`; no CDN or network dependency is needed for playback.
  If serving over HTTP, use a static server with Range request support (such as
  Nginx or Starlette `StaticFiles`) so the browser can seek through the video.
- `marked.mp4`: stream-copied audio/video/subtitles, with new chapter marks.
  This does not burn labels into frames or re-encode media. Chapter display
  depends on the player; the HTML page provides a separate clickable timeline.
- `analysis.json`: original API response, retained for reuse.
- `markers.json`: sorted timestamp groups, with exact duplicates removed.
- `chapters.ffmetadata`: chapter metadata in
  [FFmpeg's metadata format](https://ffmpeg.org/ffmpeg-formats.html#Metadata).

The source video is never modified. Existing output directories are refused;
omit `--output-dir` to create a uniquely named directory under
`data/media_reviews/` (ignored by Git). If remuxing fails, the saved response can
be reused with `--results` in a new directory. Unsupported MP4 stream codecs
cause an error instead of silent transcoding or dropping audio/subtitles.
Output chapter marks replace the source's chapter table only in the new copy.

Markers are **review cues from the API**, not verified violations. A chapter
ends at the next marker for navigation only, not because that is a violation's
end time. No markers does not establish that the video is safe. Resolution is
limited by the API's sampling interval; closely spaced timestamps remain
separate so findings are not silently discarded.
