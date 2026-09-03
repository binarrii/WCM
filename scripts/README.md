# Scripts

Run the commands below from the project root, not from the `scripts/` directory.

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
