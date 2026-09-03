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

Local video paths, `--results`, and `--output-dir` expand a leading `~` to the
current user's home directory (for example `/home/aigc`), including when the
argument is quoted and the shell does not expand it. Relative and absolute
paths still work; HTTP(S) URLs are passed through unchanged.

```bash
uv run python scripts/mark_video_timeline.py '~/videos/video.mp4' \
  --results '~/reviews/analysis.json' --output-dir '~/reviews/review-2'
```

In Docker, `~` refers to the user running the script inside the container, not
the SSH user's home directory on the host. Local input still requires `--results`.

### Person intervals and other findings

`/api/v1/analyze_media` merges consecutive sampled face hits by **category and
person name**. For example, hits at 6 and 7 seconds become one result:

```json
{
  "timestamp": "00:00:06.000~00:00:07.000",
  "category": "人物类别",
  "description": "人物 A"
}
```

A missing person in a sampled frame or a gap longer than the effective sampling
interval (at least one video frame) starts a new appearance. A single hit remains
`HH:MM:SS.mmm`. Different people
or categories never merge. Only results originating from face matching merge;
visual, OCR and other findings (including sexual/violent content) remain per
frame, even if their category and description happen to match a person's.

The tool accepts both these ranges and legacy single timestamps. It shows one
entry per distinct span, with purple interval bars and orange point markers;
overlaps are placed on separate lanes. Click to seek to the start, use previous/
next to visit marker starts, and filter by category. Person intervals stay
highlighted during playback within the range. Different people or endpoints at
the same start remain separate interval entries. MP4 chapters share one entry per start and
include the original ranges in their titles.

With `--results`, the script preserves the supplied points/ranges; it does not
guess person intervals from old point-only JSON. To obtain merged results,
analyze the video through the updated API (omit `--results`).

### Output files and interpretation

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
- `markers.json`: sorted point/interval groups (`time_ms`, `end_time_ms`), with
  exact duplicate findings within the same span removed.
- `chapters.ffmetadata`: chapter metadata in
  [FFmpeg's metadata format](https://ffmpeg.org/ffmpeg-formats.html#Metadata).

The source video is never modified. Existing output directories are refused;
omit `--output-dir` to create a uniquely named directory under
`data/media_reviews/` (ignored by Git). If remuxing fails, the saved response can
be reused with `--results` in a new directory. Unsupported MP4 stream codecs
cause an error instead of silent transcoding or dropping audio/subtitles.
Output chapter marks replace the source's chapter table only in the new copy.

Markers are **review cues from the API**, not verified violations. A person
range covers the first through last consecutive sampled hit, not exact
entry/exit times. A chapter ends at the next marker for navigation only, which
may differ from the range's end and does not establish a violation's duration.
No markers does not establish that the video is safe. Resolution is limited
by the API's sampling interval; non-person timestamps remain separate.
