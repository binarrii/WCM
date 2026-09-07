# WCM WebUI

The WebUI has two hash-routed pages in its left navigation:

- `#/people`: person-library management and face search.
- `#/video`: remote video analysis and an interactive review timeline. It calls
  `/api/v1/analyze_media`, supports category filters and JSON import/export, and
  keeps the standalone chapter-export tool under `scripts/` unchanged.

## Development

```bash
pnpm install
pnpm dev
pnpm test
pnpm build
```

## Face positions during review

New analysis results preserve normalized query-frame coordinates in
`face_samples: [{ time_ms, bbox: { x, y, w, h }, similarity }]` on each person
finding. Each coordinate is relative to the analyzed frame, including any bars
already encoded into the source video. Timeline merging retains each sampled
position; it does not create a tracking path.

The player shows four-corner markers while paused within 45 ms of a measured
sample, uses only the nearest sample when samples are dense, and hides positions
during playback/seeking. Hover or keyboard-focus shows candidate names and
similarities; selecting a candidate highlights its finding. Clicking a finding
pauses at its first sample. Previous/next sample buttons step through measured
frames under the current category filter. Fullscreen uses the player wrapper
so the overlay remains visible. The layer follows the video's contained picture
area on resize.

Old JSON results remain readable. They must be analyzed again to obtain positions;
names and time intervals alone cannot recover the original face boxes.

For the synthetic component preview, run `pnpm dev` and open
`/face-overlay-preview.html`. This fixture is not included in the production build.
