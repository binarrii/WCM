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
