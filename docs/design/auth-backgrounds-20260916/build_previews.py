"""Assemble self-contained review previews; does not modify the live UI."""
from pathlib import Path
import re

out = Path(__file__).parent
visual = Path('/Users/binarii/.codex/visualizations/2026/09/16/01a0a8d7-1208-7d20-bc69-586a15fbb56a')
fragment = (out / 'preview.template.html').read_text()
for key, name in [('a', 'a-optical-depth'), ('b', 'b-silver-frames'), ('c', 'c-signal-scan'), ('d', 'd-spectral-flow')]:
    art = (out / f'{name}.svg').read_text().replace('<svg ', '<svg preserveAspectRatio="xMidYMid slice" ', 1)
    art = re.sub(r'id="([^"]+)"', lambda m: f'id="{key}-{m[1]}"', art)
    art = re.sub(r'url\(#([^)]+)\)', lambda m: f'url(#{key}-{m[1]})', art)
    art = art.replace('aria-labelledby="title desc"', f'aria-labelledby="{key}-title {key}-desc"')
    fragment = fragment.replace('{{SVG_' + key.upper() + '}}', art)
visual.mkdir(parents=True, exist_ok=True)
(visual / 'wcm-auth-directions.html').write_text(fragment)
for key in 'abcd':
    content = fragment.replace("theme: 'a'", f"theme: '{key}'")
    document = '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>WCM · ' + key.upper() + '</title><style>body{margin:0;background:#101522}</style></head><body>' + content + '</body></html>'
    (out / f'preview-{key}.html').write_text(document)
assert '{{SVG_' not in fragment
assert len(fragment.encode()) < 1_000_000
print(f'Interactive preview: {visual / "wcm-auth-directions.html"} ({len(fragment.encode()):,} bytes)')
