"""One-shot backfill: write ``file_path`` into IFS category-collection metadata.

Context
-------
The webui renders a face card's thumbnail from ``metadata.file_path`` on the
IFS Person. Historical bulk-imports wrote the ``corrupt-officials`` collection
(4424 persons) with metadata that has no ``file_path`` — only
``{category, occupation, remarks}`` — so those cards show a placeholder
("暂无预览图片"). This script fills the gap by matching each person to its
local image under ``/tmp/wcm/<category>/``.

Matching rule
-------------
Local filenames look like ``<name>_<md5>.<ext>`` (e.g. ``阿布_abcd1234.jpg``).
For each IFS person, we pick the *first* jpg/png whose basename starts with the
exact person name followed by ``_``. We deliberately require the underscore so a
short name like ``阿布`` never matches someone else's file (``阿布·XXX``). A
``Name``-style fallback list maps category-collection-name → category folder.

Safety
------
* Merges (never clobbers) existing metadata keys — an existing ``type`` /
  ``remarks`` / anything else is preserved.
* Idempotent: rerunning updates already-backfilled persons to the same value.
* Dry-run by default (``--apply`` to write). Reports how many matches were
  found vs. missing.

Usage
-----
    uv run python scripts/backfill_file_path.py               # dry run
    uv run python scripts/backfill_file_path.py --apply        # write
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

BASE = "http://10.252.25.251:18097"
# category-collection id -> local folder that holds that category's images
COLLECTION_DIR = {
    "bad-artists": "/tmp/wcm/劣迹艺人",
    "political": "/tmp/wcm/时政敏感",
    "corrupt-officials": "/tmp/wcm/落马官员",
}
CATEGORY_BY_COLLECTION = {
    "bad-artists": "劣迹艺人",
    "political": "时政敏感",
    "corrupt-officials": "落马官员",
}


def get_json(url: str):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def patch_json(url: str, payload: dict):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="PATCH",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def all_persons(cid: str) -> list[dict]:
    out = []
    cursor = None
    while True:
        url = f"{BASE}/v1/collections/{cid}/persons?limit=100"
        if cursor:
            url += f"&cursor={cursor}"
        d = get_json(url)
        out.extend(d.get("persons", []))
        cursor = d.get("next_cursor")
        if not cursor or not d.get("persons"):
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually PATCH; default is dry-run")
    ap.add_argument("--limit", type=int, default=0, help="only process first N persons (debug)")
    args = ap.parse_args()

    for cid, folder in COLLECTION_DIR.items():
        category = CATEGORY_BY_COLLECTION[cid]
        folder_path = Path(folder)
        if not folder_path.is_dir():
            print(f"[SKIP] {cid}: {folder} not a dir")
            continue
        # Index local files: {name_prefix: file_path} preferring .jpg over .png
        local: dict[str, str] = {}
        for fp in folder_path.iterdir():
            if fp.is_file() and fp.suffix.lower() in (".jpg", ".jpeg", ".png"):
                stem = fp.name.rsplit("_", 1)[0]
                cur = local.get(stem)
                if cur is None or cur.endswith(".png"):  # prefer .jpg
                    local[stem] = str(fp)

        persons = all_persons(cid)
        matched = 0
        already = 0
        missing: list[str] = []
        for i, p in enumerate(persons):
            if args.limit and i >= args.limit:
                break
            name = (p.get("name") or "").strip()
            md = p.get("metadata") or {}
            if name in local:
                fp = local[name]
                if md.get("file_path") == fp:
                    already += 1
                    continue
                new_md = dict(md)
                new_md["file_path"] = fp
                if not new_md.get("type"):
                    new_md["type"] = category
                if args.apply:
                    try:
                        patch_json(f"{BASE}/v1/collections/{cid}/persons/{p['id']}", {"metadata": new_md})
                    except Exception as e:  # noqa: BLE001
                        print(f"  [ERR] {cid}/{name}: {e}")
                        continue
                matched += 1
            else:
                missing.append(name)

        print(f"\n=== {cid} ({category}) ===")
        print(f"  persons: {len(persons)} | local files indexed: {len(local)}")
        print(f"  matched (would update): {matched} | already correct: {already}")
        print(f"  missing (no local file): {len(missing)}")
        if missing and not args.apply:
            print(f"  missing sample: {missing[:10]}")
        if not args.apply:
            print("  [DRY RUN] pass --apply to write")

    print("\nDone.")


if __name__ == "__main__":
    sys.exit(main())
