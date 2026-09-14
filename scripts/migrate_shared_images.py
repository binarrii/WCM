"""Copy an existing image library into S3 without deleting or rewriting originals."""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from wcm_facerec import image_store, person_operations
from wcm_facerec.config import settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if settings.image_storage != "s3" or not settings.cluster_enabled:
        raise SystemExit("Require S3 cluster configuration")
    source = args.source.resolve(strict=True)
    client = image_store.client()
    client.head_bucket(Bucket=settings.s3_bucket)
    existing = set()
    for page in client.get_paginator("list_objects_v2").paginate(
        Bucket=settings.s3_bucket, Prefix=settings.s3_prefix.strip("/") + "/"
    ):
        existing.update(item["Key"] for item in page.get("Contents", []))
    files = [
        p
        for p in source.rglob("*")
        if p.is_file()
        and not p.is_symlink()
        and not any(part.startswith(".") for part in p.relative_to(source).parts)
        and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
    ]

    def copy(path):
        data = path.read_bytes()
        logical = Path("/tmp/wcm") / path.relative_to(source)
        key = image_store.object_key(logical)
        digest = hashlib.sha256(data).hexdigest()
        if key in existing:
            saved = image_store.stat(logical)
            if saved.get("Metadata", {}).get("sha256") != digest:
                raise RuntimeError(f"Existing object differs; do not overwrite: {key}")
            return "existing", len(data)
        image_store.write_bytes(logical, data)
        saved = image_store.stat(logical)
        if saved["ContentLength"] != len(data) or saved.get("Metadata", {}).get("sha256") != digest:
            raise RuntimeError(f"Object verification failed: {key}")
        return "copied", len(data)

    counts = {"copied": 0, "existing": 0, "bytes": 0, "journals": 0}
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 16))) as executor:
        for index, (state, size) in enumerate(executor.map(copy, files), 1):
            counts[state] += 1
            counts["bytes"] += size
            if index % 1000 == 0:
                print(f"Verified {index}/{len(files)} images", flush=True)
    person_operations.initialize()
    for path in (source / ".person-operations").glob("*.json"):
        payload = json.loads(path.read_text())
        identity = "legacy-" + hashlib.sha256(path.name.encode()).hexdigest()[:48]
        person_operations.save_journal(identity, payload)
        counts["journals"] += 1
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
