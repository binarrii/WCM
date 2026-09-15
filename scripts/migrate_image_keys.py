"""Plan or migrate image metadata without moving images or reenrolling faces.

The plan records exact before-metadata hashes and is written before any mutation.
Every person is updated through WCM's idempotent journal and committed outbox.
Rerunning discovers already-migrated records and skips them. Native-only legacy
records use metadata-only compensation and face fingerprints.
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from uuid import uuid4

from api import parameter_store
from wcm_facerec import face_sync_store as store
from wcm_facerec import image_store
from wcm_facerec import person_operations as ops
from wcm_facerec.cluster import cluster_slot, run_sync
from wcm_facerec.config import settings
from wcm_facerec.face_engine import FaceEngine
from wcm_facerec.image_key_migration import migrate_person_image_keys


async def run(args):
    await parameter_store.initialize()
    try:
        await run_sync(ops.initialize)
        await run_sync(store.initialize)
        if not settings.insightface_replication_enabled or settings.image_storage != "s3":
            raise RuntimeError("Replication and S3 must be enabled")
        engine = FaceEngine()
        plan = []
        scanned = 0
        source = {
            "source": store.source(),
            "database": settings.review_tasks_db_name,
            "bucket": settings.s3_bucket,
            "prefix": settings.s3_prefix,
        }
        async with cluster_slot("person-library"):
            await run_sync(store.assert_primary_clean)
            if args.rollback_from:
                original = json.loads(Path(args.rollback_from).read_text())
                if original.get("source") != source or original.get("rollback"):
                    raise RuntimeError("Rollback plan does not match this deployment")
                for row in original["records"]:
                    scanned += 1
                    item = await engine._run(
                        engine._primary_adapter.get_person,
                        row["person_id"],
                        collection_id=row["collection"],
                    )
                    if item and item["metadata"] == row["before_metadata"]:
                        continue
                    after = image_store.with_images(
                        row["before_metadata"], image_store.image_refs(row["before_metadata"])
                    )
                    if not item or item["metadata"] != after:
                        raise RuntimeError("人物在迁移后被修改，拒绝覆盖，请先人工核对")
                    plan.append(
                        {
                            **row,
                            "metadata_hash": store.digest(item["metadata"]),
                            "restore_metadata": row["before_metadata"],
                            "before_metadata": item["metadata"],
                        }
                    )
            for cid in [] if args.rollback_from else store.collections():
                cursor = None
                while True:
                    items, cursor = await engine._run(
                        engine._primary_adapter.list_persons,
                        collection_id=cid,
                        cursor=cursor,
                        limit=100,
                    )
                    for item in items:
                        scanned += 1
                        metadata = image_store.with_images(
                            item["metadata"], image_store.image_refs(item)
                        )
                        if metadata != item["metadata"]:
                            plan.append(
                                {
                                    "collection": cid,
                                    "person_id": item["id"],
                                    "metadata_hash": store.digest(item["metadata"]),
                                    "before_metadata": item["metadata"],
                                }
                            )
                    if not cursor:
                        break
        report = Path(args.report)
        plan_id = uuid4().hex
        report.parent.mkdir(parents=True, exist_ok=True)
        with report.open("x", encoding="utf-8") as handle:
            os.fchmod(handle.fileno(), 0o600)
            json.dump(
                {
                    "plan_id": plan_id,
                    "source": source,
                    "rollback": bool(args.rollback_from),
                    "scanned": scanned,
                    "planned": len(plan),
                    "records": plan,
                },
                handle,
                ensure_ascii=False,
            )
        print(
            json.dumps({"scanned": scanned, "planned": len(plan), "apply": args.apply}), flush=True
        )
        if not args.apply:
            return
        started = time.monotonic()
        for index, row in enumerate(plan, 1):
            fingerprint = store.digest(
                [
                    "image-keys-rollback-v1" if args.rollback_from else "image-keys-v1",
                    plan_id,
                    store.source(),
                    settings.s3_bucket,
                    settings.s3_prefix,
                    row["collection"],
                    row["person_id"],
                    row["metadata_hash"],
                    store.digest(row.get("restore_metadata")),
                ]
            )
            token = ops.request_key.set((fingerprint, fingerprint))
            try:
                await migrate_person_image_keys(
                    engine,
                    row["person_id"],
                    row["collection"],
                    row["metadata_hash"],
                    row.get("restore_metadata"),
                )
            finally:
                ops.request_key.reset(token)
            if index % 100 == 0 or index == len(plan):
                print(
                    json.dumps(
                        {
                            "migrated": index,
                            "planned": len(plan),
                            "elapsed_seconds": round(time.monotonic() - started, 1),
                        }
                    ),
                    flush=True,
                )
        print("MIGRATION_COMPLETE", flush=True)
    finally:
        await parameter_store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--rollback-from", help="Original migration report; reject records edited since migration"
    )
    parser.add_argument(
        "--report", required=True, help="New file for the rollback/audit plan (0600)"
    )
    asyncio.run(run(parser.parse_args()))
