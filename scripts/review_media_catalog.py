"""Migrate legacy references, or unshare files before rolling back old code.

Default is a read-only count. --apply requires no processing/cancelling tasks;
hold the review admission row lock throughout to prevent new workers claiming.
Run with the candidate/new image before restarting workers on an older image.
"""

import argparse
import json
import shutil

from api import media_objects as objects
from api import review_task_store as store
from wcm_facerec import image_store
from wcm_facerec.config import settings


def unshare(media):
    with objects.locked("legacy", "") as (_, legacy):
        legacy.execute("SELECT * FROM review_media WHERE id=%s", (media["id"],))
        media = legacy.fetchone()
        if not media or not media["object_id"]:
            return
        legacy.execute("SELECT * FROM review_media_objects WHERE id=%s", (media["object_id"],))
        obj = legacy.fetchone()
        assert obj and obj["storage_scope"] == objects.storage_scope(media["storage"])
        with objects.locked(obj["storage_scope"], obj["sha256"]) as (connection, cursor):
            assert objects.inspect_object(media) == (obj["size_bytes"], obj["sha256"])
            target = dict(
                media,
                object_key=settings.review_media_prefix.strip("/") + "/" + media["id"] + ".mp4",
            )
            if target["object_key"] != media["object_key"]:
                if objects.inspect_object(target) != (obj["size_bytes"], obj["sha256"]):
                    if media["storage"] == "s3":
                        image_store.client().copy(
                            {"Bucket": settings.s3_bucket, "Key": media["object_key"]},
                            settings.s3_bucket,
                            target["object_key"],
                            ExtraArgs={"MetadataDirective": "COPY"},
                        )
                    else:
                        shutil.copyfile(objects.local_path(media), objects.local_path(target))
                assert objects.inspect_object(target) == (obj["size_bytes"], obj["sha256"])
            connection.begin()
            try:
                cursor.execute(
                    "UPDATE review_media SET object_id=NULL,object_key=%s WHERE id=%s",
                    (target["object_key"], media["id"]),
                )
                cursor.execute(
                    """UPDATE review_tasks SET media=JSON_REMOVE(
                    JSON_SET(media,'$.object_key',%s),'$.object_id')
                    WHERE id=%s AND JSON_UNQUOTE(JSON_EXTRACT(media,'$.id'))=%s""",
                    (target["object_key"], media["task_id"], media["id"]),
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise


def main(args):
    with store._connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT COUNT(*) AS total FROM review_media WHERE object_id "
            + ("IS NULL" if args.action == "migrate" else "IS NOT NULL")
        )
        print(
            json.dumps(
                {
                    "action": args.action,
                    "references": cursor.fetchone()["total"],
                    "apply": args.apply,
                }
            ),
            flush=True,
        )
        if not args.apply:
            return
        connection.begin()
        cursor.execute("SELECT id FROM review_admission WHERE id=1 FOR UPDATE")
        cursor.execute(
            "SELECT COUNT(*) AS total FROM review_tasks WHERE status IN ('processing','cancelling')"
        )
        assert cursor.fetchone()["total"] == 0, (
            "Wait for active reviews before media migration/rollback"
        )
        if args.action == "migrate":
            previous = None
            while True:
                items = objects.legacy_candidates()
                ids = [item["id"] for item in items]
                if not ids or ids == previous:
                    break
                previous = ids
                for item in items:
                    objects.migrate_legacy(item)
        else:
            cursor.execute("SELECT * FROM review_media WHERE object_id IS NOT NULL")
            for item in cursor.fetchall():
                unshare(item)
        connection.rollback()
    print("MEDIA CATALOG OPERATION COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["migrate", "unshare"])
    parser.add_argument("--apply", action="store_true")
    main(parser.parse_args())
