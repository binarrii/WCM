"""Real MySQL/S3 deduplication and GC checks in an isolated verification namespace."""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

from api import media_objects as objects
from api import parameter_store, review_media, task_queue
from api import review_task_store as store
from scripts.verify_review_media import playback
from wcm_facerec import image_store
from wcm_facerec.config import settings
from wcm_facerec.execution import execution_scope


def sql(query, args=()):
    with store._connect() as connection, connection.cursor() as cursor:
        cursor.execute(query, args)
        return cursor.fetchall()


async def claim():
    task_id = await store.create("https://source/same-url.mp4", {"sample_interval": 1})
    task = await task_queue.claim("dedup-verification")
    assert task["id"] == task_id
    return task


async def archive(task, path):
    with execution_scope(task["id"], task["lease_token"], task["attempts"]):
        await review_media.archive(task["id"], path, {})
        assert await store.complete(task["id"], [])
    public = (await store.get(task["id"]))["media"]
    return review_media._get(task["id"], public["id"])


def keys():
    return [
        item["Key"]
        for item in image_store.client()
        .list_objects_v2(
            Bucket=settings.s3_bucket, Prefix=settings.review_media_prefix.rstrip("/") + "/"
        )
        .get("Contents", [])
    ]


async def main(args):
    assert settings.review_tasks_db_name.startswith("wcm_verify_")
    assert settings.review_media_prefix.startswith("wcm/verify-media/")
    if args.child:
        task_id, token, path = args.child
        original = review_media._upload

        def upload(file, media):
            # Hold the content lock long enough for another process to contend.
            time.sleep(1)
            with open(str(file) + ".uploads", "a") as log:
                log.write(media["object_key"] + "\n")
            return original(file, media)

        review_media._upload = upload
        await archive({"id": task_id, "lease_token": token, "attempts": 1}, Path(path))
        return
    await store.initialize()
    await task_queue.initialize()
    await parameter_store.initialize()
    await parameter_store.close()
    assert not sql("SELECT id FROM review_tasks LIMIT 1") and not keys()
    with tempfile.TemporaryDirectory(prefix="wcm-dedup-check-") as folder:
        path = Path(folder) / "video.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=160x96:rate=25:duration=3",
                "-c:v",
                "libx264",
                "-threads",
                "1",
                str(path),
            ],
            check=True,
        )
        first, second = await claim(), await claim()
        children = [
            await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "scripts.verify_media_dedup",
                "--child",
                task["id"],
                task["lease_token"],
                str(path),
            )
            for task in (first, second)
        ]
        assert await asyncio.gather(*(child.wait() for child in children)) == [0, 0]
        rows = sql("SELECT * FROM review_media ORDER BY created_at")
        assert len(rows) == 2 and rows[0]["object_id"] == rows[1]["object_id"]
        assert rows[0]["id"] != rows[1]["id"]
        assert len(keys()) == 1 and len(Path(str(path) + ".uploads").read_text().splitlines()) == 1
        await playback(first["id"])
        await playback(second["id"])
        print(
            "PASS concurrent processes: one upload, two independently playable task references",
            flush=True,
        )

        # Seed a real legacy duplicate and confirm that migration preserves its
        # public media ID and independent expiration, without another upload.
        legacy_task = await claim()
        original_media = review_media._get(first["id"], rows[0]["id"])
        if original_media is None:
            original_media = review_media._get(first["id"], rows[1]["id"])
        legacy = dict(original_media, id=uuid.uuid4().hex, task_id=legacy_task["id"])
        legacy.pop("object_id")
        legacy["object_key"] = (
            settings.review_media_prefix.rstrip("/") + "/" + legacy["id"] + ".mp4"
        )
        review_media._register(legacy_task["id"], legacy)
        review_media._upload(path, legacy)
        with execution_scope(legacy_task["id"], legacy_task["lease_token"], 1):
            assert review_media._publish(legacy_task["id"], legacy)
            assert await store.complete(legacy_task["id"], [])
        assert len(keys()) == 2
        await review_media.collect_expired()
        migrated = review_media._get(legacy_task["id"], legacy["id"])
        assert migrated["object_key"] == original_media["object_key"]
        assert migrated["expires_at"] == legacy["expires_at"]
        assert len(keys()) == 2  # The obsolete duplicate has a 24h rollback grace.
        await playback(legacy_task["id"])
        print(
            "PASS legacy deduplication preserves task URLs/expiry and retains a rollback grace",
            flush=True,
        )

        sql(
            "UPDATE review_media SET expires_at=TIMESTAMPADD(DAY,-1,UTC_TIMESTAMP(3)) WHERE task_id=%s",
            (first["id"],),
        )
        await review_media.collect_expired()
        assert not sql("SELECT id FROM review_media WHERE task_id=%s", (first["id"],))
        await playback(second["id"])
        await playback(legacy_task["id"])
        await store.delete_many([first["id"], second["id"], legacy_task["id"]])
        for reference in review_media._collectable():
            review_media._delete(reference)
        stale_gc = objects.unreferenced()
        assert len(stale_gc) == 1
        replacement_task = await claim()
        replacement = await archive(replacement_task, path)
        objects.delete_unreferenced(stale_gc[0])
        await playback(replacement_task["id"])
        assert replacement["object_key"] == original_media["object_key"]
        print(
            "PASS expired/deleted references and stale GC snapshots cannot delete a reused object",
            flush=True,
        )

        path.write_bytes(path.read_bytes() + b"changed")
        changed_task = await claim()
        changed = await archive(changed_task, path)
        assert changed["sha256"] != replacement["sha256"]
        assert changed["object_key"] != replacement["object_key"]
        print("PASS changed content at the same URL receives a different object", flush=True)
        stale_task = await claim()
        with execution_scope(stale_task["id"], "invalid-token", 1):
            try:
                await review_media.archive(stale_task["id"], path, {})
            except RuntimeError as error:
                assert "租约" in str(error)
            else:
                raise AssertionError("Stale lease published a shared object")
        with execution_scope(stale_task["id"], stale_task["lease_token"], 1):
            assert await store.complete(stale_task["id"], [])
        assert (await store.get(stale_task["id"]))["media"] is None
        await store.delete_many([stale_task["id"]])
        await review_media.collect_expired()
        await playback(changed_task["id"])
        print(
            "PASS stale task lease cannot publish or delete another task's shared media", flush=True
        )
        for action in ("unshare", "migrate"):
            child = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "scripts.review_media_catalog", action, "--apply"
            )
            assert await child.wait() == 0
            await playback(replacement_task["id"])
            await playback(changed_task["id"])
            refs = sql("SELECT object_id FROM review_media")
            assert all((row["object_id"] is None) == (action == "unshare") for row in refs)
        print("PASS rollback unsharing and re-migration preserve playback", flush=True)
        await store.delete_many([replacement_task["id"], changed_task["id"]])
        # Accelerate only the disposable test object's grace period.
        sql("UPDATE review_media_objects SET created_at=TIMESTAMPADD(DAY,-2,UTC_TIMESTAMP(3))")
        await review_media.collect_expired()
        assert not sql("SELECT id FROM review_media")
        assert not sql("SELECT id FROM review_media_objects")
        assert not keys()
        print("ALL MEDIA DEDUP INTEGRATION CHECKS PASSED; TEST OBJECTS CLEANED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", nargs=3)
    asyncio.run(main(parser.parse_args()))
