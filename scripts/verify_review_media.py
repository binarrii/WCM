"""Real MySQL/S3/FFmpeg checks, restricted to a fresh wcm_verify_* database.

No model inference, user accounts, face collections or production task writes.
Pass --source-url to also ingest an actual video through the same pipeline.
"""

import argparse
import asyncio
import contextlib
import functools
import http.server
import json
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import httpx
from fastapi import FastAPI

from api import handlers, parameter_store, review_media, task_queue
from api import review_task_store as store
from api.review_progress import ReviewProgress
from api.review_tasks import review_tasks_bp
from wcm_facerec import runtime_parameters
from wcm_facerec.config import settings
from wcm_facerec.execution import execution_scope


async def playback(task_id):
    task = await store.get(task_id)
    media = task["media"]
    app = FastAPI()
    app.include_router(review_tasks_bp, prefix="/api/v1")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        head = await client.head(media["url"])
        assert (
            head.status_code == 200 and int(head.headers["content-length"]) == media["size_bytes"]
        )
        chunk = await client.get(media["url"], headers={"Range": "bytes=0-1023"})
        assert chunk.status_code == 206 and len(chunk.content) == 1024
        assert b"ftyp" in chunk.content[:32]
        assert (await client.get(media["url"], headers={"Range": "bytes=-512"})).status_code == 206
        assert (
            await client.get(media["url"], headers={"Range": "bytes=999999999999-"})
        ).status_code == 416
    print("PASS independent-process private S3 playback, HEAD and Range", flush=True)


async def ingest(url, directory):
    task_id = await store.create(url, {"sample_interval": 1, "top_k": 10, "threshold": 0.5})
    task = await task_queue.claim("media-verification")
    assert task["id"] == task_id

    async def heartbeat():
        while True:
            await asyncio.sleep(5)
            assert await task_queue.renew(task_id, task["lease_token"])

    lease = asyncio.create_task(heartbeat())
    try:
        with execution_scope(task_id, task["lease_token"], task["attempts"]):
            async with ReviewProgress(task_id) as progress:
                path = directory / f"{task_id}.mp4"
                await handlers._download_review_video(
                    url, path, settings.max_video_size_mb * 1048576, progress=progress
                )
                assert path.is_file()
                assert await store.complete(task_id, [])
                # Playback must survive losing the worker's local copy.
                path.unlink()
            child = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "scripts.verify_review_media", "--playback", task_id
            )
            assert await child.wait() == 0
        item = await store.get(task_id)
        print(json.dumps({"verified_media": item["media"]}, ensure_ascii=False), flush=True)
        await store.delete_many([task_id])
        await review_media.collect_expired()
    finally:
        lease.cancel()
        await asyncio.gather(lease, return_exceptions=True)


async def main(args):
    assert settings.review_tasks_db_name.startswith("wcm_verify_"), (
        "A disposable test database is required"
    )
    assert settings.review_media_prefix.startswith("wcm/verify-media/"), (
        "A disposable media prefix is required"
    )
    if args.playback:
        return await playback(args.playback)
    await store.initialize()
    await task_queue.initialize()
    await parameter_store.initialize()
    await parameter_store.close()
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS total FROM review_tasks")
        assert cur.fetchone()["total"] == 0, "A fresh test database is required"
    with tempfile.TemporaryDirectory(prefix="wcm-media-check-") as folder:
        root = Path(folder)
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=160x90:rate=25:duration=3",
                "-c:v",
                "libx264",
                "-threads",
                "1",
                "-g",
                "25",
                "-f",
                "hls",
                "-hls_time",
                "1",
                "-hls_playlist_type",
                "vod",
                str(root / "index.m3u8"),
            ],
            check=True,
        )
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=folder)
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            await ingest(f"http://127.0.0.1:{server.server_port}/index.m3u8?token=test", root)
            if args.source_url:
                await ingest(args.source_url, root)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS total FROM review_media")
        assert cur.fetchone()["total"] == 0, "Test assets were not cleaned up"
    print("ALL REVIEW MEDIA INTEGRATION CHECKS PASSED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-url")
    parser.add_argument("--playback")
    asyncio.run(main(parser.parse_args()))
