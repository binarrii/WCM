"""Real MySQL/S3/FFmpeg checks, restricted to a fresh wcm_verify_* database.

No model inference, user accounts, face collections or production task writes.
Pass --source-url to also ingest an actual video through the same pipeline.
"""

import argparse
import asyncio
import functools
import hashlib
import http.server
import json
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import httpx
from fastapi import FastAPI

from api import handlers, parameter_store, review_media, task_queue
from api import review_task_store as store
from api.review_progress import ReviewProgress
from api.review_tasks import review_tasks_bp
from api.utils import VideoFrameSampler
from wcm_facerec.config import settings
from wcm_facerec.execution import execution_scope


def verify_sampling(path):
    signatures = set()
    selected = 0
    last_sample = 0.0
    started = time.monotonic()
    with VideoFrameSampler(path, 1, max_dimension=1080, sampling_mode="scene") as sampler:
        for window in sampler:
            if window.review_visual:
                selected += 1
                signatures.add(hashlib.sha256(window[0].image.tobytes()).digest())
                last_sample = window[0].timestamp
        assert sampler.fixed_samples >= int(sampler.duration_seconds) - 1
        assert last_sample >= sampler.duration_seconds - 1.1, "Video tail was not sampled"
        # Both our generated test pattern and the supplied real acceptance video
        # contain motion. Previously the server decoded every frame as black.
        assert len(signatures) > 1, "Acceptance footage was decoded as one repeated image"
        print(
            json.dumps(
                {
                    "sampling_verified": True,
                    "decoded": sampler.frames_read,
                    "fixed_samples": sampler.fixed_samples,
                    "selected": selected,
                    "distinct_selected_images": len(signatures),
                    "scene_cuts": sampler.scene_cuts,
                    "last_sample_seconds": last_sample,
                    "sampling_seconds": round(time.monotonic() - started, 2),
                }
            ),
            flush=True,
        )


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


async def ingest(url, directory, *, expect_video_copy=False):
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
                started = time.monotonic()
                await handlers._download_review_video(
                    url, path, settings.max_video_size_mb * 1048576, progress=progress
                )
                assert path.is_file()
                media = (await store.get(task_id))["media"]
                if expect_video_copy:
                    assert not media["video_transcoded"] and not media["deinterlaced"]
                print(
                    json.dumps(
                        {"prepare_and_archive_seconds": round(time.monotonic() - started, 2)}
                    ),
                    flush=True,
                )
                await asyncio.to_thread(verify_sampling, path)
                assert await store.complete(task_id, [])
                # Playback must survive losing the worker's local copy.
                path.unlink()
            child = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "scripts.verify_review_media", "--playback", task_id
            )
            assert await child.wait() == 0
        item = await store.get(task_id)
        assert item["media"]["decoder_verified"]
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
            await ingest(
                f"http://127.0.0.1:{server.server_port}/index.m3u8?token=test",
                root,
                expect_video_copy=args.expect_video_copy,
            )
            if args.source_url:
                await ingest(args.source_url, root, expect_video_copy=args.expect_video_copy)
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
    parser.add_argument("--expect-video-copy", action="store_true")
    asyncio.run(main(parser.parse_args()))
