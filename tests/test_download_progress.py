import asyncio
import gzip
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from api import handlers, review_progress, utils
from api.review_progress import ReviewProgress


def install_download(monkeypatch, chunks, headers=None):
    original_client = httpx.Client

    class Stream(httpx.SyncByteStream):
        closed = False

        def __iter__(self):
            for chunk in chunks:
                if isinstance(chunk, Exception):
                    raise chunk
                yield chunk

        def close(self):
            self.closed = True

    stream = Stream()
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, headers=headers, stream=stream)
    )
    monkeypatch.setattr(
        utils.httpx, "Client", lambda **kw: original_client(transport=transport, **kw)
    )
    ticks = iter(range(100))
    monkeypatch.setattr(utils, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    return stream


@pytest.mark.parametrize("length", ["196608", None, "invalid"])
def test_download_reports_saved_bytes_and_optional_total(monkeypatch, tmp_path, length):
    chunks = [b"a" * 65536, b"b" * 65536, b"c" * 65536]
    stream = install_download(monkeypatch, chunks, {"Content-Length": length} if length else {})
    updates = []
    path = tmp_path / "movie.mp4"
    utils._download_video_safe_sync(
        "http://fixture/movie.mp4", path, 300000, on_progress=lambda *values: updates.append(values)
    )
    total = 196608 if length == "196608" else None
    assert updates[0] == (0, total)
    assert updates[-1] == (196608, total)
    assert (65536, total) in updates and (131072, total) in updates
    assert path.read_bytes() == b"".join(chunks)
    assert stream.closed


def test_compressed_download_does_not_compare_decoded_bytes_to_wire_size(monkeypatch, tmp_path):
    body = b"movie" * 40000
    compressed = gzip.compress(body)
    stream = install_download(
        monkeypatch,
        [compressed],
        {"Content-Length": str(len(compressed)), "Content-Encoding": "gzip"},
    )
    updates = []
    path = tmp_path / "movie.mp4"
    utils._download_video_safe_sync(
        "http://fixture/movie.mp4",
        path,
        len(body),
        on_progress=lambda *values: updates.append(values),
    )
    assert updates[-1] == (len(body), None)
    assert path.read_bytes() == body and stream.closed


@pytest.mark.asyncio
async def test_thread_callbacks_reach_progress_without_completing_the_review(monkeypatch, tmp_path):
    writes = []

    async def write(task_id, data):
        writes.append(data)

    monkeypatch.setattr(review_progress.review_task_store, "update_progress", write)
    progress = ReviewProgress("task", interval=0)
    paused = threading.Event()
    release = threading.Event()

    def download(*args, on_progress, **kwargs):
        on_progress(50, 100)
        paused.set()
        assert release.wait(2)
        on_progress(100, 100)

    monkeypatch.setattr(handlers, "_download_video_safe_sync", download)
    task = asyncio.create_task(
        handlers._download_review_video(
            "fixture.mp4", tmp_path / "movie.mp4", 100, progress=progress
        )
    )
    try:
        assert await asyncio.to_thread(paused.wait, 1)
        await progress.report(force=True)
        assert writes[-1]["sub_progress"]["percent"] == 50
        assert writes[-1]["percent"] is None
    finally:
        release.set()
    await asyncio.wait_for(task, 1)
    assert progress.snapshot()["sub_progress"]["percent"] == 100
    assert progress.snapshot()["percent"] is None
    await progress.begin_review(120)
    progress.update_download(100, 100)  # A delayed callback must not restore download UI.
    assert progress.snapshot()["sub_progress"] is None
    assert progress.snapshot()["percent"] == 0


@pytest.mark.asyncio
async def test_failed_download_retains_last_bytes_and_never_finishes(monkeypatch, tmp_path):
    stream = install_download(monkeypatch, [b"a" * 65536, httpx.ReadError("broken")])
    monkeypatch.setattr(review_progress.review_task_store, "update_progress", AsyncMock())
    progress = ReviewProgress("task", interval=0)
    with pytest.raises(httpx.ReadError):
        await handlers._download_review_video(
            "http://fixture/movie.mp4", tmp_path / "movie.mp4", 200000, progress=progress
        )
    sub = progress.snapshot()["sub_progress"]
    assert sub["completed"] == 65536 and sub["total"] is None
    assert sub["percent"] is None and not sub["complete"]
    await progress.set_phase("failed")
    progress.update_download(200000, 200000)
    assert progress.snapshot()["sub_progress"] == sub
    assert stream.closed


def test_download_size_limit_still_stops_unknown_length_stream(monkeypatch, tmp_path):
    stream = install_download(monkeypatch, [b"a" * 65536, b"b" * 65536])
    with pytest.raises(ValueError, match="too large"):
        utils._download_video_safe_sync("http://fixture/movie.mp4", tmp_path / "movie.mp4", 100000)
    assert stream.closed


@pytest.mark.asyncio
async def test_unknown_download_total_has_no_percentage_until_success(monkeypatch):
    monkeypatch.setattr(review_progress.review_task_store, "update_progress", AsyncMock())
    progress = ReviewProgress("task")
    await progress.begin_download()
    progress.update_download(300, None)
    assert progress.snapshot()["sub_progress"]["percent"] is None
    await progress.finish_download()
    assert progress.snapshot()["sub_progress"]["percent"] == 100
    assert progress.snapshot()["sub_progress"]["completed"] == 300
