"""Resource lifetime, backpressure and evidence-preserving speed paths."""

import asyncio
import threading
from pathlib import Path
from unittest.mock import AsyncMock

import cv2
import httpx
import numpy as np
import pytest

from api import handlers, model_clients, review_windows, utils
from api.video_workers import threaded_iterator


@pytest.mark.asyncio
async def test_model_pool_reuses_connections_and_applies_current_timeouts(monkeypatch):
    actual_client = httpx.AsyncClient
    requests = []
    clients = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "Safety: Safe"}}]})

    def client_factory(**kwargs):
        client = actual_client(transport=httpx.MockTransport(respond), **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(httpx, "AsyncClient", client_factory)
    async with model_clients.model_client_pool():
        monkeypatch.setattr(handlers.settings, "guard_timeout_s", 10)
        await handlers._request_guard("first")
        monkeypatch.setattr(handlers.settings, "guard_timeout_s", 3)
        await handlers._request_guard("second")
        assert len(clients) == 1 and not clients[0].is_closed
        timeouts = [request.extensions["timeout"]["read"] for request in requests]
        assert 9 < timeouts[0] <= 10
        assert 2 < timeouts[1] <= 3
        async with model_clients.model_client("visual", 50) as visual:
            assert visual is not clients[0]
    assert all(client.is_closed for client in clients)
    # A direct/CLI call owns its client; no closed pool leaks into another run.
    async with model_clients.model_client("guard", 3) as direct:
        assert direct not in clients[:2]
    assert direct.is_closed


@pytest.mark.asyncio
async def test_threaded_decoder_keeps_loop_responsive_and_closes_after_cancel():
    started = threading.Event()
    release = threading.Event()
    closed = threading.Event()
    thread_ids = []

    def decode():
        thread_ids.append(threading.get_ident())
        try:
            started.set()
            assert release.wait(2)
            yield "frame"
        finally:
            thread_ids.append(threading.get_ident())
            closed.set()

    async def consume():
        async with threaded_iterator(decode) as frames:
            async for _ in frames:
                pass

    task = asyncio.create_task(consume())
    assert await asyncio.to_thread(started.wait, 1)
    # This runs while the decoder is blocked, and close must wait for its read.
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done() and not closed.is_set()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 1)
    assert closed.is_set()
    assert thread_ids[0] == thread_ids[1] != threading.get_ident()


@pytest.mark.asyncio
async def test_threaded_iterator_does_not_decode_ahead_and_closes_on_break():
    produced = []
    closed = threading.Event()

    def decode():
        try:
            for value in range(100):
                produced.append(value)
                yield value
        finally:
            closed.set()

    async with threaded_iterator(decode) as frames:
        async for value in frames:
            assert value == 0
            await asyncio.sleep(0.01)
            assert produced == [0]
            break
    assert closed.is_set()


def test_nearby_targets_share_seek_keep_nearest_pts_and_deduplicate(monkeypatch):
    class Capture:
        times = [0, 0.04, 0.10, 0.17, 0.3, 0.4, 0.6, 0.8, 1.0, 2, 2.1, 2.4]

        def __init__(self):
            self.index = 0
            self.seeks = 0
            self.closed = False

        def isOpened(self):
            return True

        def set(self, prop, value):
            self.seeks += 1
            self.index = next(
                (i for i, pts in enumerate(self.times) if pts >= value / 1000), len(self.times)
            )

        def read(self):
            if self.index >= len(self.times):
                return False, None
            self.index += 1
            return True, np.full((8, 8, 3), self.index, np.uint8)

        def get(self, prop):
            return {
                cv2.CAP_PROP_FPS: 25,
                cv2.CAP_PROP_POS_MSEC: self.times[max(0, self.index - 1)] * 1000,
                cv2.CAP_PROP_POS_FRAMES: self.index,
            }.get(prop, 0)

        def release(self):
            self.closed = True

    cap = Capture()
    monkeypatch.setattr(utils.cv2, "VideoCapture", lambda _: cap)
    targets = [0.13, 0.18, 0.19, 0.33, 0.51, 0.75, 2.12]
    rows = list(utils.iter_video_frames_near(Path("fixture"), targets))
    used = set()
    expected = []
    for target in targets:
        nearest = min(cap.times, key=lambda pts: abs(pts - target))
        expected.append(None if nearest in used else nearest)
        used.add(nearest)
    assert [frame.timestamp if frame else None for _, frame in rows] == expected
    assert cap.seeks == 2  # Seven targets, two disjoint decode intervals.
    assert cap.closed


@pytest.mark.asyncio
async def test_visual_fast_path_encodes_off_loop_once_without_jpeg_decode(monkeypatch):
    pixels = np.zeros((1080, 1920, 3), np.uint8)
    pixels[:, 600:800] = 180
    frame = utils.VideoFrame(1, pixels)
    original_encode = handlers._encode_nsfw_frame
    threads = []

    def encode(image):
        threads.append(threading.get_ident())
        return original_encode(image)

    def unexpected_decode(*args):
        pytest.fail("prepared visual input must not be decoded/recompressed")

    monkeypatch.setattr(handlers, "_encode_nsfw_frame", encode)
    monkeypatch.setattr(handlers, "_decode_nsfw_frame", unexpected_decode)
    caption = AsyncMock(return_value="ordinary image")
    monkeypatch.setattr(handlers, "_request_nsfw_caption", caption)
    images, _ = await asyncio.to_thread(review_windows._prepare_visual_frames, [frame])
    assert await handlers._call_nsfw_analysis(images, [1], review_all=True) == "ordinary image"
    assert len(threads) == 1 and threads[0] != threading.get_ident()
    import base64

    image = cv2.imdecode(np.frombuffer(base64.b64decode(images[0]), np.uint8), cv2.IMREAD_COLOR)
    assert max(image.shape[:2]) == 896
    assert frame.image.shape == (1080, 1920, 3)
    assert "b64" not in frame.__dict__  # OCR input has not been generated for visual.


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("limit", [1, 2, 0, -3])
async def test_auxiliary_decode_respects_optional_limit_and_cleans_up(monkeypatch, cancel, limit):
    from tests.test_window_review import install_video, sample

    install_video(monkeypatch, [sample(i, i * 20) for i in range(9)])
    monkeypatch.setattr(handlers.settings, "face_profile_optimization", True)
    monkeypatch.setattr(handlers.settings, "face_neighbor_concurrency", limit)
    monkeypatch.setattr(handlers.settings, "face_max_extra_call_ratio", 0.3)
    monkeypatch.setattr(
        review_windows,
        "aggregate_face_candidates",
        lambda *a, **kw: (
            [],
            {"trigger_times": [1], "confirmed": 0, "probable": 0},
        ),
    )
    started = threading.Event()
    closed = threading.Event()
    ready = asyncio.Event()
    release = asyncio.Event()
    active = 0
    peak = 0
    finished = 0
    expected = min(limit, 3) if limit > 0 else 3
    produced = []

    def neighbors(path, targets, **kwargs):
        try:
            for i, target in enumerate(targets):
                if i:
                    assert started.wait(1), "model must start before remaining frames decode"
                produced.append(target)
                yield target, utils.VideoFrame(target, np.full((16, 16, 3), i, np.uint8))
        finally:
            closed.set()

    async def face(*args, auxiliary=False, **kwargs):
        nonlocal active, peak, finished
        if auxiliary:
            started.set()
            active += 1
            peak = max(peak, active)
            if active == expected:
                ready.set()
            try:
                await release.wait()
            finally:
                active -= 1
                finished += 1
        return handlers.FaceFrameResult([])

    monkeypatch.setattr(review_windows, "iter_video_frames_near", neighbors)
    monkeypatch.setattr(handlers, "_face_task", face)
    monkeypatch.setattr(handlers, "_call_ocr_api", AsyncMock(return_value=""))
    monkeypatch.setattr(handlers, "_call_nsfw_analysis", AsyncMock(return_value="ordinary"))
    monkeypatch.setattr(handlers, "_call_llm_guard", AsyncMock(return_value={"safe": True}))
    task = asyncio.create_task(handlers._process_analyze_media("fixture.mp4", 1, 10, 0.5))
    try:
        await asyncio.wait_for(ready.wait(), 2)
        assert len(produced) == expected
        assert active == expected
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 2)
            assert finished == expected
        else:
            release.set()
            await asyncio.wait_for(task, 2)
            assert len(produced) == finished == 3  # ceil(9 * 0.3), unchanged budget.
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert closed.is_set() and active == 0 and peak == expected
