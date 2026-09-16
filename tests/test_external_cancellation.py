"""Exercise actual HTTP clients/SDK hooks when external operations are interrupted."""

import asyncio
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
from botocore.exceptions import EndpointConnectionError

from api import handlers, model_clients, review_task_store
from wcm_facerec import cluster, image_store, model_budget
from wcm_facerec.config import settings


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["ocr", "visual", "guard"])
@pytest.mark.parametrize("interrupt", ["cancel", "timeout"])
async def test_model_http_streams_close_before_retry_and_release_slot(
    monkeypatch, model, interrupt
):
    started = asyncio.Event()
    active, calls, closed = [], [], []

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            started.set()
            yield (
                b'data: {"choices":[{"delta":{"content":"text"}}]}\n\n' if model == "ocr" else b"{"
            )
            await asyncio.Event().wait()

        async def aclose(self):
            closed.append(self)

    def respond(request):
        assert len(closed) == len(calls)  # Previous attempt has already closed.
        calls.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream" if model == "ocr" else "application/json"},
            stream=Stream(),
        )

    @asynccontextmanager
    async def slot(name):
        active.append(name)
        try:
            yield
        finally:
            assert len(closed) == len(calls)
            active.remove(name)

    actual = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kw: actual(transport=httpx.MockTransport(respond), **kw)
    )
    monkeypatch.setattr(model_clients, "model_slot", slot)
    monkeypatch.setattr(settings, f"{model}_timeout_s", 0.05 if interrupt == "timeout" else 10)
    operation = {
        "ocr": lambda: handlers._request_ocr("encoded"),
        "guard": lambda: handlers._request_guard("text"),
        "visual": lambda: handlers._call_nsfw_analysis(handlers.PreparedVisualFrames(["encoded"])),
    }[model]
    async with model_clients.model_client_pool():
        task = asyncio.create_task(operation())
        await asyncio.wait_for(started.wait(), 1)
        if interrupt == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 1)
        else:
            with pytest.raises(httpx.ReadTimeout):
                await asyncio.wait_for(task, 1)
    assert not active
    assert len(calls) == len(closed) == (1 if interrupt == "cancel" else 2)


@pytest.mark.asyncio
async def test_expired_budget_prevents_another_async_http_request(monkeypatch):
    sent = Mock(return_value=httpx.Response(200))
    actual = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kw: actual(transport=httpx.MockTransport(sent), **kw)
    )
    with model_budget.model_request_budget(10):
        async with model_clients.model_client("visual", 50) as client:
            await client.post("http://fixture/first")
            assert sent.call_args.args[0].extensions["timeout"]["read"] <= 10
            model_budget.cancel_model_requests()
            with pytest.raises(httpx.ReadTimeout):
                await client.post("http://fixture/fallback")
    sent.assert_called_once()


@pytest.mark.asyncio
async def test_visual_cancellation_during_fallback_preparation_sends_no_more_requests(monkeypatch):
    started, release = threading.Event(), threading.Event()
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(400, text="only one image supported")

    def compose(*args, **kwargs):
        started.set()
        assert release.wait(2)
        return "sheet", "layout"

    actual = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kw: actual(transport=httpx.MockTransport(respond), **kw)
    )
    monkeypatch.setattr(handlers, "_compose_nsfw_frames", compose)
    monkeypatch.setattr(settings, "nsfw_image_mode", "auto")
    task = asyncio.create_task(
        handlers._call_nsfw_analysis(handlers.PreparedVisualFrames(["first", "second"]))
    )
    try:
        assert await asyncio.to_thread(started.wait, 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
    finally:
        release.set()
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["faces", "text", "visual"])
async def test_legacy_video_entrypoints_cancel_download_and_remove_partial_file(
    monkeypatch, tmp_path, entry
):
    started, closed = asyncio.Event(), asyncio.Event()

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"x" * 65536
            started.set()
            await asyncio.Event().wait()

        async def aclose(self):
            closed.set()

    actual = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: actual(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, stream=Stream())), **kw
        ),
    )
    monkeypatch.setattr(settings, "nsfw_review_mode", "target")
    monkeypatch.setattr(handlers, "Path", lambda path: tmp_path / Path(path).name)
    operation = {
        "faces": lambda: handlers._search_video_frames(
            Mock(), "http://fixture/video.mp4", None, 10, 0.5, 1
        ),
        "text": lambda: handlers._process_detect_sensitive("http://fixture/video.mp4", 1),
        "visual": lambda: handlers._process_detect_nsfw("http://fixture/video.mp4", 1),
    }[entry]
    task = asyncio.create_task(operation())
    await asyncio.wait_for(started.wait(), 1)
    assert len(list(tmp_path.iterdir())) == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 1)
    assert closed.is_set() and list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_s3_internal_retry_stops_when_upload_is_cancelled(monkeypatch):
    started, release = threading.Event(), threading.Event()
    calls, budgets = [], []
    client = image_store._client("http://fixture", "us-east-1", "test-key", "test-secret")

    def send(request):
        calls.append(request)
        budgets.append(model_budget._current_budget.get())
        started.set()
        assert release.wait(2)
        raise EndpointConnectionError(endpoint_url="http://fixture")

    monkeypatch.setattr(client._endpoint, "_send", send)
    monkeypatch.setattr("botocore.endpoint.time.sleep", lambda seconds: None)
    task = asyncio.create_task(
        review_task_store._run(lambda: client.put_object(Bucket="test", Key="image", Body=b"data"))
    )
    try:
        assert await asyncio.to_thread(started.wait, 1)
        task.cancel()
        assert await asyncio.to_thread(budgets[0].cancelled.wait, 1)
        task.cancel()  # Repeat cancellation while the current HTTP call drains.
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        client.close()
        image_store._client.cache_clear()
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_resource_acquired_during_cancellation_is_released(monkeypatch):
    started, release = threading.Event(), threading.Event()
    connection = Mock()

    def acquire(*args):
        started.set()
        assert release.wait(2)
        return connection, "slot"

    monkeypatch.setattr(settings, "cluster_enabled", True)
    monkeypatch.setattr(cluster, "_try_slot", acquire)

    async def operation():
        async with cluster.cluster_slot("fixture"):
            pytest.fail("Cancelled acquisition must not enter its caller")

    task = asyncio.create_task(operation())
    try:
        assert await asyncio.to_thread(started.wait, 1)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
    connection.close.assert_called_once()
