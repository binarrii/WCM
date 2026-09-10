"""Visual failure budgets stop whole videos without leaking workers or task slots."""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import httpx
import pytest

from api import handlers, review_windows, routes
from api.model_health import (
    ModelHealth,
    ModelServiceUnavailable,
    call_model,
    model_call,
    protect_video_review,
)
from api.review_scheduler import review_task_slot
from tests.test_window_review import install_video, sample


def test_rolling_ten_calls_and_early_five_failures():
    health = ModelHealth()
    for failed in [True, False] * 4:
        health.record("visual", failed)
    health.record("visual", False)
    with pytest.raises(ModelServiceUnavailable, match="最近 10 次调用中 5 次"):
        health.record("visual", True)
    assert health.stopped.is_set()

    # Old failures expire; four failures in each disjoint batch can still trip
    # when they overlap within a rolling ten-call window.
    health = ModelHealth()
    for failed in [True] * 4 + [False] * 10 + [True] * 4:
        health.record("visual", failed)
    assert not health.stopped.is_set()
    with pytest.raises(ModelServiceUnavailable):
        health.record("visual", True)

    health = ModelHealth()
    for _ in range(4):
        health.record("visual", True)
    with pytest.raises(ModelServiceUnavailable, match="最近 5 次调用中 5 次"):
        health.record("visual", True)


@pytest.mark.asyncio
async def test_cache_reuse_and_guard_failures_do_not_consume_visual_budget(monkeypatch):
    caption = AsyncMock(return_value="caption")
    monkeypatch.setattr(handlers, "_call_nsfw_analysis", caption)
    monkeypatch.setattr(
        handlers, "_call_llm_guard", AsyncMock(side_effect=httpx.ReadTimeout("guard"))
    )

    @protect_video_review
    async def review(url):
        for _ in range(4):
            with pytest.raises(httpx.ReadTimeout):
                await handlers._review_visual(["frame"], [0])
        cache = review_windows.AsyncMemo()
        calls = AsyncMock(return_value="cached caption")
        try:
            for _ in range(12):
                assert (
                    await cache.get("same", lambda: call_model("visual", calls)) == "cached caption"
                )
        finally:
            await cache.close()
        calls.assert_awaited_once()

    await review("https://fixture/video.mp4")
    assert caption.await_count == 4


@pytest.mark.asyncio
async def test_coalesced_failed_requests_count_once():
    @protect_video_review
    async def review(url):
        cache = review_windows.AsyncMemo()
        calls = 0

        async def caption():
            nonlocal calls
            calls += 1
            await asyncio.sleep(0)
            raise httpx.ReadTimeout("fixture")

        try:
            for _ in range(4):
                results = await asyncio.gather(
                    *(cache.get("same", lambda: call_model("visual", caption)) for _ in range(3)),
                    return_exceptions=True,
                )
                assert all(isinstance(result, httpx.ReadTimeout) for result in results)
        finally:
            await cache.close()
        assert calls == 4  # Twelve waiters, but only four model calls.

    await review("fixture.mp4")


@pytest.mark.asyncio
async def test_parallel_videos_have_independent_budgets_and_cancellation_is_not_failure():
    @protect_video_review
    async def review(url):
        failure = AsyncMock(side_effect=httpx.ReadTimeout("fixture"))
        for _ in range(4):
            with pytest.raises(httpx.ReadTimeout):
                await call_model("visual", failure)
            await asyncio.sleep(0)
        for _ in range(6):
            with pytest.raises(asyncio.CancelledError):
                await call_model("visual", AsyncMock(side_effect=asyncio.CancelledError()))
        return "completed"

    assert await asyncio.gather(review("a.mp4"), review("b.mp4")) == ["completed"] * 2
    assert await review("next.mp4") == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["window", "target"])
@pytest.mark.parametrize("failed_model", ["visual", "ocr", "guard", "face"])
async def test_breaker_stops_producer_models_and_persists_failure(
    monkeypatch, tmp_path, mode, failed_model
):
    samples = [sample(i, i, scene=i) for i in range(100)]
    install_video(monkeypatch, samples)

    class Sampler:
        interval = 1

        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __iter__(self):
            return iter(samples)

    monkeypatch.setattr(handlers, "VideoFrameSampler", Sampler)
    monkeypatch.setattr(review_windows, "VideoFrameSampler", Sampler)
    monkeypatch.setattr(handlers.settings, "nsfw_review_mode", mode)
    monkeypatch.setattr(handlers.settings, "review_window_concurrency", 2)
    writes = AsyncMock()
    failed = AsyncMock()
    complete = AsyncMock()
    monkeypatch.setattr(routes.review_task_store, "update_progress", writes)
    monkeypatch.setattr(routes.review_task_store, "fail", failed)
    monkeypatch.setattr(routes.review_task_store, "complete", complete)

    @asynccontextmanager
    async def slot():
        async with review_task_slot(directory=tmp_path, limit=1):
            yield

    monkeypatch.setattr(routes, "review_task_slot", slot)
    calls = 0
    blocked_stages = set()
    cancelled_stages = set()

    async def invoke(name):
        nonlocal calls
        if name == failed_model:
            calls += 1
            await asyncio.sleep(0.005)
            raise httpx.ReadTimeout("private upstream details")
        if calls >= 5:
            blocked_stages.add(name)
            try:
                await asyncio.Event().wait()
            finally:
                cancelled_stages.add(name)
        return {"visual": "caption", "ocr": "subtitle", "face": [], "guard": {"safe": True}}[name]

    for name, attribute in {
        "visual": "_call_nsfw_analysis",
        "ocr": "_call_ocr_api",
        "face": "_face_task",
        "guard": "_call_llm_guard",
    }.items():

        async def operation(*args, model=name, **kwargs):
            return await invoke(model)

        monkeypatch.setattr(handlers, attribute, model_call(name)(operation))
    before = asyncio.all_tasks()
    with pytest.raises(ModelServiceUnavailable, match="已提前终止"):
        await asyncio.wait_for(routes._run_review_task("broken", "fixture.mp4", 1, 10, 0.5), 1)
    assert 5 <= calls <= 8  # Other windows may already have a model request in flight.
    failed.assert_awaited_once()
    assert "5 次超时或错误" in failed.await_args.args[1]
    assert failed_model in failed.await_args.args[1].lower()
    assert "private" not in failed.await_args.args[1]
    complete.assert_not_awaited()
    if failed_model != "guard":
        assert blocked_stages
    assert cancelled_stages == blocked_stages
    assert not [task for task in asyncio.all_tasks() - before if not task.done()]

    # The next task can acquire the same sole slot and finish normally.
    monkeypatch.setattr(handlers, "_call_nsfw_analysis", AsyncMock(return_value="caption"))
    monkeypatch.setattr(handlers, "_call_llm_guard", AsyncMock(return_value={"safe": True}))
    monkeypatch.setattr(handlers, "_face_task", AsyncMock(return_value=[]))
    monkeypatch.setattr(handlers, "_call_ocr_api", AsyncMock(return_value=""))
    await asyncio.wait_for(routes._run_review_task("healthy", "fixture.mp4", 1, 10, 0.5), 3)
    complete.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["visual", "ocr", "guard", "face"])
async def test_each_real_model_entry_enforces_its_total_deadline(monkeypatch, model):
    from types import SimpleNamespace

    from api.model_health import TIMEOUT_SETTINGS
    from tests.test_nsfw_target_review import image

    monkeypatch.setattr(handlers.settings, TIMEOUT_SETTINGS[model], 0.01)
    cancelled = asyncio.Event()

    async def slow(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    client = AsyncMock()
    client.__aenter__.return_value = client
    client.post.side_effect = slow
    monkeypatch.setattr(handlers.httpx, "AsyncClient", lambda **kwargs: client)
    monkeypatch.setattr(handlers, "_request_nsfw_caption", slow)
    monkeypatch.setattr(handlers.ocr, "is_uniform_image", lambda _: False)
    monkeypatch.setattr(handlers.ocr, "request_ocr", slow)
    operations = {
        "visual": lambda: handlers._call_nsfw_analysis([image(30)], [0]),
        "ocr": lambda: handlers._call_ocr_api(image(30)),
        "guard": lambda: handlers._call_llm_guard("caption"),
        "face": lambda: handlers._face_task(
            SimpleNamespace(search_multi_face=slow), None, 10, 0.5, 0
        ),
    }
    with pytest.raises(httpx.ReadTimeout, match="deadline"):
        await asyncio.wait_for(operations[model](), 0.5)
    assert cancelled.is_set()


def test_failure_counters_are_separate_for_each_model():
    health = ModelHealth()
    for _ in range(4):
        for name in ("visual", "ocr", "guard", "face"):
            health.record(name, True)
    assert not health.stopped.is_set()
    with pytest.raises(ModelServiceUnavailable, match="guard"):
        health.record("guard", True)
