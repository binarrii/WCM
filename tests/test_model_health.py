"""Visual failure budgets stop whole videos without leaking workers or task slots."""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import httpx
import pytest

from api import handlers, model_health, review_windows, routes
from api.model_health import (
    ModelHealth,
    ModelServiceUnavailable,
    call_model,
    model_call,
    protect_video_review,
)
from api.review_scheduler import review_task_slot
from tests.test_window_review import install_video, sample


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["visual", "ocr", "guard", "face"])
@pytest.mark.parametrize("error_type", [httpx.ReadTimeout, httpx.ConnectError, ValueError])
@pytest.mark.parametrize("recover", [True, False])
async def test_only_final_retry_outcome_counts(model, error_type, recover, caplog):
    health = ModelHealth()
    token = model_health._current_health.set(health)
    attempts = 0

    async def invoke():
        nonlocal attempts
        attempts += 1
        assert not health.outcomes  # The first failed attempt is not recorded.
        if recover and attempts == 2:
            return "result"
        raise error_type(f"private attempt {attempts}")

    try:
        if recover:
            assert await call_model(model, invoke) == "result"
        else:
            with pytest.raises(error_type, match="attempt 2"):
                await call_model(model, invoke)
        assert attempts == 2
        assert list(health.outcomes[model]) == [not recover]
        assert not health.stopped.is_set()
        assert "retrying once" in caplog.text
        assert "private" not in caplog.text
    finally:
        model_health._current_health.reset(token)


@pytest.mark.asyncio
async def test_recovered_errors_do_not_trip_but_five_final_failures_do():
    @protect_video_review
    async def review(url):
        recovered = AsyncMock(side_effect=[httpx.ReadTimeout("first"), "ok"] * 10)
        for _ in range(10):
            assert await call_model("visual", recovered) == "ok"
        failure = AsyncMock(side_effect=httpx.ReadTimeout("offline"))
        for _ in range(4):
            with pytest.raises(httpx.ReadTimeout):
                await call_model("visual", failure)
        with pytest.raises(ModelServiceUnavailable, match="调用频繁错误或超时"):
            await call_model("visual", failure)
        assert failure.await_count == 10
        # An open circuit never makes another upstream request.
        with pytest.raises(ModelServiceUnavailable):
            await call_model("visual", failure)
        assert failure.await_count == 10

    with pytest.raises(ModelServiceUnavailable):
        await review("fixture.mp4")


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_on", [1, 2])
async def test_cancellation_never_retries_or_records_failure(cancel_on):
    health = ModelHealth()
    token = model_health._current_health.set(health)
    operation = AsyncMock(
        side_effect=([httpx.ReadTimeout("first")] if cancel_on == 2 else [])
        + [asyncio.CancelledError()]
    )
    try:
        with pytest.raises(asyncio.CancelledError):
            await call_model("visual", operation)
        assert operation.await_count == cancel_on
        assert not health.outcomes
    finally:
        model_health._current_health.reset(token)


@pytest.mark.asyncio
async def test_other_model_opening_circuit_prevents_retry():
    health = ModelHealth()
    token = model_health._current_health.set(health)

    async def invoke():
        health.error = ModelServiceUnavailable("guard stopped video")
        health.stopped.set()
        raise httpx.ReadTimeout("visual first attempt")

    operation = AsyncMock(side_effect=invoke)
    try:
        with pytest.raises(ModelServiceUnavailable, match="guard stopped"):
            await call_model("visual", operation)
        operation.assert_awaited_once()
        assert not health.outcomes
    finally:
        model_health._current_health.reset(token)


@pytest.mark.asyncio
async def test_timeout_retry_has_a_fresh_deadline(monkeypatch):
    monkeypatch.setattr(handlers.settings, "visual_timeout_s", 0.03)
    attempts = 0
    closed = []

    async def operation():
        nonlocal attempts
        attempts += 1
        try:
            if attempts == 1:
                await asyncio.Event().wait()
            await asyncio.sleep(0.005)
            return "recovered"
        finally:
            closed.append(attempts)

    assert await call_model("visual", operation) == "recovered"
    assert attempts == 2 and closed == [1, 2]


def test_rolling_ten_calls_and_early_five_failures(caplog):
    health = ModelHealth()
    for failed in [True, False] * 4:
        health.record("visual", failed)
    health.record("visual", False)
    with pytest.raises(ModelServiceUnavailable, match="调用频繁错误或超时"):
        health.record("visual", True)
    assert health.stopped.is_set()
    assert "completed_calls=10 failures=5" in caplog.text

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
    with pytest.raises(ModelServiceUnavailable, match="调用频繁错误或超时"):
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
        assert calls == 8  # Twelve waiters, four operations with two attempts each.

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
        if calls >= 9:  # Let four operations exhaust both attempts before blocking siblings.
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
    assert 10 <= calls <= 16  # Five final failures plus other windows already in flight.
    failed.assert_awaited_once()
    assert failed.await_args.args[1] == (
        f"{model_health.MODEL_LABELS[failed_model]}模型调用频繁错误或超时，已提前终止该审核任务。"
    )
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
    attempts = 0
    cancellations = 0

    async def slow(*args, **kwargs):
        nonlocal attempts, cancellations
        attempts += 1
        try:
            await asyncio.Event().wait()
        finally:
            cancellations += 1

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
    assert attempts == cancellations == 2


def test_failure_counters_are_separate_for_each_model():
    health = ModelHealth()
    for _ in range(4):
        for name in ("visual", "ocr", "guard", "face"):
            health.record(name, True)
    assert not health.stopped.is_set()
    with pytest.raises(ModelServiceUnavailable, match="guard"):
        health.record("guard", True)
