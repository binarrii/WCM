import asyncio
import base64
from unittest.mock import AsyncMock

import cv2
import numpy as np
import pytest

from api import handlers, review_progress, routes
from api.review_progress import ReviewProgress
from tests.test_nsfw_target_review import caption, install_client
from tests.test_window_review import install_video, sample


@pytest.mark.asyncio
async def test_out_of_order_completion_keeps_active_samples_and_contiguous_position(monkeypatch):
    writes = AsyncMock()
    monkeypatch.setattr(review_progress.review_task_store, "update_progress", writes)
    progress = ReviewProgress("task", interval=0)
    await progress.begin_review(20)
    progress.enqueue(3)
    progress.enqueue(3)
    await progress.start_window(0, 0, 2, [0, 1, 2])
    await progress.start_window(1, 3, 5, [3, 4, 5])
    await progress.start_stage(0, "ocr", [1])
    assert progress.snapshot()["active_windows"][0]["stages"] == {"ocr": [1]}
    await progress.complete_window(1, 5, 3)
    assert progress.snapshot()["processed_seconds"] == 0
    assert progress.snapshot()["completed_windows"] == 1
    assert len(progress.snapshot()["active_windows"]) == 1
    await progress.complete_window(0, 2, 3)
    assert progress.snapshot()["processed_seconds"] == 5
    assert progress.snapshot()["percent"] == 25
    await progress.sampled()
    assert progress.snapshot()["total_samples"] == 6
    await progress.set_phase("saving")
    assert progress.snapshot()["percent"] == 99
    await progress.set_phase("finished", persist=False)
    assert progress.snapshot()["percent"] == 100
    assert writes.await_args.args[1]["phase"] == "saving"


@pytest.mark.asyncio
async def test_progress_is_throttled_isolated_and_cleans_up_heartbeat(monkeypatch):
    writes = AsyncMock()
    monkeypatch.setattr(review_progress.review_task_store, "update_progress", writes)
    async with ReviewProgress("first", interval=100) as progress:
        for _ in range(10):
            await progress.report()
        assert writes.await_count == 1
    assert progress.heartbeat.done()
    assert ReviewProgress("second").snapshot()["percent"] is None
    assert ReviewProgress("second").snapshot()["active_windows"] == []


@pytest.mark.asyncio
async def test_database_failure_does_not_abort_progress_or_audit(monkeypatch):
    monkeypatch.setattr(
        review_progress.review_task_store,
        "update_progress",
        AsyncMock(side_effect=RuntimeError("offline")),
    )
    progress = ReviewProgress("task", interval=0)
    await progress.begin_review()
    progress.enqueue(1)
    await progress.complete_window(0, 1, 1)
    await progress.sampled()
    assert progress.snapshot()["percent"] == 99
    await progress.set_phase("failed", persist=False)
    assert progress.snapshot()["percent"] == 99  # An interrupted task never becomes 100%.


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [1, 2, 4])
async def test_window_workers_limit_concurrency_and_report_stage_positions(monkeypatch, limit):
    monkeypatch.setattr(handlers.settings, "review_window_concurrency", limit)
    install_video(monkeypatch, [sample(i, i * 10) for i in range(18)])
    snapshots = []

    async def write(task_id, data):
        snapshots.append(data)

    monkeypatch.setattr(review_progress.review_task_store, "update_progress", write)
    active = maximum = 0

    async def visual(*args, **kwargs):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        try:
            await asyncio.sleep(0.02)
            return "description"
        finally:
            active -= 1

    monkeypatch.setattr(handlers, "_call_nsfw_analysis", visual)
    monkeypatch.setattr(handlers, "_call_ocr_api", AsyncMock(return_value=""))
    monkeypatch.setattr(handlers, "_face_task", AsyncMock(return_value=[]))
    monkeypatch.setattr(handlers, "_call_llm_guard", AsyncMock(return_value={"safe": True}))
    progress = ReviewProgress("task", interval=0)
    await handlers._process_analyze_media("https://fixture/video.mp4", 1, 5, 0.5, progress=progress)
    assert maximum == limit
    assert progress.snapshot()["completed_windows"] == 6
    assert progress.snapshot()["completed_samples"] == 18
    assert any(len(snapshot["active_windows"]) == limit for snapshot in snapshots)
    assert any(
        window["stages"].get("visual") == [0, 1, 2]
        for snapshot in snapshots
        for window in snapshot["active_windows"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["multi_image", "montage", "fallback"])
async def test_every_visual_request_image_including_montage_is_bounded(monkeypatch, mode):
    import httpx

    monkeypatch.setattr(
        handlers.settings, "nsfw_image_mode", "auto" if mode == "fallback" else mode
    )
    replies = [caption("description")]
    if mode == "fallback":
        request = httpx.Request("POST", "https://fixture/model")
        response = httpx.Response(400, request=request, text="Only one image is supported")
        replies.insert(0, httpx.HTTPStatusError("unsupported", request=request, response=response))
    calls = install_client(monkeypatch, replies)
    images = []
    for height, width in [(1600, 1200), (1080, 1920), (480, 640)]:
        ok, encoded = cv2.imencode(".jpg", np.zeros((height, width, 3), np.uint8))
        assert ok
        images.append(base64.b64encode(encoded).decode())
    await handlers._call_nsfw_analysis(images, [0, 1, 2], review_all=True)
    shapes = []
    for payload in calls:
        for item in payload["messages"][-1]["content"]:
            if item["type"] == "image_url":
                pixels = cv2.imdecode(
                    np.frombuffer(
                        base64.b64decode(item["image_url"]["url"].split(",")[1]), np.uint8
                    ),
                    cv2.IMREAD_COLOR,
                )
                assert max(pixels.shape[:2]) <= 960
                shapes.append(pixels.shape[:2])
    if mode == "multi_image":
        assert shapes == [(960, 720), (540, 960), (480, 640)]


@pytest.mark.asyncio
async def test_result_persistence_precedes_finished_progress(monkeypatch):
    events = []

    async def write(task_id, progress):
        events.append(progress["phase"])

    async def complete(*args):
        events.append("persisted")

    monkeypatch.setattr(routes, "_process_analyze_media", AsyncMock(return_value=[]))
    monkeypatch.setattr(review_progress.review_task_store, "update_progress", write)
    monkeypatch.setattr(routes.review_task_store, "complete", complete)
    await routes._run_review_task("task", "https://fixture/video.mp4", 1, 10, 0.5)
    assert events == ["queued", "downloading", "saving", "persisted"]
