import asyncio
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient

from api import handlers, main, review_cancellation, review_task_store, routes
from api.main import create_app
from api.review_cancellation import ReviewTaskCancelled, run_cancellable_review
from api.review_events import ReviewEventBus
from api.review_scheduler import review_task_slot


@pytest.mark.asyncio
async def test_cancel_before_start_never_downloads(monkeypatch):
    monkeypatch.setattr(review_task_store, "is_enabled", lambda: True)
    monkeypatch.setattr(review_task_store, "cancellation_requested", AsyncMock(return_value=True))
    operation = AsyncMock()
    with pytest.raises(ReviewTaskCancelled):
        await run_cancellable_review("task", operation)
    operation.assert_not_called()


@pytest.mark.asyncio
async def test_cancel_cross_worker_waits_for_cleanup_and_releases_slot(monkeypatch, tmp_path):
    import tempfile

    # macOS Unix socket paths have a short length limit.
    with tempfile.TemporaryDirectory(prefix="wcm-cancel-", dir="/tmp") as directory:
        owner, requester = ReviewEventBus(directory), ReviewEventBus(directory)
        await owner.start()
        await requester.start()
        monkeypatch.setattr(review_cancellation, "review_events", owner)
        monkeypatch.setattr(review_task_store, "is_enabled", lambda: True)
        requested = False

        async def check(task_id):
            return requested

        monkeypatch.setattr(review_task_store, "cancellation_requested", check)
        started, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def operation():
            async with review_task_slot(directory=tmp_path, limit=1):
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cleaning.set()
                    await release.wait()

        task = asyncio.create_task(run_cancellable_review("task", operation, poll_interval=30))
        try:
            await asyncio.wait_for(started.wait(), 1)
            requested = True
            await requester.publish(
                {"type": "changed", "reason": "cancelling", "task_ids": ["task"]}
            )
            await asyncio.wait_for(cleaning.wait(), 1)  # IPC, not the 30-second fallback.
            assert not task.done()
            release.set()
            with pytest.raises(ReviewTaskCancelled):
                await asyncio.wait_for(task, 1)
            async with review_task_slot(directory=tmp_path, limit=1):
                pass
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await owner.close()
            await requester.close()


@pytest.mark.asyncio
async def test_queued_cancel_has_poll_fallback_without_taking_a_slot(monkeypatch, tmp_path):
    monkeypatch.setattr(review_task_store, "is_enabled", lambda: True)
    check = AsyncMock(side_effect=[False, True])
    monkeypatch.setattr(review_task_store, "cancellation_requested", check)
    downloaded = AsyncMock()

    async def operation():
        async with review_task_slot(directory=tmp_path, limit=1):
            await downloaded()

    async with review_task_slot(directory=tmp_path, limit=1):
        with pytest.raises(ReviewTaskCancelled):
            await asyncio.wait_for(run_cancellable_review("task", operation, poll_interval=0.01), 1)
    downloaded.assert_not_called()


@pytest.mark.asyncio
async def test_cancel_during_slow_download_closes_stream_and_file(monkeypatch, tmp_path):
    started = asyncio.Event()
    closed = False

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"a" * 65536
            started.set()
            await asyncio.Event().wait()

        async def aclose(self):
            nonlocal closed
            closed = True

    original = httpx.AsyncClient
    transport = httpx.MockTransport(lambda request: httpx.Response(200, stream=Stream()))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: original(transport=transport, **kw))
    progress = MagicMock()
    progress.begin_download = AsyncMock()
    progress.finish_download = AsyncMock()
    path = tmp_path / "movie.mp4"
    task = asyncio.create_task(
        handlers._download_review_video(
            "http://fixture/movie.mp4", path, 1000000, progress=progress
        )
    )
    await asyncio.wait_for(started.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 1)
    assert closed and not path.exists()
    assert not list(tmp_path.glob("wcm-ingest-*"))
    progress.finish_download.assert_not_called()
    path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_completion_losing_cancel_race_never_reports_finished(monkeypatch):
    monkeypatch.setattr(review_task_store, "is_enabled", lambda: False)
    monkeypatch.setattr(routes, "_process_analyze_media", AsyncMock(return_value=[]))
    monkeypatch.setattr(review_task_store, "complete", AsyncMock(return_value=False))
    cancelled, failed = AsyncMock(), AsyncMock()
    monkeypatch.setattr(review_task_store, "cancelled", cancelled)
    monkeypatch.setattr(review_task_store, "fail", failed)
    with pytest.raises(ReviewTaskCancelled):
        await routes._run_review_task("task", "fixture.mp4", 1, 10, 0.5)
    cancelled.assert_awaited_once()
    failed.assert_not_called()


@pytest.mark.parametrize(
    "status,expected",
    [
        ("cancelling", 200),
        ("cancelled", 200),
        ("completed", 409),
        ("partial", 409),
        ("failed", 409),
        (None, 404),
    ],
)
def test_cancel_endpoint_is_idempotent_and_rejects_terminal_tasks(monkeypatch, status, expected):
    request = AsyncMock(return_value={"id": "task", "status": status} if status else None)
    monkeypatch.setattr(review_task_store, "request_cancel", request)
    with TestClient(create_app()) as client:
        response = client.post("/api/v1/review_tasks/task/cancel")
    assert response.status_code == expected
    request.assert_awaited_once_with("task")


def test_cancel_endpoint_reports_unavailable_storage(monkeypatch):
    monkeypatch.setattr(
        review_task_store,
        "request_cancel",
        AsyncMock(side_effect=review_task_store.ReviewTaskStoreUnavailable("offline")),
    )
    with TestClient(create_app()) as client:
        assert client.post("/api/v1/review_tasks/task/cancel").status_code == 503


@pytest.mark.parametrize("status", ["processing", "cancelling"])
def test_active_task_cannot_be_deleted(monkeypatch, status):
    connection = MagicMock()
    connection.__enter__.return_value = connection
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = [{"status": "completed"}, {"status": status}]
    monkeypatch.setattr(review_task_store, "_connect", lambda: connection)
    with pytest.raises(review_task_store.ReviewTaskConflict):
        review_task_store._delete_many_sync(["finished", "active"])
    connection.rollback.assert_called_once()
    assert all("DELETE" not in call.args[0] for call in cursor.execute.call_args_list)


def test_late_progress_and_completion_cannot_overwrite_cancelled_status(monkeypatch):
    connection = MagicMock()
    connection.__enter__.return_value = connection
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.rowcount = 0
    monkeypatch.setattr(review_task_store, "_connect", lambda: connection)
    assert review_task_store._complete_sync("task", []) is False
    assert review_task_store._fail_sync("task", "late failure") is False
    review_task_store._update_progress_sync("task", {"sequence": 12})
    for call in cursor.execute.call_args_list:
        assert "status = 'processing'" in call.args[0]


def test_websocket_cancellation_keeps_record_and_does_not_emit_results(monkeypatch):
    states = {}
    monkeypatch.setattr(review_task_store, "is_enabled", lambda: True)
    monkeypatch.setattr(main, "initialize_review_tasks", AsyncMock())
    monkeypatch.setattr(review_task_store, "update_progress", AsyncMock())
    monkeypatch.setattr(review_task_store, "complete", AsyncMock())
    monkeypatch.setattr(review_task_store, "fail", AsyncMock())

    async def create(url, parameters, task_id):
        states[task_id] = "processing"
        return task_id

    async def check(task_id):
        return states[task_id] == "cancelling"

    async def request(task_id):
        states[task_id] = "cancelling"
        await review_cancellation.review_events.publish(
            {"type": "changed", "reason": "cancelling", "task_ids": [task_id]}
        )
        return {"id": task_id, "status": states[task_id]}

    async def cancelled(task_id, progress):
        states[task_id] = "cancelled"

    async def review(*args, **kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(review_task_store, "create", create)
    monkeypatch.setattr(review_task_store, "cancellation_requested", check)
    monkeypatch.setattr(review_task_store, "request_cancel", request)
    monkeypatch.setattr(review_task_store, "cancelled", cancelled)
    monkeypatch.setattr(routes, "_process_analyze_media", review)
    with TestClient(create_app()) as client:
        with client.websocket_connect("/api/v1/ws/analyze_media") as socket:
            socket.send_json({"url": "https://fixture/movie.mp4"})
            task_id = socket.receive_json()["taskId"]
            assert client.post(f"/api/v1/review_tasks/{task_id}/cancel").status_code == 200
            for _ in range(5):
                event = socket.receive_json()
                if event.get("status") == "cancelled":
                    break
            assert event == {"status": "cancelled", "taskId": task_id}
            assert states[task_id] == "cancelled"
    review_task_store.complete.assert_not_called()
    review_task_store.fail.assert_not_called()
