import asyncio
import json
import sys
import tempfile
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from api import main, review_stream, review_task_store, routes
from api.review_events import ReviewEventBus


@pytest.fixture
def bus(monkeypatch):
    with tempfile.TemporaryDirectory(prefix="wcm-ws-", dir="/tmp") as directory:
        events = ReviewEventBus(directory)
        for module in (main, review_stream, review_task_store):
            monkeypatch.setattr(module, "review_events", events)
        monkeypatch.setattr(main, "initialize_review_tasks", AsyncMock())
        yield events


def test_submission_socket_delivers_accepted_progress_and_final_result(monkeypatch, bus):
    monkeypatch.setattr(review_task_store, "is_enabled", lambda: True)
    monkeypatch.setattr(review_task_store, "create", AsyncMock())
    monkeypatch.setattr(review_task_store, "complete", AsyncMock())
    monkeypatch.setattr(review_task_store, "_run", AsyncMock(return_value=True))

    async def summaries(ids):
        return [{"id": ids[0], "status": "completed", "progress": {"percent": 100}}]

    monkeypatch.setattr(review_task_store, "get_summaries", summaries)

    async def analyze(*args, progress, **kwargs):
        await progress.begin_review(20)
        await progress.start_window(0, 3, 5, [3, 4, 5])
        await progress.start_stage(0, "ocr", [4])
        await progress.report(force=True)
        await asyncio.sleep(0.03)
        return []

    monkeypatch.setattr(routes, "_process_analyze_media", analyze)
    with TestClient(main.create_app()) as client:
        with client.websocket_connect("/api/v1/ws/analyze_media") as socket:
            socket.send_json({"url": "https://fixture/video.mp4"})
            accepted = socket.receive_json()
            assert accepted["status"] == "accepted"
            messages = []
            while True:
                message = socket.receive_json()
                assert message.get("status") != "error", message
                messages.append(message)
                if message.get("status") == "completed":
                    break
            assert any(
                message.get("progress", {}).get("active_windows", [{}])[0].get("stages")
                == {"ocr": [4]}
                for message in messages
                if message.get("progress", {}).get("active_windows")
            )
            assert messages[-1]["results"] == []
            assert messages[-1]["task"]["status"] == "completed"
            assert messages[-1]["taskId"] == accepted["taskId"]
    assert not bus.listeners


def test_subscription_reads_one_snapshot_then_pushes_without_polling(monkeypatch, bus):
    async def snapshot(ids):
        await bus.publish({"type": "progress", "task_id": "task", "progress": {"sequence": 2}})
        return [{"id": "task", "status": "processing", "progress": {"sequence": 1}}]

    read = AsyncMock(side_effect=snapshot)
    monkeypatch.setattr(review_task_store, "get_summaries", read)
    monkeypatch.setattr(review_stream, "HEARTBEAT_SECONDS", 0.02)
    with TestClient(main.create_app()) as client:
        with client.websocket_connect("/api/v1/ws/analyze_media") as socket:
            socket.send_json({"type": "subscribe", "task_ids": ["task"]})
            assert socket.receive_json()["type"] == "snapshot"
            assert socket.receive_json()["progress"]["sequence"] == 2
            for _ in range(3):
                assert socket.receive_json()["type"] == "heartbeat"
            read.assert_awaited_once_with(["task"])
            client.portal.call(
                bus.publish, {"type": "progress", "task_id": "other", "progress": {}}
            )
            client.portal.call(
                bus.publish, {"type": "changed", "task_ids": ["task"], "reason": "completed"}
            )
            assert socket.receive_json()["reason"] == "completed"
    assert not bus.listeners


def test_task_list_subscription_gets_new_task_events_without_submitting(monkeypatch, bus):
    create = AsyncMock()
    monkeypatch.setattr(review_task_store, "create", create)
    with TestClient(main.create_app()) as client:
        with client.websocket_connect("/api/v1/ws/analyze_media") as socket:
            socket.send_json({"type": "subscribe", "task_ids": [], "watch_list": True})
            assert socket.receive_json()["tasks"] == []
            client.portal.call(
                bus.publish, {"type": "changed", "task_ids": ["new"], "reason": "created"}
            )
            assert socket.receive_json()["task_ids"] == ["new"]
    create.assert_not_awaited()


@pytest.mark.asyncio
async def test_worker_events_cross_process_without_external_service():
    with tempfile.TemporaryDirectory(prefix="wcm-ipc-", dir="/tmp") as directory:
        bus = ReviewEventBus(directory)
        await bus.start()
        try:
            async with bus.subscribe() as queue:
                event = {"type": "progress", "task_id": "cross-worker", "progress": {"sequence": 3}}
                code = f"""import asyncio
from api.review_events import ReviewEventBus
async def run():
 bus=ReviewEventBus({directory!r})
 await bus.start()
 try: await bus.publish({event!r})
 finally: await bus.close()
asyncio.run(run())
"""
                process = await asyncio.create_subprocess_exec(sys.executable, "-c", code)
                assert await asyncio.wait_for(queue.get(), timeout=5) == event
                assert await process.wait() == 0
        finally:
            await bus.close()


@pytest.mark.asyncio
async def test_disconnected_progress_socket_does_not_cancel_audit(monkeypatch, bus):
    class BrokenSocket:
        async def send_json(self, payload):
            raise OSError("disconnected")

    completed = False
    async with review_stream.push_review_progress(BrokenSocket(), "task"):
        await bus.publish({"type": "progress", "task_id": "task", "progress": {"sequence": 1}})
        await asyncio.sleep(0.01)
        completed = True
    assert completed
    assert not bus.listeners


@pytest.mark.asyncio
async def test_slow_subscriber_resyncs_and_stale_progress_is_not_published(monkeypatch, bus):
    async with bus.subscribe() as queue:
        for _ in range(129):
            await bus.publish({"type": "changed", "task_ids": ["task"]})
        assert (await queue.get())["type"] == "resync"
    monkeypatch.setattr(review_task_store, "is_enabled", lambda: True)
    monkeypatch.setattr(review_task_store, "_run", AsyncMock(side_effect=[False, True]))
    async with bus.subscribe() as queue:
        await review_task_store.update_progress("task", {"sequence": 2})
        assert queue.empty()
        await review_task_store.update_progress("task", {"sequence": 3})
        assert (await queue.get())["progress"]["sequence"] == 3
