import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock

import pymysql
import pytest

from api import review_task_store as store
from api import routes, task_queue
from api.review_cancellation import ReviewTaskCancelled
from wcm_facerec.config import settings
from wcm_facerec.execution import execution_scope


@pytest.mark.asyncio
@pytest.mark.parametrize("saved", [True, False])
async def test_deadlock_retries_save_without_repeating_review_or_evidence(monkeypatch, saved):
    monkeypatch.setattr(settings, "cluster_enabled", True)
    monkeypatch.setattr(store, "is_enabled", lambda: True)
    monkeypatch.setattr(store, "_DEADLOCK_RETRY_DELAYS", (0,))
    review = AsyncMock(return_value=[])
    archive = Mock(side_effect=lambda task_id, results: results)
    failed, cancelled, publish = AsyncMock(), AsyncMock(), AsyncMock()
    monkeypatch.setattr(routes, "_process_analyze_media", review)
    monkeypatch.setattr(store, "archive_evidence", archive)
    monkeypatch.setattr(store, "fail", failed)
    monkeypatch.setattr(store, "cancelled", cancelled)
    monkeypatch.setattr(store, "cancellation_requested", AsyncMock(return_value=False))
    monkeypatch.setattr(store, "update_progress", AsyncMock())
    monkeypatch.setattr(store.review_events, "publish", publish)
    connections = [MagicMock(), MagicMock()]
    cursors = []
    for connection in connections:
        connection.__enter__.return_value = connection
        cursors.append(connection.cursor.return_value.__enter__.return_value)
    cursors[0].execute.side_effect = pymysql.err.OperationalError(1213, "deadlock")
    # A cancellation or expired/replaced lease can win before the second write.
    cursors[1].rowcount = int(saved)
    connect = Mock(side_effect=connections)
    monkeypatch.setattr(store, "_connect", connect)

    with execution_scope("task", "token", 1):
        if saved:
            assert await routes._run_review_task("task", "fixture.mp4", 1, 10, 0.5) == []
        else:
            with pytest.raises(ReviewTaskCancelled):
                await routes._run_review_task("task", "fixture.mp4", 1, 10, 0.5)

    review.assert_awaited_once()
    archive.assert_called_once()
    failed.assert_not_called()
    assert cancelled.await_count == int(not saved)
    assert connect.call_count == 2
    assert cursors[0].execute.call_args == cursors[1].execute.call_args
    for connection, cursor in zip(connections, cursors, strict=True):
        connection.__exit__.assert_called_once()
        sql, values = cursor.execute.call_args.args
        assert "status = 'processing'" in sql
        assert "lease_token = %s AND lease_expires > UTC_TIMESTAMP(3)" in sql
        assert values[-2:] == ("task", "token")
    completed = [c for c in publish.await_args_list if c.args[0].get("reason") == "completed"]
    assert len(completed) == int(saved)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "errno,retry,expected_calls",
    [(1213, True, 4), (1213, False, 1), (1205, True, 1), (2013, True, 1)],
)
async def test_database_retry_is_bounded_and_only_replays_rolled_back_deadlocks(
    monkeypatch, caplog, errno, retry, expected_calls
):
    monkeypatch.setattr(store, "_DEADLOCK_RETRY_DELAYS", (0, 0, 0))
    calls = 0

    def operation():
        nonlocal calls
        calls += 1
        raise pymysql.err.OperationalError(errno, "private SQL payload")

    with pytest.raises(store.ReviewTaskStoreUnavailable) as caught:
        await store._run(operation, retry_deadlocks=retry)
    assert calls == expected_calls
    assert caught.value.__cause__.args[0] == errno
    assert f"mysql_errno={errno}" in caplog.text
    assert "private SQL payload" not in caplog.text


@pytest.mark.asyncio
async def test_cancellation_stops_deadlock_retry(monkeypatch):
    monkeypatch.setattr(store, "_DEADLOCK_RETRY_DELAYS", (60,))
    entered = asyncio.Event()
    loop = asyncio.get_running_loop()
    calls = 0

    def operation():
        nonlocal calls
        calls += 1
        loop.call_soon_threadsafe(entered.set)
        raise pymysql.err.OperationalError(1213, "deadlock")

    task = asyncio.create_task(store._run(operation, retry_deadlocks=True))
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 1)
    assert calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["claim", "renew"])
async def test_queue_deadlock_retries_the_database_operation(monkeypatch, operation):
    monkeypatch.setattr(store, "_DEADLOCK_RETRY_DELAYS", (0,))
    calls = 0

    def database_operation(*args):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise pymysql.err.OperationalError(1213, "deadlock")
        return None if operation == "claim" else True

    monkeypatch.setattr(task_queue, f"_{operation}_sync", database_operation)
    if operation == "claim":
        assert await task_queue.claim("worker") is None
    else:
        assert await task_queue.renew("task", "token") is True
    assert calls == 2
