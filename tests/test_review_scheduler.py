import asyncio
import sys
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from api import routes
from api.review_scheduler import review_task_slot
from wcm_facerec.config import Settings


async def acquire_once(directory, limit=1):
    async with review_task_slot(directory=directory, limit=limit):
        return True


@pytest.mark.asyncio
async def test_slots_are_shared_with_other_processes_and_released_on_exit(tmp_path):
    script = """
import asyncio, sys
from api.review_scheduler import review_task_slot
async def main():
    async with review_task_slot(directory=sys.argv[1], limit=1):
        print('acquired', flush=True)
        await asyncio.Event().wait()
asyncio.run(main())
"""
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-c", script, str(tmp_path), stdout=asyncio.subprocess.PIPE
    )
    try:
        assert await asyncio.wait_for(process.stdout.readline(), 5) == b"acquired\n"
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(acquire_once(tmp_path), 0.35)
    finally:
        process.kill()
        await process.wait()
    assert await asyncio.wait_for(acquire_once(tmp_path), 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [1, 4])
async def test_review_entry_queues_and_releases_slots_after_failure_and_cancellation(
    monkeypatch, tmp_path, limit
):
    @asynccontextmanager
    async def slot():
        async with review_task_slot(directory=tmp_path, limit=limit):
            yield

    monkeypatch.setattr(routes, "review_task_slot", slot)
    writes = AsyncMock()
    monkeypatch.setattr(routes.review_task_store, "update_progress", writes)
    monkeypatch.setattr(routes.review_task_store, "complete", AsyncMock())
    monkeypatch.setattr(routes.review_task_store, "fail", AsyncMock())
    entered = asyncio.Queue()
    releases = {str(i): asyncio.Event() for i in range(limit + 2)}
    active = peak = 0

    async def analyze(url, *args, **kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        entered.put_nowait(url)
        try:
            await releases[url].wait()
            if url == "0":
                raise ValueError("fixture failure")
            return []
        finally:
            active -= 1

    monkeypatch.setattr(routes, "_process_analyze_media", analyze)
    tasks = []
    try:
        for i in range(limit):
            tasks.append(asyncio.create_task(routes._run_review_task(str(i), str(i), 1, 5, 0.5)))
            assert await asyncio.wait_for(entered.get(), 1) == str(i)
        waiting = asyncio.create_task(routes._run_review_task("waiting", str(limit), 1, 5, 0.5))
        tasks.append(waiting)
        await asyncio.sleep(0.05)
        assert entered.empty()
        assert any(c.args[0] == "waiting" and c.args[1]["phase"] == "queued" for c in writes.call_args_list)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        successor = asyncio.create_task(routes._run_review_task("next", str(limit + 1), 1, 5, 0.5))
        tasks.append(successor)
        releases["0"].set()
        with pytest.raises(ValueError):
            await tasks[0]
        assert await asyncio.wait_for(entered.get(), 1) == str(limit + 1)
        assert peak == limit
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    # Cancelling active and waiting reviews must not leak capacity.
    assert await asyncio.wait_for(acquire_once(tmp_path, limit), 1)


def test_concurrency_configuration(monkeypatch):
    for field in ("review_task_concurrency", "review_window_concurrency"):
        assert Settings.model_fields[field].default == (4 if field == "review_task_concurrency" else 2)
        with pytest.raises(ValidationError):
            Settings(_env_file=None, **{field: 0})
        monkeypatch.setenv(f"WCM_{field.upper()}", "3")
        assert getattr(Settings(_env_file=None), field) == 3
