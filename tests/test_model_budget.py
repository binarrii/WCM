import asyncio
import threading
import time

import httpx
import pytest

from api.model_health import call_model
from wcm_facerec import model_budget
from wcm_facerec.cluster import run_sync
from wcm_facerec.config import settings
from wcm_facerec.ifs_adapter import InsightFaceAdapter


@pytest.mark.asyncio
async def test_timeout_stops_composed_sdk_before_more_requests_and_retry_gets_fresh_budget(
    monkeypatch,
):
    calls = []

    def request(self, *args, **kwargs):
        calls.append((model_budget._current_budget.get(), kwargs["timeout"]))
        # Simulate one already-running HTTP call draining after the deadline.
        time.sleep(0.06)
        return {}

    monkeypatch.setattr("wcm_facerec.ifs_adapter.Client._request", request)
    monkeypatch.setattr(settings, "insightface_timeout_s", 0.03)
    target = InsightFaceAdapter("http://example.invalid", "people", timeout=10)

    def composed():
        for _ in range(10):
            target._client._request("POST", "/v1/search")

    try:
        with pytest.raises(httpx.ReadTimeout):
            await asyncio.wait_for(call_model("face", lambda: run_sync(composed)), 2)
    finally:
        target._client.close()
    assert len(calls) == 2  # One HTTP request per attempt, not all 20.
    assert calls[0][0] is not calls[1][0]
    assert all(0 < timeout <= 0.03 for _, timeout in calls)


@pytest.mark.asyncio
async def test_cancel_drains_current_sdk_request_before_releasing_scope_but_stops_followups(
    monkeypatch,
):
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    calls = []

    def request(self, *args, **kwargs):
        calls.append(model_budget._current_budget.get())
        started.set()
        assert release.wait(2)
        finished.set()
        return {}

    monkeypatch.setattr("wcm_facerec.ifs_adapter.Client._request", request)
    target = InsightFaceAdapter("http://example.invalid", "people", timeout=10)

    def composed():
        target._client._request("POST", "/v1/detect")
        target._client._request("POST", "/v1/search")

    task = asyncio.create_task(call_model("face", lambda: run_sync(composed)))
    try:
        assert await asyncio.to_thread(started.wait, 1)
        task.cancel()
        assert await asyncio.to_thread(calls[0].cancelled.wait, 1)
        assert not task.done() and not finished.is_set()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
        assert finished.is_set() and len(calls) == 1
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        target._client.close()


def test_sdk_outside_model_attempt_keeps_its_existing_timeout(monkeypatch):
    calls = []

    def request(self, *args, **kwargs):
        calls.append(kwargs)
        return {}

    monkeypatch.setattr("wcm_facerec.ifs_adapter.Client._request", request)
    target = InsightFaceAdapter("http://example.invalid", "people", timeout=10)
    try:
        target._client._request("GET", "/v1/health")
    finally:
        target._client.close()
    assert calls == [{}]  # No new budget is imposed on library writes/sync.


@pytest.mark.asyncio
async def test_cancellation_budget_is_isolated_between_parallel_operations():
    async def operation(cancel):
        with model_budget.model_request_budget(10):
            if cancel:
                model_budget.cancel_model_requests()
            await asyncio.sleep(0)
            if cancel:
                with pytest.raises(httpx.ReadTimeout):
                    await asyncio.to_thread(model_budget.remaining_request_time)
            else:
                assert await asyncio.to_thread(model_budget.remaining_request_time) > 0

    await asyncio.gather(operation(True), operation(False))
    assert model_budget.remaining_request_time() is None
