import asyncio
import base64
from contextlib import asynccontextmanager
from unittest.mock import Mock

import pytest

from api import review_task_store
from api.review_evidence import archive_evidence
from wcm_facerec import cluster, image_store, runtime_parameters
from wcm_facerec.cluster import run_sync
from wcm_facerec.config import settings
from wcm_facerec.execution import execution_scope


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["insightface", "ocr", "visual", "guard"])
@pytest.mark.parametrize("limit", [0, -3])
async def test_unlimited_models_skip_cluster_admission_and_propagate_cancel(
    monkeypatch, model, limit
):
    monkeypatch.setattr(settings, "cluster_enabled", True)
    admission = Mock(side_effect=AssertionError("unlimited calls must not acquire database slots"))
    monkeypatch.setattr(cluster, "cluster_slot", admission)
    before = runtime_parameters.snapshot()
    key = f"{model}_concurrency"
    entered = 0
    ready = asyncio.Event()
    tasks = []

    async def operation():
        nonlocal entered
        # Model admission follows live limits even for an older submitted task.
        with runtime_parameters.frozen({key: 1}):
            async with cluster.model_slot(model):
                entered += 1
                if entered == 5:
                    ready.set()
                await asyncio.Event().wait()

    try:
        runtime_parameters.install({key: limit})
        tasks = [asyncio.create_task(operation()) for _ in range(5)]
        await asyncio.wait_for(ready.wait(), 1)
        admission.assert_not_called()
    finally:
        for task in tasks:
            task.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        runtime_parameters.install(before)
    assert all(isinstance(result, asyncio.CancelledError) for result in results)


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["insightface", "ocr", "visual", "guard"])
async def test_positive_model_limit_uses_live_cluster_admission_and_releases(monkeypatch, model):
    monkeypatch.setattr(settings, "cluster_enabled", True)
    calls = []

    @asynccontextmanager
    async def admission(resource, limit):
        calls.append((resource, limit))
        try:
            yield
        finally:
            calls.append("released")

    monkeypatch.setattr(cluster, "cluster_slot", admission)
    before = runtime_parameters.snapshot()
    key = f"{model}_concurrency"
    try:
        runtime_parameters.install({key: 2})
        with runtime_parameters.frozen({key: 0}), pytest.raises(RuntimeError, match="upstream"):
            async with cluster.model_slot(model):
                raise RuntimeError("upstream")
    finally:
        runtime_parameters.install(before)
    assert calls == [(f"model:{model}", 2), "released"]


def test_execution_cannot_write_another_tasks_results(monkeypatch):
    monkeypatch.setattr(settings, "cluster_enabled", True)
    assert review_task_store._ownership("task")[0] == " AND 1 = 0"
    with execution_scope("task", "lease-token", 1):
        assert review_task_store._ownership("another-task")[0] == " AND 1 = 0"
        sql, values = review_task_store._ownership("task")
        assert "lease_expires > UTC_TIMESTAMP(3)" in sql
        assert values == ("lease-token",)
    assert review_task_store._ownership("task")[0] == " AND 1 = 0"


@pytest.mark.asyncio
async def test_frozen_parameters_are_isolated_between_tasks_and_threads():
    before = runtime_parameters.snapshot()
    try:
        runtime_parameters.install({"jpeg_quality": 95})

        async def read(value):
            with runtime_parameters.frozen({"jpeg_quality": value}):
                await asyncio.sleep(0)
                assert runtime_parameters.get_live("jpeg_quality") == 95
                return await asyncio.to_thread(lambda: settings.jpeg_quality)

        assert await asyncio.gather(read(61), read(72)) == [61, 72]
        assert settings.jpeg_quality == 95
    finally:
        runtime_parameters.install(before)


def test_embedded_evidence_becomes_shared_references_without_mutating_input(monkeypatch):
    monkeypatch.setattr(settings, "image_storage", "s3")
    write = Mock()
    monkeypatch.setattr(image_store, "write_bytes", write)
    original = [{"evidence": [{"face_image_b64": base64.b64encode(b"jpeg-evidence").decode()}]}]
    result = archive_evidence("task", original)
    assert result[0]["evidence"][0]["face_image_url"].startswith("/images/evidence/task/")
    assert "face_image_b64" in original[0]["evidence"][0]
    assert write.call_args.args[1] == b"jpeg-evidence"


@pytest.mark.parametrize(
    "value", ["/etc/passwd", "/tmp/wcm/../private.jpg", "/tmp/wcm/.person-operations/op.json"]
)
def test_public_image_keys_reject_traversal_and_private_journals(value):
    with pytest.raises(ValueError):
        image_store.object_key(value)


def test_s3_streaming_body_serves_actual_bytes_and_head_without_context_manager_unwrapping(
    monkeypatch,
):
    import io

    from botocore.response import StreamingBody
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.images import images_bp

    body = b"persisted-image"
    monkeypatch.setattr(
        image_store, "stat", lambda path: {"ContentLength": len(body), "ETag": '"v1"'}
    )
    client = Mock()
    client.get_object.side_effect = lambda **kw: {
        "ContentLength": len(body),
        "Body": StreamingBody(io.BytesIO(body), len(body)),
        "ContentType": "image/jpeg",
    }
    monkeypatch.setattr(image_store, "client", lambda: client)
    app = FastAPI()
    app.include_router(images_bp)
    with TestClient(app) as http:
        response = http.get("/images/gallery/example.jpg")
        assert response.content == body
        assert response.headers["content-length"] == str(len(body))
        assert http.head("/images/gallery/example.jpg").content == b""
        assert (
            http.get("/images/gallery/example.jpg", headers={"If-None-Match": '"v1"'}).status_code
            == 304
        )


@pytest.mark.asyncio
async def test_cancelled_sdk_thread_is_drained_before_releasing_its_caller():
    import threading

    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def operation():
        started.set()
        release.wait(2)
        finished.set()

    task = asyncio.create_task(run_sync(operation))
    await asyncio.to_thread(started.wait, 2)
    task.cancel()
    await asyncio.sleep(0.01)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished.is_set()
