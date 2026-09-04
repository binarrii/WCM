import io
import json
import zipfile
from datetime import datetime
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from api import review_task_store, review_tasks, routes
from api.main import create_app


def test_review_task_list_and_detail_routes(monkeypatch):
    created = datetime(2026, 9, 4, 10, 30)
    item = {
        "id": "task-1",
        "video_url": "https://example.com/video.mp4",
        "parameters": {"sample_interval": 1, "top_k": 10, "threshold": 0.5},
        "status": "completed",
        "result_count": 2,
        "error": None,
        "created_at": created.isoformat() + "Z",
        "updated_at": created.isoformat() + "Z",
    }
    list_tasks = AsyncMock(return_value={"items": [item], "total": 1, "page": 1, "page_size": 30})
    get_task = AsyncMock(return_value={**item, "results": [{"timestamp": "00:00:01.000"}]})
    monkeypatch.setattr(review_tasks.review_task_store, "list_tasks", list_tasks)
    monkeypatch.setattr(review_tasks.review_task_store, "get", get_task)

    with TestClient(create_app()) as client:
        response = client.get("/api/v1/review_tasks?q=video&status=completed")
        detail = client.get("/api/v1/review_tasks/task-1")

    assert response.status_code == 200
    assert response.json()["items"][0]["id"] == "task-1"
    list_tasks.assert_awaited_once_with("video", "completed", 1, 30)
    assert detail.json()["results"][0]["timestamp"] == "00:00:01.000"


def test_review_task_route_rejects_unknown_status(monkeypatch):
    monkeypatch.setattr(review_tasks.review_task_store, "list_tasks", AsyncMock())
    with TestClient(create_app()) as client:
        response = client.get("/api/v1/review_tasks?status=unknown")
    assert response.status_code == 422


def test_single_and_batch_delete_routes(monkeypatch):
    delete_many = AsyncMock(side_effect=[1, 2, 0])
    monkeypatch.setattr(review_tasks.review_task_store, "delete_many", delete_many)

    with TestClient(create_app()) as client:
        single = client.delete("/api/v1/review_tasks/task-1")
        batch = client.request(
            "DELETE", "/api/v1/review_tasks", json={"ids": ["task-2", "task-3", "task-2"]}
        )
        missing = client.delete("/api/v1/review_tasks/missing")

    assert single.json() == {"deleted": 1}
    assert batch.json() == {"deleted": 2, "requested": 2}
    assert missing.status_code == 404
    assert [call.args[0] for call in delete_many.await_args_list] == [
        ["task-1"],
        ["task-2", "task-3"],
        ["missing"],
    ]


def test_batch_delete_requires_at_least_one_id(monkeypatch):
    monkeypatch.setattr(review_tasks.review_task_store, "delete_many", AsyncMock())
    with TestClient(create_app()) as client:
        response = client.request("DELETE", "/api/v1/review_tasks", json={"ids": []})
    assert response.status_code == 422


def test_single_and_batch_result_downloads(monkeypatch):
    results = [{"timestamp": "00:00:01.000", "category": "人物", "description": "测试"}]
    task = {"id": "task-1", "status": "completed", "results": results}
    monkeypatch.setattr(review_tasks.review_task_store, "get", AsyncMock(return_value=task))
    monkeypatch.setattr(
        review_tasks.review_task_store,
        "get_many",
        AsyncMock(return_value=[task, {**task, "id": "task-2"}]),
    )

    with TestClient(create_app()) as client:
        single = client.get("/api/v1/review_tasks/task-1/results/download")
        batch = client.post(
            "/api/v1/review_tasks/results/download", json={"ids": ["task-1", "task-2"]}
        )

    assert single.status_code == 200
    assert single.json() == results
    assert single.headers["content-disposition"] == 'attachment; filename="analysis-task-1.json"'
    assert batch.status_code == 200
    assert batch.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(batch.content)) as archive:
        assert archive.namelist() == ["analysis-task-1.json", "analysis-task-2.json"]
        assert json.loads(archive.read("analysis-task-2.json")) == results


def test_result_download_rejects_unfinished_task(monkeypatch):
    monkeypatch.setattr(
        review_tasks.review_task_store,
        "get",
        AsyncMock(return_value={"id": "task-1", "status": "processing", "results": None}),
    )
    with TestClient(create_app()) as client:
        response = client.get("/api/v1/review_tasks/task-1/results/download")
    assert response.status_code == 409


def test_analyze_media_persists_task_and_keeps_legacy_response(monkeypatch):
    results = [{"timestamp": "00:00:01.000", "category": "待复核", "description": "内容"}]
    create = AsyncMock(return_value="task-1")
    complete = AsyncMock()
    monkeypatch.setattr(routes.review_task_store, "create", create)
    monkeypatch.setattr(routes.review_task_store, "complete", complete)
    monkeypatch.setattr(routes, "_process_analyze_media", AsyncMock(return_value=results))

    with TestClient(create_app()) as client:
        response = client.post(
            "/api/v1/analyze_media",
            json={
                "url": "https://example.com/video.mp4",
                "sample_interval": 2,
                "top_k": 5,
                "threshold": 0.4,
            },
        )

    assert response.status_code == 200
    assert response.json() == results
    assert response.headers["x-review-task-id"] == "task-1"
    create.assert_awaited_once_with(
        "https://example.com/video.mp4",
        {"sample_interval": 2.0, "top_k": 5, "threshold": 0.4},
    )
    complete.assert_awaited_once_with("task-1", results)


def test_analyze_media_records_failure(monkeypatch):
    monkeypatch.setattr(routes.review_task_store, "create", AsyncMock(return_value="task-2"))
    fail = AsyncMock()
    monkeypatch.setattr(routes.review_task_store, "fail", fail)
    monkeypatch.setattr(
        routes, "_process_analyze_media", AsyncMock(side_effect=RuntimeError("boom"))
    )

    with TestClient(create_app()) as client:
        response = client.post(
            "/api/v1/analyze_media", json={"url": "https://example.com/video.mp4"}
        )

    assert response.status_code == 400
    await_args = fail.await_args.args
    assert await_args == ("task-2", "boom")


def test_store_public_row_decodes_json_and_utc_timestamps():
    row = {
        "id": "task-1",
        "video_url": "https://example.com/video.mp4",
        "parameters": '{"top_k":5}',
        "status": "completed",
        "results": '[{"timestamp":"00:00:01.000"}]',
        "result_count": 1,
        "error": None,
        "created_at": datetime(2026, 9, 4, 10, 30),
        "updated_at": datetime(2026, 9, 4, 10, 31),
    }
    item = review_task_store._public_row(row, include_results=True)
    assert item["parameters"] == {"top_k": 5}
    assert item["results"][0]["timestamp"] == "00:00:01.000"
    assert item["created_at"] == "2026-09-04T10:30:00Z"
