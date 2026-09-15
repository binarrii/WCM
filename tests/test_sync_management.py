from unittest.mock import Mock

import httpx
import pymysql
import pytest
import pytest_asyncio
from fastapi import FastAPI

from api.insightface_management import insightface_management_bp
from wcm_facerec import face_sync_store as store
from wcm_facerec.config import settings


@pytest_asyncio.fixture
async def client():
    app = FastAPI()
    app.include_router(insightface_management_bp, prefix="/api/v1")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@pytest.mark.asyncio
async def test_disabled_replication_status_does_not_connect_to_mysql(client, monkeypatch):
    monkeypatch.setattr(settings, "insightface_replication_enabled", False)
    connect = Mock(side_effect=AssertionError("No MySQL for disabled replication"))
    monkeypatch.setattr(store, "connect", connect)
    assert (await client.get("/api/v1/insightface/replication")).json() == {"enabled": False}
    response = await client.post("/api/v1/insightface/replication/sync", json={})
    assert response.status_code == 409
    connect.assert_not_called()


@pytest.mark.asyncio
async def test_manual_request_returns_accepted_not_completed(client, monkeypatch):
    request = Mock(return_value={"requested": [{"id": "a", "request_id": "r"}], "skipped": []})
    monkeypatch.setattr(store, "request_sync", request)
    response = await client.post("/api/v1/insightface/replication/sync", json={"node_id": "a"})
    assert response.status_code == 202 and "completed" not in response.json()
    request.assert_called_once_with("a")


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [{"node_id": ""}, {"node_id": "x" * 65}, {"force": True}])
async def test_request_cannot_include_force_recovery(client, monkeypatch, payload):
    request = Mock()
    monkeypatch.setattr(store, "request_sync", request)
    assert (
        await client.post("/api/v1/insightface/replication/sync", json=payload)
    ).status_code == 422
    request.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "status"),
    [(ValueError("副本不存在"), 404), (store.ReplicationUnavailable("副本隔离"), 409)],
)
async def test_request_rejects_unavailable_targets(client, monkeypatch, error, status):
    monkeypatch.setattr(store, "request_sync", Mock(side_effect=error))
    response = await client.post("/api/v1/insightface/replication/sync", json={})
    assert response.status_code == status


@pytest.mark.asyncio
async def test_database_error_does_not_expose_connection_details(client, monkeypatch):
    monkeypatch.setattr(settings, "insightface_replication_enabled", True)
    failure = pymysql.OperationalError(2003, "connection-secret")
    monkeypatch.setattr(store, "status", Mock(side_effect=failure))
    monkeypatch.setattr(store, "request_sync", Mock(side_effect=failure))
    for method, path in [("GET", ""), ("POST", "/sync")]:
        response = await client.request(method, "/api/v1/insightface/replication" + path, json={})
        assert response.status_code == 503
        assert "connection-secret" not in response.text


@pytest.mark.asyncio
async def test_caught_up_worker_acknowledges_only_observed_request(monkeypatch):
    from api.face_sync_worker import sync_once

    monkeypatch.setattr(
        store,
        "node",
        lambda name: (
            {"head": 3},
            {"state": "ready", "applied_seq": 3, "due": True, "sync_request_id": "observed"},
        ),
    )
    monkeypatch.setattr(store, "heartbeat", Mock())
    completed = Mock()
    monkeypatch.setattr(store, "complete_sync_request", completed)
    target = Mock()
    await sync_once("a", target)
    target.health.assert_called_once()
    completed.assert_called_once_with("a", "observed")


@pytest.mark.asyncio
async def test_failed_health_keeps_manual_request_pending(monkeypatch):
    from api.face_sync_worker import sync_once

    monkeypatch.setattr(
        store,
        "node",
        lambda name: (
            {"head": 3},
            {"state": "retry", "applied_seq": 3, "due": True, "sync_request_id": "pending"},
        ),
    )
    monkeypatch.setattr(store, "begin_sync", lambda *args: True)
    monkeypatch.setattr(store, "retry", Mock())
    completed = Mock()
    monkeypatch.setattr(store, "complete_sync_request", completed)
    target = Mock()
    target.health.side_effect = TimeoutError()
    await sync_once("a", target)
    completed.assert_not_called()
