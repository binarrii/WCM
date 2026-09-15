import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import pymysql
import pytest
from fastapi import FastAPI, WebSocketDisconnect
from fastapi.testclient import TestClient

from api import insightface_management as management
from wcm_facerec.config import settings

PATH = "/api/v1/insightface/replication/ws"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "insightface_replication_enabled", True)
    app = FastAPI()
    app.include_router(management.insightface_management_bp, prefix="/api/v1")
    with TestClient(app) as client:
        yield client


def test_each_pull_reads_latest_state_and_serializes_utc(client, monkeypatch):
    now = datetime(2026, 9, 15, tzinfo=timezone.utc)
    read = Mock(side_effect=[{"enabled": True, "heartbeat": now}, {"enabled": False}])
    monkeypatch.setattr(management.face_sync_store, "status", read)
    with client.websocket_connect(PATH) as socket:
        read.assert_not_called()
        socket.send_json({"type": "status", "request_id": 7})
        response = socket.receive_json()
        assert response == {
            "type": "status",
            "request_id": 7,
            "data": {"enabled": True, "heartbeat": "2026-09-15T00:00:00+00:00"},
        }
        assert read.call_count == 1
        socket.send_json({"type": "status", "request_id": 8})
        assert socket.receive_json() == {
            "type": "status",
            "request_id": 8,
            "data": {"enabled": False},
        }
        assert read.call_count == 2


def test_database_failure_can_be_retried_on_same_connection(client, monkeypatch):
    read = Mock(side_effect=[pymysql.OperationalError(2003, "secret-host"), {"enabled": True}])
    monkeypatch.setattr(management.face_sync_store, "status", read)
    with client.websocket_connect(PATH) as socket:
        socket.send_json({"type": "status", "request_id": 1})
        error = socket.receive_json()
        assert error["type"] == "error" and error["request_id"] == 1 and error["code"] == 503
        assert "secret-host" not in str(error)
        socket.send_json({"type": "status", "request_id": 2})
        assert socket.receive_json()["data"]["enabled"] is True


def test_disabled_replication_needs_no_database(client, monkeypatch):
    monkeypatch.setattr(settings, "insightface_replication_enabled", False)
    read = Mock(side_effect=AssertionError("No database needed"))
    monkeypatch.setattr(management.face_sync_store, "status", read)
    with client.websocket_connect(PATH) as socket:
        socket.send_json({"type": "status", "request_id": 1})
        assert socket.receive_json()["data"] == {"enabled": False}
    read.assert_not_called()


@pytest.mark.parametrize(
    "payload",
    [
        "not-json",
        "null",
        "x" * 513,
        '{"type":"sync","request_id":1}',
        '{"type":"status","request_id":true}',
        '{"type":"status","request_id":0}',
        '{"type":"status","request_id":9007199254740992}',
        '{"type":"status","request_id":1,"force":true}',
    ],
)
def test_invalid_messages_never_query_or_write(client, monkeypatch, payload):
    read, write = Mock(), Mock()
    monkeypatch.setattr(management.face_sync_store, "status", read)
    monkeypatch.setattr(management.face_sync_store, "request_sync", write)
    with client.websocket_connect(PATH) as socket:
        socket.send_text(payload)
        with pytest.raises(WebSocketDisconnect) as error:
            socket.receive_json()
        assert error.value.code == 1008
    read.assert_not_called()
    write.assert_not_called()


def test_binary_message_closes_with_policy_code(client):
    with client.websocket_connect(PATH) as socket:
        socket.send_bytes(b"hello")
        with pytest.raises(WebSocketDisconnect) as error:
            socket.receive_json()
        assert error.value.code == 1008


@pytest.mark.asyncio
async def test_idle_socket_never_queries_and_is_released(monkeypatch):
    socket = AsyncMock()
    socket.receive.side_effect = asyncio.Queue().get
    read = Mock()
    monkeypatch.setattr(management.face_sync_store, "status", read)
    monkeypatch.setattr(management, "PULL_IDLE_TIMEOUT", 0.01)
    await management.replication_pull(socket)
    socket.accept.assert_awaited_once()
    socket.send_json.assert_not_awaited()
    socket.close.assert_awaited_once_with(code=1001)
    read.assert_not_called()


@pytest.mark.asyncio
async def test_disconnect_does_not_leave_a_polling_task(monkeypatch):
    socket = AsyncMock()
    socket.receive.return_value = {"type": "websocket.disconnect"}
    read = Mock()
    monkeypatch.setattr(management.face_sync_store, "status", read)
    await management.replication_pull(socket)
    socket.receive.assert_awaited_once()
    socket.send_json.assert_not_awaited()
    read.assert_not_called()
