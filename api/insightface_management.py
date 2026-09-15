"""Operational status and safe, durable requests for the existing sync worker."""

import asyncio
from contextlib import suppress
from typing import Literal

import pymysql
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from wcm_facerec import face_sync_store
from wcm_facerec.cluster import run_sync
from wcm_facerec.config import settings

insightface_management_bp = APIRouter()
PULL_IDLE_TIMEOUT = 60
PULL_SEND_TIMEOUT = 5


class SyncRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    node_id: str | None = Field(default=None, min_length=1, max_length=64)


class StatusPull(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["status"]
    request_id: int = Field(strict=True, ge=1, le=9007199254740991)


@insightface_management_bp.get("/insightface/replication")
async def replication_status():
    if not settings.insightface_replication_enabled:
        return {"enabled": False}
    try:
        return await run_sync(face_sync_store.status)
    except pymysql.MySQLError:
        raise HTTPException(503, "暂时无法读取同步状态，请稍后刷新")


@insightface_management_bp.websocket("/insightface/replication/ws")
async def replication_pull(websocket: WebSocket):
    """Read once per client request; idle connections do not query the database."""
    await websocket.accept()
    close_code = 1000
    try:
        while True:
            message = await asyncio.wait_for(websocket.receive(), PULL_IDLE_TIMEOUT)
            if message["type"] == "websocket.disconnect":
                return
            raw = message.get("text")
            if raw is None or len(raw) > 512:
                close_code = 1008
                return
            try:
                request = StatusPull.model_validate_json(raw)
            except ValidationError:
                close_code = 1008
                return
            try:
                data = await replication_status()
                response = {"type": "status", "request_id": request.request_id, "data": data}
            except HTTPException as exc:
                response = {
                    "type": "error",
                    "request_id": request.request_id,
                    "code": exc.status_code,
                    "error": exc.detail,
                }
            await asyncio.wait_for(
                websocket.send_json(jsonable_encoder(response)), PULL_SEND_TIMEOUT
            )
    except asyncio.TimeoutError:
        close_code = 1001
    except WebSocketDisconnect:
        pass
    finally:
        with suppress(WebSocketDisconnect, RuntimeError, asyncio.TimeoutError):
            await asyncio.wait_for(websocket.close(code=close_code), PULL_SEND_TIMEOUT)


@insightface_management_bp.post("/insightface/replication/sync", status_code=202)
async def trigger_replication(payload: SyncRequest):
    try:
        return await run_sync(face_sync_store.request_sync, payload.node_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    except face_sync_store.ReplicationUnavailable as exc:
        raise HTTPException(409, str(exc))
    except pymysql.MySQLError:
        raise HTTPException(503, "同步请求结果暂无法确认，请刷新状态后重试；重复请求会合并")
