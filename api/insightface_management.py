"""Operational status and safe, durable requests for the existing sync worker."""

import pymysql
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from wcm_facerec import face_sync_store
from wcm_facerec.cluster import run_sync
from wcm_facerec.config import settings

insightface_management_bp = APIRouter()


class SyncRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    node_id: str | None = Field(default=None, min_length=1, max_length=64)


@insightface_management_bp.get("/insightface/replication")
async def replication_status():
    if not settings.insightface_replication_enabled:
        return {"enabled": False}
    try:
        return await run_sync(face_sync_store.status)
    except pymysql.MySQLError:
        raise HTTPException(503, "暂时无法读取同步状态，请稍后刷新")


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
