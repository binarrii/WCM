"""Review task management API."""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from . import review_task_store

review_tasks_bp = APIRouter()
_STATUSES = {"processing", "completed", "failed"}


class ReviewTaskDeleteRequest(BaseModel):
    ids: list[str] = Field(min_length=1, max_length=100)


def _storage_error(exc: Exception) -> HTTPException:
    return HTTPException(status_code=503, detail=str(exc))


@review_tasks_bp.get("/review_tasks")
async def list_review_tasks(
    q: str = Query(default="", max_length=300),
    status: str = Query(default=""),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=30, ge=1, le=100),
):
    if status and status not in _STATUSES:
        raise HTTPException(status_code=422, detail="无效的任务状态")
    try:
        return await review_task_store.list_tasks(q.strip(), status, page, page_size)
    except review_task_store.ReviewTaskStoreUnavailable as exc:
        raise _storage_error(exc) from exc


@review_tasks_bp.get("/review_tasks/{task_id}")
async def get_review_task(task_id: str):
    try:
        task = await review_task_store.get(task_id)
    except review_task_store.ReviewTaskStoreUnavailable as exc:
        raise _storage_error(exc) from exc
    if task is None:
        raise HTTPException(status_code=404, detail="审核任务不存在")
    return task


@review_tasks_bp.delete("/review_tasks/{task_id}")
async def delete_review_task(task_id: str):
    try:
        deleted = await review_task_store.delete_many([task_id])
    except review_task_store.ReviewTaskStoreUnavailable as exc:
        raise _storage_error(exc) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="审核任务不存在")
    return {"deleted": deleted}


@review_tasks_bp.delete("/review_tasks")
async def delete_review_tasks(body: ReviewTaskDeleteRequest):
    task_ids = list(dict.fromkeys(task_id.strip() for task_id in body.ids if task_id.strip()))
    if not task_ids:
        raise HTTPException(status_code=422, detail="至少选择一个审核任务")
    try:
        deleted = await review_task_store.delete_many(task_ids)
    except review_task_store.ReviewTaskStoreUnavailable as exc:
        raise _storage_error(exc) from exc
    return {"deleted": deleted, "requested": len(task_ids)}
