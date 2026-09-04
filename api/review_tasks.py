"""Review task management API."""

from fastapi import APIRouter, HTTPException, Query

from . import review_task_store

review_tasks_bp = APIRouter()
_STATUSES = {"processing", "completed", "failed"}


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
