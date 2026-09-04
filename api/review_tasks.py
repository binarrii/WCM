"""Review task management API."""

import io
import json
import re
import zipfile

from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel, Field

from . import review_task_store

review_tasks_bp = APIRouter()
_STATUSES = {"processing", "completed", "failed"}


class ReviewTaskDeleteRequest(BaseModel):
    ids: list[str] = Field(min_length=1, max_length=100)


def _storage_error(exc: Exception) -> HTTPException:
    return HTTPException(status_code=503, detail=str(exc))


def _task_ids(ids: list[str]) -> list[str]:
    return list(dict.fromkeys(task_id.strip() for task_id in ids if task_id.strip()))


def _result_bytes(task: dict) -> bytes:
    if task["status"] != "completed" or task.get("results") is None:
        raise HTTPException(status_code=409, detail=f"审核任务 {task['id']} 的分析结果尚未就绪")
    return json.dumps(task["results"], ensure_ascii=False, indent=2).encode("utf-8")


def _safe_task_id(task_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", task_id).strip("-.") or "task"


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


@review_tasks_bp.post("/review_tasks/results/download")
async def download_review_task_results(body: ReviewTaskDeleteRequest):
    task_ids = _task_ids(body.ids)
    if not task_ids:
        raise HTTPException(status_code=422, detail="至少选择一个审核任务")
    try:
        tasks = await review_task_store.get_many(task_ids)
    except review_task_store.ReviewTaskStoreUnavailable as exc:
        raise _storage_error(exc) from exc
    if len(tasks) != len(task_ids):
        found_ids = {task["id"] for task in tasks}
        missing_ids = [task_id for task_id in task_ids if task_id not in found_ids]
        raise HTTPException(status_code=404, detail=f"审核任务不存在：{', '.join(missing_ids)}")

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for task in tasks:
            output.writestr(f"analysis-{_safe_task_id(task['id'])}.json", _result_bytes(task))
    return Response(
        content=archive.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="analysis-results-{len(tasks)}.zip"'},
    )


@review_tasks_bp.get("/review_tasks/{task_id}/results/download")
async def download_review_task_result(task_id: str):
    try:
        task = await review_task_store.get(task_id)
    except review_task_store.ReviewTaskStoreUnavailable as exc:
        raise _storage_error(exc) from exc
    if task is None:
        raise HTTPException(status_code=404, detail="审核任务不存在")
    return Response(
        content=_result_bytes(task),
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="analysis-{_safe_task_id(task_id)}.json"'
        },
    )


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
    task_ids = _task_ids(body.ids)
    if not task_ids:
        raise HTTPException(status_code=422, detail="至少选择一个审核任务")
    try:
        deleted = await review_task_store.delete_many(task_ids)
    except review_task_store.ReviewTaskStoreUnavailable as exc:
        raise _storage_error(exc) from exc
    return {"deleted": deleted, "requested": len(task_ids)}
