"""MySQL persistence for media review tasks."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Any

import pymysql
from pymysql.cursors import DictCursor

from wcm_facerec.config import settings

from .review_coverage import completion_status, coverage_message
from .review_results import consolidate_results, flatten_findings


class ReviewTaskStoreUnavailable(RuntimeError):
    """Raised when review task persistence is disabled or unreachable."""


def is_enabled() -> bool:
    return settings.review_tasks_db_enabled


def _connect():
    if not is_enabled():
        raise ReviewTaskStoreUnavailable("审核任务数据库未启用")
    try:
        return pymysql.connect(
            host=settings.review_tasks_db_host,
            port=settings.review_tasks_db_port,
            user=settings.review_tasks_db_user,
            password=settings.review_tasks_db_password,
            database=settings.review_tasks_db_name,
            charset="utf8mb4",
            cursorclass=DictCursor,
            autocommit=True,
            connect_timeout=settings.review_tasks_db_connect_timeout_s,
            read_timeout=10,
            write_timeout=10,
            init_command="SET time_zone = '+00:00'",
        )
    except pymysql.MySQLError as exc:
        raise ReviewTaskStoreUnavailable(f"审核任务数据库不可用：{exc}") from exc


async def _run(function, *args):
    try:
        return await asyncio.to_thread(function, *args)
    except ReviewTaskStoreUnavailable:
        raise
    except pymysql.MySQLError as exc:
        raise ReviewTaskStoreUnavailable(f"审核任务数据库操作失败：{exc}") from exc


def _initialize_sync() -> None:
    with _connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS review_tasks (
                id CHAR(36) NOT NULL PRIMARY KEY,
                video_url TEXT NOT NULL,
                parameters JSON NOT NULL,
                status VARCHAR(20) NOT NULL,
                results JSON NULL,
                result_count INT UNSIGNED NOT NULL DEFAULT 0,
                error TEXT NULL,
                review_summary JSON NULL,
                progress JSON NULL,
                created_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
                updated_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3)
                    ON UPDATE CURRENT_TIMESTAMP(3),
                INDEX idx_review_tasks_status_created (status, created_at),
                INDEX idx_review_tasks_created (created_at)
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
            """
        )

        for column in ("review_summary", "progress"):
            cursor.execute("SHOW COLUMNS FROM review_tasks LIKE %s", (column,))
            if not cursor.fetchone():
                try:
                    cursor.execute(f"ALTER TABLE review_tasks ADD COLUMN {column} JSON NULL")
                except pymysql.err.OperationalError as exc:
                    # Multiple API workers can initialize the existing table together.
                    if exc.args[0] != 1060:
                        raise


async def initialize() -> None:
    if is_enabled():
        await _run(_initialize_sync)


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _json_load(value: Any) -> Any:
    if value is None or isinstance(value, (dict, list)):
        return value
    return json.loads(value)


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def _public_row(row: dict, *, include_results: bool) -> dict:
    item = {
        "id": row["id"],
        "video_url": row["video_url"],
        "parameters": _json_load(row["parameters"]),
        "status": row["status"],
        "result_count": row["result_count"],
        "error": row["error"],
        "review_summary": _json_load(row.get("review_summary")),
        "progress": _json_load(row.get("progress")),
        "has_results": bool(row.get("has_results", row.get("results") is not None)),
        "created_at": _iso(row["created_at"]),
        "updated_at": _iso(row["updated_at"]),
    }
    if row["status"] == "partial" and item["review_summary"] is None:
        item["error"] = (item["error"] or "含未审核项。") + "（历史任务未记录总采样数，比例未知）"
    if include_results:
        results = _json_load(row.get("results"))
        item["results"] = consolidate_results(results) if isinstance(results, list) else results
        if isinstance(item["results"], list):
            item["result_count"] = len(item["results"])
    return item


def _create_sync(video_url: str, parameters: dict, task_id: str) -> None:
    with _connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO review_tasks (id, video_url, parameters, status)
            VALUES (%s, %s, %s, 'processing')
            """,
            (task_id, video_url, _json_dump(parameters)),
        )


async def create(video_url: str, parameters: dict, task_id: str | None = None) -> str | None:
    if not is_enabled():
        return None
    resolved_id = task_id or str(uuid.uuid4())
    await _run(_create_sync, video_url, parameters, resolved_id)
    return resolved_id


def _complete_sync(task_id: str, results: list[dict], summary: dict | None = None) -> None:
    incomplete = [
        item for item in flatten_findings(results) if item.get("review_status") == "incomplete"
    ]
    results = consolidate_results(results)
    status = "partial" if incomplete else "completed"
    error = None
    if incomplete:
        timestamps = {item["timestamp"] for item in incomplete}
        error = f"{len(timestamps)} 个采样时间点、{len(incomplete)} 项审核未完成；缺少总采样数，比例未知，请人工复核。"
    if summary is not None:
        status = completion_status(summary)
        error = coverage_message(summary)
    with _connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            UPDATE review_tasks
            SET status = %s, results = %s, result_count = %s, error = %s, review_summary = %s
                , progress = JSON_SET(COALESCE(progress, JSON_OBJECT()), '$.phase', 'finished', '$.percent', 100)
            WHERE id = %s
            """,
            (
                status,
                _json_dump(results),
                len(results),
                error,
                _json_dump(summary) if summary is not None else None,
                task_id,
            ),
        )


async def complete(task_id: str | None, results: list[dict], summary: dict | None = None) -> None:
    if task_id and is_enabled():
        await _run(_complete_sync, task_id, results, summary)


def _fail_sync(task_id: str, error: str) -> None:
    with _connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            UPDATE review_tasks SET status = 'failed', error = %s,
                progress = JSON_SET(COALESCE(progress, JSON_OBJECT()), '$.phase', 'failed')
            WHERE id = %s
            """,
            (error[:65535], task_id),
        )


async def fail(task_id: str | None, error: str) -> None:
    if task_id and is_enabled():
        await _run(_fail_sync, task_id, error)


def _update_progress_sync(task_id: str, progress: dict) -> None:
    with _connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            "UPDATE review_tasks SET progress = %s WHERE id = %s AND status = 'processing' "
            "AND COALESCE(CAST(JSON_UNQUOTE(JSON_EXTRACT(progress, '$.sequence')) AS SIGNED), -1) < %s",
            (_json_dump(progress), task_id, progress["sequence"]),
        )


async def update_progress(task_id: str | None, progress: dict) -> None:
    if task_id and is_enabled():
        await _run(_update_progress_sync, task_id, progress)


def _get_sync(task_id: str) -> dict | None:
    with _connect() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT * FROM review_tasks WHERE id = %s", (task_id,))
        row = cursor.fetchone()
    return _public_row(row, include_results=True) if row else None


async def get(task_id: str) -> dict | None:
    return await _run(_get_sync, task_id)


def _get_many_sync(task_ids: list[str]) -> list[dict]:
    if not task_ids:
        return []
    placeholders = ", ".join(["%s"] * len(task_ids))
    with _connect() as connection, connection.cursor() as cursor:
        cursor.execute(f"SELECT * FROM review_tasks WHERE id IN ({placeholders})", task_ids)
        rows = cursor.fetchall()
    rows_by_id = {row["id"]: row for row in rows}
    return [
        _public_row(rows_by_id[task_id], include_results=True)
        for task_id in task_ids
        if task_id in rows_by_id
    ]


async def get_many(task_ids: list[str]) -> list[dict]:
    return await _run(_get_many_sync, task_ids)


def _delete_many_sync(task_ids: list[str]) -> int:
    placeholders = ", ".join(["%s"] * len(task_ids))
    with _connect() as connection, connection.cursor() as cursor:
        return cursor.execute(f"DELETE FROM review_tasks WHERE id IN ({placeholders})", task_ids)


async def delete_many(task_ids: list[str]) -> int:
    return await _run(_delete_many_sync, task_ids)


def _list_sync(query: str, status: str, page: int, page_size: int) -> dict:
    clauses = []
    values: list[Any] = []
    if query:
        like = f"%{query}%"
        clauses.append("(id LIKE %s OR video_url LIKE %s OR error LIKE %s)")
        values.extend([like, like, like])
    if status:
        clauses.append("status = %s")
        values.append(status)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""

    with _connect() as connection, connection.cursor() as cursor:
        cursor.execute(f"SELECT COUNT(*) AS total FROM review_tasks{where}", values)
        total = cursor.fetchone()["total"]
        cursor.execute(
            "SELECT id, video_url, parameters, status, result_count, error, review_summary, progress, "
            "results IS NOT NULL AS has_results, "
            f"created_at, updated_at FROM review_tasks{where} "
            "ORDER BY created_at DESC LIMIT %s OFFSET %s",
            [*values, page_size, (page - 1) * page_size],
        )
        rows = cursor.fetchall()
    return {
        "items": [_public_row(row, include_results=False) for row in rows],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


async def list_tasks(query: str, status: str, page: int, page_size: int) -> dict:
    return await _run(_list_sync, query, status, page, page_size)


def _health_sync() -> None:
    with _connect() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT 1")


async def health() -> str:
    if not is_enabled():
        return "disabled"
    await _run(_health_sync)
    return "healthy"
