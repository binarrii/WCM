"""Durable video queue with bounded cluster admission and fenced execution leases."""

import asyncio
import uuid

from wcm_facerec.config import settings

from . import review_task_store as store
from .review_cancellation import ReviewTaskCancelled
from .review_events import review_events


def _initialize_sync():
    with store._connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT GET_LOCK(%s, 30) AS acquired", (f"{settings.cluster_namespace}:schema",)
        )
        if cursor.fetchone()["acquired"] != 1:
            raise RuntimeError("Could not acquire schema migration lock")
        columns = {
            "runtime_parameters": "JSON NULL",
            "worker_id": "VARCHAR(160) NULL",
            "lease_token": "CHAR(36) NULL",
            "lease_expires": "DATETIME(3) NULL",
            "heartbeat_at": "DATETIME(3) NULL",
            "attempts": "INT UNSIGNED NOT NULL DEFAULT 0",
            "available_at": "DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3)",
        }
        for name, definition in columns.items():
            cursor.execute("SHOW COLUMNS FROM review_tasks LIKE %s", (name,))
            if not cursor.fetchone():
                cursor.execute(f"ALTER TABLE review_tasks ADD COLUMN {name} {definition}")
        cursor.execute("CREATE TABLE IF NOT EXISTS review_admission (id INT PRIMARY KEY)")
        cursor.execute("INSERT IGNORE INTO review_admission (id) VALUES (1)")


async def initialize():
    if not settings.cluster_enabled:
        return
    if not store.is_enabled():
        raise RuntimeError("Cluster mode requires MySQL")
    if settings.review_heartbeat_seconds * 3 >= settings.review_lease_seconds:
        raise RuntimeError("The lease must exceed three heartbeat intervals")
    await store._run(_initialize_sync)


def _recover_expired_sync(cursor):
    # A locking UPDATE by status takes the secondary-index lock before the
    # primary-key lock, opposite to result saving. Discover candidates without
    # locks, then update each by primary key in a consistent order. A heartbeat,
    # cancellation or completion may win the race, so recheck both predicates.
    cursor.execute(
        "SELECT id, status FROM review_tasks WHERE status IN ('processing', 'cancelling') "
        "AND (lease_expires IS NULL OR lease_expires <= UTC_TIMESTAMP(3)) ORDER BY id"
    )
    for row in cursor.fetchall():
        if row["status"] == "cancelling":
            cursor.execute(
                "UPDATE review_tasks SET status = 'cancelled', "
                "progress = JSON_SET(COALESCE(progress, JSON_OBJECT()), '$.phase', 'cancelled') "
                "WHERE id = %s AND status = 'cancelling' "
                "AND (lease_expires IS NULL OR lease_expires <= UTC_TIMESTAMP(3))",
                (row["id"],),
            )
        else:
            cursor.execute(
                "UPDATE review_tasks SET status = IF(attempts < %s, 'queued', 'failed'), "
                "error = '执行节点租约过期，任务已重新排队或达到重试上限', "
                "progress = JSON_OBJECT('phase', IF(attempts < %s, 'queued', 'failed'), 'attempt', attempts), "
                "lease_token = NULL, lease_expires = NULL "
                "WHERE id = %s AND status = 'processing' "
                "AND (lease_expires IS NULL OR lease_expires <= UTC_TIMESTAMP(3))",
                (settings.review_max_attempts, settings.review_max_attempts, row["id"]),
            )


def _claim_sync(worker_id):
    with store._connect() as connection, connection.cursor() as cursor:
        connection.begin()
        # Serialize the capacity check, not the expensive work. A SELECT COUNT
        # without this row lock allows concurrent claimers to exceed the limit.
        cursor.execute("SELECT id FROM review_admission WHERE id = 1 FOR UPDATE")
        _recover_expired_sync(cursor)
        cursor.execute(
            "SELECT COUNT(*) AS total FROM review_tasks "
            "WHERE status IN ('processing', 'cancelling') AND lease_expires > UTC_TIMESTAMP(3)"
        )
        if cursor.fetchone()["total"] >= settings.review_task_concurrency:
            connection.commit()
            return None
        cursor.execute(
            "SELECT * FROM review_tasks WHERE status = 'queued' AND available_at <= UTC_TIMESTAMP(3) "
            "ORDER BY created_at, id LIMIT 1 FOR UPDATE SKIP LOCKED"
        )
        row = cursor.fetchone()
        if row is None:
            connection.commit()
            return None
        token = str(uuid.uuid4())
        attempt = row["attempts"] + 1
        cursor.execute(
            "UPDATE review_tasks SET status = 'processing', worker_id = %s, lease_token = %s, "
            "lease_expires = TIMESTAMPADD(SECOND, %s, UTC_TIMESTAMP(3)), heartbeat_at = UTC_TIMESTAMP(3), "
            "attempts = %s, error = NULL, media = NULL, progress = %s WHERE id = %s",
            (
                worker_id,
                token,
                settings.review_lease_seconds,
                attempt,
                store._json_dump({"phase": "downloading", "sequence": 0, "attempt": attempt}),
                row["id"],
            ),
        )
        connection.commit()
        return {
            **row,
            "lease_token": token,
            "attempts": attempt,
            "parameters": store._json_load(row["parameters"]),
            "runtime_parameters": store._json_load(row["runtime_parameters"]) or {},
        }


async def claim(worker_id):
    task = await store._run(_claim_sync, worker_id, retry_deadlocks=True)
    if task:
        await review_events.publish(
            {"type": "changed", "task_ids": [task["id"]], "reason": "claimed"}
        )
    return task


def _renew_sync(task_id, token):
    with store._connect() as connection, connection.cursor() as cursor:
        return bool(
            cursor.execute(
                "UPDATE review_tasks SET lease_expires = TIMESTAMPADD(SECOND, %s, UTC_TIMESTAMP(3)), "
                "heartbeat_at = UTC_TIMESTAMP(3) WHERE id = %s AND lease_token = %s "
                "AND lease_expires > UTC_TIMESTAMP(3) AND status IN ('processing', 'cancelling')",
                (settings.review_lease_seconds, task_id, token),
            )
        )


async def renew(task_id, token):
    return await store._run(_renew_sync, task_id, token, retry_deadlocks=True)


async def wait_result(task_id):
    """Compatibility callers only wait; disconnecting them never cancels a job."""
    while True:
        task = await store.get(task_id)
        if task is None:
            raise ValueError("审核任务已删除")
        if task["status"] == "cancelled":
            raise ReviewTaskCancelled("审核任务已取消")
        if task["status"] in {"completed", "partial"}:
            return task["results"]
        if task["status"] == "failed":
            raise ValueError(task["error"] or "审核失败")
        await asyncio.sleep(0.5)
