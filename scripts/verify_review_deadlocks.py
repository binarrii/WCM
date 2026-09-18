"""Real MySQL regression checks, confined to a fresh wcm_verify_* database.

Run with cluster mode enabled and WCM_REVIEW_TASKS_DB_NAME set to an empty,
dedicated database. No model calls, media uploads or event-bus startup occur.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pymysql

from api import review_task_store as store
from api import task_queue
from scripts.verify_cluster import require_test_database, verify_queue
from wcm_facerec import runtime_parameters
from wcm_facerec.execution import execution_scope


def sql(statement, args=()):
    require_test_database()
    with store._connect() as connection, connection.cursor() as cursor:
        cursor.execute(statement, args)
        return cursor.fetchall()


def insert_task(status="processing", *, expired=False, attempts=1):
    task_id = str(uuid4())
    sql(
        "INSERT INTO review_tasks "
        "(id, video_url, parameters, status, attempts, lease_token, lease_expires) "
        "VALUES (%s, 'https://example.invalid/test.mp4', '{}', %s, %s, %s, "
        "TIMESTAMPADD(SECOND, %s, UTC_TIMESTAMP(3)))",
        (task_id, status, attempts, task_id, -60 if expired else 120),
    )
    return task_id


def recover():
    require_test_database()
    with store._connect() as connection, connection.cursor() as cursor:
        connection.begin()
        task_queue._recover_expired_sync(cursor)
        connection.commit()


def verify_healthy_task_not_locked():
    """Old status scan waits on a healthy primary row; new claim must not."""
    task_id = insert_task()
    pending_id = insert_task("queued", attempts=0)

    def old_recovery():
        with store._connect() as connection, connection.cursor() as cursor:
            cursor.execute("SET SESSION innodb_lock_wait_timeout = 1")
            connection.begin()
            try:
                cursor.execute(
                    "UPDATE review_tasks SET status = IF(attempts < 3, 'queued', 'failed'), "
                    "lease_token = NULL, lease_expires = NULL WHERE status = 'processing' "
                    "AND (lease_expires IS NULL OR lease_expires <= UTC_TIMESTAMP(3))"
                )
            except pymysql.err.OperationalError as exc:
                assert exc.args[0] == 1205, exc.args
                return "blocked"
            finally:
                connection.rollback()
            raise AssertionError("The old status-index scan did not reproduce the lock wait")

    with ThreadPoolExecutor(max_workers=1) as executor:
        with store._connect() as connection, connection.cursor() as cursor:
            connection.begin()
            cursor.execute("SELECT id FROM review_tasks WHERE id = %s FOR UPDATE", (task_id,))
            assert executor.submit(old_recovery).result(timeout=5) == "blocked"
            claimed = executor.submit(task_queue._claim_sync, "lock-probe").result(timeout=3)
            assert claimed["id"] == pending_id
            cursor.execute("UPDATE review_tasks SET status = 'completed' WHERE id = %s", (task_id,))
            connection.commit()
    sql("DELETE FROM review_tasks")
    print("PASS old recovery blocks a healthy save; new claim does not lock that row", flush=True)


def verify_expiry_and_races():
    queued = insert_task(expired=True)
    failed = insert_task(expired=True, attempts=3)
    cancelled = insert_task("cancelling", expired=True)
    healthy = insert_task()
    null_lease = insert_task()
    sql("UPDATE review_tasks SET lease_expires = NULL WHERE id = %s", (null_lease,))
    recover()
    rows = {row["id"]: row for row in sql("SELECT id, status, lease_token FROM review_tasks")}
    for task_id, expected in (
        (queued, "queued"),
        (failed, "failed"),
        (cancelled, "cancelled"),
        (healthy, "processing"),
        (null_lease, "queued"),
    ):
        assert rows[task_id]["status"] == expected
    assert rows[queued]["lease_token"] is None and rows[failed]["lease_token"] is None
    sql("DELETE FROM review_tasks")

    renewed = insert_task(expired=True)
    completed = insert_task(expired=True)
    cancelling = insert_task(expired=True)
    already_cancelled = insert_task("cancelling", expired=True)

    class RacingCursor:
        def __init__(self, cursor):
            self.cursor = cursor

        def execute(self, *args):
            return self.cursor.execute(*args)

        def fetchall(self):
            candidates = self.cursor.fetchall()
            assert len(candidates) == 4
            # Another connection wins after candidate discovery, before recovery.
            sql(
                "UPDATE review_tasks SET lease_expires = UTC_TIMESTAMP(3) + INTERVAL 120 SECOND "
                "WHERE id = %s",
                (renewed,),
            )
            for task_id, status in (
                (completed, "completed"),
                (cancelling, "cancelling"),
                (already_cancelled, "cancelled"),
            ):
                sql("UPDATE review_tasks SET status = %s WHERE id = %s", (status, task_id))
            return candidates

    with store._connect() as connection, connection.cursor() as cursor:
        connection.begin()
        task_queue._recover_expired_sync(RacingCursor(cursor))
        connection.commit()
    states = {row["id"]: row["status"] for row in sql("SELECT id, status FROM review_tasks")}
    assert states == {
        renewed: "processing",
        completed: "completed",
        cancelling: "cancelling",
        already_cancelled: "cancelled",
    }, states
    sql("DELETE FROM review_tasks")
    print(
        "PASS expiry recovery, retry limit, null leases and concurrent state/lease changes",
        flush=True,
    )


async def verify_concurrent_saves():
    ids = [insert_task("queued", attempts=0) for _ in range(100)]
    completed = set()

    async def worker(number):
        while len(completed) < len(ids):
            task = await task_queue.claim(f"stress-{number}")
            if task is None:
                await asyncio.sleep(0.005)
                continue
            assert task["id"] not in completed
            assert task["attempts"] == 1
            with execution_scope(task["id"], task["lease_token"], task["attempts"]):
                assert await task_queue.renew(task["id"], task["lease_token"])
                await store.update_progress(task["id"], {"phase": "saving", "sequence": 1})
                assert await store.complete(task["id"], [])
            completed.add(task["id"])

    await asyncio.wait_for(asyncio.gather(*(worker(i) for i in range(8))), timeout=60)
    assert completed == set(ids)
    rows = sql(
        "SELECT status, attempts, COUNT(*) AS total FROM review_tasks GROUP BY status, attempts"
    )
    assert rows == [{"status": "completed", "attempts": 1, "total": 100}], rows
    sql("DELETE FROM review_tasks")
    print(
        "PASS 100 tasks, 8 concurrent claimers, heartbeat/progress/save; all attempts remain 1",
        flush=True,
    )


async def main():
    require_test_database()
    await store.initialize()
    await task_queue.initialize()
    assert sql("SELECT COUNT(*) AS total FROM review_tasks")[0]["total"] == 0, (
        "Use a fresh database"
    )
    runtime_parameters.install({"review_task_concurrency": 2, "review_max_attempts": 3})
    await asyncio.to_thread(verify_healthy_task_not_locked)
    await asyncio.to_thread(verify_expiry_and_races)
    await verify_queue()
    await verify_concurrent_saves()
    print("ALL REVIEW DEADLOCK CHECKS PASSED", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
