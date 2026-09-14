"""Destructive integration checks confined to a freshly created wcm_verify_* database.

Uses real MySQL, Redis and S3 plus separate API/worker processes. Model work is
replaced inside test child processes; no InsightFace mutations or inference run.
"""

import asyncio
import contextlib
import copy
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path
from uuid import uuid4

import httpx
import websockets

from api import parameter_store, task_queue
from api import review_task_store as store
from api.review_cancellation import ReviewTaskCancelled, run_cancellable_review
from wcm_facerec import image_store, person_operations, runtime_parameters
from wcm_facerec.cluster import cluster_slot, connect, run_sync
from wcm_facerec.config import settings
from wcm_facerec.execution import current_execution, execution_scope


def require_test_database():
    assert settings.review_tasks_db_name.startswith("wcm_verify_"), (
        "Refusing to touch a non-test database"
    )


async def fake_review(task_id, url, sample_interval, top_k, threshold):
    async def work():
        execution = current_execution.get()
        await store.update_progress(
            task_id, {"phase": "reviewing", "sequence": 1, "attempt": execution.attempt}
        )
        await asyncio.sleep(30 if "slow" in url and execution.attempt == 1 else 0.3)
        await store.complete(
            task_id,
            [
                {
                    "timestamp": "00:00:01.000",
                    "category": "test",
                    "description": str(settings.jpeg_quality),
                }
            ],
        )

    try:
        await run_cancellable_review(task_id, work, poll_interval=0.1)
    except ReviewTaskCancelled:
        await store.cancelled(task_id, {"phase": "cancelled"})


def run_test_worker():
    from api import worker

    require_test_database()
    worker._run_review_task = fake_review
    asyncio.run(worker.serve())


async def until(operation, predicate, seconds=25):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        value = await operation()
        if predicate(value):
            return value
        await asyncio.sleep(0.1)
    raise AssertionError("Condition did not become true before deadline")


async def verify_queue():
    runtime_parameters.install({"review_task_concurrency": 2, "jpeg_quality": 61})
    params = {"sample_interval": 1, "top_k": 10, "threshold": 0.5}
    ids = [await store.create(f"https://example.invalid/{i}.mp4", params) for i in range(7)]
    claims = await asyncio.gather(*(task_queue.claim(f"worker-{i}") for i in range(8)))
    claims = [task for task in claims if task]
    assert len(claims) == 2, "Cluster capacity exceeded"
    assert len({task["id"] for task in claims}) == 2
    first = claims[0]
    await store.update_progress(first["id"], {"phase": "wrong", "sequence": 99})
    assert (await store.get(first["id"]))["progress"]["phase"] != "wrong"
    with execution_scope(first["id"], first["lease_token"], first["attempts"]):
        await store.update_progress(
            first["id"], {"phase": "reviewing", "sequence": 2, "attempt": 1}
        )
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            "UPDATE review_tasks SET lease_expires = UTC_TIMESTAMP(3) - INTERVAL 1 SECOND WHERE id = %s",
            (first["id"],),
        )
    replacement = await task_queue.claim("replacement")
    assert replacement["id"] == first["id"] and replacement["attempts"] == 2
    with execution_scope(first["id"], first["lease_token"], 1):
        assert await store.complete(first["id"], []) is False
        assert await store.fail(first["id"], "stale failure") is False
        await store.update_progress(first["id"], {"phase": "stale", "sequence": 999})
    assert (await store.get(first["id"]))["progress"]["phase"] != "stale"
    runtime_parameters.install({"jpeg_quality": 99})
    with runtime_parameters.frozen(replacement["runtime_parameters"]):
        assert settings.jpeg_quality == 61
        assert await asyncio.to_thread(lambda: settings.jpeg_quality) == 61
    with execution_scope(replacement["id"], replacement["lease_token"], 2):
        assert await store.complete(replacement["id"], []) is True
    pending = next(task_id for task_id in ids if task_id not in {task["id"] for task in claims})
    assert (await store.request_cancel(pending))["status"] == "cancelled"
    assert (await store.request_cancel(pending))["status"] == "cancelled"
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute("DELETE FROM review_tasks")
    print(
        "PASS concurrent claims, cluster capacity, stale-write fencing, frozen parameters, queued cancellation",
        flush=True,
    )


def lock_probe():
    require_test_database()

    async def run():
        async with cluster_slot("integration-model", 2):
            with connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    "UPDATE verification_counter SET active = active + 1, peak = GREATEST(peak, active)"
                )
            await asyncio.sleep(0.3)
            with connect() as connection, connection.cursor() as cursor:
                cursor.execute("UPDATE verification_counter SET active = active - 1")

    asyncio.run(run())


async def verify_locks():
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute("CREATE TABLE verification_counter (active INT, peak INT)")
        cursor.execute("INSERT INTO verification_counter VALUES (0, 0)")
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", "from scripts.verify_cluster import lock_probe; lock_probe()"]
        )
        for _ in range(5)
    ]
    for process in processes:
        assert await asyncio.to_thread(process.wait, 30) == 0
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT * FROM verification_counter")
        row = cursor.fetchone()
        assert row["active"] == 0 and row["peak"] == 2, row
    print("PASS model admission is shared across independent processes", flush=True)


class FakeAdapter:
    def __init__(self, item):
        self.items = {item["id"]: copy.deepcopy(item)}

    def get_person(self, person_id, *, collection_id=None):
        return copy.deepcopy(self.items.get(person_id))

    def delete_person(self, person_id, *, collection_id=None):
        self.items.pop(person_id, None)

    def register_person(
        self, *, name, image_bytes, metadata, external_id=None, person_id=None, collection_id=None
    ):
        self.items[person_id] = {"id": person_id, "name": name, "face_count": 1, **metadata}
        return person_id, "face"

    def add_person_image(self, person_id, image_bytes, *, collection_id=None):
        self.items[person_id]["face_count"] += 1


async def verify_person_recovery(logical):
    from wcm_facerec.face_engine import FaceEngine

    original = {
        "id": "verification-person",
        "name": "before",
        "file_path": str(logical),
        "face_count": 1,
    }
    engine = FaceEngine.__new__(FaceEngine)
    engine._adapter = FakeAdapter(original)
    engine._adapter.items[original["id"]]["name"] = "interrupted update"
    data = {
        "id": uuid4().hex,
        "kind": "transaction",
        "status": "running",
        "version": 0,
        "before": {
            "person": {
                "collection": settings.insightface_collection_id,
                "person_id": original["id"],
                "item": original,
            }
        },
    }
    person_operations._save(data)

    async def read(engine):
        return engine._adapter.get_person(original["id"])

    key = person_operations.request_key.set((uuid4().hex, "same-request"))
    try:
        async with cluster_slot("person-library"):
            result = await person_operations.run(engine, read)
            assert result["name"] == "before"
            engine._adapter.items[original["id"]]["name"] = "later"
            assert (await person_operations.run(engine, read))["name"] == "before"
    finally:
        person_operations.request_key.reset(key)
    print("PASS interrupted person operation recovery and idempotent replay", flush=True)


async def verify_processes(logical):
    children = []
    outputs = []
    env = {
        **os.environ,
        "WCM_WORKER_CONCURRENCY": "1",
        "WCM_REVIEW_LEASE_SECONDS": "15",
        "WCM_REVIEW_HEARTBEAT_SECONDS": "1",
        "WCM_CLUSTER_NAMESPACE": "wcm-verify",
    }
    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:

        def start(command):
            log = stack.enter_context(open(Path(directory) / f"process-{len(children)}.log", "w+"))
            outputs.append(log)
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            children.append(child)
            return child

        try:
            apis = [
                start([sys.executable, "-m", "uvicorn", "api.main:app", "--port", str(port)])
                for port in (27001, 27002)
            ]
            async with httpx.AsyncClient(timeout=5) as client:

                async def ready(port):
                    try:
                        return (
                            await client.get(f"http://127.0.0.1:{port}/openapi.json")
                        ).status_code == 200
                    except httpx.HTTPError:
                        return False

                for port in (27001, 27002):
                    await until(lambda: ready(port), bool)
                urls = [
                    f"http://127.0.0.1:{port}/images/{logical.relative_to('/tmp/wcm').as_posix()}"
                    for port in (27001, 27002)
                ]
                one, two = await asyncio.gather(*(client.get(url) for url in urls))
                assert (
                    one.status_code == two.status_code == 200
                    and one.content == two.content == b"shared-image-bytes"
                )
                ranged = await client.get(urls[1], headers={"Range": "bytes=0-5"})
                assert ranged.status_code == 206 and ranged.content == b"shared"
                async with websockets.connect(
                    "ws://127.0.0.1:27001/api/v1/ws/analyze_media"
                ) as websocket:
                    await websocket.send(
                        json.dumps({"type": "subscribe", "task_ids": [], "watch_list": True})
                    )
                    assert json.loads(await websocket.recv())["type"] == "snapshot"
                    response = await client.post(
                        "http://127.0.0.1:27002/api/v1/review_tasks",
                        json={"url": "https://example.invalid/slow.mp4"},
                    )
                    assert response.status_code == 202, response.text
                    task_id = response.json()["id"]

                    async def event():
                        return json.loads(await asyncio.wait_for(websocket.recv(), 10))

                    await until(
                        event,
                        lambda value: (
                            value.get("reason") == "created"
                            and task_id in value.get("task_ids", [])
                        ),
                        10,
                    )
                workers = [
                    start(
                        [
                            sys.executable,
                            "-c",
                            "from scripts.verify_cluster import run_test_worker; run_test_worker()",
                        ]
                    )
                    for _ in range(2)
                ]

                async def row():
                    with connect() as connection, connection.cursor() as cursor:
                        cursor.execute(
                            "SELECT worker_id, status, attempts FROM review_tasks WHERE id = %s",
                            (task_id,),
                        )
                        return cursor.fetchone()

                claimed = await until(row, lambda value: value["status"] == "processing")
                owner_pid = int(claimed["worker_id"].split(":")[1])
                next(worker for worker in workers if worker.pid == owner_pid).kill()
                apis[0].terminate()
                completed = await until(
                    lambda: store.get(task_id), lambda value: value["status"] == "completed", 25
                )
                assert completed["attempt"] == 2
                assert (
                    await client.get(f"http://127.0.0.1:27002/api/v1/review_tasks/{task_id}")
                ).json()["status"] == "completed"
                response = await client.post(
                    "http://127.0.0.1:27002/api/v1/review_tasks",
                    json={"url": "https://example.invalid/slow-cancel.mp4"},
                )
                cancelled_id = response.json()["id"]
                await until(
                    lambda: store.get(cancelled_id), lambda value: value["status"] == "processing"
                )
                response = await client.post(
                    f"http://127.0.0.1:27002/api/v1/review_tasks/{cancelled_id}/cancel"
                )
                assert response.status_code == 200
                await until(
                    lambda: store.get(cancelled_id),
                    lambda value: value["status"] == "cancelled",
                    10,
                )
            print(
                "PASS cross-API images/Range, Redis events, API exit independence, worker SIGKILL takeover, cross-API cancellation",
                flush=True,
            )
        except BaseException:
            for log in outputs:
                log.flush()
                log.seek(0)
                print(log.read()[-5000:], flush=True)
            raise
        finally:
            for child in children:
                if child.poll() is None:
                    child.terminate()
            for child in children:
                try:
                    await asyncio.to_thread(child.wait, 8)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            for log in outputs:
                log.close()


async def main():
    require_test_database()
    await store.initialize()
    await task_queue.initialize()
    await parameter_store.initialize()
    await parameter_store.close()
    person_operations.initialize()
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) AS total FROM review_tasks")
        assert cursor.fetchone()["total"] == 0, "Use a fresh test database"
    logical = Path("/tmp/wcm/verification") / f"{uuid4().hex}.png"
    try:
        image_store.write_bytes(logical, b"shared-image-bytes")
        assert image_store.read_bytes(logical) == b"shared-image-bytes"
        await verify_queue()
        await verify_locks()
        await verify_person_recovery(logical)
        await verify_processes(logical)
    finally:
        image_store.delete(logical)
    print("ALL CLUSTER INTEGRATION CHECKS PASSED", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
