"""Real-HTTP replication fault checks, confined to fresh test DB/collections.

Run against two disposable InsightFace replicas, never the production primary.
The configured source/target must both be Compose insightface-a/b endpoints.
"""

import asyncio
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

import httpx

from api import parameter_store
from api.face_sync_worker import sync_once
from scripts.face_replication import record_seed, recover, replay_primary
from wcm_facerec import face_sync_store as store
from wcm_facerec import image_store
from wcm_facerec import person_operations as ops
from wcm_facerec.cluster import cluster_slot, connect, run_sync
from wcm_facerec.config import settings
from wcm_facerec.face_engine import FaceEngine
from wcm_facerec.face_replication import node_adapter
from wcm_facerec.image_key_migration import migrate_person_image_keys


def require_isolation():
    assert settings.review_tasks_db_name.startswith("wcm_verify_"), "Fresh test database required"
    assert settings.insightface_collection_id.startswith("wcm-sync-check-"), (
        "Test collection required"
    )
    assert settings.insightface_base_url == "http://insightface-a:8080"
    assert settings.insightface_replicas == {"verify": "http://insightface-b:8080"}
    assert settings.insightface_category_collections == {}


async def kill_after_write():
    require_isolation()
    await parameter_store.initialize()
    target = node_adapter(settings.insightface_replicas["verify"])
    original = target._client.update_person

    def update(*args, **kwargs):
        original(*args, **kwargs)
        os.kill(os.getpid(), signal.SIGKILL)

    target._client.update_person = update
    async with cluster_slot("face-replica-writer:verify"):
        await sync_once("verify", target)


async def main():
    require_isolation()
    await parameter_store.initialize()
    await run_sync(ops.initialize)
    await run_sync(store.initialize)
    engine = FaceEngine()
    primary = engine._primary_adapter
    target = node_adapter(settings.insightface_replicas["verify"])
    cid = settings.insightface_collection_id
    created_collections = []
    photo_path = os.environ["WCM_VERIFY_IMAGE_PATH"]
    photo = await run_sync(image_store.read_bytes, photo_path)
    timings = {}
    created_images = set()
    try:
        for adapter in (primary, target):
            await run_sync(
                adapter._client.create_collection,
                cid,
                name="Disposable replication verification",
                capacity_rows=100,
                max_faces_per_person=20,
            )
            created_collections.append(adapter)
        await run_sync(record_seed, "verify", {})
        start = time.perf_counter()
        from api.main import app

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://wcm-test"
        ) as http:
            headers = {"Idempotency-Key": uuid4().hex}
            form = {
                "name": "__wcm_replication_verify__" + uuid4().hex[:8],
                "type": "replication-verification",
            }
            response = await http.post(
                "/api/v1/face_records",
                headers=headers,
                data=form,
                files={"file": ("face.jpg", photo, "image/jpeg")},
            )
            assert response.status_code == 200, response.text
            result = response.json()
            replay = await http.post(
                "/api/v1/face_records",
                headers=headers,
                data=form,
                files={"file": ("face.jpg", photo, "image/jpeg")},
            )
            assert replay.status_code == 200 and replay.json() == result, replay.text
        timings["primary_registration_ms"] = round((time.perf_counter() - start) * 1000, 1)
        pid = result["id"]
        record = await run_sync(primary.get_person, pid, collection_id=cid)
        created_images.update(image_store.image_refs(record))
        assert record["metadata"]["image_key"].startswith(settings.s3_prefix.strip("/") + "/")
        assert "file_path" not in record["metadata"] and "image_paths" not in record["metadata"]
        assert (await run_sync(store.status))["committed_sequence"] == 1
        assert await run_sync(store.acquire_read) is None
        await sync_once("verify", target)
        ids = await run_sync(target.person_face_ids, pid, collection_id=cid)
        assert len(ids) == 1 and ids != await run_sync(
            primary.person_face_ids, pid, collection_id=cid
        )
        result = await engine.search_multi_face(photo, threshold=1.0, include_source_bbox=True)
        match = next(x for x in result["all_results"] if x["person_id"] == pid)
        assert match["matched_face_id"] in ids and match["source_w"] > 0
        print(
            "PASS real registration, durable outbox, stale-read exclusion, replica-local face IDs and pinned bbox lookup",
            flush=True,
        )

        start = time.perf_counter()
        await engine.update_person_record(pid, name=None, metadata={"remarks": "metadata-v2"})
        timings["primary_metadata_ms"] = round((time.perf_counter() - start) * 1000, 1)
        await engine.update_person_record(pid, name=None, metadata={"remarks": "metadata-v3"})
        await engine.update_person_record(pid, name=None, metadata={"remarks": "metadata-v4"})
        start = time.perf_counter()
        await sync_once("verify", target)
        timings["coalesced_metadata_sync_ms"] = round((time.perf_counter() - start) * 1000, 1)
        assert (await run_sync(target.get_person, pid, collection_id=cid))[
            "remarks"
        ] == "metadata-v4"
        assert await run_sync(target.person_face_ids, pid, collection_id=cid) == ids
        print(
            "PASS ordered catch-up coalesces edits without reenrolling unchanged photos", flush=True
        )

        await engine.update_person_record(pid, name=None, metadata={"remarks": "after-offline"})
        original_health = target.health
        target.health = lambda: (_ for _ in ()).throw(ConnectionError("offline"))
        await sync_once("verify", target)
        target.health = original_health
        assert (await run_sync(store.node, "verify"))[1]["state"] == "retry"
        assert await run_sync(store.acquire_read) is None
        with connect() as db, db.cursor() as c:
            c.execute("UPDATE face_sync_nodes SET next_retry=NULL WHERE id='verify'")
        await sync_once("verify", target)
        assert (await run_sync(target.get_person, pid, collection_id=cid))[
            "remarks"
        ] == "after-offline"
        print(
            "PASS unreachable replica backs off, remains out of read pool and automatically catches up",
            flush=True,
        )

        lease = await run_sync(store.acquire_read)
        assert lease is not None
        await engine.update_person_record(
            pid, name=None, metadata={"remarks": "after-reader-drained"}
        )
        syncing = asyncio.create_task(sync_once("verify", target))
        await asyncio.sleep(0.3)
        assert not syncing.done()
        assert (await run_sync(target.get_person, pid, collection_id=cid))[
            "remarks"
        ] == "after-offline"
        await run_sync(store.release_read, lease["lease"])
        await syncing
        print("PASS replica waits for already-admitted readers before applying writes", flush=True)

        await engine.update_person_record(pid, name=None, metadata={"remarks": "response-lost"})
        original_update = target._client.update_person

        def lose_response(*args, **kwargs):
            original_update(*args, **kwargs)
            raise TimeoutError("Injected response loss after actual server commit")

        target._client.update_person = lose_response
        await sync_once("verify", target)
        target._client.update_person = original_update
        assert (await run_sync(store.node, "verify"))[1]["state"] == "quarantined"
        assert await run_sync(store.acquire_read) is None
        # The injection raised only after the real request completed: no old
        # request remains in flight. This proves the fencing prerequisite here.
        await recover("verify")
        await sync_once("verify", target)
        assert len(await run_sync(target.person_face_ids, pid, collection_id=cid)) == 1
        print(
            "PASS lost write response quarantines; fenced full-state recovery creates no duplicate faces",
            flush=True,
        )

        await engine.update_person_record(pid, name=None, metadata={"remarks": "worker-killed"})
        child = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import asyncio; from scripts.verify_face_replication import kill_after_write; asyncio.run(kill_after_write())",
            ]
        )
        assert await asyncio.to_thread(child.wait, 60) == -signal.SIGKILL
        assert (await run_sync(store.node, "verify"))[1]["state"] == "syncing"
        await sync_once("verify", target)
        assert (await run_sync(store.node, "verify"))[1]["state"] == "quarantined"
        await recover("verify")
        await sync_once("verify", target)
        assert (await run_sync(target.get_person, pid, collection_id=cid))[
            "remarks"
        ] == "worker-killed"
        assert len(await run_sync(target.person_face_ids, pid, collection_id=cid)) == 1
        print(
            "PASS actual synchronization process SIGKILL is detected; fenced restart converges without duplicate writes",
            flush=True,
        )

        original_connect = ops.connect

        class LostCommit:
            def __init__(self):
                self.db = original_connect()

            def __getattr__(self, key):
                return getattr(self.db, key)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.db.close()

            def commit(self):
                self.db.commit()
                raise ConnectionError("Injected lost MySQL COMMIT acknowledgement")

        ops.connect = LostCommit
        key = ops.request_key.set((uuid4().hex, "commit-ack-verification"))
        try:
            before = (await run_sync(store.status))["committed_sequence"]
            first = await engine.update_person_record(
                pid, name=None, metadata={"remarks": "commit-ack-lost"}
            )
            second = await engine.update_person_record(
                pid, name=None, metadata={"remarks": "commit-ack-lost"}
            )
            assert first == second
            assert (await run_sync(store.status))["committed_sequence"] == before + 1
        finally:
            ops.connect = original_connect
            ops.request_key.reset(key)
        await sync_once("verify", target)
        assert (await run_sync(target.get_person, pid, collection_id=cid))[
            "remarks"
        ] == "commit-ack-lost"
        print(
            "PASS lost MySQL commit acknowledgement preserves operation/outbox atomicity and idempotent response",
            flush=True,
        )

        # This test collection's sequence-0 baseline is empty. Simulate restoring
        # that old primary state, prove a wrong sequence fails closed, then replay.
        await run_sync(primary.delete_person, pid, collection_id=cid)
        head = (await run_sync(store.status))["committed_sequence"]
        try:
            await replay_primary(head)
            raise AssertionError("An incorrect backup sequence must not be accepted")
        except store.ReplicationUnavailable:
            assert (await run_sync(store.status))["primary_recovering"] is True
        await replay_primary(0)
        assert (await run_sync(primary.get_person, pid, collection_id=cid))[
            "remarks"
        ] == "commit-ack-lost"
        assert (await run_sync(store.status))["primary_recovering"] is False
        assert (await run_sync(store.status))["committed_sequence"] == head
        print(
            "PASS primary snapshot recovery replays committed state; wrong backup sequence keeps writes blocked",
            flush=True,
        )

        # Exercise the actual key-migration CLI against a legacy-format record.
        current = await run_sync(primary.get_person, pid, collection_id=cid)
        refs = image_store.image_refs(current)
        prefix = settings.s3_prefix.strip("/") + "/"
        legacy_paths = ["/tmp/wcm/" + ref[len(prefix) :] for ref in refs]
        legacy_metadata = {
            k: v for k, v in current["metadata"].items() if k not in ("image_key", "image_keys")
        }
        legacy_metadata.update(file_path=legacy_paths[0], image_paths=legacy_paths)
        await migrate_person_image_keys(
            engine, pid, cid, store.digest(current["metadata"]), legacy_metadata
        )
        await sync_once("verify", target)
        original_primary_ids = await run_sync(primary.person_face_ids, pid, collection_id=cid)
        original_replica_ids = await run_sync(target.person_face_ids, pid, collection_id=cid)
        original_patch = primary._client.update_person

        def lost_migration_response(*args, **kwargs):
            original_patch(*args, **kwargs)
            raise TimeoutError("Injected lost metadata migration response after completion")

        primary._client.update_person = lost_migration_response
        try:
            try:
                await migrate_person_image_keys(engine, pid, cid, store.digest(legacy_metadata))
                raise AssertionError("Expected quarantine after lost migration response")
            except store.ReplicationUnavailable:
                pass
        finally:
            primary._client.update_person = original_patch
        pending = [row for row in await run_sync(ops._load) if row.get("kind") == "transaction"]
        assert len(pending) == 1 and pending[0]["uncertain"]
        # The injected exception occurred after the HTTP response: no old write
        # is in flight, so this test has satisfied the external fencing condition.
        async with cluster_slot("person-library"):
            await ops.restore(engine, pending[0])
        assert (await run_sync(primary.get_person, pid, collection_id=cid))[
            "metadata"
        ] == legacy_metadata
        assert (
            await run_sync(primary.person_face_ids, pid, collection_id=cid) == original_primary_ids
        )
        print(
            "PASS uncertain metadata migration restores only metadata and preserves native face IDs",
            flush=True,
        )

        reports = []

        async def migrate_cli(*extra):
            path = "/tmp/wcm-image-key-check-" + uuid4().hex + ".json"
            reports.append(path)
            command = [sys.executable, "-m", "scripts.migrate_image_keys", "--report", path, *extra]
            completed = await asyncio.to_thread(
                subprocess.run, command, capture_output=True, text=True, timeout=120
            )
            assert completed.returncode == 0, completed.stdout + completed.stderr
            return path, json.loads(Path(path).read_text())

        try:
            plan, report = await migrate_cli("--apply")
            assert report["planned"] == 1
            await sync_once("verify", target)
            _, report = await migrate_cli()
            assert report["planned"] == 0
            await migrate_cli("--apply", "--rollback-from", plan)
            await sync_once("verify", target)
            assert (await run_sync(primary.get_person, pid, collection_id=cid))[
                "metadata"
            ] == legacy_metadata
            await migrate_cli("--apply")
            await sync_once("verify", target)
            assert (
                await run_sync(primary.person_face_ids, pid, collection_id=cid)
                == original_primary_ids
            )
            assert (
                await run_sync(target.person_face_ids, pid, collection_id=cid)
                == original_replica_ids
            )
            assert (await run_sync(target.get_person, pid, collection_id=cid))["metadata"][
                "image_keys"
            ] == refs
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://wcm-test"
            ) as http:
                photo_url = image_store.public_url(refs[0])
                response = await http.get(photo_url)
                assert response.status_code == 200 and response.content == photo
                assert (await http.head(photo_url)).status_code == 200
            print(
                "PASS actual migration CLI, repeated-run skip, rollback/remigration, preserved face IDs and stable streamed image URL",
                flush=True,
            )
        finally:
            for path in reports:
                Path(path).unlink(missing_ok=True)

        await engine.delete_person_record(pid)
        await sync_once("verify", target)
        assert await run_sync(target.get_person, pid, collection_id=cid) is None
        state = await run_sync(store.status)
        assert state["replicas"][0]["lag"] == 0 and state["replicas"][0]["state"] == "ready"
        print("PASS replicated deletion and final checkpoint convergence", flush=True)
        print(json.dumps({"timings": timings}, ensure_ascii=False), flush=True)
        print("ALL REAL INSIGHTFACE REPLICATION CHECKS PASSED", flush=True)
    finally:
        for adapter in created_collections:
            with contextlib.suppress(Exception):
                await run_sync(adapter._client.delete_collection, cid, force=True)
        for path in created_images:
            with contextlib.suppress(Exception):
                await run_sync(image_store.delete, path)
        await parameter_store.close()


if __name__ == "__main__":
    asyncio.run(main())
