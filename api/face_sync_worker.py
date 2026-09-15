"""Independent durable InsightFace synchronization worker."""

import asyncio
import contextlib
import logging
import signal
import time
from pathlib import Path
from uuid import uuid4

from wcm_facerec import face_sync_store as store
from wcm_facerec import person_operations
from wcm_facerec.cluster import cluster_slot, run_sync
from wcm_facerec.config import settings
from wcm_facerec.face_replication import AmbiguousReplicaWrite, apply_snapshot, node_adapter

from . import parameter_store

logger = logging.getLogger(__name__)


async def sync_once(name, target):
    root, row = await run_sync(store.node, name)
    if row["state"] in {"new", "quarantined", "disabled"}:
        return
    if row["state"] in {"draining", "syncing"}:
        # A previous owner disappeared. SQLite cannot enforce our fencing token.
        await run_sync(
            store.quarantine,
            name,
            "interrupted_owner: fence old worker and replica requests, then recover",
        )
        return
    if not row["due"]:
        return
    try:
        await run_sync(target.health)
    except Exception:
        # No write was attempted; automatic backoff is safe here.
        owner = uuid4().hex
        if await run_sync(store.begin_sync, name, owner):
            await run_sync(store.retry, name, owner, "replica_unreachable_before_write")
        return
    await run_sync(store.heartbeat, name)
    if row["applied_seq"] == root["head"]:
        if row["state"] == "retry":
            owner = uuid4().hex
            if await run_sync(store.begin_sync, name, owner):
                await run_sync(store.ready, name, owner)
        await run_sync(store.complete_sync_request, name, row.get("sync_request_id"))
        return
    owner = uuid4().hex
    if not await run_sync(store.begin_sync, name, owner):
        return
    try:
        while not await run_sync(store.drained, name):
            await asyncio.sleep(0.25)
        await run_sync(store.owned_update, name, owner, "state='syncing'")
        batch = await run_sync(store.changes, row["applied_seq"])
        if not batch:
            raise store.ReplicationUnavailable("同步日志缺失，不能推进副本版本")
        # Readers only see the whole committed batch. Collapse repeated edits to
        # the same person to avoid redundant inference during a long catch-up.
        latest = {}
        for event in batch:
            for snapshot in event["payload"]:
                latest[store.identity(snapshot["collection"], snapshot["person_id"])] = snapshot
        for snapshot in latest.values():
            await apply_snapshot(name, owner, target, snapshot)
        await run_sync(store.checkpoint, name, owner, batch[-1]["seq"])
        await run_sync(store.ready, name, owner)
        await run_sync(store.complete_sync_request, name, row.get("sync_request_id"))
    except AmbiguousReplicaWrite:
        logger.error("Replica %s quarantined after an ambiguous write", name)
    except asyncio.CancelledError:
        with contextlib.suppress(Exception):
            await run_sync(store.quarantine, name, "sync_cancelled: fence then recover")
        raise
    except Exception as exc:
        # This path contains reads, verification, and durable checkpoint writes.
        # An SDK mutation error is handled separately and never auto-retried.
        with contextlib.suppress(Exception):
            await run_sync(store.retry, name, owner, type(exc).__name__)
        logger.warning("Replica %s will retry after %s", name, type(exc).__name__)


async def serve():
    logging.basicConfig(level=logging.INFO)
    if not settings.insightface_replication_enabled:
        raise RuntimeError("Enable WCM_INSIGHTFACE_REPLICATION_ENABLED")
    await parameter_store.initialize()
    await run_sync(person_operations.initialize)
    await run_sync(store.initialize)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    async def replicate(name, url):
        while not stop.is_set():
            try:
                async with cluster_slot(f"face-replica-writer:{name}"):
                    target = node_adapter(url)
                    while not stop.is_set():
                        await sync_once(name, target)
                        with contextlib.suppress(TimeoutError):
                            await asyncio.wait_for(stop.wait(), settings.insightface_sync_poll_s)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Replica %s coordinator unavailable: %s", name, type(exc).__name__)
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(stop.wait(), settings.insightface_sync_poll_s)

    tasks = [
        asyncio.create_task(replicate(name, url))
        for name, url in settings.insightface_replicas.items()
    ]
    try:
        while not stop.is_set():
            if any(task.done() for task in tasks):
                for task in tasks:
                    if task.done():
                        task.result()
                raise RuntimeError("Replica coordinator exited unexpectedly")
            Path("/tmp/wcm-face-sync-health").write_text(str(time.time()))
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), 5)
    finally:
        stop.set()
        try:
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=120)
        except TimeoutError:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        await parameter_store.close()


if __name__ == "__main__":
    asyncio.run(serve())
