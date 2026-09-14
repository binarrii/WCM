"""State replication through the existing InsightFace HTTP API only.

Replicas are removed from read admission before writes. Ambiguous upstream
writes require fencing the old caller/server before an operator resumes them.
"""

import asyncio
import contextlib
import hashlib
from contextvars import ContextVar
from functools import lru_cache, wraps

from . import face_sync_store as store
from . import image_store
from .cluster import check_locks, cluster_slot, run_sync
from .config import settings
from .ifs_adapter import InsightFaceAdapter

current_adapter = ContextVar("face_read_adapter", default=None)
current_node = ContextVar("face_read_node", default=None)


@lru_cache(maxsize=16)
def adapter(url, collection, timeout, key):
    return InsightFaceAdapter(url, collection, timeout=timeout, api_key=key or None)


def node_adapter(url):
    return adapter(
        url,
        settings.insightface_collection_id,
        settings.insightface_timeout_s,
        settings.insightface_api_key,
    )


def manifest(cid, pid, item, *, hashes=None, strict=True):
    paths = list(
        dict.fromkeys(
            p
            for p in ([item.get("file_path"), *(item.get("image_paths") or [])] if item else [])
            if p
        )
    )
    incomplete = item is not None and len(paths) != int(item.get("face_count") or 0)
    if incomplete and strict:
        raise store.ReplicationUnavailable(
            f"人物 {pid} 原照片数与登记人脸数不一致，不能可靠重建副本"
        )
    images = []
    for path in paths:
        if hashes is not None and path in hashes:
            sha = hashes[path]
        else:
            sha = image_store.stat(path).get("Metadata", {}).get("sha256")
            if not sha:
                sha = hashlib.sha256(image_store.read_bytes(path)).hexdigest()
            if hashes is not None:
                hashes[path] = sha
        images.append({"path": path, "sha256": sha})
    person = (
        None
        if item is None
        else {
            "id": pid,
            "name": item.get("name"),
            "external_id": item.get("external_id"),
            "metadata": item.get("metadata") or {},
            "face_count": int(item["face_count"]),
        }
    )
    snapshot = {"collection": cid, "person_id": pid, "person": person, "images": images}
    if incomplete:
        # Baseline copies preserve native faces that cannot be regenerated from
        # the remaining originals. They are never admitted to the write log.
        snapshot["rebuildable"] = False
    return snapshot


async def capture(engine, entries):
    snapshots = []
    for entry in entries:
        cid, pid = entry["collection"], entry["person_id"]
        item = await run_sync(engine._primary_adapter.get_person, pid, collection_id=cid)
        snapshots.append(await run_sync(manifest, cid, pid, item))
    return snapshots


def matches(item, person):
    if item is None or person is None:
        return item is None and person is None
    return all(
        item.get(key) == person.get(key)
        for key in ("id", "name", "external_id", "metadata", "face_count")
    )


class AmbiguousReplicaWrite(store.ReplicationUnavailable):
    pass


async def apply_snapshot(name, owner, target, snapshot):
    """Converge a complete person; never retry an append with an unknown result."""
    if snapshot.get("rebuildable") is False:
        raise store.ReplicationUnavailable("历史人物缺少完整原照片，只能从原生备份恢复")
    cid, pid, person = snapshot["collection"], snapshot["person_id"], snapshot["person"]
    previous = await run_sync(store.applied, name, snapshot)
    current = await run_sync(target.get_person, pid, collection_id=cid)
    if (
        previous
        and previous["manifest_hash"] == store.digest(snapshot)
        and matches(current, person)
    ):
        return
    metadata_only = (
        person is not None
        and current is not None
        and previous
        and previous["images_hash"] == store.digest(snapshot["images"])
        and current.get("face_count") == person["face_count"]
        and current.get("external_id") == person["external_id"]
    )
    # Download and verify everything before the first destructive HTTP request.
    images = []
    if person and not metadata_only:
        for image in snapshot["images"]:
            data = await run_sync(image_store.read_bytes, image["path"])
            if hashlib.sha256(data).hexdigest() != image["sha256"]:
                raise store.ReplicationUnavailable("同步图片 SHA-256 校验失败")
            images.append(data)
    check_locks()
    try:
        if metadata_only:
            await run_sync(
                target._client.update_person,
                cid,
                pid,
                name=person["name"],
                metadata=person["metadata"],
            )
        else:
            if current:
                await run_sync(target.delete_person, pid, collection_id=cid)
            check_locks()
            if person:
                result = await run_sync(
                    target._client.create_person,
                    cid,
                    person_id=pid,
                    name=person["name"],
                    external_id=person["external_id"],
                    metadata=person["metadata"],
                    images=images,
                )
                if len(result.faces or []) != len(images) or result.rejected_images:
                    raise store.ReplicationUnavailable("副本未完整登记所有照片")
        actual = await run_sync(target.get_person, pid, collection_id=cid)
        if not matches(actual, person):
            raise store.ReplicationUnavailable("副本人物写入后校验失败")
    except BaseException as exc:
        # Includes timeout, response loss, process cancellation and index errors
        # after SQLite committed. An ordinary retry cannot fence an old request.
        with contextlib.suppress(Exception):
            await run_sync(
                store.quarantine, name, f"ambiguous_write:{type(exc).__name__}; fence then recover"
            )
        raise AmbiguousReplicaWrite(
            "副本写入结果不确定，已隔离；需停止旧执行者及副本在途请求后恢复"
        ) from exc
    check_locks()
    await run_sync(store.record_applied, name, owner, snapshot)


def replica_read(function):
    """Pin an entire search, including subsequent face-ID lookups, to one node."""

    @wraps(function)
    async def wrapped(*args, **kwargs):
        if not settings.insightface_replication_enabled or current_adapter.get() is not None:
            return await function(*args, **kwargs)
        selected = await run_sync(store.acquire_read)
        if selected is None:
            # Read-after-write remains correct while all replicas catch up.
            async with cluster_slot("person-library"):
                await run_sync(store.assert_primary_clean)
                return await function(*args, **kwargs)
        target = node_adapter(selected["url"])
        token = current_adapter.set(target)
        node_token = current_node.set(selected["id"])
        guard = {"lost": False}
        guard_token = store.read_guard.set(guard)
        owner_task = asyncio.current_task()
        lost = False

        async def renew():
            nonlocal lost
            try:
                while True:
                    await asyncio.sleep(settings.insightface_read_lease_s / 3)
                    await run_sync(store.renew_read, selected["lease"])
            except asyncio.CancelledError:
                raise
            except Exception:
                lost = True
                guard["lost"] = True
                owner_task.cancel()

        monitor = asyncio.create_task(renew())
        try:
            return await function(*args, **kwargs)
        except asyncio.CancelledError:
            if lost:
                raise store.ReplicationUnavailable("读取租约失效，请重试整个查询") from None
            raise
        finally:
            monitor.cancel()
            await asyncio.gather(monitor, return_exceptions=True)
            current_adapter.reset(token)
            current_node.reset(node_token)
            store.read_guard.reset(guard_token)
            with contextlib.suppress(Exception):
                await run_sync(store.release_read, selected["lease"])

    return wrapped
