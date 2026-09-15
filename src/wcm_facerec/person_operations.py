"""Durable before-images and recoverable compensation around IFS mutations.

IFS remains the person/vector authority. MySQL stores operation coordination;
original image bytes are restored through the shared image backend.
"""

import hashlib
import inspect
import json
from contextvars import ContextVar
from uuid import uuid4

from . import face_sync_store, image_store
from .cluster import check_locks, connect, model_slot, run_sync
from .config import settings
from .face_replication import capture, matches, native_faces_hash

_operation = ContextVar("person_operation", default=None)
request_key = ContextVar("person_request_key", default=None)
expected_revision = ContextVar("person_expected_revision", default=None)
_MUTATIONS = {
    "register_person",
    "update_person",
    "delete_person",
    "add_person_image",
    "delete_person_image",
}


def initialize():
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute("""CREATE TABLE IF NOT EXISTS person_operations (
            id VARCHAR(64) PRIMARY KEY, status VARCHAR(24) NOT NULL,
            version BIGINT NOT NULL DEFAULT 1, payload JSON NOT NULL,
            updated_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3)
        )""")


def save_journal(operation_id, payload):
    """Store legacy gallery journals centrally as well as the outer transaction."""
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            "INSERT INTO person_operations (id, status, payload) VALUES (%s, %s, %s) "
            "ON DUPLICATE KEY UPDATE status = VALUES(status), payload = VALUES(payload), version = version + 1",
            (operation_id, payload["status"], json.dumps(payload, ensure_ascii=False)),
        )


def _save(data):
    with connect() as connection, connection.cursor() as cursor:
        version = data["version"]
        if version == 0:
            cursor.execute(
                "INSERT INTO person_operations (id, status, payload) VALUES (%s, %s, %s)",
                (data["id"], data["status"], json.dumps(data, ensure_ascii=False)),
            )
        else:
            changed = cursor.execute(
                "UPDATE person_operations SET status = %s, payload = %s, version = version + 1 "
                "WHERE id = %s AND version = %s",
                (data["status"], json.dumps(data, ensure_ascii=False), data["id"], version),
            )
            if not changed:
                raise RuntimeError("人物操作版本冲突")
        data["version"] = version + 1


def _load(operation_id=None):
    with connect() as connection, connection.cursor() as cursor:
        if operation_id:
            cursor.execute("SELECT * FROM person_operations WHERE id = %s", (operation_id,))
        else:
            cursor.execute(
                "SELECT * FROM person_operations WHERE status IN ('running', 'recovering', 'uncertain') ORDER BY updated_at"
            )
        rows = cursor.fetchall()
    return [{**json.loads(row["payload"]), "version": row["version"]} for row in rows]


async def call(function, *args, **kwargs):
    """Persist a before-image before each first mutation, including lost responses."""
    check_locks()
    data = _operation.get()
    name = getattr(function, "__name__", "")
    if data is not None and name in _MUTATIONS:
        metadata_only = data.get("operation") == "migrate_person_image_keys"
        if metadata_only and name != "update_person":
            raise face_sync_store.ReplicationUnavailable("图片 Key 迁移只允许修改元数据")
        if settings.insightface_replication_enabled and data.get("uncertain"):
            raise face_sync_store.ReplicationUnavailable(
                "先前写入结果不确定，停止后续写入和在线补偿"
            )
        bound = inspect.signature(function).bind_partial(*args, **kwargs)
        if name == "register_person" and not bound.arguments.get("person_id"):
            bound.arguments["person_id"] = str(uuid4())
        args, kwargs = bound.args, bound.kwargs
        pid = bound.arguments["person_id"]
        cid = bound.arguments.get("collection_id") or settings.insightface_collection_id
        identity = f"{cid}:{pid}"
        if identity not in data["before"]:
            adapter = function.__self__
            item = await run_sync(adapter.get_person, pid, collection_id=cid)
            entry = {"collection": cid, "person_id": pid, "item": item}
            if item:
                paths = image_store.image_refs(item)
                face_count = int(item.get("face_count") or 0)
                if metadata_only:
                    entry["metadata_only"] = True
                    if len(paths) != face_count:
                        entry["native_faces_hash"] = await run_sync(
                            native_faces_hash, adapter, cid, pid, face_count
                        )
                elif len(paths) < face_count or (
                    settings.insightface_replication_enabled and len(paths) != face_count
                ):
                    raise ValueError("人物原照片不完整，无法建立可恢复的操作快照")
                if not metadata_only:
                    for path in paths:
                        await run_sync(image_store.read_bytes, path)
            elif metadata_only:
                raise face_sync_store.ReplicationUnavailable("待迁移人物已不存在")
            data["before"][identity] = entry
            await run_sync(_save, data)
        if settings.insightface_replication_enabled:
            data["inflight"] = {"collection": cid, "person_id": pid, "method": name}
            await run_sync(_save, data)
    check_locks()
    try:
        result = await run_sync(function, *args, **kwargs)
    except BaseException:
        if settings.insightface_replication_enabled and data is not None and name in _MUTATIONS:
            data.update(status="uncertain", uncertain=True)
            await run_sync(_save, data)
        raise
    if settings.insightface_replication_enabled and data is not None and name in _MUTATIONS:
        data["inflight"] = None
        await run_sync(_save, data)
    return result


async def restore(engine, data):
    async with model_slot("insightface"):
        await _restore(engine, data)


async def _restore(engine, data):
    """Compensate after fencing; image-key migrations restore only metadata."""
    data.update(status="recovering", uncertain=False, inflight=None)
    await run_sync(_save, data)
    adapter = engine._adapter
    for entry in reversed(list(data["before"].values())):
        check_locks()
        cid, pid, item = entry["collection"], entry["person_id"], entry["item"]
        if entry.get("metadata_only"):
            current = await run_sync(adapter.get_person, pid, collection_id=cid)
            if not current or any(
                current.get(k) != item.get(k) for k in ("id", "name", "external_id", "face_count")
            ):
                raise face_sync_store.ReplicationUnavailable("迁移回滚前人物状态不符，停止恢复")
            fingerprint = entry.get("native_faces_hash")
            if (
                fingerprint
                and await run_sync(native_faces_hash, adapter, cid, pid, item["face_count"])
                != fingerprint
            ):
                raise face_sync_store.ReplicationUnavailable("迁移回滚前原生人脸不符，停止恢复")
            await _restore_call(
                data,
                adapter.update_person,
                pid,
                metadata=engine._item_metadata(item),
                collection_id=cid,
            )
            restored = await run_sync(adapter.get_person, pid, collection_id=cid)
            if not matches(restored, item):
                raise face_sync_store.ReplicationUnavailable("图片 Key 迁移回滚校验失败")
            if (
                fingerprint
                and await run_sync(native_faces_hash, adapter, cid, pid, item["face_count"])
                != fingerprint
            ):
                raise face_sync_store.ReplicationUnavailable("图片 Key 迁移回滚后原生人脸不符")
            continue
        paths = image_store.image_refs(item)
        # Fetch every byte before deleting anything, so a storage outage is safe.
        images = [await run_sync(image_store.read_bytes, path) for path in paths]
        if item and not images:
            raise RuntimeError("人物恢复缺少原照片")
        current = await run_sync(adapter.get_person, pid, collection_id=cid)
        if current:
            check_locks()
            await _restore_call(data, adapter.delete_person, pid, collection_id=cid)
        if item:
            check_locks()
            await _restore_call(
                data,
                adapter.register_person,
                name=item["name"],
                image_bytes=images[0],
                metadata=engine._item_metadata(item),
                external_id=item.get("external_id"),
                person_id=pid,
                collection_id=cid,
            )
            for image in images[1:]:
                check_locks()
                await _restore_call(data, adapter.add_person_image, pid, image, collection_id=cid)
    data["status"] = "rolled_back"
    await run_sync(_save, data)


async def _restore_call(data, function, *args, **kwargs):
    # Recovery has the same response-loss window as the original mutation.
    if settings.insightface_replication_enabled:
        data["inflight"] = {"method": function.__name__, "recovery": True}
        await run_sync(_save, data)
    try:
        result = await run_sync(function, *args, **kwargs)
    except BaseException:
        if settings.insightface_replication_enabled:
            data.update(status="uncertain", uncertain=True)
            await run_sync(_save, data)
        raise
    if settings.insightface_replication_enabled:
        data["inflight"] = None
        await run_sync(_save, data)
    return result


async def run(engine, function, *args, **kwargs):
    # The caller holds the global library lock. Another API instance can finish
    # compensation left behind by a dead process before accepting a new write.
    await run_sync(face_sync_store.before_write)
    for pending in await run_sync(_load):
        if pending.get("kind") == "transaction":
            if settings.insightface_replication_enabled and (
                pending.get("inflight") or pending.get("uncertain")
            ):
                raise face_sync_store.ReplicationUnavailable(
                    "主节点有结果不确定的写入，必须隔离在途请求后恢复"
                )
            await restore(engine, pending)
    key = request_key.get()
    operation_id = key[0] if key else uuid4().hex
    previous = await run_sync(_load, operation_id)
    if previous:
        data = previous[0]
        if not key or data.get("fingerprint") != key[1]:
            raise ValueError("幂等键已用于其他人物操作")
        if data["status"] == "completed":
            return data["result"]
        raise ValueError("此前人物操作已回滚，请刷新后重新提交")
    revision = expected_revision.get()
    if revision and function.__name__ != "register_from_image":
        current = await run_sync(engine._adapter.get_person, args[0])
        if current is None or record_revision(current) != revision:
            raise ValueError("人物资料已被修改，请刷新后重试")
    data = {
        "id": operation_id,
        "kind": "transaction",
        "operation": function.__name__,
        "status": "running",
        "version": 0,
        "before": {},
        "fingerprint": key[1] if key else None,
    }
    await run_sync(_save, data)
    token = _operation.set(data)
    try:
        result = await function(engine, *args, **kwargs)
        check_locks()
        data.update(status="completed", result=result)
        if settings.insightface_replication_enabled:
            snapshots = await capture(engine, data["before"].values())
            await run_sync(_commit_replicated, data, snapshots)
        else:
            await run_sync(_save, data)
        return result
    except CommitUncertain:
        # Do not compensate a transaction whose COMMIT may have succeeded.
        raise
    except Exception:
        check_locks()
        if settings.insightface_replication_enabled and (
            data.get("uncertain") or data.get("inflight")
        ):
            data.update(status="uncertain", uncertain=True)
            await run_sync(_save, data)
            raise face_sync_store.ReplicationUnavailable(
                "人物写入结果不确定，已暂停后续写入；请执行主节点恢复流程"
            ) from None
        await restore(engine, data)
        raise
    finally:
        _operation.reset(token)


class CommitUncertain(face_sync_store.ReplicationUnavailable):
    pass


def _commit_replicated(data, snapshots):
    try:
        with connect() as connection:
            connection.begin()
            with connection.cursor() as cursor:
                changed = cursor.execute(
                    "UPDATE person_operations SET status='completed',payload=%s,version=version+1 WHERE id=%s AND version=%s",
                    (json.dumps(data, ensure_ascii=False), data["id"], data["version"]),
                )
                if not changed:
                    raise RuntimeError("人物操作版本冲突")
                face_sync_store.append(cursor, data["id"], snapshots)
            connection.commit()
        data["version"] += 1
    except Exception as exc:
        try:
            saved = _load(data["id"])
            if saved and saved[0]["status"] == "completed":
                return
        except Exception:
            pass
        raise CommitUncertain(
            "人物提交结果待确认；请用同一幂等键查询或重试，不要生成新请求"
        ) from exc


def record_revision(item):
    fields = {
        key: item.get(key)
        for key in (
            "id",
            "name",
            "occupation",
            "type",
            "remarks",
            "file_path",
            "image_paths",
            "image_key",
            "image_keys",
            "face_count",
        )
    }
    return hashlib.sha256(
        json.dumps(fields, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()
