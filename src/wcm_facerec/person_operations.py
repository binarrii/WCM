"""Durable before-images and recoverable compensation around IFS mutations.

IFS remains the person/vector authority. MySQL stores operation coordination;
original image bytes are restored through the shared image backend.
"""

import hashlib
import inspect
import json
from contextvars import ContextVar
from uuid import uuid4

from . import image_store
from .cluster import check_locks, connect, model_slot, run_sync
from .config import settings

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
                "SELECT * FROM person_operations WHERE status IN ('running', 'recovering') ORDER BY updated_at"
            )
        rows = cursor.fetchall()
    return [{**json.loads(row["payload"]), "version": row["version"]} for row in rows]


async def call(function, *args, **kwargs):
    """Persist a before-image before each first mutation, including lost responses."""
    check_locks()
    data = _operation.get()
    name = getattr(function, "__name__", "")
    if data is not None and name in _MUTATIONS:
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
            if item:
                paths = list(
                    dict.fromkeys(
                        p for p in [item.get("file_path"), *(item.get("image_paths") or [])] if p
                    )
                )
                if len(paths) < int(item.get("face_count") or 0):
                    raise ValueError("人物原照片不完整，无法建立可恢复的操作快照")
                for path in paths:
                    await run_sync(image_store.read_bytes, path)
            data["before"][identity] = {"collection": cid, "person_id": pid, "item": item}
            await run_sync(_save, data)
    check_locks()
    return await run_sync(function, *args, **kwargs)


async def restore(engine, data):
    async with model_slot("insightface"):
        await _restore(engine, data)


async def _restore(engine, data):
    """Idempotent full-person compensation; repeating after a crash is safe."""
    data["status"] = "recovering"
    await run_sync(_save, data)
    adapter = engine._adapter
    for entry in reversed(list(data["before"].values())):
        check_locks()
        cid, pid, item = entry["collection"], entry["person_id"], entry["item"]
        paths = list(
            dict.fromkeys(
                p
                for p in ([item.get("file_path"), *(item.get("image_paths") or [])] if item else [])
                if p
            )
        )
        # Fetch every byte before deleting anything, so a storage outage is safe.
        images = [await run_sync(image_store.read_bytes, path) for path in paths]
        if item and not images:
            raise RuntimeError("人物恢复缺少原照片")
        current = await run_sync(adapter.get_person, pid, collection_id=cid)
        if current:
            check_locks()
            await run_sync(adapter.delete_person, pid, collection_id=cid)
        if item:
            check_locks()
            await run_sync(
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
                await run_sync(adapter.add_person_image, pid, image, collection_id=cid)
    data["status"] = "rolled_back"
    await run_sync(_save, data)


async def run(engine, function, *args, **kwargs):
    # The caller holds the global library lock. Another API instance can finish
    # compensation left behind by a dead process before accepting a new write.
    for pending in await run_sync(_load):
        if pending.get("kind") == "transaction":
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
        await run_sync(_save, data)
        return result
    except Exception:
        check_locks()
        await restore(engine, data)
        raise
    finally:
        _operation.reset(token)


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
            "face_count",
        )
    }
    return hashlib.sha256(
        json.dumps(fields, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()
