"""Gallery mutations shared by the aggregate and category face collections."""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import logging
from functools import wraps
from pathlib import Path
from uuid import uuid4

from .config import settings

logger = logging.getLogger(__name__)
IMAGE_ROOT = Path("/tmp/wcm")


class SameNamePeopleError(ValueError):
    def __init__(self, people: list[dict]):
        super().__init__("已存在同名人物，请选择合并或新建")
        self.people = people


def library_write(func):
    """Serialize library writes across Gunicorn workers, without blocking reads."""

    @wraps(func)
    async def locked(*args, **kwargs):
        IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
        with (IMAGE_ROOT / ".person-library.lock").open("a") as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    await asyncio.sleep(0.1)
            try:
                task = asyncio.create_task(func(*args, **kwargs))
                try:
                    return await asyncio.shield(task)
                except asyncio.CancelledError:
                    # A disconnected client must not release the lock while an
                    # SDK thread is still writing or compensation is in progress.
                    await task
                    raise
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    return locked


def gallery(item):
    paths = [item.get("file_path"), *(item.get("image_paths") or [])]
    return list(dict.fromkeys(path for path in paths if isinstance(path, str) and path))


def read_gallery(item):
    images = {}
    for value in gallery(item):
        path = Path(value).resolve()
        if not path.is_relative_to(IMAGE_ROOT.resolve()) or not path.is_file():
            raise ValueError(f"人物 {item['id']} 的原照片缺失，无法安全合并，请先恢复照片")
        images[value] = path.read_bytes()
    if not images or int(item.get("face_count") or 0) > len(images):
        raise ValueError(f"人物 {item['id']} 的原照片不完整，无法安全合并")
    return images


async def mutate_gallery(engine, target_id, *, source_ids=None, image=None, image_ext=".jpg"):
    """Prepare all galleries before writes; compensate failures from a durable journal.

    Source files are retained: historical review results can still reference them.
    Re-enrollment uses original full images, never thumbnails or lossy crops.
    """
    adapter = engine._adapter
    run = engine._run
    aggregate = settings.insightface_collection_id
    target = await run(adapter.get_person, target_id)
    if not target:
        raise LookupError("保留的人物不存在，请刷新列表")
    sources = []
    for source_id in source_ids or []:
        source = await run(adapter.get_person, source_id)
        if not source:
            raise LookupError(f"待合并人物 {source_id} 不存在，请刷新列表")
        sources.append(source)

    # Snapshots include raw metadata so custom import fields survive mutations.
    snapshots = [(aggregate, target_id, target)]
    target_category = target.get("type") or target.get("category")
    target_cid = settings.insightface_category_collections.get(target_category)
    if target_cid and target_cid != aggregate:
        mirror = await engine._find_category_mirror(target_id, target_cid)
        snapshots.append((target_cid, mirror["id"] if mirror else target_id, mirror))
    target_slots = len(snapshots)
    for source in sources:
        for cid in dict.fromkeys(settings.insightface_category_collections.values()):
            if cid == aggregate:
                continue
            mirror = await engine._find_category_mirror(source["id"], cid)
            if mirror:
                snapshots.append((cid, mirror["id"], mirror))
        snapshots.append((aggregate, source["id"], source))

    images = {}
    for _, _, item in snapshots:
        if item:
            images.update(await run(read_gallery, item))
    paths = gallery(target)
    hashes = {hashlib.sha256(images[path]).digest() for path in paths}
    for _, _, item in snapshots[1:]:
        for path in gallery(item or {}):
            digest = hashlib.sha256(images[path]).digest()
            if digest not in hashes:
                hashes.add(digest)
                paths.append(path)
    new_file = None
    if image is not None:
        digest = hashlib.sha256(image).digest()
        if digest in hashes:
            return {"record": target, "added_images": 0, "merged_ids": []}
        # Keep paths independent of mutable names/category and of other people.
        path = IMAGE_ROOT / "uploads" / f"{uuid4().hex}{image_ext}"
        path.parent.mkdir(parents=True, exist_ok=True)
        await run(path.write_bytes, image)
        new_file = path
        paths.append(str(path))
        images[str(path)] = image

    metadata = {**engine._item_metadata(target), "image_paths": paths}
    operation_id = uuid4().hex
    journal_dir = IMAGE_ROOT / ".person-operations"
    journal_dir.mkdir(parents=True, exist_ok=True)
    journal = journal_dir / f"{operation_id}.json"
    journal_data = {"id": operation_id, "status": "prepared", "snapshots": snapshots}

    def save_journal(status):
        journal_data["status"] = status
        temporary = journal.with_suffix(".tmp")
        temporary.write_text(json.dumps(journal_data, ensure_ascii=False), encoding="utf-8")
        temporary.replace(journal)

    original_faces = {}
    created = []
    deleted = []
    touched = []
    try:
        for cid, pid, item in snapshots[:target_slots]:
            if item:
                original_faces[(cid, pid)] = await run(
                    adapter.person_face_ids, pid, collection_id=cid
                )
        journal_data["original_faces"] = [
            {"collection": cid, "person_id": pid, "face_ids": ids}
            for (cid, pid), ids in original_faces.items()
        ]
        await run(save_journal, "prepared")
        for cid, pid, item in snapshots[:target_slots]:
            if item is None:
                # Record intent before creation so a lost response can be recovered.
                created.append((cid, pid))
                await run(
                    adapter.register_person,
                    name=target["name"],
                    image_bytes=images[paths[0]],
                    metadata=metadata,
                    external_id=target_id,
                    person_id=pid,
                    collection_id=cid,
                )
                remaining = paths[1:]
            else:
                existing = {hashlib.sha256(images[p]).digest() for p in gallery(item)}
                remaining = [p for p in paths if hashlib.sha256(images[p]).digest() not in existing]
            for path in remaining:
                await run(adapter.add_person_image, pid, images[path], collection_id=cid)
            if item:
                touched.append((cid, pid, item))
                await run(adapter.update_person, pid, metadata=metadata, collection_id=cid)

        # Verify the retained record before removing any source identity.
        updated = await run(adapter.get_person, target_id)
        if not updated or gallery(updated) != paths:
            raise RuntimeError("合并后的照片信息校验失败")
        await run(save_journal, "target_ready")
        for cid, pid, item in snapshots[target_slots:]:
            deleted.append((cid, pid, item))
            await run(adapter.delete_person, pid, collection_id=cid)
        await run(save_journal, "completed")
        return {
            "record": updated,
            "added_images": len(paths) - len(gallery(target)),
            "merged_ids": [source["id"] for source in sources],
        }
    except Exception as exc:
        failures = []
        # Restore deleted source identities first, with every original sample.
        for cid, pid, item in reversed(deleted):
            try:
                if await run(adapter.get_person, pid, collection_id=cid):
                    continue
                originals = gallery(item)
                await run(
                    adapter.register_person,
                    name=item["name"],
                    image_bytes=images[originals[0]],
                    metadata=engine._item_metadata(item),
                    external_id=item.get("external_id"),
                    person_id=pid,
                    collection_id=cid,
                )
                for path in originals[1:]:
                    await run(adapter.add_person_image, pid, images[path], collection_id=cid)
            except Exception:
                failures.append(pid)
                logger.exception("Cannot restore source %s (operation %s)", pid, operation_id)
        for cid, pid, item in reversed(touched):
            try:
                await run(
                    adapter.update_person,
                    pid,
                    metadata=engine._item_metadata(item),
                    collection_id=cid,
                )
            except Exception:
                failures.append(pid)
        # Compare IDs to handle an enrollment that committed but lost its response.
        for (cid, pid), initial_ids in original_faces.items():
            try:
                current_ids = await run(adapter.person_face_ids, pid, collection_id=cid)
                for face_id in set(current_ids) - set(initial_ids):
                    await run(adapter.delete_person_image, pid, face_id, collection_id=cid)
            except Exception:
                failures.append(pid)
        for cid, pid in reversed(created):
            try:
                if await run(adapter.get_person, pid, collection_id=cid):
                    await run(adapter.delete_person, pid, collection_id=cid)
            except Exception:
                failures.append(pid)
        if new_file and not failures:
            new_file.unlink(missing_ok=True)
        await run(save_journal, "recovery_required" if failures else "rolled_back")
        if failures:
            raise RuntimeError(
                f"操作失败且部分恢复未完成，请联系管理员，操作编号：{operation_id}"
            ) from exc
        raise


async def remove_gallery_images(engine, target_id: str, image_paths: list[str]):
    """Remove selected images while keeping aggregate/category galleries identical.

    IFS face ids are collection-local and are not persisted beside the original
    image paths. Rebuilding each Person from the retained originals avoids
    guessing that the server's face order matches metadata order. Original files
    stay on disk so historical review results that already reference them remain
    renderable.
    """
    adapter = engine._adapter
    run = engine._run
    aggregate = settings.insightface_collection_id
    target = await run(adapter.get_person, target_id)
    if not target:
        raise LookupError("人物不存在，请刷新列表")

    original_paths = gallery(target)
    resolved_paths = {str(Path(path).resolve()): path for path in original_paths}
    requested = list(dict.fromkeys(str(Path(path).resolve()) for path in image_paths))
    if not requested:
        raise ValueError("请选择需要删除的照片")
    if any(path not in resolved_paths for path in requested):
        raise ValueError("所选照片不属于该人物，请刷新后重试")

    removed = {resolved_paths[path] for path in requested}
    retained = [path for path in original_paths if path not in removed]
    if not retained:
        raise ValueError("每个人物至少需要保留一张照片")

    snapshots = [(aggregate, target_id, target)]
    for cid in dict.fromkeys(settings.insightface_category_collections.values()):
        if cid == aggregate:
            continue
        mirror = await engine._find_category_mirror(target_id, cid)
        if mirror:
            snapshots.append((cid, mirror["id"], mirror))

    images = {}
    for _, _, item in snapshots:
        images.update(await run(read_gallery, item))
    for path in retained:
        if path not in images:
            raise ValueError(f"人物 {target_id} 的原照片缺失，无法安全删除")

    operation_id = uuid4().hex
    journal_dir = IMAGE_ROOT / ".person-operations"
    journal_dir.mkdir(parents=True, exist_ok=True)
    journal = journal_dir / f"{operation_id}.json"
    journal_data = {
        "id": operation_id,
        "status": "prepared",
        "operation": "remove_images",
        "removed_paths": sorted(removed),
        "snapshots": snapshots,
    }

    def save_journal(status):
        journal_data["status"] = status
        temporary = journal.with_suffix(".tmp")
        temporary.write_text(json.dumps(journal_data, ensure_ascii=False), encoding="utf-8")
        temporary.replace(journal)

    async def rebuild(cid, pid, item, paths):
        current = await run(adapter.get_person, pid, collection_id=cid)
        if current:
            await run(adapter.delete_person, pid, collection_id=cid)
        metadata = {
            **engine._item_metadata(item),
            "file_path": paths[0],
            "image_paths": paths,
        }
        await run(
            adapter.register_person,
            name=item.get("name") or target.get("name") or "",
            image_bytes=images[paths[0]],
            metadata=metadata,
            external_id=item.get("external_id"),
            person_id=pid,
            collection_id=cid,
        )
        for path in paths[1:]:
            await run(adapter.add_person_image, pid, images[path], collection_id=cid)
        updated = await run(adapter.get_person, pid, collection_id=cid)
        face_count = updated.get("face_count") if updated else None
        if (
            not updated
            or gallery(updated) != paths
            or (face_count is not None and int(face_count) != len(paths))
        ):
            raise RuntimeError("删除照片后的图库校验失败")
        return updated

    touched = []
    try:
        await run(save_journal, "prepared")
        aggregate_record = None
        for cid, pid, item in snapshots:
            touched.append((cid, pid, item))
            updated = await rebuild(cid, pid, item, retained)
            if cid == aggregate and pid == target_id:
                aggregate_record = updated
        await run(save_journal, "completed")
        return {
            "record": aggregate_record,
            "removed_images": len(removed),
        }
    except Exception as exc:
        failures = []
        for cid, pid, item in reversed(touched):
            try:
                await rebuild(cid, pid, item, gallery(item))
            except Exception:
                failures.append(pid)
                logger.exception("Cannot restore person %s (operation %s)", pid, operation_id)
        await run(save_journal, "recovery_required" if failures else "rolled_back")
        if failures:
            raise RuntimeError(
                f"照片删除失败且部分恢复未完成，请联系管理员，操作编号：{operation_id}"
            ) from exc
        raise
