"""Metadata-only image-key migration through the regular durable write journal."""

from . import face_sync_store as store
from . import image_store
from .config import settings
from .person_library import library_write


@library_write
async def migrate_person_image_keys(
    engine, person_id, collection_id, expected_metadata, restore_metadata=None
):
    if not settings.insightface_replication_enabled or settings.image_storage != "s3":
        raise store.ReplicationUnavailable("迁移需要已启用的可靠同步及 S3 存储")
    adapter = engine._primary_adapter
    before = await engine._run(adapter.get_person, person_id, collection_id=collection_id)
    if not before or store.digest(before["metadata"]) != expected_metadata:
        raise store.ReplicationUnavailable("人物元数据已改变，请重新生成迁移计划")
    refs = image_store.image_refs(before)
    metadata = (
        image_store.with_images(before["metadata"], refs)
        if restore_metadata is None
        else dict(restore_metadata)
    )
    if image_store.with_images(
        metadata, image_store.image_refs(metadata)
    ) != image_store.with_images(before["metadata"], refs):
        raise store.ReplicationUnavailable("元数据回滚只允许改变图片引用格式")
    if metadata == before["metadata"]:
        return {"changed": False}
    # A missing object must be discovered before the metadata write.
    for ref in refs:
        await engine._run(image_store.stat, ref)
    result = await engine._run(
        adapter.update_person, person_id, metadata=metadata, collection_id=collection_id
    )
    if result["metadata"] != metadata:
        raise store.ReplicationUnavailable("图片 Key 迁移后元数据校验失败")
    return {"changed": True, "collection": collection_id, "person_id": person_id}
