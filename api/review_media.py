"""Private, immutable review videos shared by API/Worker replicas."""

import logging
import re
import shutil
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from botocore.exceptions import ClientError
from fastapi import HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from starlette.background import BackgroundTask

from wcm_facerec import image_store
from wcm_facerec.cluster import cluster_slot, run_sync
from wcm_facerec.config import settings
from wcm_facerec.model_budget import model_request_budget, remaining_request_time

from . import media_objects
from . import review_task_store as store
from .review_events import review_events

logger = logging.getLogger(__name__)


def initialize_sync(cursor):
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS review_media (
            id CHAR(32) PRIMARY KEY,
            task_id CHAR(36) NOT NULL,
            storage VARCHAR(10) NOT NULL,
            object_key VARCHAR(512) NOT NULL,
            expires_at DATETIME(3) NOT NULL,
            created_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
            INDEX idx_review_media_expiry (expires_at),
            INDEX idx_review_media_task (task_id)
        ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
    """)
    media_objects.initialize_sync(cursor)


def public_media(value):
    media = store._json_load(value)
    if not media:
        return None
    return {
        **{k: v for k, v in media.items() if k not in {"storage", "object_key", "object_id"}},
        "url": f"/api/v1/review_tasks/{media['task_id']}/media/{media['id']}",
    }


def _register(task_id, media, cursor=None):
    if cursor is None:
        with store._connect() as connection, connection.cursor() as cursor:
            return _register(task_id, media, cursor)
    cursor.execute(
        "INSERT INTO review_media (id, task_id, storage, object_key, expires_at, object_id) VALUES (%s, %s, %s, %s, %s, %s)",
        (
            media["id"],
            task_id,
            media["storage"],
            media["object_key"],
            datetime.fromisoformat(media["expires_at"]).replace(tzinfo=None),
            media.get("object_id"),
        ),
    )


def _publish(task_id, media, cursor=None):
    if cursor is None:
        with store._connect() as connection, connection.cursor() as cursor:
            return _publish(task_id, media, cursor)
    fence, ownership = store._ownership(task_id)
    return bool(
        cursor.execute(
            "UPDATE review_tasks SET media = %s WHERE id = %s AND status = 'processing'" + fence,
            (store._json_dump(media), task_id, *ownership),
        )
    )


def _upload(path, media):
    if "sha256" not in media:
        media["sha256"] = media_objects.fingerprint(path)
    if media["storage"] == "s3":
        client = image_store.client()
        parameters = {"Bucket": settings.s3_bucket, "Key": media["object_key"]}
        upload = client.create_multipart_upload(
            **parameters, ContentType="video/mp4", Metadata={"sha256": media["sha256"]}
        )
        completed = False
        try:
            parts = []
            with path.open("rb") as source:
                while chunk := source.read(16 * 1048576):
                    remaining_request_time()
                    number = len(parts) + 1
                    part = client.upload_part(
                        **parameters, UploadId=upload["UploadId"], PartNumber=number, Body=chunk
                    )
                    parts.append({"PartNumber": number, "ETag": part["ETag"]})
            client.complete_multipart_upload(
                **parameters, UploadId=upload["UploadId"], MultipartUpload={"Parts": parts}
            )
            completed = True
        finally:
            if not completed:
                # Only the abort gets a fresh budget after cancellation. No further
                # media writes are permitted by the cancelled execution.
                with model_request_budget(30):
                    client.abort_multipart_upload(**parameters, UploadId=upload["UploadId"])

    else:
        target = media_objects.local_path(media)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)


def _archive(task_id, path, metadata):
    digest = media_objects.fingerprint(path)
    size = path.stat().st_size
    scope = media_objects.storage_scope(settings.image_storage)
    media_id = uuid.uuid4().hex
    media = {
        **metadata,
        "id": media_id,
        "task_id": task_id,
        "storage": settings.image_storage,
        "sha256": digest,
        "size_bytes": size,
        "object_key": f"{settings.review_media_prefix.strip('/')}/{media_id}.mp4",
        "expires_at": (
            datetime.now(timezone.utc) + timedelta(days=settings.video_retention_days)
        ).isoformat(),
    }
    with media_objects.locked(scope, digest) as (_, cursor):
        obj = media_objects.reusable(cursor, scope, digest, size)
        reused = obj is not None
        if obj is None:
            obj = media_objects.create(
                cursor, scope, digest, media["storage"], media["object_key"], size
            )
        media.update(object_id=obj["id"], object_key=obj["object_key"])
        # Register before upload, including unsuccessful/cancelled attempts.
        _register(task_id, media, cursor)
        if not reused:
            _upload(path, media)
            remaining_request_time()
            cursor.execute(
                "UPDATE review_media_objects SET state='ready' WHERE id=%s", (obj["id"],)
            )
        remaining_request_time()
        if not _publish(task_id, media, cursor):
            raise RuntimeError("媒体保存时任务已取消或执行租约失效")
    logger.info("Review media archived: task=%s reused=%s sha256=%s", task_id, reused, digest)


async def archive(task_id, path, metadata):
    if not task_id or not store.is_enabled():
        return
    await store._run(_archive, task_id, Path(path), metadata)
    await review_events.publish({"type": "changed", "task_ids": [task_id], "reason": "media_ready"})


def _collectable():
    with store._connect() as connection, connection.cursor() as cursor:
        cursor.execute("""
            SELECT m.* FROM review_media m LEFT JOIN review_tasks t ON t.id = m.task_id
            WHERE (t.id IS NULL OR t.status NOT IN ('processing', 'cancelling')) AND
                (m.expires_at <= UTC_TIMESTAMP(3) OR t.id IS NULL OR
                 (m.created_at < TIMESTAMPADD(DAY, -1, UTC_TIMESTAMP(3)) AND
                  COALESCE(JSON_UNQUOTE(JSON_EXTRACT(t.media, '$.id')), '') <> m.id))
            ORDER BY m.created_at LIMIT 20
        """)
        return cursor.fetchall()


def _delete(media):
    # Migration can attach a formerly-legacy reference after the GC snapshot.
    with media_objects.locked("legacy", "") as (_, cursor):
        cursor.execute(
            """SELECT m.* FROM review_media m LEFT JOIN review_tasks t ON t.id=m.task_id
            WHERE m.id=%s AND (t.id IS NULL OR t.status NOT IN ('processing','cancelling'))
            AND (m.expires_at<=UTC_TIMESTAMP(3) OR t.id IS NULL OR
                (m.created_at<TIMESTAMPADD(DAY,-1,UTC_TIMESTAMP(3)) AND
                 COALESCE(JSON_UNQUOTE(JSON_EXTRACT(t.media,'$.id')),'')<>m.id))""",
            (media["id"],),
        )
        current = cursor.fetchone()
        if not current:
            return
        if not current["object_id"]:
            cursor.execute(
                "SELECT id FROM review_media_objects WHERE storage_scope=%s AND object_key=%s",
                (media_objects.storage_scope(current["storage"]), current["object_key"]),
            )
            if not cursor.fetchone():
                media_objects.delete_file(current)
        cursor.execute("DELETE FROM review_media WHERE id = %s", (current["id"],))


async def collect_expired():
    if not store.is_enabled():
        return
    async with cluster_slot("media-gc"):
        for legacy in await store._run(media_objects.legacy_candidates):
            try:
                await store._run(media_objects.migrate_legacy, legacy)
            except Exception:
                logger.warning("Review media migration will retry: id=%s", legacy["id"])
        for media in await store._run(_collectable):
            try:
                await run_sync(_delete, media)
            except Exception:
                logger.warning("Review media cleanup will retry: id=%s", media["id"])
        for obj in await store._run(media_objects.unreferenced):
            try:
                await store._run(media_objects.delete_unreferenced, obj)
            except Exception:
                logger.warning("Review media object cleanup will retry: id=%s", obj["id"])


def _get(task_id, media_id):
    with store._connect() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT media FROM review_tasks WHERE id = %s", (task_id,))
        row = cursor.fetchone()
        media = store._json_load(row["media"]) if row else None
        return media if media and media["id"] == media_id else None


def parse_range(value, size):
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", value or "")
    if not match or not any(match.groups()):
        raise ValueError()
    start, end = match.groups()
    if not start:
        length = int(end)
        if length <= 0:
            raise ValueError()
        return max(0, size - length), size - 1
    start, end = int(start), min(int(end), size - 1) if end else size - 1
    if not 0 <= start <= end < size:
        raise ValueError()
    return start, end


async def playback(task_id, media_id, request: Request):
    media = await store._run(_get, task_id, media_id)
    if not media:
        raise HTTPException(404, "复核视频不存在或不是此任务的当前版本")
    if datetime.fromisoformat(media["expires_at"]) <= datetime.now(timezone.utc):
        raise HTTPException(410, "复核视频已超过保存期限，请重新提交审核")
    etag = '"' + media["sha256"] + '"'
    headers = {"Accept-Ranges": "bytes", "ETag": etag, "Cache-Control": "private, no-store"}
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=headers)
    if media["storage"] == "local":
        path = media_objects.local_path(media)
        if not path.is_file():
            raise HTTPException(410, "复核视频已清理，请重新提交审核")
        return FileResponse(path, media_type="video/mp4", headers=headers)
    parameters = {"Bucket": settings.s3_bucket, "Key": media["object_key"]}
    size = media["size_bytes"]
    status = 200
    if (
        request.method != "HEAD"
        and request.headers.get("range")
        and request.headers.get("if-range", etag) == etag
    ):
        try:
            start, end = parse_range(request.headers["range"], size)
        except ValueError:
            return Response(
                status_code=416, headers={**headers, "Content-Range": f"bytes */{size}"}
            )
        parameters["Range"] = f"bytes={start}-{end}"
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        size = end - start + 1
        status = 206
    headers["Content-Length"] = str(size)
    if request.method == "HEAD":
        return Response(headers=headers, media_type="video/mp4")
    try:
        result = await run_sync(
            image_store.client().get_object, _on_cancel=lambda r: r["Body"].close(), **parameters
        )
    except ClientError as exc:
        code = str(exc.response["Error"]["Code"])
        raise HTTPException(
            410 if code in {"404", "NoSuchKey"} else 503, "复核视频已清理或存储暂不可用"
        ) from None

    def chunks():
        try:
            yield from result["Body"].iter_chunks(256 * 1024)
        finally:
            result["Body"].close()

    return StreamingResponse(
        chunks(),
        status_code=status,
        headers=headers,
        media_type="video/mp4",
        background=BackgroundTask(result["Body"].close),
    )
