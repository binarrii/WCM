import asyncio
import io
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from botocore.response import StreamingBody
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import media_objects as objects
from api import review_media as media
from api import review_task_store as store
from api.auth_guard import required_permission
from api.review_tasks import review_tasks_bp
from wcm_facerec.config import settings
from wcm_facerec.execution import execution_scope


@pytest.fixture
def playback(monkeypatch):
    body = b"0123456789"
    value = {
        "id": "a" * 32,
        "task_id": "task",
        "storage": "s3",
        "object_key": "wcm/review-media/a.mp4",
        "sha256": "hash",
        "size_bytes": len(body),
        "expires_at": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
    }
    monkeypatch.setattr(
        media,
        "_get",
        lambda task_id, media_id: value if (task_id, media_id) == ("task", "a" * 32) else None,
    )
    client = MagicMock()
    streams = []

    def get(**kwargs):
        data = body
        if "Range" in kwargs:
            start, end = media.parse_range(kwargs["Range"], len(body))
            data = data[start : end + 1]
        stream = io.BytesIO(data)
        streams.append(stream)
        return {"Body": StreamingBody(stream, len(data))}

    client.get_object.side_effect = get
    monkeypatch.setattr(media.image_store, "client", lambda: client)
    app = FastAPI()
    app.include_router(review_tasks_bp, prefix="/api/v1")
    return TestClient(app), value, client, streams


def test_private_playback_range_head_conditional_and_closed_stream(playback):
    browser, value, s3, streams = playback
    url = f"/api/v1/review_tasks/task/media/{value['id']}"
    assert required_permission(url, "GET") == {"review.read"}
    response = browser.get(url, headers={"Range": "bytes=2-5"})
    assert response.status_code == 206 and response.content == b"2345"
    assert response.headers["content-range"] == "bytes 2-5/10"
    assert response.headers["content-type"] == "video/mp4"
    assert streams[-1].closed
    assert browser.get(url, headers={"Range": "bytes=-2"}).content == b"89"
    assert browser.get(url, headers={"Range": "bytes=10-"}).status_code == 416
    assert browser.get(url, headers={"Range": "bytes=0-1,4-5"}).status_code == 416
    count = s3.get_object.call_count
    assert browser.head(url).headers["content-length"] == "10"
    assert browser.get(url, headers={"If-None-Match": '"hash"'}).status_code == 304
    assert s3.get_object.call_count == count
    assert (
        browser.get(url, headers={"Range": "bytes=2-5", "If-Range": '"old"'}).content
        == b"0123456789"
    )
    assert browser.get(url.replace("/task/", "/other/")).status_code == 404
    value["expires_at"] = "2000-01-01T00:00:00+00:00"
    assert browser.get(url).status_code == 410


def test_media_publish_is_fenced(monkeypatch):
    connection, cursor = MagicMock(), MagicMock()
    connection.__enter__.return_value = connection
    connection.cursor.return_value.__enter__.return_value = cursor
    monkeypatch.setattr(store, "_connect", lambda: connection)
    monkeypatch.setattr(settings, "cluster_enabled", True)
    with execution_scope("task", "token", 2):
        media._publish("task", {"id": "asset"})
    sql, args = cursor.execute.call_args.args
    assert "lease_token = %s" in sql and "lease_expires > UTC_TIMESTAMP" in sql
    assert "status = 'processing'" in sql and args[-1] == "token"
    media._publish("task", {"id": "stale"})
    assert "AND 1 = 0" in cursor.execute.call_args.args[0]


def test_permanent_media_failure_skips_automatic_retries(monkeypatch):
    connection, cursor = MagicMock(), MagicMock()
    connection.__enter__.return_value = connection
    connection.cursor.return_value.__enter__.return_value = cursor
    monkeypatch.setattr(store, "_connect", lambda: connection)
    monkeypatch.setattr(settings, "cluster_enabled", True)
    with execution_scope("task", "token", 1):
        store._fail_sync("task", "video limit", False)
    args = cursor.execute.call_args.args[1]
    assert args[0] == args[2] == 0


def test_cancelled_multipart_upload_is_aborted(monkeypatch, tmp_path):
    client = MagicMock()
    client.create_multipart_upload.return_value = {"UploadId": "upload"}
    client.upload_part.side_effect = RuntimeError("cancelled")
    monkeypatch.setattr(media.image_store, "client", lambda: client)
    path = tmp_path / "v.mp4"
    path.write_bytes(b"video")
    with pytest.raises(RuntimeError):
        media._upload(path, {"id": "a", "storage": "s3", "object_key": "wcm/review-media/a.mp4"})
    client.abort_multipart_upload.assert_called_once()
    client.complete_multipart_upload.assert_not_called()


@pytest.mark.asyncio
async def test_stale_execution_cannot_publish_uploaded_media(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "is_enabled", lambda: True)
    monkeypatch.setattr(media, "_register", MagicMock())
    monkeypatch.setattr(media, "_upload", MagicMock())
    monkeypatch.setattr(media, "_publish", lambda *args: False)
    monkeypatch.setattr(media.review_events, "publish", AsyncMock())
    path = tmp_path / "video.mp4"
    path.write_bytes(b"video")

    @contextmanager
    def locked(*args):
        yield None, MagicMock()

    monkeypatch.setattr(objects, "locked", locked)
    monkeypatch.setattr(objects, "reusable", lambda *args: {"id": "object", "object_key": "key"})
    with pytest.raises(RuntimeError, match="租约"):
        await media.archive("task", path, {})
    media.review_events.publish.assert_not_awaited()


@pytest.mark.asyncio
async def test_concurrent_identical_media_share_one_upload_with_independent_references(
    monkeypatch, tmp_path
):
    path = tmp_path / "source.mp4"
    path.write_bytes(b"identical video")
    lock, catalog, references, published, uploads = threading.Lock(), {}, {}, {}, []

    @contextmanager
    def locked(*args):
        with lock:
            yield None, MagicMock()

    def reusable(cursor, scope, digest, size):
        return catalog.get(digest)

    def create(cursor, scope, digest, storage, key, size):
        item = {"id": digest[:32], "object_key": key}
        catalog[digest] = item
        return item

    def publish(task_id, value, cursor):
        published[task_id] = dict(value)
        return True

    monkeypatch.setattr(store, "is_enabled", lambda: True)
    monkeypatch.setattr(objects, "locked", locked)
    monkeypatch.setattr(objects, "reusable", reusable)
    monkeypatch.setattr(objects, "create", create)
    monkeypatch.setattr(
        media, "_register", lambda task_id, value, cursor: references.update({task_id: dict(value)})
    )
    monkeypatch.setattr(media, "_publish", publish)
    monkeypatch.setattr(media, "_upload", lambda path, value: uploads.append(value["object_key"]))
    monkeypatch.setattr(media.review_events, "publish", AsyncMock())
    await asyncio.gather(media.archive("one", path, {}), media.archive("two", path, {}))
    assert len(uploads) == 1
    assert published["one"]["id"] != published["two"]["id"]
    assert published["one"]["object_id"] == published["two"]["object_id"]
    assert published["one"]["object_key"] == published["two"]["object_key"]
    path.write_bytes(b"different video at the same URL")
    await media.archive("three", path, {})
    assert len(uploads) == 2
    assert published["three"]["object_key"] != published["one"]["object_key"]


def test_object_gc_rechecks_references_and_retires_before_storage_delete(monkeypatch):
    item = {
        "id": "asset",
        "storage": "s3",
        "storage_scope": "scope",
        "sha256": "hash",
        "object_key": "key",
    }
    cursor = MagicMock()

    @contextmanager
    def locked(*args):
        yield None, cursor

    monkeypatch.setattr(objects, "locked", locked)
    monkeypatch.setattr(objects, "storage_scope", lambda storage: "scope")
    delete = MagicMock()
    monkeypatch.setattr(objects, "delete_file", delete)
    cursor.fetchone.side_effect = [item, {"id": "live-reference"}]
    objects.delete_unreferenced(item)
    delete.assert_not_called()
    cursor.fetchone.side_effect = [item, None]

    def fail_after_retire(value):
        assert "state='deleting'" in cursor.execute.call_args.args[0]
        raise RuntimeError("storage timeout")

    delete.side_effect = fail_after_retire
    with pytest.raises(RuntimeError):
        objects.delete_unreferenced(item)
    assert not any(
        "DELETE FROM review_media_objects" in call.args[0] for call in cursor.execute.call_args_list
    )


def test_shared_local_playback_uses_object_key_not_task_media_id(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "review_media_dir", str(tmp_path))
    value = {"id": "b" * 32, "object_key": "wcm/review-media/" + "a" * 32 + ".mp4"}
    assert objects.local_path(value) == tmp_path / ("a" * 32 + ".mp4")
    with pytest.raises(ValueError):
        objects.local_path({"object_key": "../../secret"})
