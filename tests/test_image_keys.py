import copy
import hashlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from api.face_records import _image_url_to_path, _item_image_urls
from wcm_facerec import face_replication as replication
from wcm_facerec import face_sync_store as store
from wcm_facerec import image_store
from wcm_facerec import person_operations as ops
from wcm_facerec.config import settings
from wcm_facerec.face_engine import FaceEngine
from wcm_facerec.image_key_migration import migrate_person_image_keys


@pytest.fixture
def s3(monkeypatch):
    monkeypatch.setattr(settings, "image_storage", "s3")
    monkeypatch.setattr(settings, "s3_prefix", "wcm/images")
    monkeypatch.setattr(settings, "s3_bucket", "wcm-test")


def test_key_is_used_once_and_legacy_path_resolves_to_same_object(s3, monkeypatch):
    client = Mock()
    monkeypatch.setattr(image_store, "client", lambda: client)
    for ref in ("wcm/images/people/demo.jpg", "/tmp/wcm/people/demo.jpg"):
        image_store.stat(ref)
        client.head_object.assert_called_with(Bucket="wcm-test", Key="wcm/images/people/demo.jpg")
    assert image_store.with_images(
        {"file_path": "/tmp/wcm/people/demo.jpg", "custom": 1}, ["/tmp/wcm/people/demo.jpg"]
    ) == {
        "image_key": "wcm/images/people/demo.jpg",
        "image_keys": ["wcm/images/people/demo.jpg"],
        "custom": 1,
    }


@pytest.mark.parametrize(
    "key",
    [
        "wcm/images/../private",
        "wcm/images/.hidden/a",
        "wcm/images//a",
        "wcm/images/a\\b",
        "other/bucket/a",
        "wcm/images/",
        "/wcm/images/a",
    ],
)
def test_keys_cannot_escape_image_namespace(s3, key):
    with pytest.raises(ValueError):
        image_store.validate_key(key)


def test_new_metadata_is_authoritative_and_urls_round_trip_unicode_reserved_characters(s3):
    key = "wcm/images/人物/照片 #%?.jpg"
    item = {"image_key": key, "image_keys": [key], "file_path": "/tmp/wcm/stale.jpg"}
    urls = _item_image_urls(item)
    assert urls == ["/images/%E4%BA%BA%E7%89%A9/%E7%85%A7%E7%89%87%20%23%25%3F.jpg"]
    assert _image_url_to_path(urls[0]) == key
    assert _image_url_to_path("/images/%2e%2e/private") is None
    assert image_store.image_refs({"metadata": item}) == [key]


@pytest.mark.asyncio
async def test_old_checkpoint_allows_key_migration_without_reenrollment(s3, monkeypatch):
    old_images = [{"path": "/tmp/wcm/demo.jpg", "sha256": "sha"}]
    person = {
        "id": "p",
        "name": "n",
        "external_id": None,
        "face_count": 1,
        "metadata": {"image_key": "wcm/images/demo.jpg", "image_keys": ["wcm/images/demo.jpg"]},
    }
    old = {**person, "metadata": {"file_path": "/tmp/wcm/demo.jpg"}}
    snapshot = {
        "collection": "all",
        "person_id": "p",
        "person": person,
        "images": image_store.canonical_images(old_images),
    }
    actual = copy.deepcopy(old)

    def update(*args, **kwargs):
        actual.update(kwargs)

    target = SimpleNamespace(
        get_person=lambda *a, **k: actual.copy(),
        _client=SimpleNamespace(update_person=Mock(side_effect=update)),
    )
    monkeypatch.setattr(
        store,
        "applied",
        lambda *args: {"manifest_hash": "old", "images_hash": store.digest(old_images)},
    )
    monkeypatch.setattr(store, "record_applied", Mock())
    monkeypatch.setattr(
        image_store, "read_bytes", Mock(side_effect=AssertionError("must not reenroll"))
    )
    await replication.apply_snapshot("a", "owner", target, snapshot)
    assert actual == person
    target._client.update_person.assert_called_once()


@pytest.mark.asyncio
async def test_native_only_migration_and_compensation_preserve_faces(s3, monkeypatch):
    monkeypatch.setattr(settings, "insightface_replication_enabled", True)
    original = {
        "id": "p",
        "name": "n",
        "external_id": None,
        "face_count": 2,
        "metadata": {"file_path": "/tmp/wcm/a.jpg", "custom": 3},
    }
    actual = copy.deepcopy(original)
    faces = [{"id": "face-1"}, {"id": "face-2"}]

    class Adapter:
        _client = SimpleNamespace(
            list_faces=lambda *a, **k: SimpleNamespace(faces=faces, next_cursor=None)
        )

        def get_person(self, *a, **k):
            return copy.deepcopy(actual)

        def update_person(self, person_id, *, metadata, collection_id=None):
            actual["metadata"] = copy.deepcopy(metadata)
            return self.get_person(person_id)

    engine = FaceEngine.__new__(FaceEngine)
    engine._adapter = Adapter()
    data = {"id": "test", "operation": "migrate_person_image_keys", "before": {}}
    monkeypatch.setattr(ops, "_save", lambda *args: None)
    monkeypatch.setattr(image_store, "stat", lambda *a: {"Metadata": {"sha256": "sha"}})
    monkeypatch.setattr(
        image_store,
        "read_bytes",
        Mock(side_effect=AssertionError("must not rebuild native-only record")),
    )
    token = ops._operation.set(data)
    try:
        await migrate_person_image_keys.__wrapped__(
            engine, "p", "all", store.digest(original["metadata"])
        )
        assert actual["metadata"] == {
            "image_key": "wcm/images/a.jpg",
            "image_keys": ["wcm/images/a.jpg"],
            "custom": 3,
        }
        snapshots = await replication.capture(engine, data["before"].values())
        assert snapshots[0]["rebuildable"] is False
        assert snapshots[0]["images"] == [{"key": "wcm/images/a.jpg", "sha256": "sha"}]
        await ops._restore(engine, data)
        assert actual == original
        assert data["status"] == "rolled_back"
        assert len(faces) == 2
    finally:
        ops._operation.reset(token)


@pytest.mark.asyncio
async def test_migration_rejects_changed_metadata_before_writing(s3, monkeypatch):
    monkeypatch.setattr(settings, "insightface_replication_enabled", True)
    adapter = SimpleNamespace(
        get_person=lambda *a, **k: {"metadata": {"custom": "edited"}}, update_person=Mock()
    )

    async def run(func, *args, **kwargs):
        return func(*args, **kwargs)

    engine = SimpleNamespace(_primary_adapter=adapter, _run=run)
    with pytest.raises(store.ReplicationUnavailable, match="已改变"):
        await migrate_person_image_keys.__wrapped__(engine, "p", "all", "stale")
    adapter.update_person.assert_not_called()
