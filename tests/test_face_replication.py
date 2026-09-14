import asyncio
import copy
import hashlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from wcm_facerec import face_replication as replication
from wcm_facerec import face_sync_store as store
from wcm_facerec.config import settings


@pytest.fixture
def snapshot():
    return {
        "collection": "people",
        "person_id": "person-1",
        "person": {
            "id": "person-1",
            "name": "new",
            "external_id": None,
            "metadata": {"file_path": "/tmp/wcm/photo.jpg"},
            "face_count": 1,
        },
        "images": [{"path": "/tmp/wcm/photo.jpg", "sha256": hashlib.sha256(b"photo").hexdigest()}],
    }


class Target:
    def __init__(self, person):
        self.person = copy.deepcopy(person)
        self.mutations = []
        self._client = self
        self.fail_after_create = False

    def get_person(self, pid, collection_id=None):
        return copy.deepcopy(self.person)

    def delete_person(self, pid, collection_id=None):
        self.mutations.append("delete")
        self.person = None

    def create_person(self, cid, **kwargs):
        self.mutations.append("create")
        self.person = {
            "id": kwargs["person_id"],
            "name": kwargs["name"],
            "external_id": kwargs["external_id"],
            "metadata": kwargs["metadata"],
            "face_count": len(kwargs["images"]),
        }
        if self.fail_after_create:
            raise TimeoutError("Response lost after commit")
        return SimpleNamespace(faces=[{"id": "replica-local-id"}], rejected_images=[])

    def update_person(self, cid, pid, **kwargs):
        self.mutations.append("metadata")
        self.person.update(kwargs)


@pytest.fixture
def mocked_store(monkeypatch):
    saved = Mock()
    quarantined = Mock()
    monkeypatch.setattr(store, "applied", lambda *args: None)
    monkeypatch.setattr(store, "record_applied", saved)
    monkeypatch.setattr(store, "quarantine", quarantined)
    monkeypatch.setattr(replication.image_store, "read_bytes", lambda *args: b"photo")
    return saved, quarantined


@pytest.mark.asyncio
async def test_snapshot_rebuild_then_verified_checkpoint(snapshot, mocked_store):
    target = Target({**snapshot["person"], "name": "old"})
    await replication.apply_snapshot("replica", "owner", target, snapshot)
    assert target.person == snapshot["person"]
    assert target.mutations == ["delete", "create"]
    mocked_store[0].assert_called_once()


@pytest.mark.asyncio
async def test_lost_write_response_is_quarantined_and_never_acknowledged(snapshot, mocked_store):
    target = Target(None)
    target.fail_after_create = True
    with pytest.raises(replication.AmbiguousReplicaWrite):
        await replication.apply_snapshot("replica", "owner", target, snapshot)
    assert target.person == snapshot["person"]  # It actually committed remotely.
    mocked_store[0].assert_not_called()
    mocked_store[1].assert_called_once()
    # After externally fencing old requests, full-state replay leaves one face.
    target.fail_after_create = False
    await replication.apply_snapshot("replica", "new-owner", target, snapshot)
    assert target.person["face_count"] == 1
    assert target.mutations == ["create", "delete", "create"]


@pytest.mark.asyncio
async def test_image_integrity_checked_before_delete(snapshot, mocked_store, monkeypatch):
    target = Target(snapshot["person"])
    monkeypatch.setattr(replication.image_store, "read_bytes", lambda *args: b"corrupted")
    with pytest.raises(store.ReplicationUnavailable, match="SHA-256"):
        await replication.apply_snapshot("replica", "owner", target, snapshot)
    assert target.mutations == []
    mocked_store[1].assert_not_called()


@pytest.mark.asyncio
async def test_metadata_only_update_does_not_reenroll(snapshot, mocked_store, monkeypatch):
    target = Target({**snapshot["person"], "name": "old"})
    monkeypatch.setattr(
        store,
        "applied",
        lambda *args: {"manifest_hash": "old", "images_hash": store.digest(snapshot["images"])},
    )
    images = Mock(side_effect=AssertionError("metadata update must not download images"))
    monkeypatch.setattr(replication.image_store, "read_bytes", images)
    await replication.apply_snapshot("replica", "owner", target, snapshot)
    assert target.mutations == ["metadata"]
    images.assert_not_called()


@pytest.mark.asyncio
async def test_completed_person_is_not_rewritten_on_batch_retry(
    snapshot, mocked_store, monkeypatch
):
    target = Target(snapshot["person"])
    monkeypatch.setattr(
        store,
        "applied",
        lambda *args: {
            "manifest_hash": store.digest(snapshot),
            "images_hash": store.digest(snapshot["images"]),
        },
    )
    await replication.apply_snapshot("replica", "owner", target, snapshot)
    assert not target.mutations


@pytest.mark.asyncio
async def test_tombstone_deletes_only_named_person(snapshot, mocked_store):
    target = Target(snapshot["person"])
    snapshot.update(person=None, images=[])
    await replication.apply_snapshot("replica", "owner", target, snapshot)
    assert target.person is None
    await replication.apply_snapshot("replica", "owner", target, snapshot)
    assert target.mutations == ["delete"]


def test_manifest_rejects_missing_originals():
    with pytest.raises(store.ReplicationUnavailable, match="原照片数"):
        replication.manifest("people", "p", {"face_count": 2, "file_path": "/tmp/wcm/one.jpg"})


def test_legacy_baseline_requires_identical_native_faces(monkeypatch):
    from scripts.face_replication import baseline_manifest

    monkeypatch.setattr(replication.image_store, "stat", lambda p: {"Metadata": {"sha256": "sha"}})
    item = {"id": "p", "face_count": 2, "file_path": "/tmp/wcm/one.jpg"}
    faces = [{"id": "native-1", "bbox": [1, 2, 3, 4]}, {"id": "native-2"}]
    client = SimpleNamespace(
        list_faces=Mock(
            side_effect=[
                SimpleNamespace(faces=faces[:1], next_cursor="page-2"),
                SimpleNamespace(faces=faces[1:], next_cursor=None),
            ]
        )
    )
    target = SimpleNamespace(_client=client)
    before = baseline_manifest(target, "people", item, {})
    assert before["rebuildable"] is False
    assert before["person"]["face_count"] == 2
    assert len(before["images"]) == 1
    client.list_faces = Mock(
        return_value=SimpleNamespace(faces=list(reversed(faces)), next_cursor=None)
    )
    assert baseline_manifest(target, "people", item, {}) == before
    client.list_faces.return_value.faces[0] = {"id": "different-native-face"}
    assert baseline_manifest(target, "people", item, {}) != before
    client.list_faces.return_value.faces = faces[:1]
    with pytest.raises(store.ReplicationUnavailable, match="清单不完整"):
        baseline_manifest(target, "people", item, {})


@pytest.mark.asyncio
async def test_legacy_snapshot_never_causes_lossy_rebuild(snapshot, mocked_store):
    target = Target(snapshot["person"])
    snapshot["rebuildable"] = False
    with pytest.raises(store.ReplicationUnavailable, match="原生备份"):
        await replication.apply_snapshot("replica", "owner", target, snapshot)
    assert target.mutations == []
    mocked_store[0].assert_not_called()


@pytest.mark.asyncio
async def test_parallel_searches_pin_their_own_instance_across_awaits(monkeypatch):
    monkeypatch.setattr(settings, "insightface_replication_enabled", True)
    selections = iter(
        [{"id": "a", "url": "http://a", "lease": "a"}, {"id": "b", "url": "http://b", "lease": "b"}]
    )
    monkeypatch.setattr(store, "acquire_read", lambda: next(selections))
    monkeypatch.setattr(store, "release_read", Mock())
    monkeypatch.setattr(replication, "node_adapter", lambda url: url)

    @replication.replica_read
    async def search():
        first = replication.current_adapter.get()
        await asyncio.sleep(0.01)
        assert replication.current_adapter.get() == first
        return first

    assert set(await asyncio.gather(search(), search())) == {"http://a", "http://b"}
    assert replication.current_adapter.get() is None


@pytest.mark.asyncio
async def test_sync_crash_leaves_replica_quarantined(monkeypatch):
    from api.face_sync_worker import sync_once

    monkeypatch.setattr(
        store, "node", lambda name: ({"head": 2}, {"state": "syncing", "applied_seq": 1})
    )
    quarantined = Mock()
    monkeypatch.setattr(store, "quarantine", quarantined)
    target = Mock()
    await sync_once("a", target)
    quarantined.assert_called_once()
    target.health.assert_not_called()


@pytest.mark.asyncio
async def test_primary_lost_response_blocks_compensation_and_next_write(monkeypatch):
    from wcm_facerec import person_operations as ops
    from wcm_facerec.face_engine import FaceEngine

    monkeypatch.setattr(settings, "insightface_replication_enabled", True)
    monkeypatch.setattr(store, "before_write", lambda: None)
    persisted = {}

    def save(data):
        data["version"] += 1
        persisted[data["id"]] = copy.deepcopy(data)

    monkeypatch.setattr(ops, "_save", save)
    monkeypatch.setattr(
        ops,
        "_load",
        lambda operation_id=None: [
            copy.deepcopy(v) for v in persisted.values() if v["status"] in {"running", "uncertain"}
        ],
    )
    engine = FaceEngine.__new__(FaceEngine)

    class Primary:
        def get_person(self, pid, collection_id=None):
            return None

        def register_person(self, *, person_id=None, **kwargs):
            raise TimeoutError("remote committed, response lost")

    engine._adapter = Primary()
    restored = AsyncMock()
    monkeypatch.setattr(ops, "restore", restored)

    async def write(engine):
        await ops.call(engine._adapter.register_person, person_id="p")

    with pytest.raises(store.ReplicationUnavailable, match="结果不确定"):
        await ops.run(engine, write)
    restored.assert_not_awaited()
    assert next(iter(persisted.values()))["status"] == "uncertain"
    with pytest.raises(store.ReplicationUnavailable, match="结果不确定"):
        await ops.run(engine, write)


@pytest.mark.asyncio
async def test_interrupted_compensation_preserves_uncertain_marker(monkeypatch):
    from wcm_facerec import person_operations as ops

    monkeypatch.setattr(settings, "insightface_replication_enabled", True)
    states = []
    monkeypatch.setattr(ops, "_save", lambda data: states.append(copy.deepcopy(data)))
    data = {"status": "recovering", "version": 1}

    def delete_person():
        raise TimeoutError("response lost during compensation")

    with pytest.raises(TimeoutError):
        await ops._restore_call(data, delete_person)
    assert states[0]["inflight"]["recovery"] is True
    assert states[-1]["status"] == "uncertain"
    assert data["inflight"] and data["uncertain"]


def test_expired_lease_stops_composed_sdk_before_its_next_http_call():
    from wcm_facerec.ifs_adapter import InsightFaceAdapter

    target = InsightFaceAdapter("http://example.invalid", "people")
    token = store.read_guard.set({"lost": True})
    try:
        with pytest.raises(store.ReplicationUnavailable, match="读取租约"):
            target.health()
    finally:
        store.read_guard.reset(token)


@pytest.mark.asyncio
async def test_multipart_idempotency_ignores_boundary_but_detects_changed_intent():
    import httpx
    from starlette.requests import Request

    from api.request_identity import person_request_fingerprint

    async def fingerprint(photo, name):
        upload = httpx.Request(
            "POST",
            "http://test/api/v1/face_records",
            data={"name": name},
            files={"file": ("face.jpg", photo, "image/jpeg")},
        )
        body = upload.read()

        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        request = Request(
            {
                "type": "http",
                "method": "POST",
                "scheme": "http",
                "path": "/api/v1/face_records",
                "query_string": b"",
                "headers": [(key.lower(), value) for key, value in upload.headers.raw],
                "server": ("test", 80),
            },
            receive,
        )
        result = await person_request_fingerprint(request)
        assert await request.body() == body
        return result, upload.headers["content-type"]

    first, boundary_a = await fingerprint(b"photo", "name")
    same, boundary_b = await fingerprint(b"photo", "name")
    assert boundary_a != boundary_b and first == same
    assert (await fingerprint(b"different", "name"))[0] != first
    assert (await fingerprint(b"photo", "different"))[0] != first
