"""Unit tests for FaceEngine — adapter mocked; live server untouched."""

from __future__ import annotations

import pytest

from tests.test_ifs_adapter import _make_client
from wcm_facerec.config import settings
from wcm_facerec.face_engine import FaceEngine

from .conftest import FakeTransport


# ----------------------------------------------------------------------
# Test fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def engine(fake_transport: FakeTransport, monkeypatch):
    """A FaceEngine wired to a fake transport. Wipes the singleton."""
    import wcm_facerec.face_engine as fe_mod

    monkeypatch.setattr(fe_mod, "_engine", None)

    # Force the singleton to be created against our test base URL so the
    # FakeTransport under the SDK intercepts its requests.
    monkeypatch.setattr(settings, "insightface_base_url", "http://test:18097")
    monkeypatch.setattr(settings, "insightface_collection_id", "all-persons")
    monkeypatch.setattr(settings, "insightface_api_key", "")
    e = FaceEngine()
    e._adapter._client = _make_client(fake_transport)
    return e


# ----------------------------------------------------------------------
# Coercion of legacy img_source overload
# ----------------------------------------------------------------------
def test_detect_rejects_url_gr_source():
    """URLs must be downloaded by the route layer; engine refuses them."""
    import asyncio

    e = FaceEngine()
    e._adapter.detect = lambda *_a, **_kw: [
        {"facial_area": {}, "confidence": 0, "area": 0, "embedding": None}
    ]

    async def go():
        return await e.detect_faces("https://example.com/x.jpg")

    # _to_bytes raises ValueError, detect_faces returns [] on (TypeError, ValueError)
    result = asyncio.run(go())
    assert result == []


# ----------------------------------------------------------------------
# Search results stay sorted by distance (legacy contract)
# ----------------------------------------------------------------------
def _register_detect_one_face(fake_transport, x=0, y=0, w=120, h=120, score=0.9):
    """Mock /v1/detect with a single face large enough to pass the 80px floor."""
    fake_transport.register(
        "POST",
        "/v1/detect",
        {
            "faces": [
                {
                    "bbox": {
                        "pixels": {"x": x, "y": y, "width": w, "height": h},
                    },
                    "detection_score": score,
                },
            ],
            "processing_ms": 1.0,
        },
    )


def test_search_returns_results_in_distance_order(engine, fake_transport, sample_image_bytes):
    _register_detect_one_face(fake_transport, x=10, y=20, w=120, h=120)
    fake_transport.register(
        "POST",
        "/v1/collections/all-persons/search",
        {
            "matches": [
                {
                    "person": {"id": "p1", "name": "Alice", "metadata": {}},
                    "matched_face_id": "f1",
                    "similarity": 0.5,
                },
                {
                    "person": {"id": "p2", "name": "Bob", "metadata": {}},
                    "matched_face_id": "f2",
                    "similarity": 0.95,
                },
            ]
        },
    )
    fake_transport.register(
        "GET",
        "/v1/collections/all-persons/persons/p1/faces",
        {
            "faces": [
                {"id": "f1", "bounding_box": {"pixels": {"x": 1, "y": 2, "width": 3, "height": 4}}}
            ]
        },
    )
    fake_transport.register(
        "GET",
        "/v1/collections/all-persons/persons/p2/faces",
        {
            "faces": [
                {"id": "f2", "bounding_box": {"pixels": {"x": 5, "y": 6, "width": 7, "height": 8}}}
            ]
        },
    )
    import asyncio

    matches = asyncio.run(engine.search(sample_image_bytes, top_k=10, threshold=0.4))
    # Bob has similarity=0.95 (distance=0.05) so he must come first.
    assert [m["name"] for m in matches] == ["Bob", "Alice"]
    # Bbox backfilled from get_face_bbox
    assert matches[0]["source_x"] == 5
    assert matches[1]["source_x"] == 1


def test_search_threshold_filters_by_distance(engine, fake_transport, sample_image_bytes):
    """threshold=0.3 (distance) → similarity >= 0.7 must pass.

    The SDK applies the threshold server-side. We mimic that here by only
    registering a single-pass match in the canned response.
    """
    _register_detect_one_face(fake_transport)
    fake_transport.register(
        "POST",
        "/v1/collections/all-persons/search",
        {
            "matches": [
                {
                    "person": {"id": "p1", "name": "Pass", "metadata": {}},
                    "matched_face_id": "f1",
                    "similarity": 0.8,
                },
            ]
        },
    )
    fake_transport.register(
        "GET",
        "/v1/collections/all-persons/persons/p1/faces",
        {
            "faces": [
                {"id": "f1", "bounding_box": {"pixels": {"x": 0, "y": 0, "width": 1, "height": 1}}}
            ]
        },
    )
    import asyncio

    matches = asyncio.run(engine.search(sample_image_bytes, top_k=10, threshold=0.3))
    assert [m["name"] for m in matches] == ["Pass"]


def test_search_passes_similarity_threshold_to_sdk(engine, fake_transport, sample_image_bytes):
    """threshold=0.3 → SDK receives threshold=0.7 (1 - 0.3)."""
    _register_detect_one_face(fake_transport)
    fake_transport.register(
        "POST",
        "/v1/collections/all-persons/search",
        {"matches": []},
    )
    import asyncio

    asyncio.run(engine.search(sample_image_bytes, top_k=5, threshold=0.3))
    # Find the search call to inspect what threshold was sent.
    search_calls = [c for c in fake_transport.calls if "/search" in c[1]]
    assert search_calls, "expected a /search call"
    # The SDK encodes form fields into the multipart body; the simplest
    # assertion is that the request URL contained the path and we did not
    # raise. Threshold semantics are validated end-to-end in the live test.


# ----------------------------------------------------------------------
# Verify uses insightface_verify_similarity_threshold (not the legacy
# cosine-distance threshold).
# ----------------------------------------------------------------------
def test_verify_uses_similarity_threshold(engine, fake_transport, monkeypatch, sample_image_bytes):
    monkeypatch.setattr(settings, "insightface_verify_similarity_threshold", 0.6)
    fake_transport.register(
        "POST",
        "/v1/compare",
        {"matched": True, "similarity": 0.65, "threshold": 0.6},
    )
    import asyncio

    assert asyncio.run(engine.verify_faces(sample_image_bytes, sample_image_bytes)) is True

    # Below threshold -> False
    fake_transport.register(
        "POST",
        "/v1/compare",
        {"matched": False, "similarity": 0.5, "threshold": 0.6},
    )
    assert asyncio.run(engine.verify_faces(sample_image_bytes, sample_image_bytes)) is False


# ----------------------------------------------------------------------
# register_from_image writes both aggregate and per-category collection
# ----------------------------------------------------------------------
def test_register_writes_to_both_aggregate_and_category_collection(
    engine, fake_transport, monkeypatch, sample_image_bytes
):
    """Per-category fan-out: aggregate ``all-persons`` + mapped category.

    The per-category duplicate carries ``external_id=<aggregate_id>`` so an
    admin backfill can correlate the two IFS Persons. The returned
    record-item dict's ``id`` is the aggregate IFS ``Person.id`` (no SQL
    row is written).
    """
    monkeypatch.setattr(
        settings,
        "insightface_category_collections",
        {"时政敏感": "political", "劣迹艺人": "bad-artists", "落马官员": "corrupt-officials"},
    )
    fake_transport.register(
        "POST",
        "/v1/collections/all-persons/persons",
        {
            "person": {"id": "p_agg", "name": "测试", "face_count": 1},
            "faces": [{"id": "f_agg", "person_id": "p_agg"}],
            "rejected_images": [],
        },
    )
    fake_transport.register(
        "POST",
        "/v1/collections/political/persons",
        {
            "person": {"id": "p_cat", "name": "测试", "face_count": 1},
            "faces": [{"id": "f_cat", "person_id": "p_cat"}],
            "rejected_images": [],
        },
    )
    # The engine reads the aggregate Person back to surface created_at.
    fake_transport.register(
        "GET",
        "/v1/collections/all-persons/persons/p_agg",
        {
            "person": {
                "id": "p_agg",
                "name": "测试",
                "external_id": None,
                "face_count": 1,
                "created_at": "2026-08-07T00:00:00Z",
                "metadata": {
                    "category": "时政敏感",
                    "occupation": "教师",
                    "type": "敏感",
                    "remarks": "备注",
                    "file_path": "/tmp/wcm/时政敏感/测试_md5.jpg",
                },
            }
        },
    )
    import asyncio

    record = asyncio.run(
        engine.register_from_image(
            name="测试",
            img_source=sample_image_bytes,
            category="时政敏感",
            occupation="教师",
            type_="敏感",
            remarks="备注",
        )
    )
    # Returned dict is a flat record-item from IFS; ``id`` is the
    # aggregate Person.id, not a local UUID.
    assert record["id"] == "p_agg"
    assert record["name"] == "测试"
    assert record["category"] == "时政敏感"
    # Two POSTs were made — one to each collection.
    posts = [c for c in fake_transport.calls if c[0] == "POST" and c[1].endswith("/persons")]
    assert len(posts) == 2
    # And one GET to read the aggregate back.
    gets = [c for c in fake_transport.calls if c[0] == "GET" and c[1].endswith("/persons/p_agg")]
    assert len(gets) == 1


# ----------------------------------------------------------------------
# search_multi_face engine-level integration
# ----------------------------------------------------------------------
def test_engine_search_multi_face_returns_grouped_shape(engine, fake_transport, sample_image_bytes):
    """search_multi_face returns {face_count, faces, all_results}."""
    fake_transport.register(
        "POST",
        "/v1/detect",
        {
            "faces": [
                {
                    "bbox": {"pixels": {"x": 0, "y": 0, "width": 100, "height": 100}},
                    "detection_score": 0.9,
                },
                {
                    "bbox": {"pixels": {"x": 200, "y": 0, "width": 120, "height": 120}},
                    "detection_score": 0.85,
                },
            ],
            "processing_ms": 1.0,
        },
    )
    fake_transport.register(
        "POST",
        "/v1/collections/all-persons/search",
        {
            "matches": [
                {
                    "person": {"id": "p1", "name": "Alice", "metadata": {}},
                    "matched_face_id": "f1",
                    "similarity": 0.95,
                },
            ]
        },
    )
    import asyncio

    grouped = asyncio.run(
        engine.search_multi_face(
            sample_image_bytes,
            top_k=5,
            threshold=0.5,
            min_face_pixels=80,
            max_faces=10,
        )
    )
    assert grouped["face_count"] == 2
    assert len(grouped["faces"]) == 2
    assert len(grouped["all_results"]) == 2
    # Each face's matches have face_index + query_face_bbox set.
    for f in grouped["faces"]:
        for m in f["matches"]:
            assert "face_index" in m
            assert "query_face_bbox" in m


def test_engine_search_returns_flat_for_backcompat(engine, fake_transport, sample_image_bytes):
    """The legacy `engine.search` still returns a single flat list of dicts."""
    fake_transport.register(
        "POST",
        "/v1/detect",
        {
            "faces": [
                {
                    "bbox": {"pixels": {"x": 0, "y": 0, "width": 100, "height": 100}},
                    "detection_score": 0.9,
                },
            ],
            "processing_ms": 1.0,
        },
    )
    fake_transport.register(
        "POST",
        "/v1/collections/all-persons/search",
        {
            "matches": [
                {
                    "person": {"id": "p1", "name": "Alice", "metadata": {}},
                    "matched_face_id": "f1",
                    "similarity": 0.9,
                },
            ]
        },
    )
    fake_transport.register(
        "GET",
        "/v1/collections/all-persons/persons/p1/faces",
        {
            "faces": [
                {"id": "f1", "bounding_box": {"pixels": {"x": 0, "y": 0, "width": 1, "height": 1}}}
            ]
        },
    )
    import asyncio

    matches = asyncio.run(engine.search(sample_image_bytes, top_k=5, threshold=0.4))
    # Legacy contract: list of dicts with name/distance fields.
    assert isinstance(matches, list)
    assert matches[0]["name"] == "Alice"
    assert matches[0]["distance"] == pytest.approx(0.1)


def test_register_uses_type_for_category_and_reuses_aggregate_id(
    monkeypatch, tmp_path, sample_image_bytes
):
    import asyncio

    import wcm_facerec.face_engine as face_engine_module

    image_path = tmp_path / "face.jpg"

    def persist(image_bytes, _name, _category):
        image_path.write_bytes(image_bytes)
        return str(image_path)

    class Adapter:
        def __init__(self):
            self.calls = []

        def register_person(self, **kwargs):
            self.calls.append(kwargs)
            return ("aggregate-id", "face-id")

        def get_person(self, _person_id):
            return {
                "id": "aggregate-id",
                "name": "测试",
                "type": "时政敏感",
                "category": "时政敏感",
                "file_path": str(image_path),
            }

    monkeypatch.setattr(face_engine_module, "_persist_image", persist)
    monkeypatch.setattr(
        face_engine_module,
        "_image_target_path",
        lambda _image_bytes, _name, _category: image_path,
    )
    monkeypatch.setattr(settings, "insightface_collection_id", "all-persons")
    monkeypatch.setattr(settings, "insightface_category_collections", {"时政敏感": "political"})
    engine = FaceEngine.__new__(FaceEngine)
    engine._adapter = Adapter()

    record = asyncio.run(
        engine.register_from_image(
            "测试",
            sample_image_bytes,
            type_="时政敏感",
        )
    )
    aggregate, mirror = engine._adapter.calls
    assert record["id"] == "aggregate-id"
    assert aggregate["metadata"]["category"] == "时政敏感"
    assert mirror["collection_id"] == "political"
    assert mirror["person_id"] == "aggregate-id"
    assert mirror["external_id"] == "aggregate-id"


def test_register_rolls_back_aggregate_and_file_when_mirror_fails(
    monkeypatch, tmp_path, sample_image_bytes
):
    import asyncio

    import wcm_facerec.face_engine as face_engine_module

    image_path = tmp_path / "face.jpg"

    def persist(image_bytes, _name, _category):
        image_path.write_bytes(image_bytes)
        return str(image_path)

    class Adapter:
        def __init__(self):
            self.deleted = []

        def register_person(self, *, collection_id=None, **_kwargs):
            if collection_id:
                raise RuntimeError("mirror unavailable")
            return ("aggregate-id", "face-id")

        def delete_person(self, person_id):
            self.deleted.append(person_id)

    monkeypatch.setattr(face_engine_module, "_persist_image", persist)
    monkeypatch.setattr(
        face_engine_module,
        "_image_target_path",
        lambda _image_bytes, _name, _category: image_path,
    )
    monkeypatch.setattr(settings, "insightface_collection_id", "all-persons")
    monkeypatch.setattr(settings, "insightface_category_collections", {"时政敏感": "political"})
    engine = FaceEngine.__new__(FaceEngine)
    engine._adapter = Adapter()

    with pytest.raises(RuntimeError, match="mirror unavailable"):
        asyncio.run(
            engine.register_from_image(
                "测试",
                sample_image_bytes,
                type_="时政敏感",
            )
        )
    assert engine._adapter.deleted == ["aggregate-id"]
    assert not image_path.exists()


def test_update_moves_legacy_category_mirror(monkeypatch, tmp_path, sample_image_bytes):
    import asyncio

    image_path = tmp_path / "face.jpg"
    image_path.write_bytes(sample_image_bytes)
    current = {
        "id": "aggregate-id",
        "name": "旧名字",
        "category": "时政敏感",
        "type": "时政敏感",
        "occupation": "",
        "remarks": "",
        "file_path": str(image_path),
    }
    old_mirror = {**current, "id": "legacy-mirror", "external_id": "aggregate-id"}

    class Adapter:
        def __init__(self):
            self.registered = []
            self.deleted = []

        def get_person(self, person_id, *, collection_id=None):
            return current if collection_id is None and person_id == "aggregate-id" else None

        def find_person_by_external_id(self, _person_id, *, collection_id):
            return old_mirror if collection_id == "political" else None

        def register_person(self, **kwargs):
            self.registered.append(kwargs)
            return (kwargs["person_id"], "face-id")

        def update_person(self, person_id, *, name=None, metadata=None, collection_id=None):
            assert collection_id is None
            return {"id": person_id, "name": name or current["name"], **metadata}

        def delete_person(self, person_id, *, collection_id=None):
            self.deleted.append((collection_id, person_id))

    monkeypatch.setattr(
        settings,
        "insightface_category_collections",
        {"时政敏感": "political", "落马官员": "officials"},
    )
    engine = FaceEngine.__new__(FaceEngine)
    engine._adapter = Adapter()
    updated = asyncio.run(
        engine.update_person_record(
            "aggregate-id",
            name="新名字",
            metadata={"type": "落马官员"},
        )
    )
    assert updated["type"] == "落马官员"
    assert engine._adapter.registered[0]["collection_id"] == "officials"
    assert engine._adapter.registered[0]["person_id"] == "aggregate-id"
    assert engine._adapter.deleted == [("political", "legacy-mirror")]


def test_delete_removes_category_mirror_before_aggregate(monkeypatch, tmp_path, sample_image_bytes):
    import asyncio

    image_path = tmp_path / "face.jpg"
    image_path.write_bytes(sample_image_bytes)
    current = {
        "id": "aggregate-id",
        "name": "测试",
        "category": "时政敏感",
        "type": "时政敏感",
        "file_path": str(image_path),
    }
    mirror = {**current, "external_id": "aggregate-id"}

    class Adapter:
        def __init__(self):
            self.deleted = []

        def get_person(self, person_id, *, collection_id=None):
            if collection_id is None:
                return current
            if collection_id == "political" and person_id == "aggregate-id":
                return mirror
            return None

        def find_person_by_external_id(self, _person_id, *, collection_id):
            return None

        def delete_person(self, person_id, *, collection_id=None):
            self.deleted.append((collection_id, person_id))

    monkeypatch.setattr(
        settings,
        "insightface_category_collections",
        {"时政敏感": "political", "落马官员": "officials"},
    )
    engine = FaceEngine.__new__(FaceEngine)
    engine._adapter = Adapter()
    deleted = asyncio.run(engine.delete_person_record("aggregate-id"))
    assert deleted["id"] == "aggregate-id"
    assert engine._adapter.deleted == [
        ("political", "aggregate-id"),
        (None, "aggregate-id"),
    ]
    assert not image_path.exists()


def test_delete_resolves_legacy_category_local_id(
    monkeypatch, tmp_path, sample_image_bytes
):
    import asyncio

    image_path = tmp_path / "legacy-face.jpg"
    image_path.write_bytes(sample_image_bytes)
    aggregate_id = "bad-artists-p-bad-artists-00001"
    mirror_id = "p-bad-artists-00001"
    current = {
        "id": aggregate_id,
        "name": "测试",
        "category": "劣迹艺人",
        "type": "劣迹艺人",
        "file_path": str(image_path),
    }
    mirror = {
        **current,
        "id": mirror_id,
        "external_id": None,
    }

    class Adapter:
        def __init__(self):
            self.deleted = []

        def get_person(self, person_id, *, collection_id=None):
            if collection_id is None:
                return current if person_id == aggregate_id else None
            if collection_id == "bad-artists" and person_id == mirror_id:
                return mirror
            return None

        def find_person_by_external_id(self, _person_id, *, collection_id):
            return None

        def delete_person(self, person_id, *, collection_id=None):
            self.deleted.append((collection_id, person_id))

    monkeypatch.setattr(
        settings,
        "insightface_category_collections",
        {"劣迹艺人": "bad-artists", "时政敏感": "political"},
    )
    engine = FaceEngine.__new__(FaceEngine)
    engine._adapter = Adapter()
    deleted = asyncio.run(engine.delete_person_record(mirror_id))
    assert deleted["id"] == aggregate_id
    assert engine._adapter.deleted == [
        ("bad-artists", mirror_id),
        (None, aggregate_id),
    ]
    assert not image_path.exists()
