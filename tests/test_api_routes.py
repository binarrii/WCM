from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api import face_records, handlers, routes
from api.main import create_app


class StubEngine:
    def __init__(self, adapter=None):
        self._adapter = adapter or SimpleNamespace()

    async def _run(self, func, *args, **kwargs):
        return func(*args, **kwargs)


@pytest.fixture
def client_for(monkeypatch):
    def factory(engine):
        monkeypatch.setattr(face_records, "get_face_engine", lambda: engine)
        monkeypatch.setattr(routes, "get_face_engine", lambda: engine)
        return TestClient(create_app())

    return factory


def test_image_url_is_only_returned_for_existing_local_file(tmp_path, monkeypatch):
    image_dir = tmp_path / "劣迹艺人"
    image_dir.mkdir()
    image = image_dir / "face.jpg"
    image.write_bytes(b"image")
    monkeypatch.setattr(face_records, "_IMAGE_ROOT", tmp_path)

    assert face_records._path_to_image_url(str(image)) == "/images/劣迹艺人/face.jpg"
    assert face_records._path_to_image_url(str(image_dir / "missing.jpg")) is None


def test_image_url_rejects_path_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr(face_records, "_IMAGE_ROOT", tmp_path)

    assert face_records._path_to_image_url(str(tmp_path / ".." / "secret.jpg")) is None


def test_face_records_exposes_image_url_without_file_path(tmp_path, monkeypatch, client_for):
    image = tmp_path / "时政敏感" / "face.jpg"
    image.parent.mkdir()
    image.write_bytes(b"image")
    monkeypatch.setattr(face_records, "_IMAGE_ROOT", tmp_path)

    def list_persons(**_kwargs):
        return [{"id": "p1", "name": "测试", "file_path": str(image)}], None

    response = client_for(StubEngine(SimpleNamespace(list_persons=list_persons))).get(
        "/api/v1/face_records"
    )
    item = response.json()["items"][0]
    assert item["image_url"] == "/images/时政敏感/face.jpg"
    assert "file_path" not in item


def test_search_returns_canonical_fields_without_internal_paths(
    tmp_path, monkeypatch, client_for, sample_image_bytes
):
    image = tmp_path / "劣迹艺人" / "face.jpg"
    image.parent.mkdir()
    image.write_bytes(b"image")
    monkeypatch.setattr(face_records, "_IMAGE_ROOT", tmp_path)

    class SearchEngine(StubEngine):
        async def search(self, **_kwargs):
            return [
                {
                    "id": "p1",
                    "person_id": "p1",
                    "name": "测试",
                    "person_name": "测试",
                    "file_path": str(image),
                    "category": "劣迹艺人",
                    "type": "劣迹艺人",
                    "similarity": 0.9,
                    "distance": 0.1,
                    "effective_similarity": 0.95,
                    "effective_distance": 0.05,
                    "quality_factor": 1.1,
                    "passes_adaptive": True,
                    "matched_face_id": "face-1",
                    "query_face_bbox": {"x": 1, "y": 2, "w": 3, "h": 4},
                }
            ]

    response = client_for(SearchEngine()).post(
        "/api/v1/search",
        files={"file": ("query.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result == {
        "id": "p1",
        "name": "测试",
        "similarity": 0.95,
        "distance": 0.05,
        "image_url": "/images/劣迹艺人/face.jpg",
        "type": "劣迹艺人",
        "matched_face_id": "face-1",
        "query_face_bbox": {"x": 1, "y": 2, "w": 3, "h": 4},
    }


def test_health_checks_insightface_dependency(client_for):
    adapter = SimpleNamespace(health=lambda: {"status": "ok"})
    response = client_for(StubEngine(adapter)).get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["model"] == "buffalo_m"
    assert response.json()["dependencies"] == {"insightface": "ok"}


def test_health_returns_503_when_insightface_is_down(client_for):
    def unavailable():
        raise RuntimeError("offline")

    response = client_for(StubEngine(SimpleNamespace(health=unavailable))).get("/api/v1/health")
    assert response.status_code == 503
    assert response.json()["detail"]["status"] == "unhealthy"


def test_face_records_cursor_fetches_next_server_page(client_for):
    calls = []

    def list_persons(*, limit, cursor, search, collection_id):
        calls.append(cursor)
        if cursor is None:
            return [
                {"id": "p1", "name": "A", "type": ""},
                {"id": "p2", "name": "B", "type": ""},
            ], "server-page-2"
        return [{"id": "p3", "name": "C", "type": ""}], None

    engine = StubEngine(SimpleNamespace(list_persons=list_persons))
    client = client_for(engine)
    first = client.get("/api/v1/face_records", params={"limit": 2}).json()
    assert [item["id"] for item in first["items"]] == ["p1", "p2"]
    assert first["next_cursor"]

    second = client.get(
        "/api/v1/face_records",
        params={"limit": 2, "cursor": first["next_cursor"]},
    ).json()
    assert [item["id"] for item in second["items"]] == ["p3"]
    assert calls == [None, "server-page-2"]


def test_filtered_cursor_resumes_inside_an_ifs_page(client_for):
    people = [
        {"id": "category", "name": "skip", "type": "劣迹艺人"},
        {"id": "other-1", "name": "A", "type": "普通人物"},
        {"id": "other-2", "name": "B", "type": ""},
        {"id": "other-3", "name": "C", "type": "普通人物"},
    ]

    def list_persons(**_kwargs):
        return [dict(person) for person in people], None

    client = client_for(StubEngine(SimpleNamespace(list_persons=list_persons)))
    first = client.get("/api/v1/face_records", params={"limit": 2, "type": "其它"}).json()
    second = client.get(
        "/api/v1/face_records",
        params={"limit": 2, "type": "其它", "cursor": first["next_cursor"]},
    ).json()
    assert [item["id"] for item in first["items"]] == ["other-1", "other-2"]
    assert [item["id"] for item in second["items"]] == ["other-3"]


def test_category_list_exposes_aggregate_id(client_for):
    def list_persons(**_kwargs):
        return [
            {
                "id": "legacy-mirror-id",
                "external_id": "aggregate-id",
                "name": "测试",
                "type": "时政敏感",
            }
        ], None

    client = client_for(StubEngine(SimpleNamespace(list_persons=list_persons)))
    response = client.get("/api/v1/face_records", params={"type": "时政敏感"}).json()
    assert response["items"][0]["id"] == "aggregate-id"
    assert response["items"][0]["person"]["id"] == "aggregate-id"


def test_aggregate_list_uses_person_id_not_external_id(client_for):
    def list_persons(**_kwargs):
        return [
            {
                "id": "ifs-person-id",
                "external_id": "legacy-source-id",
                "name": "测试",
                "type": "落马官员",
            }
        ], None

    client = client_for(StubEngine(SimpleNamespace(list_persons=list_persons)))
    response = client.get("/api/v1/face_records", params={"type": "All"}).json()
    assert response["items"][0]["id"] == "ifs-person-id"
    assert response["items"][0]["person"]["id"] == "ifs-person-id"


def test_category_list_reconstructs_legacy_aggregate_id(client_for):
    def list_persons(**_kwargs):
        return [
            {
                "id": "p-bad-artists-00001",
                "external_id": None,
                "name": "测试",
                "type": "劣迹艺人",
            }
        ], None

    client = client_for(StubEngine(SimpleNamespace(list_persons=list_persons)))
    response = client.get("/api/v1/face_records", params={"type": "劣迹艺人"}).json()
    assert response["items"][0]["id"] == "bad-artists-p-bad-artists-00001"
    assert response["items"][0]["person"]["id"] == "bad-artists-p-bad-artists-00001"


def test_create_uses_type_as_category(client_for, sample_image_bytes):
    class CreateEngine(StubEngine):
        async def detect_faces(self, _contents):
            return [{"facial_area": {}}]

        async def register_from_image(self, **kwargs):
            self.kwargs = kwargs
            return {
                "id": "aggregate-id",
                "name": kwargs["name"],
                "type": kwargs["type_"],
                "category": kwargs["category"],
            }

    engine = CreateEngine()
    response = client_for(engine).post(
        "/api/v1/face_records",
        data={"name": "测试", "type": "时政敏感"},
        files={"file": ("face.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    assert engine.kwargs["category"] == "时政敏感"
    assert engine.kwargs["type_"] == "时政敏感"


def test_update_and_delete_delegate_to_consistent_engine_methods(client_for):
    class CrudEngine(StubEngine):
        async def update_person_record(self, person_id, *, name, metadata):
            self.updated = (person_id, name, metadata)
            return {"id": person_id, "name": name, **metadata}

        async def delete_person_record(self, person_id):
            self.deleted = person_id
            return {"id": person_id}

    engine = CrudEngine()
    client = client_for(engine)
    updated = client.put(
        "/api/v1/face_records/p1",
        json={"name": "新名字", "type": "落马官员", "remarks": "备注"},
    )
    deleted = client.delete("/api/v1/face_records/p1")
    assert updated.status_code == 200
    assert engine.updated == (
        "p1",
        "新名字",
        {"type": "落马官员", "remarks": "备注"},
    )
    assert deleted.status_code == 200
    assert engine.deleted == "p1"


def test_video_search_awaits_async_engine_and_handles_zero_interval(monkeypatch, tmp_path):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    class Capture:
        def __init__(self, _path):
            self.reads = iter([(True, frame), (False, None)])

        def isOpened(self):
            return True

        def get(self, prop):
            return 25.0 if prop == handlers.cv2.CAP_PROP_FPS else 0.0

        def read(self):
            return next(self.reads)

        def release(self):
            pass

    class SearchEngine:
        async def search(self, **_kwargs):
            self.called = True
            return [{"name": "match", "person_id": "p1", "distance": 0.1}]

    monkeypatch.setattr(handlers.cv2, "VideoCapture", Capture)
    engine = SearchEngine()
    frames, results = asyncio.run(
        handlers._search_video_frames(
            engine,
            "unused",
            None,
            5,
            0.4,
            0.0,
            local_video_path=tmp_path / "video.mp4",
        )
    )
    assert engine.called is True
    assert frames == 1
    assert results[0]["person_id"] == "p1"
