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

    async def compare_gallery(self, image_bytes, paths, query_bbox):
        return [None] * len(paths)


@pytest.fixture
def client_for(monkeypatch):
    def factory(engine):
        monkeypatch.setattr(face_records, "get_face_engine", lambda: engine)
        monkeypatch.setattr(routes, "get_face_engine", lambda: engine)
        return TestClient(create_app())

    return factory


@pytest.mark.parametrize("threshold", [None, 0.7, 0.9, 0.0])
@pytest.mark.parametrize(
    "mode",
    ["upload", "image", "video", "ws_image", "ws_video", "analyze", "ws_analyze"],
)
def test_matching_threshold_defaults_and_explicit_overrides(
    mode, threshold, client_for, monkeypatch, sample_image_bytes
):
    received = []

    class SearchEngine(StubEngine):
        async def search(self, **kwargs):
            received.append(kwargs["threshold"])
            return []

    async def download(*args):
        return sample_image_bytes

    async def search_video(engine, url, name, top_k, threshold, sample_interval):
        received.append(threshold)
        return 1, []

    async def analyze(url, sample_interval, top_k, threshold):
        received.append(threshold)
        return {"results": []}

    monkeypatch.setattr(routes, "_download_url_safe", download)
    monkeypatch.setattr(routes, "_search_video_frames", search_video)
    monkeypatch.setattr(routes, "_process_analyze_media", analyze)
    client = client_for(SearchEngine())
    payload = {} if threshold is None else {"threshold": threshold}
    if mode == "upload":
        response = client.post(
            "/api/v1/search",
            data=payload,
            files={"file": ("query.jpg", sample_image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200, response.text
    else:
        payload["url"] = "https://example.com/query." + ("mp4" if "video" in mode else "jpg")
        endpoint = "analyze_media" if "analyze" in mode else "search"
        if mode.startswith("ws_"):
            with client.websocket_connect(f"/api/v1/ws/{endpoint}") as socket:
                socket.send_json(payload)
                assert socket.receive_json()["status"] == "accepted"
                assert socket.receive_json()["status"] == "completed"
        else:
            response = client.post(f"/api/v1/{endpoint}", json=payload)
            assert response.status_code == 200, response.text
    assert received == [0.5 if threshold is None else threshold]


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


def test_face_records_exposes_all_person_images(tmp_path, monkeypatch, client_for):
    image_dir = tmp_path / "落马官员"
    image_dir.mkdir()
    images = [image_dir / "face-1.jpg", image_dir / "face-2.jpg"]
    for image in images:
        image.write_bytes(b"image")
    monkeypatch.setattr(face_records, "_IMAGE_ROOT", tmp_path)

    def list_persons(**_kwargs):
        return [
            {
                "id": "p1",
                "name": "艾宝俊",
                "face_count": 2,
                "file_path": str(images[0]),
                "image_paths": [str(image) for image in images],
            }
        ], None

    response = client_for(StubEngine(SimpleNamespace(list_persons=list_persons))).get(
        "/api/v1/face_records", params={"search": "艾宝俊"}
    )
    item = response.json()["items"][0]
    assert item["face_count"] == 2
    assert item["image_urls"] == [
        "/images/落马官员/face-1.jpg",
        "/images/落马官员/face-2.jpg",
    ]


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
        "image_urls": ["/images/劣迹艺人/face.jpg"],
        "image_similarities": {"/images/劣迹艺人/face.jpg": None},
        "type": "劣迹艺人",
        "matched_face_id": "face-1",
        "query_face_bbox": {"x": 1, "y": 2, "w": 3, "h": 4},
    }


def test_search_merges_sample_hits_and_returns_full_gallery(
    tmp_path, monkeypatch, client_for, sample_image_bytes
):
    images = [tmp_path / f"face-{index}.jpg" for index in range(4)]
    for image in images:
        image.write_bytes(b"image")
    monkeypatch.setattr(face_records, "_IMAGE_ROOT", tmp_path)
    base = {
        "id": "p1",
        "name": "测试",
        "file_path": str(images[0]),
        "image_paths": [str(image) for image in images]
        + [str(images[0]), str(tmp_path / "missing.jpg"), None, "../secret.jpg"],
        "face_count": 4,
        "face_index": 0,
        "query_face_bbox": {"x": 1, "y": 2, "w": 3, "h": 4},
    }

    class SearchEngine(StubEngine):
        async def compare_gallery(self, image_bytes, paths, query_bbox):
            assert paths == images
            assert query_bbox == base["query_face_bbox"]
            return [0.76, 0.65, 0.97, 0.22]

        async def search(self, **kwargs):
            assert kwargs["threshold"] == 0.7  # minimum similarity 30%
            return [
                {**base, "similarity": 0.7, "distance": 0.3, "matched_face_id": "f1"},
                {
                    **base,
                    "similarity": 0.9,
                    "distance": 0.1,
                    "matched_face_id": "f2",
                    "source_x": 42,
                },
            ]

    response = client_for(SearchEngine()).post(
        "/api/v1/search",
        files={"file": ("query.jpg", sample_image_bytes, "image/jpeg")},
        data={"threshold": "0.7"},
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 1
    result = results[0]
    assert result["matched_face_id"] == "f2"
    assert result["source_x"] == 42
    assert result["similarity"] == 0.9
    assert result["distance"] == 0.1
    assert result["face_count"] == 4
    assert result["image_urls"] == [f"/images/face-{index}.jpg" for index in range(4)]
    assert result["image_url"] == result["image_urls"][0]
    assert result["image_similarities"] == {
        "/images/face-0.jpg": 0.76,
        "/images/face-1.jpg": 0.65,
        "/images/face-2.jpg": 0.97,
        "/images/face-3.jpg": 0.22,
    }
    assert "file_path" not in response.text
    assert "image_paths" not in response.text
    assert str(tmp_path) not in response.text


def test_search_deduplication_keeps_query_faces_frames_and_distinct_ids():
    base = {"id": "p1", "name": "same name", "face_index": 0, "similarity": 0.8}
    matches = [
        base,
        {**base, "similarity": 0.99, "effective_similarity": 0.5},
        {**base, "face_index": 1},
        {**base, "frame_time": 1.0},
        {**base, "id": "p2", "similarity": 0.9},
    ]
    results = routes._public_search_results(matches)
    assert len(results) == 4
    assert results[0]["id"] == "p2"
    assert [result["similarity"] for result in results] == [0.9, 0.8, 0.8, 0.8]
    assert {(r["id"], r["face_index"], r.get("frame_time")) for r in results} == {
        ("p1", 0, None),
        ("p1", 1, None),
        ("p1", 0, 1.0),
        ("p2", 0, None),
    }


def test_search_gallery_falls_back_when_cover_is_missing(tmp_path, monkeypatch):
    image = tmp_path / "other.jpg"
    image.write_bytes(b"image")
    monkeypatch.setattr(face_records, "_IMAGE_ROOT", tmp_path)
    result = routes._public_search_result(
        {"file_path": str(tmp_path / "missing.jpg"), "image_paths": [str(image)]}
    )
    assert result["image_url"] == "/images/other.jpg"
    assert result["image_urls"] == ["/images/other.jpg"]
    assert routes._public_search_result({"image_paths": "not-a-list"})["image_urls"] == []
    assert len(routes._public_search_results([{}, {}])) == 2


def test_gallery_url_resolution_never_reads_outside_image_root(tmp_path, monkeypatch):
    root = tmp_path / "images"
    root.mkdir()
    image = root / "face.jpg"
    image.write_bytes(b"image")
    secret = tmp_path / "secret.jpg"
    secret.write_bytes(b"not-public")
    (root / "link.jpg").symlink_to(secret)
    monkeypatch.setattr(face_records, "_IMAGE_ROOT", root)
    assert face_records._image_url_to_path("/images/face.jpg") == image
    for url in ("/images/../secret.jpg", "/images/link.jpg", "/images/missing.jpg", str(secret)):
        assert face_records._image_url_to_path(url) is None


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


def test_stats_exposes_people_and_images_without_double_counting_mirrors(monkeypatch, client_for):
    monkeypatch.setattr(face_records.settings, "insightface_collection_id", "aggregate")
    monkeypatch.setattr(
        face_records.settings,
        "insightface_category_collections",
        {
            "劣迹艺人": "artists",
            "时政敏感": "political",
            "落马官员": "officials",
        },
    )
    libraries = {
        "aggregate": {"person_count": 12, "face_count": 32},
        "artists": {"person_count": 2, "face_count": 7},
        "political": {"person_count": 3, "face_count": 8},
        "officials": {"person_count": 4, "face_count": 9},
    }
    calls = []

    def collection_stats(collection_id):
        calls.append(collection_id)
        return libraries[collection_id]

    client = client_for(StubEngine(SimpleNamespace(collection_stats=collection_stats)))
    response = client.get("/api/v1/face_records/stats")
    assert response.status_code == 200
    assert response.json() == {
        "total": 12,
        "bad_artists": 2,
        "political": 3,
        "officials": 4,
        "total_images": 32,
        "bad_artists_images": 7,
        "political_images": 8,
        "officials_images": 9,
    }
    assert calls == ["aggregate", "artists", "political", "officials"]


def test_stats_empty_library_and_unconfigured_categories_are_zero(monkeypatch, client_for):
    monkeypatch.setattr(face_records.settings, "insightface_category_collections", {})
    adapter = SimpleNamespace(collection_stats=lambda _: {"person_count": 0, "face_count": 0})
    response = client_for(StubEngine(adapter)).get("/api/v1/face_records/stats")
    assert response.status_code == 200
    assert len(response.json()) == 8
    assert set(response.json().values()) == {0}


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
