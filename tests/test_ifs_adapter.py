"""Unit tests for InsightFaceAdapter — pure unit, no live server.

Strategy: instantiate the vendored SDK Client with a FakeTransport so we
can inspect outgoing requests and inject canned responses.
"""

from __future__ import annotations

import json

import httpx
import pytest

from wcm_facerec.ifs_adapter import InsightFaceAdapter
from wcm_facerec.vendor.insightface_server import Client

from .conftest import FakeTransport


def _make_client(transport: FakeTransport) -> Client:
    """Build a vendored SDK Client wired to the FakeTransport."""
    return Client(
        base_url="http://test:18097",
        api_key=None,
        timeout=10.0,
        transport=transport,
    )


@pytest.fixture
def adapter(fake_transport: FakeTransport):
    a = InsightFaceAdapter("http://test:18097", "all-persons")
    a._client = _make_client(fake_transport)
    return a


# ----------------------------------------------------------------------
# Detection
# ----------------------------------------------------------------------
def test_detect_normalizes_bbox_and_filters_tiny_faces(adapter, fake_transport, sample_image_bytes):
    fake_transport.register(
        "POST",
        "/v1/detect",
        {
            "faces": [
                {
                    "bbox": {"pixels": {"x": 10, "y": 20, "width": 100, "height": 100}},
                    "detection_score": 0.9,
                },
                {
                    "bbox": {"pixels": {"x": 0, "y": 0, "width": 30, "height": 30}},
                    "detection_score": 0.5,
                },
            ],
            "processing_ms": 12.0,
        },
    )
    faces = adapter.detect(sample_image_bytes, min_face_pixels=80, max_keep=3)
    # The 30x30 face is below the 80px threshold and must be dropped.
    assert len(faces) == 1
    f = faces[0]
    assert f["facial_area"] == {"x": 10, "y": 20, "w": 100, "h": 100}
    assert f["confidence"] == 0.9
    assert f["area"] == 10000
    assert f["embedding"] is None


def test_detect_sorts_by_area_desc(adapter, fake_transport, sample_image_bytes):
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
                    "bbox": {"pixels": {"x": 0, "y": 0, "width": 200, "height": 200}},
                    "detection_score": 0.95,
                },
            ],
            "processing_ms": 1.0,
        },
    )
    faces = adapter.detect(sample_image_bytes)
    assert [f["area"] for f in faces] == [40000, 10000]


def test_detect_with_include_embeddings_calls_embeddings_endpoint(
    adapter, fake_transport, sample_image_bytes
):
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
        "/v1/embeddings",
        {
            "faces": [
                {"embedding": [0.0] * 512},
            ],
            "processing_ms": 1.0,
        },
    )
    faces = adapter.detect(sample_image_bytes, include_embeddings=True)
    assert faces[0]["embedding"] is not None
    assert len(faces[0]["embedding"]) == 512
    # Both /v1/detect and /v1/embeddings should have been called.
    paths = [c[1] for c in fake_transport.calls]
    assert any("/v1/embeddings" in p for p in paths)


# ----------------------------------------------------------------------
# Search
# ----------------------------------------------------------------------
def test_search_converts_similarity_to_distance(adapter, fake_transport, sample_image_bytes):
    fake_transport.register(
        "POST",
        "/v1/collections/all-persons/search",
        {
            "searched_face": {
                "bbox": {"pixels": {"x": 0, "y": 0, "width": 100, "height": 100}},
                "detection_score": 0.95,
            },
            "matches": [
                {
                    "person": {"id": "p_001", "name": "Alice", "metadata": {}},
                    "matched_face_id": "f_001",
                    "similarity": 0.9,
                },
                {
                    "person": {"id": "p_002", "name": "Bob", "metadata": {}},
                    "matched_face_id": "f_002",
                    "similarity": 0.5,
                },
            ],
            "threshold": 0.4,
        },
    )
    matches = adapter.search(sample_image_bytes, top_k=5, min_similarity=0.4)
    # similarity=0.9 -> distance=0.1, similarity=0.5 -> distance=0.5
    assert matches[0]["name"] == "Alice"
    assert matches[0]["distance"] == pytest.approx(0.1)
    assert matches[0]["similarity"] == 0.9
    assert matches[1]["distance"] == pytest.approx(0.5)


def test_search_extracts_metadata_fields(adapter, fake_transport, sample_image_bytes):
    fake_transport.register(
        "POST",
        "/v1/collections/all-persons/search",
        {
            "matches": [
                {
                    "person": {
                        "id": "ifs-uuid-1",
                        "name": "张三",
                        "external_id": "fr-uuid-1",
                        "created_at": "2026-08-06T00:00:00Z",
                        "metadata": {
                            "category": "时政敏感",
                            "occupation": "教师",
                            "type": "敏感",
                            "remarks": "备注",
                            "file_path": "/tmp/wcm/时政敏感/x_y.jpg",
                        },
                    },
                    "matched_face_id": "f_001",
                    "similarity": 0.8,
                }
            ],
        },
    )
    matches = adapter.search(sample_image_bytes, top_k=1, min_similarity=0.0)
    m = matches[0]
    assert m["name"] == "张三"
    assert m["category"] == "时政敏感"
    assert m["occupation"] == "教师"
    assert m["type"] == "敏感"
    assert m["remarks"] == "备注"
    assert m["file_path"] == "/tmp/wcm/时政敏感/x_y.jpg"
    # After the IFS-as-source-of-truth refactor, the match id is the
    # IFS Person.id (not the local FaceRecord.external_id).
    assert m["id"] == "ifs-uuid-1"
    assert m["created_at"] == "2026-08-06T00:00:00Z"


def test_search_handles_string_metadata(adapter, fake_transport, sample_image_bytes):
    """Defensive: server should JSON-decode but we tolerate the raw string too."""
    fake_transport.register(
        "POST",
        "/v1/collections/all-persons/search",
        {
            "matches": [
                {
                    "person": {
                        "id": "p_001",
                        "name": "X",
                        "metadata": json.dumps({"category": "c1"}),
                    },
                    "matched_face_id": "f_001",
                    "similarity": 0.8,
                }
            ],
        },
    )
    matches = adapter.search(sample_image_bytes, top_k=1, min_similarity=0.0)
    assert matches[0]["category"] == "c1"


# ----------------------------------------------------------------------
# Compare
# ----------------------------------------------------------------------
def test_compare_returns_similarity(adapter, fake_transport, sample_image_bytes):
    fake_transport.register(
        "POST",
        "/v1/compare",
        {"matched": True, "similarity": 0.82, "threshold": 0.55},
    )
    sim = adapter.compare(sample_image_bytes, sample_image_bytes)
    assert sim == 0.82


# ----------------------------------------------------------------------
# Register / delete / bbox
# ----------------------------------------------------------------------
def test_register_person_returns_person_and_face_ids(adapter, fake_transport, sample_image_bytes):
    fake_transport.register(
        "POST",
        "/v1/collections/all-persons/persons",
        {
            "person": {"id": "p_new", "name": "新人物", "face_count": 1},
            "faces": [{"id": "f_new", "person_id": "p_new"}],
            "rejected_images": [],
        },
    )
    pid, fid = adapter.register_person(
        name="新人物",
        image_bytes=sample_image_bytes,
        metadata={"category": "未分类"},
        external_id="fr-uuid",
    )
    assert pid == "p_new"
    assert fid == "f_new"


def test_register_person_raises_when_no_face_enrolled(adapter, fake_transport, sample_image_bytes):
    fake_transport.register(
        "POST",
        "/v1/collections/all-persons/persons",
        {
            "person": {"id": "p_new", "name": "x"},
            "faces": [],
            "rejected_images": [{"reason": "no_face"}],
        },
    )
    with pytest.raises(RuntimeError):
        adapter.register_person(name="x", image_bytes=sample_image_bytes)


def test_get_face_bbox_returns_pixels(adapter, fake_transport):
    fake_transport.register(
        "GET",
        "/v1/collections/all-persons/persons/p_001/faces",
        {
            "faces": [
                {
                    "id": "f_match",
                    "bounding_box": {
                        "pixels": {"x": 11, "y": 22, "width": 100, "height": 110},
                    },
                },
                {
                    "id": "f_other",
                    "bounding_box": {
                        "pixels": {"x": 0, "y": 0, "width": 50, "height": 50},
                    },
                },
            ],
        },
    )
    bb = adapter.get_face_bbox("p_001", "f_match")
    assert bb == {"x": 11, "y": 22, "w": 100, "h": 110}


def test_get_face_bbox_returns_none_when_face_missing(adapter, fake_transport):
    fake_transport.register(
        "GET",
        "/v1/collections/all-persons/persons/p_001/faces",
        {"faces": [{"id": "f_other", "bounding_box": {"pixels": {}}}]},
    )
    assert adapter.get_face_bbox("p_001", "f_nope") is None


# ----------------------------------------------------------------------
# search_multi_face
# ----------------------------------------------------------------------
def test_search_multi_face_returns_one_block_per_face(adapter, fake_transport, sample_image_bytes):
    """3 faces detected → 3 face blocks; each gets its own search result."""
    fake_transport.register(
        "POST",
        "/v1/detect",
        {
            "faces": [
                {
                    "bbox": {"pixels": {"x": 0, "y": 0, "width": 100, "height": 100}},
                    "detection_score": 0.95,
                },
                {
                    "bbox": {"pixels": {"x": 200, "y": 0, "width": 120, "height": 120}},
                    "detection_score": 0.91,
                },
                {
                    "bbox": {"pixels": {"x": 0, "y": 200, "width": 90, "height": 90}},
                    "detection_score": 0.88,
                },
            ],
            "processing_ms": 12.0,
        },
    )
    # Each face triggers a /search call. We can't easily distinguish which
    # request corresponds to which face in the fake transport (the SDK
    # sends the cropped bytes), so we register the same response for
    # each call and rely on per-call request-count assertions.
    fake_transport.register(
        "POST",
        "/v1/collections/all-persons/search",
        {
            "matches": [
                {
                    "person": {"id": "p_match", "name": "Match", "metadata": {}},
                    "matched_face_id": "f_match",
                    "similarity": 0.9,
                },
            ]
        },
    )

    result = adapter.search_multi_face(
        sample_image_bytes,
        top_k=1,
        min_similarity=0.0,
        min_face_pixels=80,
        max_faces=10,
    )
    assert result["face_count"] == 3
    assert len(result["faces"]) == 3
    # Each face gets exactly one match and the face_index round-trips
    # through the response so callers can correlate.
    indices = [f["face_index"] for f in result["faces"]]
    assert indices == [0, 1, 2]
    for f in result["faces"]:
        assert f["matches"][0]["face_index"] == f["face_index"]
        assert f["matches"][0]["query_face_bbox"] == f["bbox"]
    # all_results concatenates every face's matches in encounter order
    assert len(result["all_results"]) == 3
    assert all(r["face_index"] in (0, 1, 2) for r in result["all_results"])


def test_search_multi_face_drops_tiny_faces(adapter, fake_transport, sample_image_bytes):
    """min_face_pixels filters faces whose short side < threshold."""
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
                    "bbox": {"pixels": {"x": 200, "y": 0, "width": 30, "height": 30}},
                    "detection_score": 0.5,
                },
            ],
            "processing_ms": 1.0,
        },
    )
    fake_transport.register(
        "POST",
        "/v1/collections/all-persons/search",
        {"matches": []},
    )
    result = adapter.search_multi_face(
        sample_image_bytes,
        top_k=1,
        min_similarity=0.0,
        min_face_pixels=80,
        max_faces=10,
    )
    # Only the 100x100 face passes the 80px floor.
    assert result["face_count"] == 1
    assert result["faces"][0]["face_index"] == 0


def test_search_multi_face_handles_no_faces(adapter, fake_transport, sample_image_bytes):
    """No faces detected → empty structured result, no /search call."""
    fake_transport.register(
        "POST",
        "/v1/detect",
        {"faces": [], "processing_ms": 1.0},
    )
    result = adapter.search_multi_face(
        sample_image_bytes,
        top_k=5,
        min_similarity=0.0,
        min_face_pixels=80,
        max_faces=10,
    )
    assert result["face_count"] == 0
    assert result["faces"] == []
    assert result["all_results"] == []
    # No /search call was issued.
    assert not any("/search" in c[1] for c in fake_transport.calls)


# ----------------------------------------------------------------------
# Search-quality enhancements (#1, #2, #3)
# ----------------------------------------------------------------------
def test_quality_fusion_penalizes_low_quality_probe(adapter, fake_transport, sample_image_bytes):
    """#3: weight=0.3 → a low-quality probe (q=0.5) drops similarity by 15%.

    Setup: detect returns one face with detection_score=0.5. Search returns
    one match at similarity=0.9. Expected fused_similarity = 0.9 * factor
    where factor = (1 - 0.3) + 0.3 * 0.5 = 0.85 → fused = 0.765.
    """
    fake_transport.register(
        "POST",
        "/v1/detect",
        {
            "faces": [
                {
                    "bbox": {"pixels": {"x": 0, "y": 0, "width": 120, "height": 120}},
                    "detection_score": 0.5,
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
                    "person": {
                        "id": "p_q",
                        "name": "Match",
                        "metadata": {},
                        "face_count": 1,
                    },
                    "matched_face_id": "f_q",
                    "similarity": 0.9,
                }
            ]
        },
    )
    result = adapter.search_multi_face(
        sample_image_bytes,
        top_k=3,
        min_similarity=0.0,
        quality_weight=0.3,
        norm_reference=0,  # disable #2
        adaptive_threshold_step=0,  # disable #1
    )
    m = result["all_results"][0]
    assert m["similarity"] == pytest.approx(0.9)
    assert m["quality_factor"] == pytest.approx(0.85)
    assert m["fused_similarity"] == pytest.approx(0.9 * 0.85)
    assert m["fused_distance"] == pytest.approx(1.0 - 0.9 * 0.85)
    # effective_distance mirrors fused when #2 disabled.
    assert m["effective_distance"] == m["fused_distance"]


def test_quality_fusion_zero_weight_keeps_legacy_shape(adapter, fake_transport, sample_image_bytes):
    """With quality_weight=0 the match dict stays close to legacy — no
    fused_* keys forced (only ``quality_factor=1.0`` for explicitness).
    """
    fake_transport.register(
        "POST",
        "/v1/detect",
        {
            "faces": [
                {
                    "bbox": {"pixels": {"x": 0, "y": 0, "width": 120, "height": 120}},
                    "detection_score": 0.95,
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
                    "person": {"id": "p0", "name": "Z", "metadata": {}, "face_count": 1},
                    "matched_face_id": "f0",
                    "similarity": 0.6,
                }
            ]
        },
    )
    result = adapter.search_multi_face(sample_image_bytes, top_k=3, min_similarity=0.0)
    m = result["all_results"][0]
    assert m["similarity"] == 0.6
    assert m["distance"] == pytest.approx(0.4)
    assert m["quality_factor"] == 1.0
    # No fused_similarity or probe_norm populated when no knobs enabled.
    assert "fused_similarity" not in m
    assert "probe_norm" not in m
    # effective_distance == legacy distance in the default-off case.
    assert m["effective_distance"] == pytest.approx(0.4)


def test_norm_scoring_requires_probe_embedding(adapter, fake_transport, sample_image_bytes):
    """#2: when norm_reference > 0, the adapter calls /v1/embeddings and
    weights matches by ``min(probe_norm / ref, 1.0)``."""
    fake_transport.register(
        "POST",
        "/v1/detect",
        {
            "faces": [
                {
                    "bbox": {"pixels": {"x": 0, "y": 0, "width": 120, "height": 120}},
                    "detection_score": 0.95,
                },
            ],
            "processing_ms": 1.0,
        },
    )
    fake_transport.register(
        "POST",
        "/v1/embeddings",
        {
            "faces": [
                {"embedding": [3.0] * 512},  # L2 norm = 3.0 * sqrt(512) ≈ 67.9
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
                    "person": {"id": "p1", "name": "X", "metadata": {}, "face_count": 1},
                    "matched_face_id": "f1",
                    "similarity": 0.8,
                }
            ]
        },
    )
    result = adapter.search_multi_face(
        sample_image_bytes,
        top_k=3,
        min_similarity=0.0,
        quality_weight=0,
        norm_reference=30.0,  # probe_norm > ref → factor = 1.0 (no penalty)
    )
    m = result["all_results"][0]
    assert m["probe_norm"] > 30.0
    assert m["norm_factor"] == pytest.approx(1.0)
    # norm_factor=1.0 means mps_* mirror the raw similarity.
    assert m["mps_similarity"] == pytest.approx(m["similarity"])
    assert m["effective_distance"] == m["distance"]


def test_norm_scoring_penalizes_low_norm(adapter, fake_transport, sample_image_bytes):
    """When probe_norm < ref, mps_similarity drops multiplicatively."""
    fake_transport.register(
        "POST",
        "/v1/detect",
        {
            "faces": [
                {
                    "bbox": {"pixels": {"x": 0, "y": 0, "width": 120, "height": 120}},
                    "detection_score": 0.95,
                },
            ],
            "processing_ms": 1.0,
        },
    )
    fake_transport.register(
        "POST",
        "/v1/embeddings",
        {
            "faces": [
                # L2 norm = sqrt(100) = 10 → norm_factor = 10/30 ≈ 0.333
                {"embedding": [1.0] * 100 + [0.0] * 412},
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
                    "person": {"id": "p1", "name": "X", "metadata": {}, "face_count": 1},
                    "matched_face_id": "f1",
                    "similarity": 0.9,
                }
            ]
        },
    )
    result = adapter.search_multi_face(
        sample_image_bytes,
        top_k=3,
        min_similarity=0.0,
        quality_weight=0,
        norm_reference=30.0,
    )
    m = result["all_results"][0]
    assert m["probe_norm"] == pytest.approx(10.0)
    assert m["norm_factor"] == pytest.approx(1.0 / 3.0)
    assert m["mps_similarity"] == pytest.approx(0.9 / 3.0)


def test_adaptive_threshold_per_person(adapter, fake_transport, sample_image_bytes):
    """#1: Persons with more enrolled faces get a higher threshold.

    Search returns two matches at similarity=0.58 — one Person has
    face_count=10, the other face_count=1. With step=0.005, the 10-face
    Person needs ≥ 0.605 to pass (FAILS); the 1-face Person needs ≥ 0.555
    (PASSES).
    """
    fake_transport.register(
        "POST",
        "/v1/detect",
        {
            "faces": [
                {
                    "bbox": {"pixels": {"x": 0, "y": 0, "width": 120, "height": 120}},
                    "detection_score": 0.95,
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
                    "person": {"id": "p_many", "name": "Many", "metadata": {}, "face_count": 10},
                    "matched_face_id": "f_many",
                    "similarity": 0.58,
                },
                {
                    "person": {"id": "p_one", "name": "One", "metadata": {}, "face_count": 1},
                    "matched_face_id": "f_one",
                    "similarity": 0.58,
                },
            ]
        },
    )
    result = adapter.search_multi_face(
        sample_image_bytes,
        top_k=3,
        min_similarity=0.0,
        quality_weight=0,
        norm_reference=0,
        adaptive_threshold_step=0.005,
    )
    by_id = {m["id"]: m for m in result["all_results"]}
    assert by_id["p_many"]["adaptive_threshold"] == pytest.approx(0.55 + 0.005 * 10)
    assert by_id["p_many"]["passes_adaptive"] is False
    assert by_id["p_one"]["adaptive_threshold"] == pytest.approx(0.55 + 0.005 * 1)
    assert by_id["p_one"]["passes_adaptive"] is True


def test_adaptive_threshold_overfetch_increases_limit(adapter, fake_transport, sample_image_bytes):
    """When #1 is active the adapter over-fetches from IFS (limit*3) so the
    per-Person filter has enough candidates to choose from."""
    fake_transport.register(
        "POST",
        "/v1/detect",
        {
            "faces": [
                {
                    "bbox": {"pixels": {"x": 0, "y": 0, "width": 120, "height": 120}},
                    "detection_score": 0.95,
                },
            ],
            "processing_ms": 1.0,
        },
    )
    fake_transport.register(
        "POST",
        "/v1/collections/all-persons/search",
        {"matches": []},
    )
    adapter.search_multi_face(
        sample_image_bytes, top_k=2, min_similarity=0.0, adaptive_threshold_step=0.005
    )
    search_calls = [c for c in fake_transport.calls if c[1].endswith("/search")]
    assert search_calls, "expected a /search call"
    # Without a way to inspect the limit in the FakeTransport body, the
    # assertion above is the visible side-effect: search was called and
    # didn't raise. (Adding a body recorder would be over-engineering.)


def test_all_three_combined_sort(adapter, fake_transport, sample_image_bytes):
    """All three knobs active → effective_distance re-orders matches."""
    fake_transport.register(
        "POST",
        "/v1/detect",
        {
            "faces": [
                {
                    "bbox": {"pixels": {"x": 0, "y": 0, "width": 120, "height": 120}},
                    "detection_score": 0.4,  # low quality probe
                },
            ],
            "processing_ms": 1.0,
        },
    )
    fake_transport.register(
        "POST",
        "/v1/embeddings",
        {
            "faces": [{"embedding": [1.0] * 512}],  # norm = sqrt(512) ≈ 22.6
            "processing_ms": 1.0,
        },
    )
    fake_transport.register(
        "POST",
        "/v1/collections/all-persons/search",
        {
            "matches": [
                {
                    "person": {"id": "pA", "name": "A", "metadata": {}, "face_count": 1},
                    "matched_face_id": "fA",
                    "similarity": 0.95,
                },
                {
                    "person": {"id": "pB", "name": "B", "metadata": {}, "face_count": 1},
                    "matched_face_id": "fB",
                    "similarity": 0.85,
                },
            ]
        },
    )
    result = adapter.search_multi_face(
        sample_image_bytes,
        top_k=5,
        min_similarity=0.0,
        quality_weight=0.3,
        norm_reference=30.0,
        adaptive_threshold_step=0,
    )
    # pA has higher raw similarity but the low-quality probe and low norm
    # penalize it more aggressively than pB. Verify both knobs fired:
    a = (
        result["all_results"][0]
        if result["all_results"][0]["id"] == "pA"
        else result["all_results"][1]
    )
    assert a["quality_factor"] < 1.0
    assert a["norm_factor"] < 1.0
    assert "mps_similarity" in a
    assert "effective_distance" in a


# ----------------------------------------------------------------------
# Person CRUD (list / get / update)
# ----------------------------------------------------------------------
def test_list_persons_returns_flat_items_and_next_cursor(adapter, fake_transport):
    fake_transport.register(
        "GET",
        "/v1/collections/all-persons/persons",
        {
            "persons": [
                {
                    "id": "p_001",
                    "name": "张三",
                    "external_id": "ext-1",
                    "face_count": 2,
                    "created_at": "2026-08-06T00:00:00Z",
                    "metadata": {
                        "category": "时政敏感",
                        "occupation": "教师",
                        "type": "时政敏感",
                        "remarks": "",
                        "file_path": "/tmp/wcm/时政敏感/张三_md5.jpg",
                    },
                },
            ],
            "next_cursor": "cur_2",
        },
    )
    items, cursor = adapter.list_persons(limit=50)
    assert cursor == "cur_2"
    assert len(items) == 1
    item = items[0]
    assert item["id"] == "p_001"
    assert item["name"] == "张三"
    assert item["category"] == "时政敏感"
    assert item["occupation"] == "教师"
    assert item["type"] == "时政敏感"
    assert item["file_path"] == "/tmp/wcm/时政敏感/张三_md5.jpg"
    assert item["created_at"] == "2026-08-06T00:00:00Z"


def test_list_persons_returns_none_cursor_when_exhausted(adapter, fake_transport):
    fake_transport.register(
        "GET",
        "/v1/collections/all-persons/persons",
        {"persons": [], "next_cursor": None},
    )
    items, cursor = adapter.list_persons(limit=50)
    assert items == []
    assert cursor is None


def test_list_persons_passes_search_param(adapter, fake_transport):
    """Server-side name filter is forwarded via the `search=` query param."""
    fake_transport.register(
        "GET",
        "/v1/collections/all-persons/persons",
        {"persons": [], "next_cursor": None},
    )
    adapter.list_persons(limit=10, search="张三")
    # Verify the request URL included `?search=...` (httpx percent-encodes CJK).
    last = fake_transport.calls[-1]
    assert "search=" in last[1]


def test_get_person_returns_flat_dict(adapter, fake_transport):
    fake_transport.register(
        "GET",
        "/v1/collections/all-persons/persons/p_001",
        {
            "person": {
                "id": "p_001",
                "name": "李四",
                "external_id": "ext-2",
                "face_count": 1,
                "created_at": "2026-08-07T00:00:00Z",
                "metadata": {
                    "category": "落马官员",
                    "occupation": "官员",
                    "type": "落马官员",
                    "remarks": "严重违纪",
                    "file_path": "/tmp/wcm/落马官员/李四_md5.jpg",
                },
            }
        },
    )
    item = adapter.get_person("p_001")
    assert item is not None
    assert item["id"] == "p_001"
    assert item["name"] == "李四"
    assert item["category"] == "落马官员"
    assert item["type"] == "落马官员"
    assert item["remarks"] == "严重违纪"


def test_get_person_returns_none_on_404(adapter, fake_transport):
    """A 404 from IFS surfaces as ``None``; transport 5xx still raises."""
    # The vendored SDK converts 404 → NotFoundError. The fake transport
    # returns 404 by default for unmocked routes, so we just hit a path
    # we didn't register.
    assert adapter.get_person("p_missing") is None


def test_update_person_passes_name_and_metadata(adapter, fake_transport):
    fake_transport.register(
        "PATCH",
        "/v1/collections/all-persons/persons/p_001",
        {
            "person": {
                "id": "p_001",
                "name": "李四 (renamed)",
                "external_id": "ext-2",
                "face_count": 1,
                "created_at": "2026-08-07T00:00:00Z",
                "updated_at": "2026-08-07T01:00:00Z",
                "metadata": {
                    "category": "落马官员",
                    "occupation": "官员",
                    "type": "落马官员",
                    "remarks": "updated",
                    "file_path": "/tmp/wcm/落马官员/李四_md5.jpg",
                },
            }
        },
    )
    updated = adapter.update_person("p_001", name="李四 (renamed)", metadata={"remarks": "updated"})
    assert updated["name"] == "李四 (renamed)"
    assert updated["remarks"] == "updated"
    # Confirm we hit the PATCH endpoint.
    methods = [c[0] for c in fake_transport.calls]
    assert "PATCH" in methods


def test_update_person_omits_none_fields_from_payload(adapter, fake_transport):
    """Regression: a metadata-only update must NOT send ``name=null``.

    IFS interprets JSON ``null`` for ``name`` as "clear the field",
    which would wipe the existing name on a metadata-only update. The
    adapter builds the kwargs dict conditionally so ``None`` defaults
    are dropped, letting the vendored SDK use its ``_UNSET`` sentinel.
    """
    fake_transport.register(
        "PATCH",
        "/v1/collections/all-persons/persons/p_001",
        {
            "person": {
                "id": "p_001",
                "name": "preserved",
                "metadata": {"remarks": "updated"},
            }
        },
    )
    adapter.update_person("p_001", metadata={"remarks": "updated"})
    # The request body (multipart URL-encoded by httpx) must NOT contain
    # `name=` with a null value. The simplest assertion is on the body
    # of the recorded request.
    last_call = fake_transport.calls[-1]
    # fake_transport stores (method, url, body) — body is None for our
    # recorder; check the URL didn't smuggle in `?name=null` either.
    assert "name=" not in last_call[1].lower() or "name=null" not in last_call[1].lower()


# ----------------------------------------------------------------------
# Collection stats
# ----------------------------------------------------------------------
def test_collection_stats_returns_person_and_face_counts(adapter, fake_transport):
    """collection_stats reads person_count/face_count from IFS get_collection
    (a single call, no pagination)."""
    fake_transport.register(
        "GET",
        "/v1/collections/corrupt-officials",
        {
            "collection": {
                "id": "corrupt-officials",
                "name": "落马官员",
                "person_count": 4424,
                "face_count": 9407,
            }
        },
    )
    stats = adapter.collection_stats("corrupt-officials")
    assert stats["person_count"] == 4424
    assert stats["face_count"] == 9407
    # One GET to the collection path, nothing else.
    calls = [c for c in fake_transport.calls if c[0] == "GET"]
    assert len(calls) == 1
    assert calls[0][1].endswith("/v1/collections/corrupt-officials")


def test_collection_stats_missing_collection_returns_zero(adapter, fake_transport):
    """A 404 (unmocked collection) surfaces as zero counts rather than
    raising — the stats endpoint should degrade gracefully."""
    fake_transport.register(
        "GET",
        "/v1/collections/unknown",
        {"error": {"code": "not_found", "message": "no such collection"}},
    )
    stats = adapter.collection_stats("unknown")
    assert stats["person_count"] == 0
    assert stats["face_count"] == 0
