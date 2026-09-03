"""Media analysis must filter/crop query faces, not enrolled face samples."""

import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock

import cv2
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api import handlers
from api.main import create_app
from wcm_facerec.face_engine import FaceEngine
from wcm_facerec.vendor.insightface_server import Client


@pytest.mark.asyncio
@pytest.mark.parametrize("sample_size", [12, 160])
@pytest.mark.parametrize(
    "width,height,keep",
    [
        (47, 90, False),
        (90, 47, False),
        (48, 48, True),
        (60, 60, True),
        (64, 64, True),
        (79, 79, True),
        (120, 120, True),
    ],
)
async def test_face_filter_uses_query_dimensions(width, height, keep, sample_size):
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    match = {
        "query_face_bbox": {"x": 10, "y": 20, "w": width, "h": height},
        "source_x": 0,
        "source_y": 0,
        "source_w": sample_size,
        "source_h": sample_size,
    }
    search = AsyncMock(return_value={"all_results": [match]})
    engine = SimpleNamespace(search_multi_face=search)

    results = await handlers._face_task(engine, frame, 3, 0.5, 2.25)

    assert len(results) == int(keep)
    search.assert_awaited_once_with(img_source=frame, top_k=3, threshold=0.5, min_face_pixels=48)
    if keep:
        assert results[0]["frame_time"] == 2.25
        assert results[0]["timestamp"] == "00:00:02.250"


@pytest.mark.asyncio
async def test_face_crop_uses_query_coordinates_and_preserves_sample_metadata():
    frame = np.zeros((180, 200, 3), dtype=np.uint8)
    frame[60:120, 80:128] = (10, 120, 240)
    match = {
        "query_face_bbox": {"x": 80, "y": 60, "w": 48, "h": 60},
        "source_x": 5,
        "source_y": 5,
        "source_w": 12,
        "source_h": 20,
    }
    engine = SimpleNamespace(search_multi_face=AsyncMock(return_value={"all_results": [match]}))

    results = await handlers._face_task(engine, frame, 10, 0.5, 0.0)

    assert len(results) == 1
    _, expected = cv2.imencode(".jpg", frame[60:120, 80:128])
    assert base64.b64decode(results[0]["face_image_b64"]) == expected.tobytes()
    assert results[0]["source_w"] == 12
    assert results[0]["source_h"] == 20


@pytest.mark.asyncio
@pytest.mark.parametrize("bbox", [None, {}, {"w": 48, "h": 48}])
async def test_missing_query_coordinates_never_fall_back_to_sample(bbox):
    match = {
        "query_face_bbox": bbox,
        "source_x": 0,
        "source_y": 0,
        "source_w": 12,
        "source_h": 12,
    }
    engine = SimpleNamespace(search_multi_face=AsyncMock(return_value={"all_results": [match]}))
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    results = await handlers._face_task(engine, frame, 10, 0.5, 0.0)

    assert len(results) == 1
    assert "face_image_b64" not in results[0]


@pytest.mark.parametrize(
    "width,height,keep", [(47, 80, False), (80, 47, False), (48, 48, True), (60, 60, True)]
)
def test_analyze_media_allows_48px_faces_through_real_engine_and_adapter(
    monkeypatch, width, height, keep
):
    """Exercise the real route/handler/engine/adapter with a fake IFS transport."""
    search_calls = []

    def respond(request):
        if request.url.path == "/v1/detect":
            return httpx.Response(
                200,
                json={
                    "faces": [
                        {
                            "bbox": {
                                "pixels": {"x": 20, "y": 20, "width": width, "height": height}
                            },
                            "detection_score": 0.99,
                        }
                    ]
                },
            )
        assert request.url.path.endswith("/search")
        search_calls.append(request.read())
        return httpx.Response(
            200,
            json={
                "matches": [
                    {
                        "person": {"id": "test-person", "name": "test-match", "metadata": {}},
                        "matched_face_id": "test-sample",
                        "similarity": 0.9,
                    }
                ]
            },
        )

    engine = FaceEngine()
    with Client(base_url="http://test", transport=httpx.MockTransport(respond)) as client:
        engine._adapter._client = client
        monkeypatch.setattr(
            engine._adapter, "get_face_bbox", lambda *args: {"x": 0, "y": 0, "w": 12, "h": 12}
        )
        monkeypatch.setattr(handlers, "get_face_engine", lambda: engine)
        monkeypatch.setattr(handlers.settings, "insightface_quality_weight", 0.0)
        monkeypatch.setattr(handlers.settings, "insightface_norm_reference", 0.0)
        monkeypatch.setattr(handlers.settings, "insightface_adaptive_threshold_step", 0.0)
        _, image = cv2.imencode(".png", np.zeros((150, 150, 3), dtype=np.uint8))
        monkeypatch.setattr(handlers, "_download_url_safe", AsyncMock(return_value=image.tobytes()))
        monkeypatch.setattr(handlers, "_call_nsfw_analysis", AsyncMock(return_value="safe scene"))
        monkeypatch.setattr(handlers, "_call_ocr_api", AsyncMock(return_value=""))
        monkeypatch.setattr(handlers, "_call_llm_guard", AsyncMock(return_value={"safe": True}))

        with TestClient(create_app()) as api:
            response = api.post(
                "/api/v1/analyze_media", json={"url": "https://example.com/query.png"}
            )

    assert response.status_code == 200, response.text
    assert len(response.json()) == int(keep)
    assert len(search_calls) == int(keep)
    if keep:
        assert response.json()[0]["description"] == "test-match"
        assert b'name="threshold"\r\n\r\n0.5\r\n' in search_calls[0]
