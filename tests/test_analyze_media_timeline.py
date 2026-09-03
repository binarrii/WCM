"""Person intervals use face-task provenance; other findings remain per frame."""

import asyncio
from unittest.mock import AsyncMock

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api import handlers
from api.main import create_app


def frame(time, *people):
    return (
        [{"category": category, "name": name} for category, name in people],
        None,
        None,
        None,
        time,
    )


def test_merge_people_by_category_and_name_despite_out_of_order_completion():
    a = ("person-a", "甲")
    b = ("person-a", "乙")
    c = ("person-b", "甲")
    frames = [frame(7, a, b, c), frame(6, a, a, b), frame(8, a, c), frame(10, a)]
    assert handlers._merge_person_timelines(frames, 1) == [
        {"timestamp": "00:00:06.000~00:00:08.000", "category": "person-a", "description": "甲"},
        {"timestamp": "00:00:06.000~00:00:07.000", "category": "person-a", "description": "乙"},
        {"timestamp": "00:00:07.000~00:00:08.000", "category": "person-b", "description": "甲"},
        {"timestamp": "00:00:10.000", "category": "person-a", "description": "甲"},
    ]


def test_empty_frame_breaks_person_interval_even_within_time_tolerance():
    person = ("person-a", "甲")
    frames = [frame(0, person), frame(0.25), frame(0.5, person)]
    assert [r["timestamp"] for r in handlers._merge_person_timelines(frames, 1)] == [
        "00:00:00.000",
        "00:00:00.500",
    ]


@pytest.mark.parametrize("interval,start,end", [(2, 6, 8), (0.5, 6.5, 7), (1, 6, 7.0005)])
def test_merge_uses_sampling_interval_with_millisecond_tolerance(interval, start, end):
    person = ("person-a", "甲")
    result = handlers._merge_person_timelines([frame(start, person), frame(end, person)], interval)
    assert len(result) == 1
    assert (
        result[0]["timestamp"]
        == f"{handlers._format_timestamp(start)}~{handlers._format_timestamp(end)}"
    )


def test_empty_and_single_image_keep_legacy_point_shape():
    assert handlers._merge_person_timelines([], 1) == []
    assert handlers._merge_person_timelines([frame(0, ("", "甲"))], 1) == [
        {"timestamp": "00:00:00.000", "category": "敏感人物", "description": "甲"}
    ]


@pytest.mark.parametrize("sample_interval", [1, 0.5])
def test_http_video_response_merges_faces_but_not_other_sources(monkeypatch, sample_interval):
    class Video:
        index = 0

        def read(self):
            self.index += 1
            return (
                (True, np.zeros((64, 64, 3), dtype=np.uint8)) if self.index <= 4 else (False, None)
            )

        def get(self, property_id):
            return 1 if property_id == cv2.CAP_PROP_FPS else (self.index - 1) * 1000

        def release(self):
            pass

    async def faces(engine, image, top_k, threshold, time):
        await asyncio.sleep((3 - time) * 0.001)  # Consumers finish out of order.
        return [] if time == 2 else [{"category": "person-a", "name": "review-subject"}]

    monkeypatch.setattr(handlers, "get_face_engine", lambda: object())
    monkeypatch.setattr(handlers.cv2, "VideoCapture", lambda path: Video())
    monkeypatch.setattr(
        handlers, "_download_video_safe_sync", lambda url, path, *a, **k: path.touch()
    )
    monkeypatch.setattr(handlers, "_face_task", faces)
    # Same category/description as a face: provenance, not text matching, controls merging.
    monkeypatch.setattr(handlers, "_call_nsfw_analysis", AsyncMock(return_value="review-subject"))
    monkeypatch.setattr(handlers, "_call_ocr_api", AsyncMock(return_value="review-subject"))
    monkeypatch.setattr(
        handlers, "_call_llm_guard", AsyncMock(return_value={"safe": False, "category": "person-a"})
    )
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/v1/analyze_media",
            json={"url": "https://example.com/video.mp4", "sample_interval": sample_interval},
        )

    assert response.status_code == 200, response.text
    results = response.json()
    assert len(results) == 10  # Two person appearances + eight untouched OCR/visual findings.
    assert [r["timestamp"] for r in results if "~" in r["timestamp"]] == [
        "00:00:00.000~00:00:01.000"
    ]
    for second in range(4):
        item = {
            "timestamp": handlers._format_timestamp(second),
            "category": "person-a",
            "description": "review-subject",
        }
        assert results.count(item) == (3 if second == 3 else 2)
