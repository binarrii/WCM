"""All video tasks share window anchors, including short videos and EOF tails."""

import asyncio
import base64
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import cv2
import numpy as np
import pytest

from api import handlers, utils


class Capture:
    def __init__(self, count, fps=2, fail_at=None):
        self.count, self.fps, self.fail_at = count, fps, fail_at
        self.index = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        return self.fps if prop == cv2.CAP_PROP_FPS else 0.0

    def read(self):
        if self.index == self.fail_at:
            raise RuntimeError("decode failed")
        if self.index >= self.count:
            return False, None
        image = np.full((64, 64, 3), self.index, dtype=np.uint8)
        self.index += 1
        return True, image

    def release(self):
        self.released = True


@pytest.mark.asyncio
async def test_model_failure_fails_review_and_releases_capture(monkeypatch):
    cap = Capture(2)
    monkeypatch.setattr(utils.cv2, "VideoCapture", lambda _: cap)
    monkeypatch.setattr(
        handlers, "_download_video_safe_sync", lambda url, path, *a, **kw: path.touch()
    )
    monkeypatch.setattr(handlers, "get_face_engine", lambda: object())
    monkeypatch.setattr(handlers, "_face_task", AsyncMock(return_value=[]))
    monkeypatch.setattr(handlers, "_call_ocr_api", AsyncMock(return_value=""))
    guard = AsyncMock(return_value={"safe": True})
    monkeypatch.setattr(handlers, "_call_llm_guard", guard)
    monkeypatch.setattr(
        handlers,
        "_call_nsfw_analysis",
        AsyncMock(side_effect=handlers.NsfwAnalysisError("offline")),
    )
    with pytest.raises(handlers.NsfwAnalysisError, match="offline"):
        await handlers._process_analyze_media("http://test/video.mp4", 0.5, 5, 0.5)
    guard.assert_not_called()
    assert cap.released


@pytest.mark.asyncio
async def test_cancellation_releases_capture_and_consumers(monkeypatch):
    cap = Capture(50)
    monkeypatch.setattr(utils.cv2, "VideoCapture", lambda _: cap)
    monkeypatch.setattr(
        handlers, "_download_video_safe_sync", lambda url, path, *a, **kw: path.touch()
    )
    monkeypatch.setattr(handlers, "get_face_engine", lambda: object())
    monkeypatch.setattr(handlers, "_face_task", AsyncMock(return_value=[]))
    monkeypatch.setattr(handlers, "_call_ocr_api", AsyncMock(return_value=""))
    started = asyncio.Event()

    async def wait_for_model(*args):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(handlers, "_call_nsfw_analysis", wait_for_model)
    task = asyncio.create_task(
        handlers._process_analyze_media("http://test/video.mp4", 0.5, 5, 0.5)
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cap.released
    assert not [t for t in asyncio.all_tasks() if t.get_coro().__qualname__.endswith(".consumer")]


@pytest.mark.parametrize("count", range(8))
@pytest.mark.parametrize("interval", [0, 0.5, 1, 2])
def test_window_anchors_cover_every_sample_once_without_padding(monkeypatch, count, interval):
    cap = Capture(count)
    monkeypatch.setattr(utils.cv2, "VideoCapture", lambda _: cap)
    with utils.VideoFrameSampler(Path("unused"), interval) as sampler:
        windows = list(sampler)
        assert sampler.frames_read == count
        assert sampler.interval == interval
    sampled = list(range(0, count, max(int(2 * interval), 1)))
    assert [[int(f.image[0, 0, 0]) for f in w] for w in windows] == [
        sampled[i : i + 3] for i in range(len(sampled))
    ]
    assert [w[0].timestamp for w in windows] == [i / 2 for i in sampled]
    assert cap.released


def test_waits_for_three_frames_and_shares_context_encoding(monkeypatch):
    cap = Capture(5)
    monkeypatch.setattr(utils.cv2, "VideoCapture", lambda _: cap)
    with utils.VideoFrameSampler(Path("unused"), 0.5) as sampler:
        iterator = iter(sampler)
        first = next(iterator)
        assert cap.index == 3
        second = next(iterator)
        assert cap.index == 4
        assert first[1] is second[0]
        assert first[1].b64 is second[0].b64


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf")])
def test_invalid_fps_uses_common_fallback(monkeypatch, fps):
    cap = Capture(26, fps)
    monkeypatch.setattr(utils.cv2, "VideoCapture", lambda _: cap)
    with utils.VideoFrameSampler(Path("unused"), 1) as sampler:
        assert [w[0].timestamp for w in sampler] == [0, 1]


def test_capture_released_on_decode_error(monkeypatch):
    cap = Capture(8, fail_at=3)
    monkeypatch.setattr(utils.cv2, "VideoCapture", lambda _: cap)
    with pytest.raises(RuntimeError, match="decode failed"):
        with utils.VideoFrameSampler(Path("unused"), 0) as sampler:
            list(sampler)
    assert cap.released


def pixel(b64):
    frame = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
    return int(frame[0, 0, 0])


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [0, 1, 2, 3, 4, 5])
async def test_combined_and_standalone_tasks_use_identical_windows(monkeypatch, count, tmp_path):
    captures = []

    def capture(_):
        cap = Capture(count)
        captures.append(cap)
        return cap

    monkeypatch.setattr(utils.cv2, "VideoCapture", capture)
    monkeypatch.setattr(
        handlers, "_download_video_safe_sync", lambda url, path, *a, **kw: path.touch()
    )
    nsfw_calls, ocr_calls, face_calls = [], [], []

    async def nsfw(images, times):
        nsfw_calls.append(([pixel(image) for image in images], times))
        return "scene"

    async def ocr(image):
        ocr_calls.append(pixel(image))
        return ""

    async def face(engine, image, top_k, threshold, ts):
        face_calls.append((int(image[0, 0, 0]), ts))
        return []

    async def search(**kwargs):
        face_calls.append((int(kwargs["img_source"][0, 0, 0]), None))
        return []

    engine = SimpleNamespace(search=search)
    monkeypatch.setattr(handlers, "get_face_engine", lambda: engine)
    monkeypatch.setattr(handlers, "_call_nsfw_analysis", nsfw)
    monkeypatch.setattr(handlers, "_call_ocr_api", ocr)
    monkeypatch.setattr(handlers, "_face_task", face)
    monkeypatch.setattr(handlers, "_call_llm_guard", AsyncMock(return_value={"safe": True}))
    expected = [
        (list(range(i, min(i + 3, count))), [j / 2 for j in range(i, min(i + 3, count))])
        for i in range(count)
    ]
    await handlers._process_analyze_media("http://test/video.mp4", 0.5, 5, 0.5)
    assert sorted(nsfw_calls) == expected
    assert sorted(ocr_calls) == list(range(count))
    assert sorted(face_calls) == [(i, i / 2) for i in range(count)]
    nsfw_calls.clear()
    ocr_calls.clear()
    face_calls.clear()
    await handlers._process_detect_nsfw("http://test/video.mp4", 0.5)
    assert nsfw_calls == expected
    assert ocr_calls == list(range(count))
    ocr_calls.clear()
    await handlers._process_detect_sensitive("http://test/video.mp4", 0.5)
    assert ocr_calls == list(range(count))
    frames, _ = await handlers._search_video_frames(
        engine, "unused", None, 5, 0.5, 0.5, local_video_path=tmp_path / "video.mp4"
    )
    assert frames == count
    assert face_calls == [(i, None) for i in range(count)]
    assert all(cap.released for cap in captures)


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [1, 2, 3])
async def test_nsfw_request_sends_one_contact_sheet_with_context_prompt(monkeypatch, count):
    captured = []

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, **kwargs):
            captured.append(kwargs["json"])
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {
                    "choices": [
                        {
                            "message": {
                                "content": "普通场景"
                                if len(captured) == (2 if count > 1 else 1)
                                else "后续画面线索"
                            }
                        }
                    ]
                },
            )

    monkeypatch.setattr(handlers.httpx, "AsyncClient", Client)
    images = [encoded_frame((80, 120), value) for value in (60, 130, 220)][:count]
    assert await handlers._call_nsfw_analysis(images, [i / 2 for i in range(count)]) == "普通场景"
    assert len(captured) == (2 if count > 1 else 1)
    for payload in captured:
        assert payload["temperature"] == 0
        assert len([x for x in payload["messages"][-1]["content"] if x["type"] == "image_url"]) == 1
    final_content = captured[-1]["messages"][-1]["content"]
    final_image = [x["image_url"]["url"] for x in final_content if x["type"] == "image_url"][
        0
    ].split(",", 1)[1]
    # The verification request contains only the original target scene.
    assert decode_frame(final_image).shape == (80, 120, 3)
    assert np.all(abs(decode_frame(final_image).astype(int) - 60) <= 2)
    if count > 1:
        prompt = final_content[0]["text"]
        assert "后续画面线索" not in prompt
        assert len(captured[-1]["messages"]) == 2
        assert all("后续画面线索" not in str(message) for message in captured[-1]["messages"])
        first_prompt = captured[0]["messages"][1]["content"][0]["text"]
        assert (
            "TARGET" in first_prompt and "CONTEXT" in first_prompt and "scene cut" in first_prompt
        )


def encoded_frame(shape, value):
    ok, encoded = cv2.imencode(".png", np.full((*shape, 3), value, dtype=np.uint8))
    assert ok
    return base64.b64encode(encoded).decode()


def decode_frame(encoded):
    return cv2.imdecode(np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_COLOR)


def scene_pixels(sheet, value):
    # Ignore isolated text antialiasing pixels with the same grey value.
    mask = np.all(abs(sheet.astype(int) - value) < 4, axis=2).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    assert count > 1
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return np.argwhere(labels == largest)


@pytest.mark.parametrize("count", [2, 3])
@pytest.mark.parametrize("shape", [(400, 800), (800, 400), (1600, 1600)])
def test_contact_sheet_preserves_aspect_ratio_order_and_bounds(count, shape):
    values = (60, 130, 220)[:count]
    images = [encoded_frame(shape, value) for value in values]
    encoded, layout = handlers._compose_nsfw_frames(images, None)
    sheet = decode_frame(encoded)
    assert max(sheet.shape[:2]) <= 3 * (1024 + 36 + 8)
    positions = []
    for value in values:
        pixels = scene_pixels(sheet, value)
        assert len(pixels) > 0
        h, w = np.ptp(pixels, axis=0) + 1
        assert w / h == pytest.approx(shape[1] / shape[0], rel=0.03)
        positions.append(pixels.mean(axis=0))
    target_area = len(scene_pixels(sheet, values[0]))
    assert all(target_area > 3.5 * len(scene_pixels(sheet, value)) for value in values[1:])
    axis = 0 if shape[1] >= shape[0] else 1
    assert all(positions[0][axis] < point[axis] for point in positions[1:])
    if count == 3:
        assert positions[1][1 - axis] < positions[2][1 - axis]
    assert "TARGET 1" in layout and "CONTEXT" in layout


def test_contact_sheet_accepts_different_frame_sizes_without_cropping():
    images = [encoded_frame((80, 120), 60), encoded_frame((160, 60), 130)]
    encoded, _ = handlers._compose_nsfw_frames(images, [0, 0.5])
    sheet = decode_frame(encoded)
    for shape, value in [((80, 120), 60), ((160, 60), 130)]:
        pixels = scene_pixels(sheet, value)
        actual_h, actual_w = np.ptp(pixels, axis=0) + 1
        assert actual_w / actual_h == pytest.approx(shape[1] / shape[0], rel=0.12)
        assert actual_w <= shape[1] and actual_h <= shape[0]


@pytest.mark.asyncio
async def test_corrupt_context_frame_fails_instead_of_sending_incomplete_window(monkeypatch):
    client = AsyncMock()
    monkeypatch.setattr(handlers.httpx, "AsyncClient", client)
    with pytest.raises(handlers.NsfwAnalysisError, match="拼接失败"):
        await handlers._call_nsfw_analysis([encoded_frame((80, 120), 60), "broken"], [0, 1])
    client.assert_not_called()
