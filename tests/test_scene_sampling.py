"""Scene boundaries and visual deduplication preserve fixed face/OCR coverage."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import cv2
import httpx
import numpy as np
import pytest

from api import handlers, utils


class SceneCapture:
    def __init__(self, frames, times):
        self.frames, self.times = frames, times
        self.index = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        return 25 if prop == cv2.CAP_PROP_FPS else self.times[self.index - 1] * 1000

    def read(self):
        if self.index == len(self.frames):
            return False, None
        frame = self.frames[self.index].copy()
        self.index += 1
        return True, frame

    def release(self):
        self.released = True


def solid(value):
    return np.full((90, 160, 3), value, np.uint8)


def sample(monkeypatch, frames, times, interval=1, **kwargs):
    cap = SceneCapture(frames, times)
    monkeypatch.setattr(utils.cv2, "VideoCapture", lambda _: cap)
    with utils.VideoFrameSampler(
        Path("fixture"), interval, sampling_mode="scene", **kwargs
    ) as sampler:
        windows = list(sampler)
    assert cap.released
    assert all(1 <= len(window) <= 3 for window in windows)
    assert all(len({frame.scene_id for frame in window}) == 1 for window in windows)
    return windows, sampler


@pytest.mark.parametrize("count", [0, 1, 2, 3, 10])
def test_static_scene_keeps_grid_and_bounds_visual_stride(monkeypatch, count):
    windows, sampler = sample(monkeypatch, [solid(100)] * count, list(range(count)))
    assert [w[0].timestamp for w in windows if w.sampled] == list(range(count))
    targets = [w[0].timestamp for w in windows if w.review_visual]
    if count:
        assert targets[0] == 0 and targets[-1] == count - 1
    assert all(b - a <= 3 for a, b in zip(targets, targets[1:]))
    if count == 10:
        assert targets == [0, 3, 6, 9]
        assert sampler.visual_skipped == 6


def test_short_shot_between_grid_points_is_retained_without_extra_long_shot_targets(monkeypatch):
    times = [0, 0.4, 0.5, 0.6, 0.7, 1, 1.4]
    frames = [solid(v) for v in [20, 20, 200, 200, 20, 20, 20]]
    windows, sampler = sample(monkeypatch, frames, times)
    assert [w[0].timestamp for w in windows if w.sampled] == [0, 1]
    assert [w[0].timestamp for w in windows if w.review_visual] == [0, 0.5, 1]
    assert sampler.scene_cuts == 2
    assert [[f.timestamp for f in w] for w in windows] == [[0], [0.5], [1]]
    assert sampler.boundary_samples == 1


def test_adjacent_single_frame_shots_never_mix_context(monkeypatch):
    windows, sampler = sample(
        monkeypatch, [solid(v) for v in [20, 200, 20, 200]], [0, 0.1, 0.2, 0.3]
    )
    assert [len(w) for w in windows] == [1, 1, 1, 1]
    assert all(w.review_visual for w in windows)
    assert sampler.scene_cuts == 3


def test_local_change_is_retained_despite_almost_identical_background(monkeypatch):
    background = solid(100)
    change = background.copy()
    change[40:50, 70:80] = 220
    windows, sampler = sample(
        monkeypatch, [background, change, background, background], [0, 1, 2, 3]
    )
    assert sampler.scene_cuts == 0
    assert [w[0].timestamp for w in windows if w.review_visual] == [0, 1, 2, 3]


def test_slow_drift_compares_against_last_reviewed_target(monkeypatch):
    windows, _ = sample(monkeypatch, [solid(v) for v in range(100, 107)], list(range(7)))
    assert [w[0].timestamp for w in windows if w.review_visual] == [0, 2, 4, 6]


def test_strong_color_cut_without_brightness_change_is_detected(monkeypatch):
    before, after = solid(0), solid(0)
    before[:, :, 2] = 255
    after[:, :, 0] = 255
    windows, sampler = sample(monkeypatch, [before, after], [0, 1])
    assert sampler.scene_cuts == 1
    assert [len(w) for w in windows] == [1, 1]


def test_vfr_scene_targets_keep_actual_pts_and_no_duplicate_tail(monkeypatch):
    windows, _ = sample(monkeypatch, [solid(100)] * 4, [0, 1.02, 3.12, 3.4])
    assert [w[0].timestamp for w in windows] == [0, 1.02, 3.12]
    assert [w[0].timestamp for w in windows if w.sampled] == [0, 1.02, 3.12]
    assert [w[0].timestamp for w in windows if w.review_visual] == [0, 3.12]


def test_dark_compression_noise_does_not_create_false_shots(monkeypatch):
    rng = np.random.default_rng(42)
    frames = [rng.integers(0, 5, (90, 160, 3), np.uint8) for _ in range(5)]
    _, sampler = sample(monkeypatch, frames, list(range(5)))
    assert sampler.scene_cuts == 0


def test_sampling_choices_are_independent_of_model_image_resize(monkeypatch):
    rng = np.random.default_rng(7)
    frames = [rng.integers(0, 255, (320, 480, 3), np.uint8) for _ in range(4)]
    signatures = []
    for limit in [None, 108]:
        windows, _ = sample(monkeypatch, frames, [0, 0.5, 1, 1.5], max_dimension=limit)
        signatures.append(
            [(tuple(f.timestamp for f in w), w.review_visual, w.sampled) for w in windows]
        )
    assert signatures[0] == signatures[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_visual", [False, True])
async def test_combined_and_standalone_keep_same_visual_targets_and_fixed_ocr(
    monkeypatch, fail_visual
):
    monkeypatch.setattr(handlers.settings, "nsfw_sampling_mode", "scene")
    times = [0, 0.4, 0.5, 0.6, 0.7, 1, 1.4]
    frames = [solid(v) for v in [20, 20, 200, 200, 20, 20, 20]]
    captures = []

    def capture(_):
        cap = SceneCapture(frames, times)
        captures.append(cap)
        return cap

    monkeypatch.setattr(utils.cv2, "VideoCapture", capture)
    monkeypatch.setattr(handlers, "_download_video_safe_sync", lambda *a, **kw: None)
    monkeypatch.setattr(handlers, "get_face_engine", lambda: object())
    face_times, visual_calls, ocr_values = [], [], []

    async def face(engine, frame, top_k, threshold, timestamp):
        face_times.append(timestamp)
        return [{"category": "人物", "name": "person"}]

    async def visual(images, timestamps):
        visual_calls.append(timestamps)
        if fail_visual and timestamps[0] == 0.5:
            raise httpx.ReadTimeout("fixture failure")
        return "visual"

    async def ocr(image):
        ocr_values.append(image)
        return ""

    monkeypatch.setattr(handlers, "_face_task", face)
    monkeypatch.setattr(handlers, "_call_nsfw_analysis", visual)
    monkeypatch.setattr(handlers, "_call_ocr_api", ocr)
    monkeypatch.setattr(handlers, "_call_llm_guard", AsyncMock(return_value={"safe": True}))
    combined = await handlers._process_analyze_media("http://fixture/video.mp4", 1, 5, 0.5)
    expected_visual = sorted(visual_calls)
    assert [times[0] for times in expected_visual] == [0, 0.5, 1]
    assert sorted(face_times) == [0, 1]
    assert len(ocr_values) == 2
    # Visual-only boundaries must not split the original face timeline.
    assert [r["timestamp"] for r in combined if r.get("category") == "人物"] == [
        "00:00:00.000~00:00:01.000"
    ]
    gaps = [r for r in combined if r.get("review_status") == "incomplete"]
    assert len(gaps) == int(fail_visual)
    if gaps:
        assert gaps[0]["timestamp"] == "00:00:00.500"
    visual_calls.clear()
    ocr_values.clear()
    standalone = await handlers._process_detect_nsfw("http://fixture/video.mp4", 1)
    assert sorted(visual_calls) == expected_visual
    assert len(ocr_values) == 2
    assert len(standalone.get("errors", [])) == int(fail_visual)
    assert all(cap.released for cap in captures)


@pytest.mark.asyncio
async def test_cancellation_in_scene_mode_closes_sampler_and_consumers(monkeypatch):
    monkeypatch.setattr(handlers.settings, "nsfw_sampling_mode", "scene")
    cap = SceneCapture([solid(100)] * 100, list(range(100)))
    monkeypatch.setattr(utils.cv2, "VideoCapture", lambda _: cap)
    monkeypatch.setattr(handlers, "_download_video_safe_sync", lambda *a, **kw: None)
    monkeypatch.setattr(handlers, "get_face_engine", lambda: object())
    monkeypatch.setattr(handlers, "_face_task", AsyncMock(return_value=[]))
    monkeypatch.setattr(handlers, "_call_ocr_api", AsyncMock(return_value=""))
    started = asyncio.Event()

    async def wait(*args):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(handlers, "_call_nsfw_analysis", wait)
    task = asyncio.create_task(
        handlers._process_analyze_media("http://fixture/video.mp4", 1, 5, 0.5)
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cap.released
    assert not [t for t in asyncio.all_tasks() if t.get_coro().__qualname__.endswith(".consumer")]


@pytest.fixture(autouse=True)
def legacy_target_review_mode(monkeypatch):
    """Keep regression coverage of the configurable target-mode fallback."""
    monkeypatch.setattr(handlers.settings, "nsfw_review_mode", "target")
