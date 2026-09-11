"""Whole-window coverage, request reuse, failure isolation and interval contracts."""

import asyncio
from unittest.mock import AsyncMock

import httpx
import numpy as np
import pytest

from api import handlers, review_progress, review_windows
from api.review_coverage import ReviewCoverage
from api.utils import ReviewWindowPlanner, VideoFrame, VideoWindow
from tests.test_nsfw_target_review import caption, image, install_client


def sample(time, value=0, scene=0, selected=True):
    return VideoWindow(
        (VideoFrame(time, np.full((64, 96, 3), value, np.uint8), scene_id=scene),),
        review_visual=selected,
    )


def plan(samples, max_span=10):
    planner = ReviewWindowPlanner(max_span)
    windows = []
    for item in samples:
        ready = planner.push(item)
        if ready:
            windows.append(ready)
    tail = planner.flush()
    if tail:
        windows.append(tail)
    return windows


def test_windows_cover_selected_changes_once_and_end_at_scene_boundary():
    samples = [sample(i, i, selected=i not in {1, 3, 4}) for i in range(8)]
    samples += [sample(8, 50, scene=1), sample(9, 60, scene=1)]
    windows = plan(samples)
    assert [(w.start, w.end) for w in windows] == [(0, 5), (6, 7), (8, 9)]
    assert [[f.timestamp for f in w.frames] for w in windows] == [[0, 2, 5], [6, 7], [8, 9]]
    assert all(len(w.frames) <= 3 and w.end - w.start <= 10 for w in windows)
    assert all({f.scene_id for f in w.frames} == {w.scene_id} for w in windows)


def test_stable_shot_is_bounded_and_new_window_has_its_own_evidence():
    windows = plan([sample(i, selected=i == 0) for i in range(24)])
    assert [(w.start, w.end) for w in windows] == [(0, 10), (11, 21), (22, 23)]
    assert [w.frames[0].timestamp for w in windows] == [0, 11, 22]
    assert plan([]) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["multi_image", "montage"])
async def test_all_images_are_targets_without_a_second_verification(monkeypatch, mode):
    monkeypatch.setattr(handlers.settings, "nsfw_image_mode", mode)
    monkeypatch.setattr(handlers.settings, "nsfw_verify_target", True)
    calls = install_client(monkeypatch, [caption("第三张图片有需要审核的细节")])
    result = await handlers._call_nsfw_analysis(
        [image(0), image(90), image(200)], [0, 1, 2], review_all=True
    )
    assert result == "第三张图片有需要审核的细节"
    assert len(calls) == 1
    payload = calls[0]
    assert payload["max_tokens"] == 300
    assert "500个中文字符" in payload["messages"][0]["content"]
    content = payload["messages"][-1]["content"]
    assert sum(item["type"] == "image_url" for item in content) == (
        3 if mode == "multi_image" else 1
    )
    assert "TARGET" not in str(payload) and "CONTEXT" not in str(payload)
    assert "综合所有图片中的可见证据" in payload["messages"][0]["content"]


@pytest.mark.asyncio
async def test_window_unsupported_images_fall_back_once(monkeypatch):
    monkeypatch.setattr(handlers.settings, "nsfw_image_mode", "auto")
    request = httpx.Request("POST", "https://fixture/model")
    response = httpx.Response(400, request=request, text="Only one image is supported")
    error = httpx.HTTPStatusError("unsupported", request=request, response=response)
    calls = install_client(monkeypatch, [error, caption("第三个面板的可见内容")])
    assert await handlers._call_nsfw_analysis(
        [image(0), image(90), image(200)], [0, 1, 2], review_all=True
    )
    assert len(calls) == 2
    assert "equally important FRAME panels" in str(calls[1])
    assert "TARGET" not in str(calls[1])


@pytest.mark.asyncio
async def test_cache_coalesces_concurrent_successes_but_retries_failures_and_evicts():
    memo = review_windows.AsyncMemo(1)
    operation = AsyncMock(return_value="result")
    assert await asyncio.gather(*(memo.get("a", operation) for _ in range(5))) == ["result"] * 5
    operation.assert_awaited_once()
    failure = AsyncMock(side_effect=ValueError("bad"))
    for _ in range(2):
        with pytest.raises(ValueError):
            await memo.get("b", failure)
    assert failure.await_count == 2
    await memo.get("b", operation)
    await memo.get("a", operation)
    assert operation.await_count == 3  # The older success was evicted.
    await memo.close()
    assert not memo.pending and not memo.values


@pytest.mark.asyncio
async def test_cache_cancellation_cleans_up_in_flight_work():
    memo = review_windows.AsyncMemo()
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def pending():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    task = asyncio.create_task(memo.get("a", pending))
    await started.wait()
    await memo.close()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stopped.is_set() and not memo.pending


def install_video(monkeypatch, samples):
    class Sampler:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return iter(samples)

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(review_windows, "VideoFrameSampler", Sampler)
    monkeypatch.setattr(handlers, "_download_video_safe_sync", lambda *a, **kw: None)
    monkeypatch.setattr(handlers.settings, "nsfw_review_mode", "window")
    monkeypatch.setattr(handlers, "get_face_engine", lambda: object())


@pytest.mark.asyncio
async def test_changed_subtitles_and_new_people_in_later_frames_are_kept_and_reused(monkeypatch):
    install_video(
        monkeypatch, [sample(i, value) for i, value in enumerate([0, 80, 160, 0, 80, 160])]
    )
    face = AsyncMock(
        return_value=[
            {
                "category": "人物",
                "name": "甲",
                "frame_time": -1,
                "face_location": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4},
                "similarity": 0.9,
            }
        ]
    )
    ocr = AsyncMock(return_value="相同字幕")
    visual = AsyncMock(return_value="视觉细节")
    guard = AsyncMock(return_value={"safe": False, "category": "需审核"})
    for name, mock in [
        ("_face_task", face),
        ("_call_ocr_api", ocr),
        ("_call_nsfw_analysis", visual),
        ("_call_llm_guard", guard),
    ]:
        monkeypatch.setattr(handlers, name, mock)
    rows = await handlers._process_analyze_media("https://fixture/video.mp4", 1, 5, 0.5)
    assert face.await_count == ocr.await_count == 3
    assert visual.await_count == 1 and guard.await_count == 2
    assert visual.await_args.args[1] == [0, 1, 2]
    assert len(rows) == 3
    assert {row["timestamp"] for row in rows} == {"00:00:00.000~00:00:05.000"}
    person = next(row for row in rows if row["source"] == "face")
    assert [f["time_ms"] for f in person["face_samples"]] == [0, 1000, 2000, 3000, 4000, 5000]
    assert all(f["bbox"]["x"] == 0.1 for f in person["face_samples"])
    assert [f["pts_seconds"] for f in person["face_samples"]] == [0, 1, 2, 3, 4, 5]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["ocr", "face", "visual"])
async def test_failed_module_keeps_other_findings_and_later_windows(monkeypatch, failure):
    install_video(monkeypatch, [sample(i, i * 30) for i in range(6)])

    async def visual(images, times, **kwargs):
        if failure == "visual" and times[0] == 0:
            raise httpx.ReadTimeout("private error")
        return "later frame detail"

    async def ocr(image):
        if failure == "ocr" and image == sample(1, 30)[0].b64:
            raise httpx.ReadTimeout("private error")
        return "subtitle"

    async def face(engine, image, top_k, threshold, time):
        if failure == "face" and time == 1:
            raise httpx.ReadTimeout("private error")
        return [{"category": "人物", "name": "甲"}]

    monkeypatch.setattr(handlers, "_call_nsfw_analysis", visual)
    monkeypatch.setattr(handlers, "_call_ocr_api", ocr)
    monkeypatch.setattr(handlers, "_face_task", face)
    monkeypatch.setattr(
        handlers, "_call_llm_guard", AsyncMock(return_value={"safe": False, "category": "复核"})
    )
    coverage = ReviewCoverage()
    rows = await asyncio.wait_for(
        handlers._process_analyze_media("https://fixture/video.mp4", 1, 5, 0.5, coverage=coverage),
        2,
    )
    summary = coverage.summarize(rows)
    assert summary["total_samples"] == 6
    assert summary["incomplete_samples"] == (3 if failure == "visual" else 1)
    errors = [r for r in rows if r.get("review_status") == "incomplete"]
    assert len(errors) == 1 and errors[0]["stage"] == failure
    assert errors[0]["timestamp"] == (
        "00:00:00.000~00:00:02.000" if failure == "visual" else "00:00:01.000"
    )
    assert {r["source"] for r in rows if r not in errors} == {"ocr", "visual", "face"}
    assert "private" not in str(rows)


def test_intervals_never_bridge_safe_failed_or_other_shot_windows():
    hit = {"source": "ocr", "category": "复核", "description": "字幕"}
    completed = [
        {
            "index": i,
            "scene_id": scene,
            "start": i * 3,
            "end": i * 3 + 2,
            "hits": [hit] if found else [],
        }
        for i, scene, found in [
            (0, 0, True),
            (1, 0, False),
            (2, 0, True),
            (4, 0, True),
            (5, 1, True),
        ]
    ]
    rows = review_windows.merge_window_results(list(reversed(completed)), 10)
    assert len(rows) == 4
    assert rows[-1]["timestamp"] == "00:00:15.000~00:00:17.000"


def test_adjacent_windows_bridge_sampling_gap_despite_different_descriptions():
    first = {
        "index": 0,
        "scene_id": 4,
        "start": 15,
        "end": 17,
        "hits": [
            {"source": "visual", "category": "复核", "description": "画面甲"},
            {"source": "ocr", "category": "复核", "description": "字幕甲"},
            {"source": "ocr", "category": "复核", "description": "字幕乙"},
            {"source": "face", "category": "人物", "description": "甲"},
        ],
    }
    second = {
        "index": 1,
        "scene_id": 4,
        "start": 18,
        "end": 19,
        "hits": [
            {"source": "visual", "category": "复核", "description": "画面乙"},
            {"source": "ocr", "category": "复核", "description": "字幕丙"},
            {"source": "face", "category": "人物", "description": "乙"},
        ],
    }
    rows = review_windows.merge_window_results([second, first], 1.1)
    continuous = [r for r in rows if r["source"] != "face"]
    assert len(continuous) == 2
    assert all(r["timestamp"] == "00:00:15.000~00:00:19.000" for r in continuous)
    ocr = next(r for r in continuous if r["source"] == "ocr")
    assert ocr["description"] == "字幕甲\n\n字幕乙\n\n字幕丙"
    assert [e["timestamp"] for e in ocr["evidence"]] == [
        "00:00:15.000~00:00:17.000",
        "00:00:15.000~00:00:17.000",
        "00:00:18.000~00:00:19.000",
    ]
    assert len([r for r in rows if r["source"] == "face"]) == 2
    for change in ({"scene_id": 5}, {"index": 2}, {"start": 19}):
        assert len(review_windows.merge_window_results([first, {**second, **change}], 1.1)) == 6


def test_other_categories_or_sources_cannot_bridge_a_safe_category_window():
    completed = [
        {
            "index": i,
            "scene_id": 0,
            "start": i * 3,
            "end": i * 3 + 2,
            "hits": [{"source": source, "category": category, "description": str(i)}],
        }
        for i, source, category in [
            (0, "ocr", "甲"),
            (1, "visual", "甲"),
            (2, "ocr", "甲"),
            (3, "ocr", "乙"),
        ]
    ]
    assert len(review_windows.merge_window_results(completed, 1.1)) == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["_process_detect_nsfw", "_process_detect_sensitive"])
async def test_standalone_video_returns_ranges_and_empty_ocr_skips_guard(monkeypatch, endpoint):
    install_video(monkeypatch, [sample(i, i * 50) for i in range(3)])
    monkeypatch.setattr(handlers, "_call_ocr_api", AsyncMock(return_value=""))
    monkeypatch.setattr(handlers, "_call_nsfw_analysis", AsyncMock(return_value="可见细节"))
    guard = AsyncMock(return_value={"safe": False, "category": "复核"})
    monkeypatch.setattr(handlers, "_call_llm_guard", guard)
    result = await getattr(handlers, endpoint)("https://fixture/video.mp4", 1)
    assert result["unsafe_text_frames"] == []
    if endpoint == "_process_detect_nsfw":
        assert result["visual_analysis"][0]["timestamp"] == "00:00:00.000~00:00:02.000"
        guard.assert_awaited_once_with("可见细节")
    else:
        guard.assert_not_awaited()


@pytest.mark.asyncio
async def test_only_third_frame_has_subtitle_and_person_both_cover_the_window(monkeypatch):
    install_video(monkeypatch, [sample(i, i * 80) for i in range(3)])
    third = sample(2, 160)[0].b64

    async def ocr(image):
        return "变化后的字幕" if image == third else ""

    async def face(engine, image, top_k, threshold, time):
        return [{"name": "新入镜人物"}] if time == 2 else []

    monkeypatch.setattr(handlers, "_call_ocr_api", ocr)
    monkeypatch.setattr(handlers, "_face_task", face)
    monkeypatch.setattr(handlers, "_call_nsfw_analysis", AsyncMock(return_value="普通画面"))
    monkeypatch.setattr(
        handlers,
        "_call_llm_guard",
        AsyncMock(side_effect=lambda text: {"safe": text == "普通画面", "category": "复核"}),
    )
    rows = await handlers._process_analyze_media("https://fixture/video.mp4", 1, 5, 0.5)
    assert {row["description"] for row in rows} == {"变化后的字幕", "新入镜人物"}
    assert all(row["timestamp"] == "00:00:00.000~00:00:02.000" for row in rows)


@pytest.mark.asyncio
async def test_difficult_face_adds_only_budgeted_neighbor_frame(monkeypatch):
    install_video(monkeypatch, [sample(i, i * 30) for i in range(3)])
    monkeypatch.setattr(handlers.settings, "face_profile_optimization", True)
    monkeypatch.setattr(handlers.settings, "face_max_extra_call_ratio", 0.30)
    monkeypatch.setattr(handlers.settings, "face_max_extra_frames_per_window", 3)
    monkeypatch.setattr(handlers.settings, "face_neighbor_offsets_s", (-0.4, -0.2, 0.2, 0.4))
    requested = []

    def neighbors(path, timestamps, **kwargs):
        requested.extend(timestamps)
        return [
            (
                timestamps[0],
                VideoFrame(
                    0.8,
                    np.full((64, 96, 3), 99, np.uint8),
                    sampled=False,
                    frame_index=24,
                ),
            )
        ]

    monkeypatch.setattr(review_windows, "read_video_frames_near", neighbors)
    calls = []

    async def face(
        engine,
        image,
        top_k,
        threshold,
        time,
        *,
        profile_optimization=False,
        auxiliary=False,
    ):
        calls.append((time, profile_optimization, auxiliary))
        if time not in {0.8, 1}:
            return handlers.FaceFrameResult([], observations=[])
        pose = 0.8 if auxiliary else 0.3
        similarity = 0.72 if auxiliary else 0.68
        bbox = {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}
        record = {
            "person_id": "p1",
            "name": "侧脸人物",
            "category": "人物",
            "similarity": similarity,
            "face_index": 0,
            "query_face_bbox": {"x": 10, "y": 10, "w": 50, "h": 50},
            "query_quality": {"score": 0.8, "sharpness": 0.7, "pose": pose},
            "face_location": bbox,
        }
        observation = {
            "frame_time": time,
            "face_index": 0,
            "query_quality": record["query_quality"],
            "face_location": bbox,
        }
        return handlers.FaceFrameResult([record], observations=[observation])

    monkeypatch.setattr(handlers, "_face_task", face)
    monkeypatch.setattr(handlers, "_call_ocr_api", AsyncMock(return_value=""))
    monkeypatch.setattr(handlers, "_call_nsfw_analysis", AsyncMock(return_value="普通画面"))
    monkeypatch.setattr(handlers, "_call_llm_guard", AsyncMock(return_value={"safe": True}))
    snapshots = []

    async def update_progress(task_id, data):
        snapshots.append(data)

    monkeypatch.setattr(review_progress.review_task_store, "update_progress", update_progress)
    progress = review_progress.ReviewProgress("task", interval=0)

    rows = await handlers._process_analyze_media(
        "https://fixture/video.mp4", 1, 5, 0.5, progress=progress
    )

    # Three primary samples allow ceil(3 * 0.30) == one auxiliary call.
    assert requested == [0.8]
    assert len(calls) == 4
    assert calls[-1] == (0.8, True, True)
    person = next(row for row in rows if row.get("source") == "face")
    assert person["recognition_status"] == "confirmed"
    assert person["evidence_count"] == 2
    assert any(sample.get("auxiliary") for sample in person["face_samples"])
    resampling = [item for item in snapshots if item["phase"] == "resampling"]
    assert resampling[0]["sub_progress"]["completed"] == 0
    assert resampling[-1]["sub_progress"] == {
        "stage": "face_resampling",
        "completed": 1,
        "total": 1,
        "percent": 100.0,
    }


@pytest.mark.asyncio
async def test_cancelling_window_pipeline_releases_consumers_and_model_calls(monkeypatch):
    install_video(monkeypatch, [sample(i, i % 255) for i in range(120)])
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def visual(*args, **kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    monkeypatch.setattr(handlers, "_call_nsfw_analysis", visual)
    monkeypatch.setattr(handlers, "_call_ocr_api", AsyncMock(return_value=""))
    before = asyncio.all_tasks()
    task = asyncio.create_task(handlers._process_detect_nsfw("https://fixture/video.mp4", 1))
    await asyncio.wait_for(started.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 1)
    assert stopped.is_set()
    assert not [t for t in asyncio.all_tasks() - before if not t.done()]
