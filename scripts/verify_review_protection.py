"""Verify coverage and terminal model protection against an isolated MySQL queue.

Run inside a candidate image with a fresh wcm_verify_* database and matching
cluster namespace. Frames and model responses are fixtures; no inference runs.
"""

import asyncio
from contextlib import ExitStack
from unittest.mock import AsyncMock, patch

import httpx
import numpy as np

from api import handlers, parameter_store, review_windows, task_queue, worker
from api import review_task_store as store
from api.model_health import model_call
from api.utils import VideoFrame, VideoWindow
from wcm_facerec import runtime_parameters
from wcm_facerec.config import settings


async def verify_case(case):
    primary = [
        VideoWindow((VideoFrame(i, np.full((64, 96, 3), i * 7, np.uint8)),)) for i in range(30)
    ]
    face_attempts = 0
    visual_attempts = 0

    def neighbors(path, timestamps, **kwargs):
        for index, timestamp in enumerate(timestamps):
            yield (
                timestamp,
                VideoFrame(timestamp + 0.04, np.full((65, 97, 3), 101 + index, np.uint8)),
            )

    @model_call("face")
    async def face(engine, image, top_k, threshold, time, *, auxiliary=False, **kwargs):
        nonlocal face_attempts
        if auxiliary and time < 1:
            face_attempts += 1
            if case == "auxiliary-failed" or face_attempts == 1:
                raise httpx.ReadTimeout("fixture auxiliary failure")
        return [{"frame_time": time}] if time == 1 else []

    @model_call("visual")
    async def visual(*args, **kwargs):
        nonlocal visual_attempts
        visual_attempts += 1
        if case == "breaker":
            raise httpx.ReadTimeout("fixture model outage")
        return "ordinary scene"

    with ExitStack() as stack:
        for module, name, value in (
            (
                handlers,
                "_download_review_video",
                AsyncMock(
                    side_effect=httpx.ConnectError("fixture download outage")
                    if case == "transient"
                    else None
                ),
            ),
            (handlers, "get_face_engine", lambda: object()),
            (handlers, "_face_task", face),
            (handlers, "_call_ocr_api", AsyncMock(return_value="")),
            (handlers, "_call_nsfw_analysis", visual),
            (handlers, "_call_llm_guard", AsyncMock(return_value={"safe": True})),
            (review_windows, "_sample_video", lambda *args: iter([30.0, *primary])),
            (review_windows, "iter_video_frames_near", neighbors),
            (
                review_windows,
                "aggregate_face_candidates",
                lambda *args, **kwargs: ([], {"trigger_times": [1]}),
            ),
        ):
            stack.enter_context(patch.object(module, name, value))
        task_id = await store.create(
            "https://fixture.invalid/video.mp4",
            {"sample_interval": 1, "top_k": 10, "threshold": 0.5},
        )
        task = await task_queue.claim("review-protection-verifier")
        assert task and task["id"] == task_id and task["attempts"] == 1
        await asyncio.wait_for(worker.execute(task), 20)
        saved = await store.get(task_id)
        if case in {"auxiliary-failed", "auxiliary-recovered"}:
            assert saved["status"] == "completed", saved
            summary = saved["review_summary"]
            assert summary["total_samples"] == 32, summary
            failed = int(case == "auxiliary-failed")
            assert summary["incomplete_samples"] == failed, summary
            assert summary["incomplete_checks"] == failed, summary
            assert summary["incomplete_ratio"] == failed / 32
            assert face_attempts == 2
            assert saved["progress"]["total_samples"] == 30
            assert saved["progress"]["sub_progress"]["completed"] == 2
        elif case == "breaker":
            assert saved["status"] == "failed", saved
            assert "已提前终止" in saved["error"]
            assert saved["attempt"] == 1
            assert 10 <= visual_attempts <= 12, visual_attempts
            assert saved["results"] is None
        else:
            assert saved["status"] == "queued", saved
        # Make any incorrectly requeued task immediately claimable.
        with store._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "UPDATE review_tasks SET available_at=UTC_TIMESTAMP(3) WHERE id=%s", (task_id,)
            )
        next_task = await task_queue.claim("second-worker")
        if case == "transient":
            assert next_task and next_task["id"] == task_id and next_task["attempts"] == 2
        else:
            assert next_task is None, "Terminal task was claimed again"
        with store._connect() as connection, connection.cursor() as cursor:
            cursor.execute("DELETE FROM review_tasks WHERE id=%s", (task_id,))
    print(f"PASS real queue: {case}", flush=True)


async def main():
    assert settings.review_tasks_db_name.startswith("wcm_verify_")
    assert settings.cluster_namespace == settings.review_tasks_db_name
    assert settings.cluster_enabled and settings.review_max_attempts > 1
    await store.initialize()
    await task_queue.initialize()
    await parameter_store.initialize()
    await parameter_store.close()
    with store._connect() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) AS n FROM review_tasks")
        assert cursor.fetchone()["n"] == 0, "A fresh test database is required"
    runtime_parameters.install(
        {
            "nsfw_review_mode": "window",
            "review_window_concurrency": 1,
            "face_profile_optimization": True,
            "face_max_extra_call_ratio": 0.30,
            "face_max_extra_frames_per_window": 3,
            "face_neighbor_offsets_s": [-0.2, 0.2],
            "face_neighbor_concurrency": 1,
        }
    )
    for case in ("auxiliary-recovered", "auxiliary-failed", "breaker", "transient"):
        await verify_case(case)
    print("ALL REVIEW PROTECTION INTEGRATION CHECKS PASSED", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
