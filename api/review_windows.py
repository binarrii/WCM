"""Whole-window review with task-local request reuse and explicit time spans."""

import asyncio
import hashlib
import logging
import math
import os
from collections import OrderedDict
from pathlib import Path

from wcm_facerec.config import settings

from . import handlers
from .face_optimization import aggregate_face_candidates, observation_is_difficult
from .model_health import gather_stages, protect_video_review
from .utils import ReviewWindowPlanner, VideoFrameSampler, read_video_frames_near

logger = logging.getLogger(__name__)


class AsyncMemo:
    """Bounded success cache; coalesce in-flight calls but never cache failures."""

    def __init__(self, capacity=256):
        self.capacity = capacity
        self.values = OrderedDict()
        self.pending = {}
        self.hits = self.misses = 0

    async def get(self, key, operation):
        if key in self.values:
            self.hits += 1
            self.values.move_to_end(key)
            return self.values[key]
        task = self.pending.get(key)
        if task is None:
            self.misses += 1
            task = asyncio.create_task(operation())
            self.pending[key] = task
        else:
            self.hits += 1
        try:
            value = await task
        except BaseException:
            if self.pending.get(key) is task:
                self.pending.pop(key)
            raise
        if self.pending.get(key) is task:
            self.pending.pop(key)
            self.values[key] = value
            if len(self.values) > self.capacity:
                self.values.popitem(last=False)
        return value

    async def close(self):
        tasks = list(self.pending.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.pending.clear()
        self.values.clear()


def _digest(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).digest()


def merge_window_results(completed, max_gap):
    """Join adjacent same-shot category hits; keep person identity and all evidence."""
    rows, previous = [], {}
    for result in sorted(completed, key=lambda item: item["index"]):
        current, grouped = {}, {}
        for hit in result["hits"]:
            # OCR wording and visual descriptions may change within one shot.
            # Different people must never inherit each other's continuous span.
            key = (
                hit["source"],
                hit["category"],
                hit["description"] if hit["source"] == "face" else None,
            )
            grouped.setdefault(key, []).append(hit)
        for key, hits in grouped.items():
            prior = previous.get(key)
            if (
                prior is not None
                and prior["index"] + 1 == result["index"]
                and prior["scene_id"] == result["scene_id"]
                and 0 <= result["start"] - prior["end"] <= max_gap + 1e-9
            ):
                row, start = prior["row"], prior["start"]
                if "evidence" not in row:
                    row["evidence"] = [{k: v for k, v in row.items() if k != "face_samples"}]
            else:
                start = result["start"]
                row = {
                    **hits[0],
                    "timestamp": handlers._format_interval(start, result["end"]),
                    "scene_id": result["scene_id"],
                }
                if "face_samples" in row:
                    row["face_samples"] = list(row["face_samples"])
                if len(hits) > 1:
                    row["evidence"] = []
                rows.append(row)
            if "evidence" in row:
                for hit in hits:
                    evidence = {k: v for k, v in hit.items() if k != "face_samples"}
                    evidence["timestamp"] = handlers._format_interval(
                        result["start"], result["end"]
                    )
                    if evidence not in row["evidence"]:
                        row["evidence"].append(evidence)
                row["description"] = "\n\n".join(
                    dict.fromkeys(e["description"] for e in row["evidence"])
                )
            if row.get("guard_verdict") == "controversial" or any(
                hit.get("guard_verdict") == "controversial" for hit in hits
            ):
                row["guard_verdict"] = "controversial"
            row["timestamp"] = handlers._format_interval(start, result["end"])
            for hit in hits:
                for sample in hit.get("face_samples", []):
                    if sample not in row.setdefault("face_samples", []):
                        row["face_samples"].append(sample)
            current[key] = {
                "index": result["index"],
                "scene_id": result["scene_id"],
                "start": start,
                "end": result["end"],
                "row": row,
            }
        # A safe/missing category, failed window, or cut breaks continuity.
        previous = current
    return sorted(rows, key=lambda row: row["timestamp"].split("~", 1)[0])


@protect_video_review
async def analyze_video(
    url,
    sample_interval,
    top_k=10,
    threshold=0.5,
    *,
    include_faces=False,
    include_visual=True,
    coverage=None,
    progress=None,
):
    engine = handlers.get_face_engine() if include_faces else None
    ocr_cache, face_cache, guard_cache = AsyncMemo(), AsyncMemo(128), AsyncMemo(512)
    visual_cache = AsyncMemo(64)
    errors, completed = [], []
    path = Path(f"/tmp/window_review_{os.urandom(8).hex()}.mp4")
    concurrency = settings.review_window_concurrency
    queue = asyncio.Queue(maxsize=concurrency * 2)
    planner = ReviewWindowPlanner(settings.nsfw_window_max_seconds)
    selected_frames = 0
    confirm_similarity = 0.0 if threshold <= 0 else max(0.0, 1.0 - float(threshold))
    if coverage is not None:
        coverage.add([])

    async def guard(text):
        return await guard_cache.get(_digest(text), lambda: handlers._call_llm_guard(text))

    async def process_window(window):
        hits = {}
        face_records = []
        face_observations = []

        def add(source, category, description, *, guard_verdict=None):
            key = (source, category, description)
            hit = hits.setdefault(
                key, {"source": source, "category": category, "description": description}
            )
            if guard_verdict:
                hit["guard_verdict"] = guard_verdict
            return hit

        async def visual():
            if not include_visual:
                return
            if progress is not None:
                await progress.start_stage(
                    window.index, "visual", [frame.timestamp for frame in window.frames]
                )

            async def operation():
                images = [frame.b64 for frame in window.frames]
                # Only visible content is described; timestamps are annotations.
                # Preserve image order, and assign this window's own span below.
                return await visual_cache.get(
                    tuple(_digest(image) for image in images),
                    lambda: handlers._review_visual(
                        images,
                        [frame.timestamp for frame in window.frames],
                        review_all=True,
                        guard_call=guard,
                    ),
                )

            result = await handlers._review_stage(
                "visual",
                window.start,
                operation,
                errors,
                end_timestamp=window.end,
            )
            if result:
                add(
                    "visual",
                    result["category"],
                    result["text"],
                    guard_verdict=result.get("guard_verdict"),
                )
            if progress is not None:
                progress.finish_stage(window.index, "visual")

        async def text():
            for frame in window.frames:
                if progress is not None:
                    await progress.start_stage(window.index, "ocr", [frame.timestamp])

                async def operation(frame=frame):
                    content = await ocr_cache.get(
                        _digest(frame.b64), lambda: handlers._call_ocr_api(frame.b64)
                    )
                    if content:
                        verdict = await guard(content)
                        if not verdict["safe"]:
                            finding = {
                                "text": content,
                                "category": verdict.get("category", "文本违规"),
                            }
                            if verdict.get("guard_verdict"):
                                finding["guard_verdict"] = verdict["guard_verdict"]
                            return finding
                    return None

                result = await handlers._review_stage("ocr", frame.timestamp, operation, errors)
                if result:
                    add(
                        "ocr",
                        result["category"],
                        result["text"],
                        guard_verdict=result.get("guard_verdict"),
                    )
            if progress is not None:
                progress.finish_stage(window.index, "ocr")

        async def faces():
            if not include_faces:
                return
            for frame in window.frames:
                if progress is not None:
                    await progress.start_stage(window.index, "face", [frame.timestamp])
                # Face matching receives raw pixels; do not key it by lossy JPEG.
                key = (frame.image.shape, _digest(frame.image.tobytes()))

                async def operation(frame=frame, key=key):
                    return await face_cache.get(
                        key,
                        lambda: handlers._video_face_task(
                            engine,
                            frame.image,
                            top_k,
                            threshold,
                            frame.timestamp,
                        ),
                    )

                records = await handlers._review_stage(
                    "face", frame.timestamp, operation, errors, default=[]
                )
                for face in records:
                    face = dict(face)
                    face["frame_time"] = frame.timestamp
                    if face.get("face_location"):
                        sample = {
                            "time_ms": round(frame.timestamp * 1000),
                            "pts_seconds": frame.timestamp,
                            "bbox": face["face_location"],
                        }
                        if frame.duration is not None:
                            sample["duration_seconds"] = frame.duration
                        if frame.frame_index is not None:
                            sample["frame_index"] = frame.frame_index
                        if face.get("similarity") is not None:
                            sample["similarity"] = float(face["similarity"])
                        quality = face.get("query_quality") or {}
                        for source, target in (
                            ("score", "quality_score"),
                            ("sharpness", "sharpness"),
                            ("pose", "pose_score"),
                        ):
                            if quality.get(source) is not None:
                                sample[target] = float(quality[source])
                        if face.get("query_estimated_yaw") is not None:
                            sample["estimated_yaw"] = float(face["query_estimated_yaw"])
                        if face.get("candidate_rank") is not None:
                            sample["candidate_rank"] = int(face["candidate_rank"])
                        if face.get("candidate_margin") is not None:
                            sample["candidate_margin"] = float(face["candidate_margin"])
                        if face.get("auxiliary"):
                            sample["auxiliary"] = True
                        face["_face_sample"] = sample
                    face_records.append(face)
                face_observations.extend(getattr(records, "observations", ()))
            if progress is not None:
                progress.finish_stage(window.index, "face")

        await gather_stages(visual(), text(), faces())
        completed.append(
            {
                "index": window.index,
                "scene_id": window.scene_id,
                "start": window.start,
                "end": window.end,
                "hits": list(hits.values()),
                "face_records": face_records,
                "face_observations": face_observations,
            }
        )

    def rebuild_face_hits(result):
        non_face = [hit for hit in result["hits"] if hit.get("source") != "face"]
        if not settings.face_profile_optimization:
            legacy = {}
            for record in result.get("face_records", ()):
                key = (
                    record.get("category") or "敏感人物",
                    record.get("name", "敏感人物"),
                )
                hit = legacy.setdefault(
                    key,
                    {"source": "face", "category": key[0], "description": key[1]},
                )
                sample = record.get("_face_sample")
                if sample and sample not in hit.setdefault("face_samples", []):
                    hit["face_samples"].append(sample)
            result["face_diagnostics"] = {
                "tracks": 0,
                "confirmed": len(legacy),
                "probable": 0,
                "trigger_times": [],
            }
            result["hits"] = [*non_face, *legacy.values()]
            return
        face_hits, diagnostics = aggregate_face_candidates(
            result.get("face_records", ()),
            confirm_similarity,
            max_gap=max(settings.face_track_max_gap_s, sample_interval + 0.001),
        )
        difficult_times = {
            float(observation["frame_time"])
            for observation in result.get("face_observations", ())
            if observation.get("frame_time") is not None and observation_is_difficult(observation)
        }
        diagnostics["trigger_times"] = sorted(
            set(diagnostics.get("trigger_times", ())) | difficult_times
        )
        result["face_diagnostics"] = diagnostics
        result["hits"] = [*non_face, *face_hits]

    async def resample_difficult_faces():
        for result in completed:
            rebuild_face_hits(result)
        if not include_faces or not settings.face_profile_optimization or not selected_frames:
            return 0
        budget = min(
            sum(settings.face_max_extra_frames_per_window for _ in completed),
            int(math.ceil(selected_frames * settings.face_max_extra_call_ratio)),
        )
        if budget <= 0:
            return 0

        requests = []
        used_targets = set()
        offsets = sorted(settings.face_neighbor_offsets_s, key=lambda value: abs(float(value)))
        for result in sorted(completed, key=lambda item: item["index"]):
            existing = {
                round(float(record.get("frame_time")), 6)
                for record in result.get("face_records", ())
                if record.get("frame_time") is not None
            }
            count = 0
            for trigger in result.get("face_diagnostics", {}).get("trigger_times", ()):
                for offset in offsets:
                    target = round(float(trigger) + float(offset), 6)
                    if target < result["start"] - 1e-6 or target > result["end"] + 1e-6:
                        continue
                    if any(abs(target - value) < 0.05 for value in existing):
                        continue
                    key = round(target, 3)
                    if key in used_targets:
                        continue
                    used_targets.add(key)
                    requests.append((target, result))
                    count += 1
                    if (
                        count >= settings.face_max_extra_frames_per_window
                        or len(requests) >= budget
                    ):
                        break
                if count >= settings.face_max_extra_frames_per_window or len(requests) >= budget:
                    break
            if len(requests) >= budget:
                break
        if not requests:
            return 0

        if progress is not None:
            await progress.begin_face_resampling(len(requests))

        try:
            frames = await asyncio.to_thread(
                read_video_frames_near,
                path,
                [target for target, _ in requests],
                max_dimension=1080,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Optional face neighbour sampling failed: %s", exc)
            return 0
        by_target = {round(target, 6): frame for target, frame in frames}
        limit = asyncio.Semaphore(settings.face_neighbor_concurrency)

        async def review(target, result):
            try:
                frame = by_target.get(round(target, 6))
                if frame is None:
                    return
                primary_times = {
                    round(float(record.get("frame_time")), 6)
                    for record in result.get("face_records", ())
                    if record.get("frame_time") is not None
                }
                if any(abs(frame.timestamp - value) < 0.05 for value in primary_times):
                    return
                key = (frame.image.shape, _digest(frame.image.tobytes()))

                async def operation():
                    async with limit:
                        return await face_cache.get(
                            key,
                            lambda: handlers._video_face_task(
                                engine,
                                frame.image,
                                top_k,
                                threshold,
                                frame.timestamp,
                                auxiliary=True,
                            ),
                        )

                records = await handlers._review_stage(
                    "face", frame.timestamp, operation, errors, default=[]
                )
                for face in records:
                    face = dict(face)
                    face["frame_time"] = frame.timestamp
                    location = face.get("face_location")
                    if location:
                        sample = {
                            "time_ms": round(frame.timestamp * 1000),
                            "pts_seconds": frame.timestamp,
                            "bbox": location,
                            "auxiliary": True,
                        }
                        if frame.duration is not None:
                            sample["duration_seconds"] = frame.duration
                        if frame.frame_index is not None:
                            sample["frame_index"] = frame.frame_index
                        if face.get("similarity") is not None:
                            sample["similarity"] = float(face["similarity"])
                        face["_face_sample"] = sample
                    result["face_records"].append(face)
                result["face_observations"].extend(getattr(records, "observations", ()))
            finally:
                if progress is not None:
                    await progress.advance_face_resampling()

        await asyncio.gather(*(review(target, result) for target, result in requests))
        if progress is not None:
            await progress.finish_face_resampling()
        for result in completed:
            rebuild_face_hits(result)
        return len(frames)

    async def consumer():
        while True:
            window = await queue.get()
            try:
                if progress is not None:
                    await progress.start_window(
                        window.index,
                        window.start,
                        window.end,
                        [frame.timestamp for frame in window.frames],
                    )
                await handlers._review_stage(
                    "frame",
                    window.start,
                    lambda window=window: process_window(window),
                    errors,
                    end_timestamp=window.end,
                )
                if progress is not None:
                    await progress.complete_window(window.index, window.end, len(window.frames))
            finally:
                queue.task_done()

    consumers = []
    try:
        await asyncio.to_thread(
            handlers._download_video_safe_sync,
            url,
            path,
            settings.max_file_size_mb * 100 * 1024 * 1024,
            timeout=900.0,
        )
        consumers = [asyncio.create_task(consumer()) for _ in range(concurrency)]
        with VideoFrameSampler(
            path,
            sample_interval,
            max_dimension=1080,
            sampling_mode=settings.nsfw_sampling_mode,
            max_visual_stride=settings.nsfw_scene_max_stride,
            scene_cut_threshold=settings.nsfw_scene_cut_threshold,
        ) as sampler:
            if progress is not None:
                await progress.begin_review(getattr(sampler, "duration_seconds", None))
            for sample in sampler:
                ready = planner.push(sample)
                if ready is not None:
                    selected_frames += len(ready.frames)
                    if coverage is not None:
                        coverage.add(frame.timestamp for frame in ready.frames)
                    if progress is not None:
                        progress.enqueue(len(ready.frames))
                    await queue.put(ready)
                await asyncio.sleep(0)
            ready = planner.flush()
            if ready is not None:
                selected_frames += len(ready.frames)
                if coverage is not None:
                    coverage.add(frame.timestamp for frame in ready.frames)
                if progress is not None:
                    progress.enqueue(len(ready.frames))
                await queue.put(ready)
        if progress is not None:
            await progress.sampled()
        await queue.join()
        extra_face_frames = await resample_difficult_faces()
        if progress is not None:
            await progress.set_phase("saving")
        logger.info(
            "Window review complete: windows=%s selected_frames=%s visual_windows=%s "
            "visual_calls=%s visual_reused=%s "
            "ocr_calls=%s ocr_reused=%s face_calls=%s face_reused=%s "
            "extra_face_frames=%s confirmed_faces=%s probable_faces=%s "
            "guard_calls=%s guard_reused=%s",
            planner.count,
            selected_frames,
            planner.count if include_visual else 0,
            visual_cache.misses,
            visual_cache.hits,
            ocr_cache.misses,
            ocr_cache.hits,
            face_cache.misses,
            face_cache.hits,
            extra_face_frames,
            sum(item.get("face_diagnostics", {}).get("confirmed", 0) for item in completed),
            sum(item.get("face_diagnostics", {}).get("probable", 0) for item in completed),
            guard_cache.misses,
            guard_cache.hits,
        )
        rows = merge_window_results(
            completed, min(settings.nsfw_window_max_seconds, max(sample_interval, 1.0) + 0.1)
        )
        rows.extend(errors)
        rows.sort(key=lambda row: row["timestamp"].split("~", 1)[0])
        return rows
    finally:
        for task in consumers:
            task.cancel()
        await asyncio.gather(*consumers, return_exceptions=True)
        await asyncio.gather(
            ocr_cache.close(), face_cache.close(), guard_cache.close(), visual_cache.close()
        )
        path.unlink(missing_ok=True)


def standalone_results(rows, *, include_visual=True):
    visual, text, errors = [], [], []
    for row in rows:
        if row.get("review_status") == "incomplete":
            errors.append(row)
        elif row.get("source") == "visual":
            visual.append(
                {
                    "timestamp": row["timestamp"],
                    "confidence": 1.0,
                    "description": f"[{row['category']}] {row['description']}",
                }
            )
        elif row.get("source") == "ocr":
            text.append(
                {
                    "timestamp": row["timestamp"],
                    "category": row["category"],
                    "text": row["description"],
                }
            )
    result = {"unsafe_text_frames": text}
    if include_visual:
        result["visual_analysis"] = visual
    if errors:
        result["errors"] = errors
    return result
