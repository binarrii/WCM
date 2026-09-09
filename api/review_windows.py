"""Whole-window review with task-local request reuse and explicit time spans."""

import asyncio
import hashlib
import logging
import os
from collections import OrderedDict
from pathlib import Path

from wcm_facerec.config import settings

from . import handlers
from .utils import ReviewWindowPlanner, VideoFrameSampler

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
    concurrency = 2
    queue = asyncio.Queue(maxsize=concurrency * 2)
    planner = ReviewWindowPlanner(settings.nsfw_window_max_seconds)
    selected_frames = 0
    if coverage is not None:
        coverage.add([])

    async def guard(text):
        return await guard_cache.get(_digest(text), lambda: handlers._call_llm_guard(text))

    async def process_window(window):
        hits = {}

        def add(source, category, description):
            key = (source, category, description)
            return hits.setdefault(
                key, {"source": source, "category": category, "description": description}
            )

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
                add("visual", result["category"], result["text"])
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
                            return {
                                "text": content,
                                "category": verdict.get("category", "文本违规"),
                            }
                    return None

                result = await handlers._review_stage("ocr", frame.timestamp, operation, errors)
                if result:
                    add("ocr", result["category"], result["text"])
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
                        lambda: handlers._face_task(
                            engine, frame.image, top_k, threshold, frame.timestamp
                        ),
                    )

                records = await handlers._review_stage(
                    "face", frame.timestamp, operation, errors, default=[]
                )
                for face in records:
                    hit = add(
                        "face", face.get("category") or "敏感人物", face.get("name", "敏感人物")
                    )
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
                        if sample not in hit.setdefault("face_samples", []):
                            hit["face_samples"].append(sample)
            if progress is not None:
                progress.finish_stage(window.index, "face")

        await asyncio.gather(visual(), text(), faces())
        completed.append(
            {
                "index": window.index,
                "scene_id": window.scene_id,
                "start": window.start,
                "end": window.end,
                "hits": list(hits.values()),
            }
        )

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
        if progress is not None:
            await progress.set_phase("saving")
        logger.info(
            "Window review complete: windows=%s selected_frames=%s visual_windows=%s "
            "visual_calls=%s visual_reused=%s "
            "ocr_calls=%s ocr_reused=%s face_calls=%s face_reused=%s guard_calls=%s guard_reused=%s",
            planner.count,
            selected_frames,
            planner.count if include_visual else 0,
            visual_cache.misses,
            visual_cache.hits,
            ocr_cache.misses,
            ocr_cache.hits,
            face_cache.misses,
            face_cache.hits,
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
