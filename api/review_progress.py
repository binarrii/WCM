"""Throttled task progress, independent of whether content passed its checks."""

import asyncio
import json
import logging
import math
import time

from . import review_task_store

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
logger.propagate = False


class ReviewProgress:
    def __init__(self, task_id=None, *, interval=2.0):
        self.task_id = task_id
        self.interval = interval
        self.started = time.monotonic()
        self.last_report = -math.inf
        self.phase = "downloading"
        self.duration = None
        self.processed_seconds = 0.0
        self.completed_windows = self.completed_samples = 0
        self.queued_windows = self.queued_samples = 0
        self.sampling_complete = False
        self.next_window = 0
        self.finished = {}
        self.active_windows = {}
        self.last_completed_window = None
        self.sub_progress = None
        self.percent = None
        self.lock = asyncio.Lock()
        self.sequence = 0

    async def __aenter__(self):
        await self.report(force=True)
        self.heartbeat = asyncio.create_task(self._heartbeat())
        return self

    async def __aexit__(self, *args):
        self.heartbeat.cancel()
        await asyncio.gather(self.heartbeat, return_exceptions=True)

    async def _heartbeat(self):
        while True:
            await asyncio.sleep(max(2.0, self.interval))
            await self.report()

    async def begin_review(self, duration=None):
        self.phase = "reviewing"
        self.duration = (
            duration
            if isinstance(duration, (int, float)) and math.isfinite(duration) and duration > 0
            else None
        )
        await self.report(force=True)

    def enqueue(self, samples):
        self.queued_windows += 1
        self.queued_samples += samples

    async def start_window(self, index, start, end, timestamps):
        self.active_windows[index] = {
            "index": index + 1,
            "start_seconds": start,
            "end_seconds": end,
            "sample_timestamps": list(timestamps),
            "stages": {},
        }
        await self.report()

    async def start_stage(self, index, stage, timestamps):
        window = self.active_windows.get(index)
        if window is not None:
            window["stages"][stage] = list(timestamps)
        await self.report()

    def finish_stage(self, index, stage):
        window = self.active_windows.get(index)
        if window is not None:
            window["stages"].pop(stage, None)

    async def complete_window(self, index, end, samples):
        window = self.active_windows.pop(index, None)
        if window is not None:
            self.last_completed_window = window
        self.completed_windows += 1
        self.completed_samples += samples
        self.finished[index] = end
        # A fast later window must not hide an unfinished earlier window.
        while self.next_window in self.finished:
            self.processed_seconds = max(
                self.processed_seconds, self.finished.pop(self.next_window)
            )
            self.next_window += 1
        await self.report()

    async def sampled(self):
        self.sampling_complete = True
        await self.report(force=True)

    async def begin_face_resampling(self, total):
        self.phase = "resampling"
        self.sub_progress = {
            "stage": "face_resampling",
            "completed": 0,
            "total": max(0, int(total)),
        }
        await self.report(force=True)

    async def advance_face_resampling(self):
        if self.sub_progress is None:
            return
        self.sub_progress["completed"] = min(
            self.sub_progress["total"], self.sub_progress["completed"] + 1
        )
        await self.report()

    async def finish_face_resampling(self):
        if self.sub_progress is None:
            return
        self.sub_progress["completed"] = self.sub_progress["total"]
        await self.report(force=True)

    def snapshot(self):
        percent = None
        if self.phase == "finished":
            percent = 100.0
        elif self.phase in {"resampling", "saving"}:
            percent = 99.0
        elif self.phase == "reviewing":
            if self.duration:
                percent = min(99.0, self.processed_seconds / self.duration * 100)
            elif self.sampling_complete and self.queued_windows:
                percent = min(99.0, self.completed_windows / self.queued_windows * 100)
        if percent is not None:
            self.percent = max(self.percent or 0, percent)
        sub_progress = None
        if self.sub_progress is not None:
            total = self.sub_progress["total"]
            completed = self.sub_progress["completed"]
            sub_progress = {
                **self.sub_progress,
                "percent": round(completed / total * 100, 1) if total else 100.0,
            }
        return {
            "phase": self.phase,
            "percent": round(self.percent, 1) if self.percent is not None else None,
            "completed_windows": self.completed_windows,
            "total_windows": self.queued_windows if self.sampling_complete else None,
            "completed_samples": self.completed_samples,
            "total_samples": self.queued_samples if self.sampling_complete else None,
            "processed_seconds": round(self.processed_seconds, 3),
            "duration_seconds": self.duration,
            "active_windows": [
                {**window, "stages": dict(window["stages"])}
                for _, window in sorted(self.active_windows.items())
            ],
            "last_completed_window": self.last_completed_window,
            "sub_progress": sub_progress,
            "elapsed_seconds": round(time.monotonic() - self.started, 1),
        }

    async def set_phase(self, phase, *, persist=True):
        self.phase = phase
        await self.report(force=True, persist=persist)

    async def report(self, *, force=False, persist=True):
        async with self.lock:
            now = time.monotonic()
            if not force and now - self.last_report < self.interval:
                return
            self.last_report = now
            self.sequence += 1
            data = self.snapshot()
            data["sequence"] = self.sequence
            percent = f"{data['percent']:.1f}%" if data["percent"] is not None else "unknown"
            logger.info(
                "Review progress: task=%s phase=%s progress=%s windows=%s/%s samples=%s/%s "
                "video_seconds=%.3f/%s elapsed_seconds=%.1f active_windows=%s sub_progress=%s",
                self.task_id or "standalone",
                self.phase,
                percent,
                self.completed_windows,
                data["total_windows"],
                self.completed_samples,
                data["total_samples"],
                self.processed_seconds,
                self.duration,
                data["elapsed_seconds"],
                json.dumps(data["active_windows"], ensure_ascii=False),
                json.dumps(data["sub_progress"], ensure_ascii=False),
            )
            if persist:
                try:
                    await asyncio.wait_for(
                        review_task_store.update_progress(self.task_id, data), timeout=1.0
                    )
                except Exception as exc:
                    # Reporting must never turn a successfully reviewed frame into a failure.
                    logger.warning(
                        "Progress persistence unavailable: task=%s error=%s",
                        self.task_id,
                        type(exc).__name__,
                    )
