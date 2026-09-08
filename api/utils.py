# pyright: ignore[reportUnusedFunction]

import base64
import logging
import math
from collections import deque
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import cv2
import httpx

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"}
MIN_FACE_PIXELS = 32 * 32
logger = logging.getLogger(__name__)


@dataclass
class VideoFrame:
    timestamp: float
    image: object

    @cached_property
    def b64(self) -> str:
        ok, buffer = cv2.imencode(".jpg", self.image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise ValueError("Could not encode sampled video frame")
        return base64.b64encode(buffer).decode("utf-8")


class VideoFrameSampler:
    """One sampling clock and stride-one, three-frame windows for every task.

    Drain partial windows at EOF: [a,b,c], [b,c], [c]. Each sampled
    frame heads exactly one window; context frames are shared, never padded.
    """

    def __init__(self, path: Path, sample_interval: float, *, max_dimension: int | None = None):
        if not math.isfinite(sample_interval) or sample_interval < 0:
            raise ValueError("sample_interval must be finite and nonnegative")
        self.cap = cv2.VideoCapture(str(path))
        try:
            if not self.cap.isOpened():
                raise ValueError("Could not open video file")
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.fps = fps if math.isfinite(fps) and fps > 0 else 25.0
        except BaseException:
            self.cap.release()
            raise
        self.interval = sample_interval
        self.frames_read = 0
        self.max_dimension = max_dimension
        self.estimated_timestamps = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cap.release()

    def __iter__(self):
        window = deque()
        previous_time = None
        last_valid_time, last_valid_index = 0.0, 0
        next_sample_time = 0.0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            index = self.frames_read
            self.frames_read += 1
            # Read after decoding: POS_MSEC is the current frame's presentation
            # position, including variable frame durations, not frame_index/FPS.
            timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if not (
                math.isfinite(timestamp)
                and timestamp >= 0
                and (previous_time is None or timestamp > previous_time)
            ):
                if not self.estimated_timestamps:
                    logger.warning(
                        "Video decoder timestamps unavailable/non-monotonic; estimating affected frames from FPS"
                    )
                self.estimated_timestamps += 1
                timestamp = last_valid_time + (index - last_valid_index) / self.fps
            else:
                last_valid_time, last_valid_index = timestamp, index
            previous_time = timestamp
            if self.interval > 0 and timestamp + 1e-9 < next_sample_time:
                continue
            if self.interval > 0:
                # Advance the absolute sampling grid, not timestamp + interval.
                # A long frame may cross several deadlines; never duplicate it.
                next_sample_time = (
                    math.floor((timestamp + 1e-9) / self.interval) + 1
                ) * self.interval
            if self.max_dimension and max(frame.shape[:2]) > self.max_dimension:
                height, width = frame.shape[:2]
                scale = self.max_dimension / max(height, width)
                frame = cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))))
            window.append(VideoFrame(timestamp, frame))
            if len(window) == 3:
                yield tuple(window)
                window.popleft()
        while window:
            yield tuple(window)
            window.popleft()


async def _download_url_safe(url: str, max_size: int, timeout: float = 60.0) -> bytes:
    """Download a URL safely, enforcing a maximum file size in bytes to prevent OOM."""
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > max_size:
                raise ValueError(f"File too large. Max allowed: {max_size} bytes")

            chunks = bytearray()
            async for chunk in response.aiter_bytes():
                chunks.extend(chunk)
                if len(chunks) > max_size:
                    raise ValueError(f"File too large. Max allowed: {max_size} bytes")
            return bytes(chunks)


def _download_video_safe_sync(url: str, file_path: Path, max_size: int, timeout: float = 120.0):
    """Synchronously download a video to disk safely, enforcing max size."""
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > max_size:
                raise ValueError(f"Video file too large. Max allowed: {max_size} bytes")

            downloaded = 0
            with open(file_path, "wb") as f:
                for chunk in response.iter_bytes():
                    f.write(chunk)
                    downloaded += len(chunk)
                    if downloaded > max_size:
                        raise ValueError(f"Video file too large. Max allowed: {max_size} bytes")


def _extract_video_frames_for_ocr(
    video_path: Path, sample_interval: float
) -> list[tuple[float, str]]:
    """OCR uses only the head of each shared sampling window."""
    with VideoFrameSampler(video_path, sample_interval) as sampler:
        return [(window[0].timestamp, window[0].b64) for window in sampler]


def _extract_video_windows(video_path: Path, sample_interval: float):
    with VideoFrameSampler(video_path, sample_interval) as sampler:
        return [tuple((frame.timestamp, frame.b64) for frame in window) for window in sampler]
