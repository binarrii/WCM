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
import numpy as np

from wcm_facerec.config import settings

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"}
MIN_FACE_PIXELS = 32 * 32
logger = logging.getLogger(__name__)


@dataclass
class VideoFrame:
    timestamp: float
    image: object
    sampled: bool = True
    scene_id: int = 0

    @cached_property
    def appearance(self):
        return cv2.resize(self.image, (160, 90), interpolation=cv2.INTER_LINEAR)

    @cached_property
    def b64(self) -> str:
        ok, buffer = cv2.imencode(".jpg", self.image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise ValueError("Could not encode sampled video frame")
        return base64.b64encode(buffer).decode("utf-8")


class VideoWindow(tuple):
    """Tuple-compatible frames with independent visual and fixed-grid scheduling."""

    def __new__(cls, frames, *, review_visual=True, sampled=True):
        window = super().__new__(cls, frames)
        window.review_visual = review_visual
        window.sampled = sampled
        return window


@dataclass(frozen=True)
class ReviewWindow:
    frames: tuple[VideoFrame, ...]
    start: float
    end: float
    scene_id: int
    index: int


class ReviewWindowPlanner:
    """Batch every selected visual target once, with bounded same-shot spans."""

    def __init__(self, max_span: float = 10.0):
        if not math.isfinite(max_span) or max_span <= 0:
            raise ValueError("max_span must be finite and positive")
        self.max_span = max_span
        self.frames = []
        self.start = self.end = None
        self.scene_id = None
        self.count = 0

    def flush(self):
        if not self.frames:
            return None
        result = ReviewWindow(tuple(self.frames), self.start, self.end, self.scene_id, self.count)
        self.count += 1
        self.frames = []
        self.start = self.end = self.scene_id = None
        return result

    def push(self, sample: VideoWindow):
        head = sample[0]
        selected = sample.review_visual
        ready = None
        if self.frames and (
            head.scene_id != self.scene_id
            or head.timestamp - self.start > self.max_span
            or (selected and len(self.frames) == 3)
        ):
            ready = self.flush()
        if self.start is None:
            self.start = head.timestamp
            self.scene_id = head.scene_id
            selected = True
        self.end = head.timestamp
        if selected:
            self.frames.append(head)
        return ready


def _scene_change_score(previous, current):
    before = cv2.cvtColor(previous, cv2.COLOR_BGR2HSV).astype(np.float32)
    after = cv2.cvtColor(current, cv2.COLOR_BGR2HSV).astype(np.float32)
    delta = np.abs(after - before)
    # Hue wraps at 180 in OpenCV; normalize its circular distance to 0..255.
    delta[:, :, 0] = np.minimum(delta[:, :, 0], 180 - delta[:, :, 0]) * (255 / 90)
    # Hue/saturation are unstable in near-black or achromatic pixels. Avoid
    # turning dark-scene compression noise into dozens of artificial shots.
    light = np.minimum(np.minimum(before[:, :, 2], after[:, :, 2]) / 32, 1)
    delta[:, :, 0] *= np.minimum(before[:, :, 1], after[:, :, 1]) / 255 * light
    delta[:, :, 1] *= light
    return float(delta.mean())


def _near_duplicate(previous, current):
    delta = cv2.absdiff(previous, current).max(axis=2)
    # Local tiles keep a small changing region from being diluted by a static
    # background. Compare against the last selected target, not its predecessor.
    tiles = delta.reshape(9, 10, 16, 10).mean(axis=(1, 3))
    return float(delta.mean()) <= 1.5 and float(tiles.max()) <= 5.0


class VideoFrameSampler:
    """PTS sampling with fixed windows or scene-aware visual scheduling.

    Every grid sample heads one candidate window. Scene mode also retains
    otherwise-unsampled short shots and limits context to the shot. Consumers
    can keep legacy heads or batch selected targets with ReviewWindowPlanner.
    """

    def __init__(
        self,
        path: Path,
        sample_interval: float,
        *,
        max_dimension: int | None = None,
        sampling_mode: str = "fixed",
        max_visual_stride: int = 3,
        scene_cut_threshold: float = 27.0,
    ):
        if not math.isfinite(sample_interval) or sample_interval < 0:
            raise ValueError("sample_interval must be finite and nonnegative")
        if sampling_mode not in {"fixed", "scene"}:
            raise ValueError("sampling_mode must be fixed or scene")
        if not 1 <= max_visual_stride <= 10:
            raise ValueError("max_visual_stride must be between 1 and 10")
        if not math.isfinite(scene_cut_threshold) or not 0 < scene_cut_threshold <= 255:
            raise ValueError("scene_cut_threshold must be finite and in (0, 255]")
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
        self.sampling_mode = sampling_mode
        self.max_visual_gap = max(sample_interval, 1 / self.fps) * max_visual_stride
        self.scene_cut_threshold = scene_cut_threshold
        self.scene_cuts = 0
        self.fixed_samples = 0
        self.boundary_samples = 0
        self.visual_targets = 0
        self.visual_skipped = 0
        self._visual_anchor = None

    def _window(self, frames, *, scene_end=False):
        head = frames[0]
        anchor = self._visual_anchor
        review = (
            self.sampling_mode == "fixed"
            or anchor is None
            or anchor.scene_id != head.scene_id
            or scene_end
            or head.timestamp - anchor.timestamp + 1e-9 >= self.max_visual_gap
            or not _near_duplicate(anchor.appearance, head.appearance)
        )
        if review:
            self._visual_anchor = head
            self.visual_targets += 1
        else:
            self.visual_skipped += 1
        return VideoWindow(frames, review_visual=review, sampled=head.sampled)

    def _resize(self, frame):
        if self.max_dimension and max(frame.image.shape[:2]) > self.max_dimension:
            height, width = frame.image.shape[:2]
            scale = self.max_dimension / max(height, width)
            frame.image = cv2.resize(
                frame.image, (max(1, int(width * scale)), max(1, int(height * scale)))
            )
        return frame

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cap.release()

    def __iter__(self):
        window = deque()
        previous_time = None
        last_valid_time, last_valid_index = 0.0, 0
        next_sample_time = 0.0
        previous_frame = None
        scene_first = None
        scene_sampled = False
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
            sampled = self.interval == 0 or timestamp + 1e-9 >= next_sample_time
            if sampled and self.interval > 0:
                # Advance the absolute sampling grid, not timestamp + interval.
                # A long frame may cross several deadlines; never duplicate it.
                next_sample_time = (
                    math.floor((timestamp + 1e-9) / self.interval) + 1
                ) * self.interval
            current = VideoFrame(timestamp, frame, sampled, self.scene_cuts)
            appearance = current.appearance if self.sampling_mode == "scene" else None
            cut = (
                self.sampling_mode == "scene"
                and previous_frame is not None
                and (
                    _scene_change_score(previous_frame.appearance, appearance)
                    >= self.scene_cut_threshold
                )
            )
            if cut:
                if not scene_sampled and scene_first is not None:
                    window.append(self._resize(scene_first))
                    self.boundary_samples += 1
                while window:
                    yield self._window(tuple(window), scene_end=len(window) == 1)
                    window.popleft()
                self.scene_cuts += 1
                current.scene_id = self.scene_cuts
                scene_first = current
                scene_sampled = False
            if scene_first is None and not scene_sampled:
                scene_first = current
            previous_frame = current
            if not sampled:
                continue
            scene_sampled = True
            scene_first = None
            self.fixed_samples += 1
            window.append(self._resize(current))
            if len(window) == 3:
                yield self._window(tuple(window))
                window.popleft()
        if self.sampling_mode == "scene" and not scene_sampled and scene_first is not None:
            window.append(self._resize(scene_first))
            self.boundary_samples += 1
        while window:
            yield self._window(tuple(window), scene_end=len(window) == 1)
            window.popleft()
        logger.info(
            "Video sampling complete: mode=%s decoded=%s fixed_samples=%s scene_cuts=%s "
            "boundary_samples=%s visual_targets=%s visual_skipped=%s",
            self.sampling_mode,
            self.frames_read,
            self.fixed_samples,
            self.scene_cuts,
            self.boundary_samples,
            self.visual_targets,
            self.visual_skipped,
        )


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
    with VideoFrameSampler(
        video_path,
        sample_interval,
        sampling_mode=settings.nsfw_sampling_mode,
        max_visual_stride=settings.nsfw_scene_max_stride,
        scene_cut_threshold=settings.nsfw_scene_cut_threshold,
    ) as sampler:
        return [
            VideoWindow(
                ((frame.timestamp, frame.b64) for frame in window),
                review_visual=window.review_visual,
                sampled=window.sampled,
            )
            for window in sampler
        ]
