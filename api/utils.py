# pyright: ignore[reportUnusedFunction]

import asyncio
import httpx
import cv2
import base64
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from wcm_facerec.config import settings
from wcm_facerec import __version__

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"}
MIN_FACE_PIXELS = 64 * 64

# Dedicated single-thread pool for CUDA/DeepFace inference
inference_executor = ThreadPoolExecutor(max_workers=1)

async def run_in_inference_thread(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        inference_executor,
        partial(func, *args, **kwargs)
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


def _extract_video_frames_for_ocr(video_path: Path, sample_interval: float) -> list[tuple[float, str]]:
    """Extract frames from video and encode to base64 for OCR."""
    frames_b64 = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return frames_b64
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
        
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % int(max(fps * sample_interval, 1)) == 0:
            _, buffer = cv2.imencode('.jpg', frame)
            b64_img = base64.b64encode(buffer).decode('utf-8')
            frames_b64.append((frame_idx/fps, b64_img))
        frame_idx += 1
    cap.release()
    return frames_b64


