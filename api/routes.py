"""API routes for face recognition service."""

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Union
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import cv2
import httpx
import numpy as np
from fastapi import APIRouter, Request, HTTPException, WebSocket, WebSocketDisconnect

from wcm_facerec import __version__
from wcm_facerec.config import settings
from wcm_facerec.face_engine import FaceEngine, get_face_engine


MIN_FACE_PIXELS = 64 * 64

api_bp = APIRouter()


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


@api_bp.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": settings.deepface_model,
        "embedding_dim": settings.embedding_dim,
        "version": __version__,
    }


@api_bp.post("/detect")
async def detect_faces(request: Request):
    """Detect faces in an image.

    Accepts either an uploaded file or a URL via form data.
    """
    engine = get_face_engine()
    img_source: Union[str, Path, bytes]
    image_source = "unknown"

    content_type = request.headers.get("content-type", "")

    # Check if it's form data with file
    if "multipart/form-data" in content_type:
        form = await request.form()
        file = form.get("file")
        if file and file.filename:
            contents = await file.read()
            if len(contents) > settings.max_file_size_mb * 1024 * 1024:
                raise HTTPException(status_code=413, detail="File too large")
            img_source = contents
            image_source = file.filename
        elif form.get("url"):
            url = form.get("url")
            try:
                img_source = await _download_url_safe(url, settings.max_file_size_mb * 1024 * 1024)
                image_source = url
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to fetch image: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail="Either file or url must be provided")
    else:
        # JSON body
        data = await request.json()
        url = data.get("url")
        if not url:
            raise HTTPException(status_code=400, detail="url is required")
        try:
            img_source = await _download_url_safe(url, settings.max_file_size_mb * 1024 * 1024)
            image_source = url
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch image: {str(e)}")

    temp_path = None
    try:
        # Handle bytes - decode to numpy array directly (no temp file needed)
        if isinstance(img_source, bytes):
            nparr = np.frombuffer(img_source, np.uint8)
            img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR_BGR)
            faces = await run_in_inference_thread(engine.detect_faces, img_array)
        elif isinstance(img_source, (str, Path)):
            # Local file - decode to numpy array so OpenCV fallback works
            img_array = cv2.imread(str(img_source), cv2.IMREAD_COLOR_BGR)
            faces = await run_in_inference_thread(engine.detect_faces, img_array)
        else:
            faces = await run_in_inference_thread(engine.detect_faces, img_source)

        results = []
        for i, face in enumerate(faces):
            results.append({
                "face_id": f"face_{i}",
                "confidence": face.get("confidence", 0.0),
                "facial_area": face.get("facial_area", {}),
            })

        return {
            "faces": results,
            "image_source": image_source,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()


@api_bp.post("/register")
async def register_face(request: Request):
    """Register a face in the database.

    Accepts form data with name and either file or url.
    """
    engine = get_face_engine()

    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" in content_type:
        form = await request.form()
        name = form.get("name")
        if not name:
            raise HTTPException(status_code=400, detail="name is required")

        if "file" in form:
            file = form.get("file")
            if file and file.filename:
                contents = await file.read()
                if len(contents) > settings.max_file_size_mb * 1024 * 1024:
                    raise HTTPException(status_code=413, detail="File too large")
                img_source = contents
            else:
                raise HTTPException(status_code=400, detail="Either file or url must be provided")
        elif form.get("url"):
            url = form.get("url")
            try:
                img_source = await _download_url_safe(url, settings.max_file_size_mb * 1024 * 1024)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to fetch image: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail="Either file or url must be provided")

        try:
            file_url = form.get("url") if "file" in form else None
            record = await run_in_inference_thread(
                engine.register_from_image,
                name=name,
                img_source=img_source,
                file_url=file_url,
            )
            return {
                "id": str(record.id),
                "name": record.name,
                "message": "Face registered successfully",
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="Content-Type must be multipart/form-data")


@api_bp.post("/search")
async def search_faces(request: Request):
    """Search for similar faces in the database."""
    engine = get_face_engine()

    data = await request.json()
    if not data:
        raise HTTPException(status_code=400, detail="Request body required")

    url = data.get("url")
    if not url:
        raise HTTPException(status_code=400, detail="url is required for search")

    name = data.get("name")
    top_k = int(data.get("top_k", 10))
    threshold = float(data.get("threshold", 0.4))

    try:
        is_video = any(url.lower().endswith(ext) for ext in {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"})

        if is_video:
            sample_interval = float(data.get("sample_interval", 1.0))
            frames, results = await asyncio.to_thread(
                _search_video_frames,
                engine, url, name, max(min(top_k, 10), 1),
                max(min(threshold, 1.0), 0.0), sample_interval
            )
            return {
                "results": results,
                "query_embedding_dim": settings.embedding_dim,
                "frames_processed": frames,
            }
        else:
            # Detect and crop face before generating embedding
            face_result = await _detect_and_crop_face(engine, url)
            if face_result is None:
                return {
                    "results": [],
                    "query_embedding_dim": settings.embedding_dim,
                    "message": "No face detected in image",
                }

            embedding = face_result["embedding"]

            results = await asyncio.to_thread(
                engine.search,
                embedding=embedding,
                name=name,
                top_k=max(min(top_k, 10), 1),
                threshold=max(min(threshold, 1.0), 0.0),
            )

            return {
                "results": results,
                "query_embedding_dim": settings.embedding_dim,
            }
    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


async def _detect_and_crop_face(engine: FaceEngine, url: str) -> dict | None:
    """Download image, detect face, crop and return face embedding (in-memory)."""
    try:
        img_bytes = await _download_url_safe(url, settings.max_file_size_mb * 1024 * 1024, timeout=30.0)
        return _detect_and_crop_face_from_bytes(engine, img_bytes)
    except Exception:
        return None


def _detect_and_crop_face_from_bytes(engine: FaceEngine, img_bytes: bytes) -> dict | None:
    """Detect face from image bytes, crop and return face embedding (in-memory).

    Returns dict with 'embedding' and 'confidence' if face found, None otherwise.
    """
    try:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR_BGR)
        if img_array is None:
            return None

        faces = engine.detect_faces(img_array)
        if not faces:
            return None

        def get_face_area(f):
            fa = f.get("facial_area", {})
            return (fa.get("w", 0) or 0) * (fa.get("h", 0) or 0)
        best_face = max(faces, key=get_face_area)
        face_img = best_face.get("face")
        confidence = best_face.get("confidence", 0.0)

        if face_img is None or confidence < 0.6:
            return None

        embedding = engine.generate_embedding(face_img)
        return {
            "embedding": embedding,
            "confidence": confidence,
        }
    except Exception:
        return None


def _search_video_frames(
    engine: FaceEngine,
    url: str,
    name: str | None,
    top_k: int,
    threshold: float,
    sample_interval: float,
) -> tuple[int, list[dict]]:
    """Search faces from video by sampling frames."""
    video_path = Path(f"/tmp/ws_video_{os.urandom(8).hex()}.mp4")
    try:
        _download_video_safe_sync(url, video_path, settings.max_file_size_mb * 100 * 1024 * 1024, timeout=900.0)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise Exception("Could not open video file")

        fps = cap.get(cv2.CAP_PROP_FPS)
        all_results = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % int(fps * sample_interval) == 0:
                current_frame_time = frame_idx / fps if fps > 0 else 0
                try:
                    faces = engine.detect_faces(frame)
                    for face_data in faces:
                        face_img = face_data.get("face")
                        if face_img is None:
                            continue
                        # Filter out bad detections
                        fa = face_data.get("facial_area", {})
                        area = (fa.get("w", 0) or 0) * (fa.get("h", 0) or 0)
                        conf = face_data.get("confidence") or 0
                        frame_area = frame.shape[0] * frame.shape[1]
                        # Skip if confidence is very low or face covers most of frame (detection failed)
                        if conf < 0.5 or area < MIN_FACE_PIXELS or area > frame_area * 0.8:
                            continue

                        # # --- DEBUG: save cropped face to temp dir ---
                        # import tempfile as _debug_tmp
                        # _debug_dir = _debug_tmp.mkdtemp(prefix="debug/face_search_debug_")
                        # _debug_fname = _debug_dir + f"/frame{frame_idx:06d}_t{current_frame_time:.2f}_conf{conf:.2f}_area{area}.png"
                        # _debug_face = (np.clip(face_img, 0, 1) * 255).astype(np.uint8) if face_img.dtype != np.uint8 else face_img
                        # cv2.imwrite(_debug_fname, _debug_face)
                        # print(f"[DEBUG FACE] frame={frame_idx} time={current_frame_time:.2f}s conf={conf:.2f} area={area} saved={_debug_fname}")
                        # # --- END DEBUG ---

                        _search_face_in_image(engine, face_img, name, top_k, threshold, all_results, current_frame_time)
                except Exception:
                    pass  # Skip frames that fail detection

            frame_idx += 1

        cap.release()

        # Sort and dedupe results by distance
        all_results.sort(key=lambda x: x.get("distance", 1.0))
        seen = set()
        deduped = []
        for r in all_results:
            key = (r.get("name"), r.get("person_id"))
            if key not in seen:
                seen.add(key)
                deduped.append(r)

        return frame_idx, deduped
        # return frame_idx, deduped[:top_k]
    finally:
        if video_path.exists():
            video_path.unlink()


def _search_face_in_image(
    engine: FaceEngine,
    face_img,
    name: str | None,
    top_k: int,
    threshold: float,
    all_results: list,
    frame_time: float | None = None,
):
    """Search a single face from a frame (in-memory, no temp files)."""
    try:
        embedding = engine.generate_embedding(face_img)
        results = engine.search(
            embedding=embedding,
            name=name,
            top_k=top_k,
            threshold=threshold,
        )
        for r in results:
            r["frame_time"] = frame_time
        all_results.extend(results[:1])
    except Exception:
        pass  # Skip faces that fail embedding generation


@api_bp.websocket("/ws/search")
async def websocket_search(websocket: WebSocket):
    """WebSocket endpoint for async face search.

    Accepts: {"url": "https://example.com/xxx.mp4"} or {"url": "https://example.com/xxx.png"}
    Responds immediately: {"status": "accepted", "taskId": "xxxxxxx"}
    Then sends result: {"status": "completed", "taskId": "xxxxxxx", "query_embedding_dim": 512, "results": [...]}
    """
    await websocket.accept()

    try:
        while True:
            try:
                data = await websocket.receive_text()
                if not data or len(data) < 2:  # Empty or single bracket
                    continue
                payload = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_json({"status": "error", "error": "Invalid JSON"})
                continue
            except WebSocketDisconnect:
                break

            url = payload.get("url")
            if not url:
                await websocket.send_json({"status": "error", "error": "url is required"})
                continue

            task_id = str(uuid.uuid4())
            name = payload.get("name")
            top_k = int(payload.get("top_k", 10))
            threshold = float(payload.get("threshold", 0.4))
            sample_interval = float(payload.get("sample_interval", 1.0))

            await websocket.send_json({"status": "accepted", "taskId": task_id})

            engine = get_face_engine()
            is_video = any(url.lower().endswith(ext) for ext in {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"})

            try:
                if is_video:
                    frames, results = await asyncio.to_thread(
                        _search_video_frames, engine, url, name, top_k, threshold, sample_interval
                    )
                    await websocket.send_json({
                        "status": "completed",
                        "taskId": task_id,
                        "query_embedding_dim": settings.embedding_dim,
                        "frames_processed": frames,
                        "results": results,
                    })
                else:
                    img_bytes = await _download_url_safe(url, settings.max_file_size_mb * 1024 * 1024)
                    face_result = await run_in_inference_thread(_detect_and_crop_face_from_bytes, engine, img_bytes)
                    if face_result is None:
                        await websocket.send_json({
                            "status": "completed",
                            "taskId": task_id,
                            "query_embedding_dim": settings.embedding_dim,
                            "results": [],
                            "message": "No face detected in image",
                        })
                        continue
                    embedding = face_result["embedding"]
                    results = await asyncio.to_thread(
                        engine.search,
                        embedding=embedding,
                        name=name,
                        top_k=max(min(top_k, 10), 1),
                        threshold=max(min(threshold, 1.0), 0.0),
                    )
                    await websocket.send_json({
                        "status": "completed",
                        "taskId": task_id,
                        "query_embedding_dim": settings.embedding_dim,
                        "results": results,
                    })
                    continue

            except httpx.HTTPError as e:
                await websocket.send_json({"status": "error", "taskId": task_id, "error": f"Failed to fetch: {str(e)}"})
            except Exception as e:
                await websocket.send_json({"status": "error", "taskId": task_id, "error": f"Search failed: {str(e)}"})

    except Exception as e:
        try:
            await websocket.send_json({"status": "error", "error": str(e)})
        except:
            pass
