"""API routes for face recognition service."""

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Optional, Union

import cv2
import httpx
import numpy as np
from fastapi import APIRouter, Request, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect

from wcm_facerec import __version__
from wcm_facerec.config import settings
from wcm_facerec.face_engine import FaceEngine, get_face_engine

api_bp = APIRouter()


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
                resp = httpx.get(url, timeout=60.0)
                resp.raise_for_status()
                img_source = resp.content
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
            resp = httpx.get(url, timeout=60.0)
            resp.raise_for_status()
            img_source = resp.content
            image_source = url
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch image: {str(e)}")

    temp_path = None
    try:
        # Handle bytes - save temporarily
        if isinstance(img_source, bytes):
            temp_path = Path(f"/tmp/facerec_{os.urandom(8).hex()}.jpg")
            with open(temp_path, "wb") as f:
                f.write(img_source)
            img_source = temp_path

        faces = engine.detect_faces(img_source)

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
            img_source = form.get("url")
        else:
            raise HTTPException(status_code=400, detail="Either file or url must be provided")

        try:
            file_url = form.get("url") if "file" in form else None
            record = engine.register_from_image(
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
            frames, results = _search_video_frames(
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

            results = engine.search(
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
    """Download image, detect face, crop and return face embedding (in-memory).

    Returns dict with 'embedding' and 'confidence' if face found, None otherwise.
    """
    try:
        # Download image
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()

        # Decode to numpy array via cv2 (no temp file, no PIL)
        nparr = np.frombuffer(response.content, np.uint8)
        img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Detect faces using numpy array
        faces = engine.detect_faces(img_array)
        if not faces:
            return None

        # Use the largest face (by area = w * h)
        def get_face_area(f):
            fa = f.get("facial_area", {})
            return (fa.get("w", 0) or 0) * (fa.get("h", 0) or 0)
        best_face = max(faces, key=get_face_area)
        face_img = best_face.get("face")
        confidence = best_face.get("confidence", 0.0)

        if face_img is None:
            return None

        # Generate embedding from cropped face (numpy array)
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
        resp = httpx.get(url, timeout=120.0)
        resp.raise_for_status()
        with open(video_path, "wb") as f:
            f.write(resp.content)

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
                # Pass frame numpy array directly (no temp file)
                try:
                    faces = engine.detect_faces(frame)
                    for face_data in faces:
                        face_img = face_data.get("face")
                        if face_img is not None:
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

        return frame_idx, deduped[:top_k]
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
        all_results.extend(results)
    except Exception:
        pass  # Skip faces that fail embedding generation


@api_bp.websocket("/ws/search")
async def websocket_search(websocket: WebSocket):
    """WebSocket endpoint for async face search.

    Accepts: {"url": "https://example.com/xxx.mp4"} or {"url": "https://example.com/xxx.png"}
    Responds immediately: {"status": "accepted", "taskId": "xxxxxxx"}
    Then sends result: {"status": "completed", "taskId": "xxxxxxx", "query_embedding_dim": 4096, "results": [...]}
    """
    await websocket.accept()

    try:
        while True:
            try:
                data = await websocket.receive_text()
                if not data:
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
                    frames, results = _search_video_frames(engine, url, name, top_k, threshold, sample_interval)
                    await websocket.send_json({
                        "status": "completed",
                        "taskId": task_id,
                        "query_embedding_dim": settings.embedding_dim,
                        "frames_processed": frames,
                        "results": results,
                    })
                else:
                    embedding = await engine.generate_embedding_async(url)
                    results = engine.search(
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

            except httpx.HTTPError as e:
                await websocket.send_json({"status": "error", "taskId": task_id, "error": f"Failed to fetch: {str(e)}"})
            except Exception as e:
                await websocket.send_json({"status": "error", "taskId": task_id, "error": f"Search failed: {str(e)}"})

    except Exception as e:
        try:
            await websocket.send_json({"status": "error", "error": str(e)})
        except:
            pass
