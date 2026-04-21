"""API routes for face recognition service."""

import asyncio
import io
import json
import os
import uuid
from pathlib import Path
from typing import Optional, Union

import cv2
import httpx
import numpy as np
from fastapi import APIRouter, Request, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from PIL import Image

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
        embedding = await engine.generate_embedding_async(url)

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


@api_bp.post("/video/register")
async def register_video_faces(request: Request):
    """Register all faces from a video file.

    Samples frames at the specified interval and registers all detected faces.
    """
    engine = get_face_engine()

    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" not in content_type:
        raise HTTPException(status_code=400, detail="Content-Type must be multipart/form-data")

    form = await request.form()
    name = form.get("name")
    if not name:
        raise HTTPException(status_code=400, detail="name is required")

    sample_interval = float(form.get("sample_interval", 1.0))
    if sample_interval < 0.1 or sample_interval > 10.0:
        sample_interval = 1.0

    video_path = None
    try:
        if "file" in form:
            file = form.get("file")
            if file and file.filename:
                contents = await file.read()
                if len(contents) > settings.max_file_size_mb * 1024 * 1024:
                    raise HTTPException(status_code=413, detail="File too large")
                video_path = Path(f"/tmp/facerec_video_{os.urandom(8).hex()}.mp4")
                with open(video_path, "wb") as f:
                    f.write(contents)
        elif form.get("url"):
            url = form.get("url")
            resp = httpx.get(url, timeout=120.0)
            resp.raise_for_status()
            video_path = Path(f"/tmp/facerec_video_{os.urandom(8).hex()}.mp4")
            with open(video_path, "wb") as f:
                f.write(resp.content)
        else:
            raise HTTPException(status_code=400, detail="Either file or url must be provided")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_idx = 0
        registered_count = 0
        errors = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % int(fps * sample_interval) == 0:
                try:
                    # Save frame temporarily
                    temp_frame_path = Path(f"/tmp/frame_{os.urandom(8).hex()}.jpg")
                    cv2.imwrite(str(temp_frame_path), frame)

                    # Detect and register faces
                    faces = engine.detect_faces(temp_frame_path)
                    for i, face_data in enumerate(faces):
                        try:
                            face_img = face_data["face"]
                            face_confidence = face_data.get("confidence", 0.0)

                            # Save face and get embedding
                            face_temp = Path(f"/tmp/face_{os.urandom(8).hex()}.jpg")
                            cv2.imwrite(str(face_temp), face_img)

                            embedding = engine.generate_embedding(face_temp)
                            engine.register_face(
                                name=name,
                                embedding=embedding,
                                file_path=str(video_path),
                                confidence=face_confidence,
                                face_id=f"frame_{frame_idx}_face_{i}",
                                frame_time=frame_idx / fps if fps > 0 else 0,
                            )
                            registered_count += 1
                            face_temp.unlink(missing_ok=True)
                        except Exception as e:
                            errors.append(f"Frame {frame_idx}, face {i}: {str(e)}")

                    temp_frame_path.unlink(missing_ok=True)
                except Exception as e:
                    errors.append(f"Frame {frame_idx}: {str(e)}")

            frame_idx += 1

        cap.release()
        return {
            "name": name,
            "total_frames_processed": frame_idx,
            "faces_registered": registered_count,
            "errors": errors if errors else None,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")
    finally:
        if video_path and video_path.exists():
            video_path.unlink()


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
                temp_frame_path = Path(f"/tmp/ws_frame_{os.urandom(8).hex()}.jpg")
                cv2.imwrite(str(temp_frame_path), frame)
                try:
                    faces = engine.detect_faces(temp_frame_path)
                    for face_data in faces:
                        face_img = face_data.get("face")
                        if face_img is not None:
                            _search_face_in_image(engine, face_img, name, top_k, threshold, all_results)
                finally:
                    temp_frame_path.unlink(missing_ok=True)

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
):
    """Search a single face from a frame."""
    face_temp = Path(f"/tmp/ws_face_{os.urandom(8).hex()}.jpg")
    cv2.imwrite(str(face_temp), face_img)
    try:
        embedding = engine.generate_embedding(face_temp)
        results = engine.search(
            embedding=embedding,
            name=name,
            top_k=top_k,
            threshold=threshold,
        )
        all_results.extend(results)
    finally:
        face_temp.unlink(missing_ok=True)


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
            data = await websocket.receive_text()
            payload = json.loads(data)

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

    except WebSocketDisconnect:
        pass
    except json.JSONDecodeError:
        await websocket.send_json({"status": "error", "error": "Invalid JSON"})
    except Exception as e:
        try:
            await websocket.send_json({"status": "error", "error": str(e)})
        except:
            pass
