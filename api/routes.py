"""API routes for face recognition service."""
import uuid
import json
from pathlib import Path
from typing import Union
import asyncio
import httpx
import cv2
import numpy as np

from fastapi import APIRouter, Request, HTTPException, WebSocket, WebSocketDisconnect

from wcm_facerec import __version__
from wcm_facerec.config import settings
from wcm_facerec.face_engine import get_face_engine
from .utils import (
    _download_url_safe,
    VIDEO_EXTENSIONS
)
from .handlers import (
    _search_video_frames,
    _process_detect_sensitive,
    _process_detect_nsfw,
    _process_analyze_media
)

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
            faces = await engine.detect_faces_async(img_array)
        elif isinstance(img_source, (str, Path)):
            # Local file - decode to numpy array so OpenCV fallback works
            img_array = cv2.imread(str(img_source), cv2.IMREAD_COLOR_BGR)
            faces = await engine.detect_faces_async(img_array)
        else:
            faces = await engine.detect_faces_async(img_source)

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
            category = form.get("category") or None
            record = await engine.register_from_image_async(
                name=name,
                img_source=img_source,
                file_url=file_url,
                category=category,
            )
            return {
                "id": str(record.id),
                "name": record.name,
                "file_path": record.file_path,
                "face_file_path": record.face_file_path,
                "category": category or settings.default_category,
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
        is_video = any(url.lower().endswith(ext) for ext in VIDEO_EXTENSIONS)

        if is_video:
            sample_interval = float(data.get("sample_interval", 1.0))
            frames, results = await _search_video_frames(
                engine, url, name, max(min(top_k, 10), 1),
                max(min(threshold, 1.0), 0.0), sample_interval
            )
            return {
                "results": results,
                "query_embedding_dim": settings.embedding_dim,
                "frames_processed": frames,
            }
        else:
            # Send image directly to DeepFace API via engine.search
            results = await asyncio.to_thread(
                engine.search,
                img_source=url,
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
            is_video = any(url.lower().endswith(ext) for ext in VIDEO_EXTENSIONS)

            try:
                if is_video:
                    frames, results = await _search_video_frames(
                        engine, url, name, top_k, threshold, sample_interval
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
                    results = await asyncio.to_thread(
                        engine.search,
                        img_source=img_bytes,
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


@api_bp.post("/detect_sensitive")
async def detect_sensitive(request: Request):
    """
    Extract text via OCR from an image/video URL and check for sensitive info using WasuGuard.
    """
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
        
    url = data.get("url")
    if not url:
        raise HTTPException(status_code=400, detail="url is required")
        
    sample_interval = float(data.get("sample_interval", 1.0))
    
    try:
        return await _process_detect_sensitive(url, sample_interval)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process media: {str(e)}")


@api_bp.websocket("/ws/detect_sensitive")
async def websocket_detect_sensitive(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            try:
                data = await websocket.receive_text()
                if not data or len(data) < 2: continue
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
            sample_interval = float(payload.get("sample_interval", 1.0))
            await websocket.send_json({"status": "accepted", "taskId": task_id})
            
            try:
                result = await _process_detect_sensitive(url, sample_interval)
                result["status"] = "completed"
                result["taskId"] = task_id
                await websocket.send_json(result)
            except Exception as e:
                await websocket.send_json({"status": "error", "taskId": task_id, "error": str(e)})
    except Exception as e:
        try:
            await websocket.send_json({"status": "error", "error": str(e)})
        except:
            pass

nsfw_pipeline = None




@api_bp.post("/detect_nsfw")
async def detect_nsfw(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
        
    url = body.get("url")
    if not url:
        raise HTTPException(status_code=400, detail="URL is required")
        
    sample_interval = float(body.get("sample_interval", 1.0))
    
    try:
        return await _process_detect_nsfw(url, sample_interval)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process media: {str(e)}")

@api_bp.websocket("/ws/detect_nsfw")
async def websocket_detect_nsfw(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            try:
                data = await websocket.receive_text()
                if not data or len(data) < 2: continue
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
            sample_interval = float(payload.get("sample_interval", 1.0))
            await websocket.send_json({"status": "accepted", "taskId": task_id})
            
            try:
                result = await _process_detect_nsfw(url, sample_interval)
                result["status"] = "completed"
                result["taskId"] = task_id
                await websocket.send_json(result)
            except Exception as e:
                await websocket.send_json({"status": "error", "taskId": task_id, "error": str(e)})
    except Exception as e:
        try:
            await websocket.send_json({"status": "error", "error": str(e)})
        except:
            pass


@api_bp.post("/analyze_media")
async def analyze_media(request: Request):
    """Analyze a single media file for faces, sensitive text, and NSFW content."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
        
    url = body.get("url")
    if not url:
        raise HTTPException(status_code=400, detail="URL is required")
        
    sample_interval = float(body.get("sample_interval", 1.0))
    top_k = int(body.get("top_k", 10))
    threshold = float(body.get("threshold", 0.4))
    
    try:
        return await _process_analyze_media(url, sample_interval, top_k, threshold)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process media: {str(e)}")

@api_bp.websocket("/ws/analyze_media")
async def websocket_analyze_media(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            try:
                data = await websocket.receive_text()
                if not data or len(data) < 2: continue
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
            sample_interval = float(payload.get("sample_interval", 1.0))
            top_k = int(payload.get("top_k", 10))
            threshold = float(payload.get("threshold", 0.4))
            
            await websocket.send_json({"status": "accepted", "taskId": task_id})
            
            try:
                result = await _process_analyze_media(url, sample_interval, top_k, threshold)
                result["status"] = "completed"
                result["taskId"] = task_id
                await websocket.send_json(result)
            except Exception as e:
                await websocket.send_json({"status": "error", "taskId": task_id, "error": str(e)})
    except Exception as e:
        try:
            await websocket.send_json({"status": "error", "error": str(e)})
        except:
            pass
