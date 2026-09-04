"""API routes for face recognition service."""

import contextlib
import json
import uuid
from pathlib import Path

import cv2
import httpx
import numpy as np
from fastapi import APIRouter, HTTPException, Request, Response, WebSocket, WebSocketDisconnect

from wcm_facerec import __version__
from wcm_facerec.config import DEFAULT_DISTANCE_THRESHOLD, settings
from wcm_facerec.face_engine import get_face_engine

from . import review_task_store
from .face_records import _image_url_to_path, _item_image_urls
from .handlers import (
    _process_analyze_media,
    _process_detect_nsfw,
    _process_detect_sensitive,
    _search_video_frames,
)
from .utils import VIDEO_EXTENSIONS, _download_url_safe

api_bp = APIRouter()


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _opt_float(source, key: str) -> float | None:
    """Read an optional float from a form/dict, returning None when absent.

    Empty strings and missing keys both yield ``None``. Used by the search
    endpoints to plumb the optional scoring knobs (#1/#2/#3).
    """
    if source is None:
        return None
    val = source.get(key)
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


_SEARCH_RESULT_FIELDS = (
    "matched_face_id",
    "created_at",
    "occupation",
    "remarks",
    "face_count",
    "face_index",
    "query_face_bbox",
    "source_x",
    "source_y",
    "source_w",
    "source_h",
    "frame_time",
    "source_face_b64",
)


def _public_search_result(match: dict) -> dict:
    """Convert an internal engine match to the stable public API contract."""
    image_urls = _item_image_urls(match)
    result = {
        "id": match.get("id") or match.get("person_id"),
        "name": match.get("name") or match.get("person_name"),
        "similarity": match.get("effective_similarity", match.get("similarity")),
        "distance": match.get("effective_distance", match.get("distance")),
        "image_url": image_urls[0] if image_urls else None,
        "image_urls": image_urls,
        "type": match.get("type") or match.get("category"),
    }
    result.update({key: match[key] for key in _SEARCH_RESULT_FIELDS if key in match})
    return result


def _public_search_results(results: list[dict]) -> list[dict]:
    """One best match per person/query face, with the complete person gallery.

    IFS can return multiple enrolled samples for the same person. Their
    metadata contains a person cover, NOT a matched-sample image path.
    Preserve independent query faces/video frames but collapse sample hits.
    The score, matched_face_id and source bbox always come from one best hit;
    the gallery does not imply that every image passed the search threshold.
    """
    grouped: dict[tuple, dict] = {}
    for index, match in enumerate(results):
        result = _public_search_result(match)
        key = (
            result.get("frame_time"),
            result.get("face_index", 0),
            result["id"] or ("unknown", index),
        )
        previous = grouped.get(key)
        if previous is None:
            grouped[key] = result
            continue
        image_urls = list(dict.fromkeys(previous["image_urls"] + result["image_urls"]))
        if (result["similarity"] or 0) > (previous["similarity"] or 0):
            grouped[key] = result
        grouped[key]["image_urls"] = image_urls
        grouped[key]["image_url"] = image_urls[0] if image_urls else None
    return sorted(grouped.values(), key=lambda item: item["similarity"] or 0, reverse=True)


@api_bp.get("/health")
async def health_check():
    """Readiness check for the API and its primary IFS dependency."""
    engine = get_face_engine()
    try:
        upstream = await engine._run(engine._adapter.health)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail={"status": "unhealthy", "insightface": str(exc)},
        ) from exc
    try:
        review_tasks = await review_task_store.health()
    except review_task_store.ReviewTaskStoreUnavailable as exc:
        raise HTTPException(
            status_code=503,
            detail={"status": "unhealthy", "review_tasks": str(exc)},
        ) from exc
    dependencies = {"insightface": upstream.get("status", "healthy")}
    if review_tasks != "disabled":
        dependencies["review_tasks"] = review_tasks
    return {
        "status": "healthy",
        "model": settings.insightface_model_name,
        "embedding_dim": settings.embedding_dim,
        "version": __version__,
        "dependencies": dependencies,
    }


@api_bp.post("/detect")
async def detect_faces(request: Request):
    """Detect faces in an image.

    Accepts either an uploaded file or a URL via form data.
    """
    engine = get_face_engine()
    img_source: str | Path | bytes
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
            faces = await engine.detect_faces(img_array)
        elif isinstance(img_source, (str, Path)):
            # Local file - decode to numpy array so OpenCV fallback works
            img_array = cv2.imread(str(img_source), cv2.IMREAD_COLOR_BGR)
            faces = await engine.detect_faces(img_array)
        else:
            faces = await engine.detect_faces(img_source)

        results = []
        for i, face in enumerate(faces):
            results.append(
                {
                    "face_id": f"face_{i}",
                    "confidence": face.get("confidence", 0.0),
                    "facial_area": face.get("facial_area", {}),
                }
            )

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


@api_bp.post("/register", status_code=410)
async def register_face(request: Request):
    """Deprecated. Use ``POST /face_records`` instead.

    The original endpoint was broken (referenced ``record.face_file_path``
    which the FaceRecord model never had, and passed a non-existent
    ``file_url=`` kwarg into ``register_from_image``). After the move to
    InsightFace Server this route is permanently retired; the new
    ``/face_records`` endpoint enforces single-face registration server-side
    and supports the same form fields.
    """
    raise HTTPException(
        status_code=410,
        detail="POST /register has been retired. Use POST /face_records instead.",
    )


@api_bp.post("/search")
async def search_faces(request: Request):
    """Search for similar faces in the database."""
    engine = get_face_engine()
    content_type = request.headers.get("content-type", "")

    img_bytes = None
    name = None
    top_k = 10
    threshold = DEFAULT_DISTANCE_THRESHOLD
    quality_weight: float | None = None
    norm_reference: float | None = None
    adaptive_threshold_step: float | None = None

    if "multipart/form-data" in content_type:
        form = await request.form()
        file = form.get("file")
        if not file or not file.filename:
            raise HTTPException(status_code=400, detail="file is required for multipart search")
        img_bytes = await file.read()
        name = form.get("name")
        top_k = int(form.get("top_k", 10))
        threshold = float(form.get("threshold", DEFAULT_DISTANCE_THRESHOLD))
        quality_weight = _opt_float(form, "quality_weight")
        norm_reference = _opt_float(form, "norm_reference")
        adaptive_threshold_step = _opt_float(form, "adaptive_threshold_step")
    else:
        # JSON body
        try:
            data = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body or request format")

        if not data:
            raise HTTPException(status_code=400, detail="Request body required")

        url = data.get("url")
        if not url:
            raise HTTPException(status_code=400, detail="url is required for JSON search")

        name = data.get("name")
        top_k = int(data.get("top_k", 10))
        threshold = float(data.get("threshold", DEFAULT_DISTANCE_THRESHOLD))
        quality_weight = _opt_float(data, "quality_weight")
        norm_reference = _opt_float(data, "norm_reference")
        adaptive_threshold_step = _opt_float(data, "adaptive_threshold_step")

        # Download from URL
        try:
            is_video = any(url.lower().endswith(ext) for ext in VIDEO_EXTENSIONS)
            if is_video:
                sample_interval = float(data.get("sample_interval", 1.0))
                frames, results = await _search_video_frames(
                    engine,
                    url,
                    name,
                    max(min(top_k, 10), 1),
                    max(min(threshold, 1.0), 0.0),
                    sample_interval,
                )
                return {
                    "results": _public_search_results(results),
                    "query_embedding_dim": settings.embedding_dim,
                    "frames_processed": frames,
                }
            else:
                img_bytes = await _download_url_safe(url, settings.max_file_size_mb * 1024 * 1024)
        except httpx.HTTPError as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch image: {str(e)}")

    # Execute face search
    try:
        results = await engine.search(
            img_source=img_bytes,
            name=name,
            top_k=max(min(top_k, 10), 1),
            threshold=max(min(threshold, 1.0), 0.0),
            quality_weight=quality_weight,
            norm_reference=norm_reference,
            adaptive_threshold_step=adaptive_threshold_step,
        )
        public_results = _public_search_results(results)
        for result in public_results:
            urls = result["image_urls"]
            scores = await engine.compare_gallery(
                img_bytes,
                [_image_url_to_path(url) for url in urls],
                result.get("query_face_bbox"),
            )
            result["image_similarities"] = dict(zip(urls, scores, strict=True))
        return {
            "results": public_results,
            "query_embedding_dim": settings.embedding_dim,
        }
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
            threshold = float(payload.get("threshold", DEFAULT_DISTANCE_THRESHOLD))
            sample_interval = float(payload.get("sample_interval", 1.0))

            await websocket.send_json({"status": "accepted", "taskId": task_id})

            engine = get_face_engine()
            is_video = any(url.lower().endswith(ext) for ext in VIDEO_EXTENSIONS)

            try:
                if is_video:
                    frames, results = await _search_video_frames(
                        engine, url, name, top_k, threshold, sample_interval
                    )
                    await websocket.send_json(
                        {
                            "status": "completed",
                            "taskId": task_id,
                            "query_embedding_dim": settings.embedding_dim,
                            "frames_processed": frames,
                            "results": _public_search_results(results),
                        }
                    )
                else:
                    img_bytes = await _download_url_safe(
                        url, settings.max_file_size_mb * 1024 * 1024
                    )
                    results = await engine.search(
                        img_source=img_bytes,
                        name=name,
                        top_k=max(min(top_k, 10), 1),
                        threshold=max(min(threshold, 1.0), 0.0),
                    )
                    await websocket.send_json(
                        {
                            "status": "completed",
                            "taskId": task_id,
                            "query_embedding_dim": settings.embedding_dim,
                            "results": _public_search_results(results),
                        }
                    )
                    continue

            except httpx.HTTPError as e:
                await websocket.send_json(
                    {"status": "error", "taskId": task_id, "error": f"Failed to fetch: {str(e)}"}
                )
            except Exception as e:
                await websocket.send_json(
                    {"status": "error", "taskId": task_id, "error": f"Search failed: {str(e)}"}
                )

    except Exception as e:
        with contextlib.suppress(Exception):
            await websocket.send_json({"status": "error", "error": str(e)})


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
                if not data or len(data) < 2:
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
        with contextlib.suppress(Exception):
            await websocket.send_json({"status": "error", "error": str(e)})


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
                if not data or len(data) < 2:
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
        with contextlib.suppress(Exception):
            await websocket.send_json({"status": "error", "error": str(e)})


@api_bp.post("/analyze_media")
async def analyze_media(request: Request, response: Response):
    """Analyze media for faces, sensitive text, and NSFW content.

    Consecutive video face hits with the same category/name use a timestamp
    range (HH:MM:SS.mmm~HH:MM:SS.mmm). Other findings remain individual points.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    url = body.get("url")
    if not url:
        raise HTTPException(status_code=400, detail="URL is required")

    sample_interval = float(body.get("sample_interval", 1.0))
    top_k = int(body.get("top_k", 10))
    threshold = float(body.get("threshold", DEFAULT_DISTANCE_THRESHOLD))
    parameters = {key: value for key, value in body.items() if key != "url"}
    parameters.update(
        {"sample_interval": sample_interval, "top_k": top_k, "threshold": threshold}
    )

    try:
        task_id = await review_task_store.create(url, parameters)
    except review_task_store.ReviewTaskStoreUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        result = await _process_analyze_media(url, sample_interval, top_k, threshold)
    except Exception as e:
        with contextlib.suppress(Exception):
            await review_task_store.fail(task_id, str(e))
        raise HTTPException(status_code=400, detail=f"Failed to process media: {str(e)}")

    stored_results = result.get("results", []) if isinstance(result, dict) else result
    try:
        await review_task_store.complete(task_id, stored_results)
    except review_task_store.ReviewTaskStoreUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if task_id:
        response.headers["X-Review-Task-ID"] = task_id
    return result


@api_bp.websocket("/ws/analyze_media")
async def websocket_analyze_media(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            try:
                data = await websocket.receive_text()
                if not data or len(data) < 2:
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
            sample_interval = float(payload.get("sample_interval", 1.0))
            top_k = int(payload.get("top_k", 10))
            threshold = float(payload.get("threshold", DEFAULT_DISTANCE_THRESHOLD))

            parameters = {key: value for key, value in payload.items() if key != "url"}
            parameters.update(
                {"sample_interval": sample_interval, "top_k": top_k, "threshold": threshold}
            )
            try:
                await review_task_store.create(url, parameters, task_id)
            except review_task_store.ReviewTaskStoreUnavailable as exc:
                await websocket.send_json({"status": "error", "error": str(exc)})
                continue

            await websocket.send_json({"status": "accepted", "taskId": task_id})

            try:
                result = await _process_analyze_media(url, sample_interval, top_k, threshold)
                stored_results = result.get("results", []) if isinstance(result, dict) else result
                await review_task_store.complete(task_id, stored_results)
                await websocket.send_json(
                    {"status": "completed", "taskId": task_id, "results": stored_results}
                )
            except Exception as e:
                with contextlib.suppress(Exception):
                    await review_task_store.fail(task_id, str(e))
                await websocket.send_json({"status": "error", "taskId": task_id, "error": str(e)})
    except Exception as e:
        with contextlib.suppress(Exception):
            await websocket.send_json({"status": "error", "error": str(e)})
