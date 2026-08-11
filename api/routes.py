"""API routes for face recognition service."""

import contextlib
import json
import uuid
from pathlib import Path

import cv2
import httpx
import numpy as np
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect

from wcm_facerec import __version__
from wcm_facerec.config import settings
from wcm_facerec.face_engine import get_face_engine

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


def _path_to_image_url(file_path: str | None) -> str | None:
    """Map an absolute ``/tmp/wcm/<cat>/<name>_<md5>.<ext>`` path to the
    web-served ``/images/...`` URL that Nginx fronts."""
    if not file_path:
        return None
    p = Path(file_path)
    try:
        rel = p.relative_to("/tmp/wcm")
        return f"/images/{rel}"
    except ValueError:
        return f"/images/{p.name}"


def _item_with_person(item: dict) -> dict:
    """Wrap an IFS person-item dict into the {id, name, file_path,
    image_url, created_at, person: {...}} shape the webui expects."""
    image_url = _path_to_image_url(item.get("file_path"))
    return {
        "id": item.get("id"),
        "name": item.get("name"),
        "file_path": item.get("file_path"),
        "image_url": image_url,
        "created_at": item.get("created_at"),
        "person": {
            "id": item.get("id"),
            "name": item.get("name"),
            "occupation": item.get("occupation"),
            "type": item.get("type"),
            "remarks": item.get("remarks"),
        },
    }


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
    threshold = 0.4
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
        threshold = float(form.get("threshold", 0.4))
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
        threshold = float(data.get("threshold", 0.4))
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
                    "results": results,
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
        return {
            "results": results,
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
                    await websocket.send_json(
                        {
                            "status": "completed",
                            "taskId": task_id,
                            "query_embedding_dim": settings.embedding_dim,
                            "frames_processed": frames,
                            "results": results,
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
                            "results": results,
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
        with contextlib.suppress(Exception):
            await websocket.send_json({"status": "error", "error": str(e)})


# --- CRUD Endpoints for Face Records (IFS-backed) ---


# How many persons to fetch per IFS page when listing. IFS caps
# ``list_persons`` at ``limit <= 100``; pulling 100 per page keeps the
# client-side `type=` filter efficient (fewer round-trips when most rows
# get dropped).
_LIST_PAGE_FETCH = 100

# The three webui category tabs map directly to IFS category collections
# via ``insightface_category_collections``. The ``其它`` tab and ``All``
# read the ``all-persons`` aggregate. A record "belongs to" a category if
# its ``metadata.type`` equals one of these labels; empty/unknown types
# fall into ``其它``.
_CATEGORY_LABELS = ("劣迹艺人", "时政敏感", "落马官员")


@api_bp.get("/face_records")
async def list_face_records(page: int = 1, limit: int = 12, search: str = None, type: str = None):
    """List face records, paginated, with optional name search and type filter.

    Category tabs read their own IFS collection (authoritative), rather
    than filtering the ``all-persons`` aggregate by ``metadata.type`` —
    that snapshot has empty ``type`` on almost every row, so category tabs
    matched nothing. The ``type`` label is injected into each record so
    the webui badge/filter still works.

    * ``type ∈ 劣迹艺人/时政敏感/落马官员`` → its category collection.
    * ``type = 其它`` → ``all-persons`` rows whose type is empty/unknown.
    * ``type = All`` (or absent) → all ``all-persons`` rows.

    ``search=`` matches names server-side. ``total`` is the count returned
    on this page (not a global count); pagination uses ``has_more``.
    """
    engine = get_face_engine()
    items: list[dict] = []
    cursor: str | None = None
    fetch_limit = max(_LIST_PAGE_FETCH, limit)

    # Resolve the (collection_id, inject_type) for this tab.
    cid: str | None = None
    inject_type: str | None = None
    if type and type != "All":
        if type in _CATEGORY_LABELS:
            cid = settings.insightface_category_collections.get(type)
            inject_type = type
        else:
            # 其它 / unknown label: scan all-persons for rows with no
            # recognized type (skip the type-injection; we filter below).
            pass

    try:
        while len(items) < limit:
            page_items, cursor = await engine._run(
                engine._adapter.list_persons,
                limit=fetch_limit,
                cursor=cursor,
                search=search or None,
                collection_id=cid,
            )
            if not page_items:
                break
            for p in page_items:
                if inject_type:
                    # Category tab: force the label so the badge renders.
                    p["type"] = inject_type
                elif (type not in (None, "All") and type not in _CATEGORY_LABELS
                      and (p.get("type") or "") in _CATEGORY_LABELS):
                    # 其它 (or unknown) tab: only rows with no recognized type.
                    continue
                items.append(_item_with_person(p))
                if len(items) >= limit:
                    break
            if not cursor:
                break

        return {
            "items": items,
            "total": len(items),
            "page": page,
            "limit": limit,
            "has_more": bool(cursor),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api_bp.get("/face_records/stats")
async def get_face_records_stats():
    """Return per-category person counts from the three IFS category
    collections.

    The webui's three category cards (劣迹艺人 / 时政敏感 / 落马官员)
    map to the ``insightface_category_collections`` setting. The
    ``total`` is the **sum of the three category collections' person
    counts**, not the ``all-persons`` aggregate count.

    We deliberately do NOT count ``all-persons`` by ``metadata.type``:
    historical imports left most of that aggregate snapshot with an
    empty ``type``, so a type-bucket walk undercounts badly (it showed
    4 where the real category data holds ~4700). Reading each category
    collection's authoritative ``person_count`` (one call each, no
    pagination) makes the webui match the database.
    """
    engine = get_face_engine()
    counts = {
        "total": 0,
        "bad_artists": 0,
        "political": 0,
        "officials": 0,
    }
    # (settings key, stats output key)
    buckets = [
        ("劣迹艺人", "bad_artists"),
        ("时政敏感", "political"),
        ("落马官员", "officials"),
    ]
    try:
        for category, out_key in buckets:
            cid = settings.insightface_category_collections.get(category)
            if not cid:
                continue
            stats = await engine._run(engine._adapter.collection_stats, cid)
            counts[out_key] = stats["person_count"]
            counts["total"] += stats["person_count"]
        return counts
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api_bp.post("/face_records")
async def create_face_record(request: Request):
    """Create a new face record by enrolling it into InsightFace Server.

    Requires exactly 1 face (single-face enforcement runs against IFS via
    ``engine.detect_faces``). All form fields become IFS Person
    ``name`` / ``metadata``; the webui's record id is the IFS aggregate
    ``Person.id`` returned by the engine.
    """
    engine = get_face_engine()

    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" not in content_type:
        raise HTTPException(status_code=400, detail="Content-Type must be multipart/form-data")

    form = await request.form()
    name = form.get("name")
    if not name:
        raise HTTPException(status_code=400, detail="姓名是必填项")

    occupation = form.get("occupation") or None
    type_val = form.get("type") or None
    remarks = form.get("remarks") or None
    category = form.get("category") or None

    file = form.get("file")
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="请上传图片")

    contents = await file.read()
    if len(contents) > settings.max_file_size_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail="文件过大")

    # Validate the image is well-formed before paying for an IFS round-trip.
    nparr = np.frombuffer(contents, np.uint8)
    img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_array is None:
        raise HTTPException(status_code=400, detail="图片格式错误，无法解析")

    try:
        local_faces = await engine.detect_faces(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"人脸检测失败: {str(e)}")

    if len(local_faces) == 0:
        raise HTTPException(status_code=400, detail="未检测到人脸，请上传包含单张清晰人脸的图片")
    elif len(local_faces) > 1:
        raise HTTPException(
            status_code=400,
            detail=f"检测到多个人脸({len(local_faces)}个)，请上传仅包含单张清晰人脸的图片",
        )

    try:
        record = await engine.register_from_image(
            name=name,
            img_source=contents,
            category=category,
            occupation=occupation,
            type_=type_val,
            remarks=remarks,
        )
        return _item_with_person(record)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"注册人脸失败: {str(e)}")


@api_bp.put("/face_records/{record_id}")
async def update_face_record(record_id: str, request: Request):
    """Update an existing record's name and ``metadata`` fields on IFS.

    The webui treats ``record_id`` as opaque (the IFS aggregate
    ``Person.id``). Only the aggregate ``all-persons`` Person is patched;
    per-category mirrors are intentionally NOT re-keyed on type change —
    delete + re-register to recategorize.
    """
    engine = get_face_engine()
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    new_name = data.get("name")
    # IFS PATCH replaces the entire ``metadata`` dict, so we must read the
    # current metadata and merge in only the keys the client sent. Without
    # this merge the other keys (category, occupation, type, file_path)
    # would silently disappear from the record on update.
    current = await engine._run(engine._adapter.get_person, record_id)
    if not current:
        raise HTTPException(status_code=404, detail="人脸记录不存在")
    merged_meta = {
        k: current.get(k)
        for k in ("category", "occupation", "type", "remarks", "file_path")
    }
    for key in ("occupation", "type", "remarks"):
        if key in data:
            merged_meta[key] = data[key]
    if not new_name and all(merged_meta[k] == current.get(k) for k in merged_meta):
        raise HTTPException(status_code=400, detail="no fields to update")

    try:
        updated = await engine._run(
            engine._adapter.update_person,
            record_id,
            name=new_name,
            metadata=merged_meta,
        )
        if not updated:
            raise HTTPException(status_code=404, detail="人脸记录不存在")
        return _item_with_person(updated)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api_bp.delete("/face_records/{record_id}")
async def delete_face_record(record_id: str):
    """Delete an IFS Person (aggregate) and its /tmp/wcm image file.

    Per-category mirrors (when present from legacy batch-script imports)
    are not removed by this endpoint — they remain on the server until an
    admin backfill is run. The webui treats this as acceptable.
    """
    engine = get_face_engine()
    try:
        person = await engine._run(engine._adapter.get_person, record_id)
        if not person:
            raise HTTPException(status_code=404, detail="人脸记录不存在")
        file_path = person.get("file_path")
        if file_path:
            p = Path(file_path)
            if p.exists():
                try:
                    p.unlink()
                except Exception as e:
                    print(f"Warning: Failed to delete file {p}: {e}")
        await engine._run(engine._adapter.delete_person, record_id)
        return {"message": "人脸记录及人物信息删除成功"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
