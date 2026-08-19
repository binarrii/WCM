"""IFS-backed face-record CRUD and cursor pagination."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request

from wcm_facerec.config import settings
from wcm_facerec.face_engine import get_face_engine

face_records_bp = APIRouter()

_LIST_PAGE_FETCH = 100
_CATEGORY_LABELS = tuple(settings.insightface_category_collections)
_IMAGE_ROOT = Path("/tmp/wcm")


def _path_to_image_url(file_path: str | None) -> str | None:
    if not file_path:
        return None
    path = Path(file_path)
    try:
        relative = path.relative_to(_IMAGE_ROOT)
    except ValueError:
        relative = Path(path.name)

    image_root = _IMAGE_ROOT.resolve()
    target = (image_root / relative).resolve()
    try:
        target.relative_to(image_root)
    except ValueError:
        return None
    if not target.is_file():
        return None
    return f"/images/{relative.as_posix()}"


def _item_with_person(item: dict, *, aggregate_id: str | None = None) -> dict:
    """Return the stable aggregate id even for legacy category mirrors."""
    # ``external_id`` is only a cross-collection reference on category
    # mirrors.  Some legacy aggregate records also carry an external id, but
    # it is not a valid IFS Person id and therefore cannot be used for CRUD.
    record_id = aggregate_id or item.get("id")
    return {
        "id": record_id,
        "name": item.get("name"),
        "file_path": item.get("file_path"),
        "image_url": _path_to_image_url(item.get("file_path")),
        "created_at": item.get("created_at"),
        "person": {
            "id": record_id,
            "name": item.get("name"),
            "occupation": item.get("occupation"),
            "type": item.get("type"),
            "remarks": item.get("remarks"),
        },
    }


def _encode_cursor(server_cursor: str | None, offset: int = 0) -> str:
    payload = json.dumps(
        {"server_cursor": server_cursor, "offset": offset},
        separators=(",", ":"),
    ).encode()
    return base64.urlsafe_b64encode(payload).decode().rstrip("=")


def _decode_cursor(cursor: str | None) -> tuple[str | None, int]:
    if not cursor:
        return None, 0
    try:
        padding = "=" * (-len(cursor) % 4)
        payload = json.loads(base64.urlsafe_b64decode(cursor + padding))
        server_cursor = payload.get("server_cursor")
        offset = int(payload.get("offset", 0))
        if server_cursor is not None and not isinstance(server_cursor, str):
            raise ValueError
        if offset < 0:
            raise ValueError
        return server_cursor, offset
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail="invalid cursor") from exc


@face_records_bp.get("/face_records")
async def list_face_records(
    limit: int = Query(12, ge=1, le=100),
    cursor: str | None = None,
    search: str | None = None,
    type: str | None = None,
):
    """List records using an opaque cursor that never skips filtered rows."""
    engine = get_face_engine()
    server_cursor, offset = _decode_cursor(cursor)
    items: list[dict] = []

    collection_id: str | None = None
    inject_type: str | None = None
    filter_other = bool(type and type != "All" and type not in _CATEGORY_LABELS)
    if type in _CATEGORY_LABELS:
        collection_id = settings.insightface_category_collections.get(type)
        inject_type = type

    fetch_limit = _LIST_PAGE_FETCH if filter_other else limit
    try:
        while len(items) < limit:
            request_cursor = server_cursor
            page_items, next_server_cursor = await engine._run(
                engine._adapter.list_persons,
                limit=fetch_limit,
                cursor=request_cursor,
                search=search or None,
                collection_id=collection_id,
            )
            index = offset
            while index < len(page_items):
                person = page_items[index]
                index += 1
                if inject_type:
                    person["type"] = inject_type
                elif filter_other and (person.get("type") or "") in _CATEGORY_LABELS:
                    continue
                aggregate_id = None
                if collection_id:
                    aggregate_id = person.get("external_id")
                    if not aggregate_id:
                        # Legacy category imports used a category-local id such as
                        # ``p-bad-artists-00001`` while the aggregate collection
                        # prefixed it with the category collection id.  Returning
                        # the local id makes update/delete target the wrong
                        # collection, so reconstruct the canonical aggregate id.
                        aggregate_id = f"{collection_id}-{person['id']}"
                items.append(_item_with_person(person, aggregate_id=aggregate_id))
                if len(items) == limit:
                    if index < len(page_items):
                        next_cursor = _encode_cursor(request_cursor, index)
                    elif next_server_cursor:
                        next_cursor = _encode_cursor(next_server_cursor)
                    else:
                        next_cursor = None
                    return {
                        "items": items,
                        "total": len(items),
                        "limit": limit,
                        "next_cursor": next_cursor,
                        "has_more": next_cursor is not None,
                    }

            if not next_server_cursor:
                break
            server_cursor, offset = next_server_cursor, 0

        return {
            "items": items,
            "total": len(items),
            "limit": limit,
            "next_cursor": None,
            "has_more": False,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@face_records_bp.get("/face_records/stats")
async def get_face_records_stats():
    """Return aggregate total plus authoritative category counts."""
    engine = get_face_engine()
    counts = {"total": 0, "bad_artists": 0, "political": 0, "officials": 0}
    buckets = [
        ("劣迹艺人", "bad_artists"),
        ("时政敏感", "political"),
        ("落马官员", "officials"),
    ]
    try:
        total = await engine._run(
            engine._adapter.collection_stats,
            settings.insightface_collection_id,
        )
        counts["total"] = total["person_count"]
        for category, output_key in buckets:
            collection_id = settings.insightface_category_collections.get(category)
            if collection_id:
                stats = await engine._run(engine._adapter.collection_stats, collection_id)
                counts[output_key] = stats["person_count"]
        return counts
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@face_records_bp.post("/face_records")
async def create_face_record(request: Request):
    engine = get_face_engine()
    if "multipart/form-data" not in request.headers.get("content-type", ""):
        raise HTTPException(status_code=400, detail="Content-Type must be multipart/form-data")

    form = await request.form()
    name = form.get("name")
    if not name:
        raise HTTPException(status_code=400, detail="姓名是必填项")

    file = form.get("file")
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="请上传图片")
    contents = await file.read()
    if len(contents) > settings.max_file_size_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail="文件过大")
    if cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR) is None:
        raise HTTPException(status_code=400, detail="图片格式错误，无法解析")

    try:
        faces = await engine.detect_faces(contents)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"人脸检测失败: {exc}") from exc
    if not faces:
        raise HTTPException(status_code=400, detail="未检测到人脸，请上传包含单张清晰人脸的图片")
    if len(faces) > 1:
        raise HTTPException(
            status_code=400,
            detail=f"检测到多个人脸({len(faces)}个)，请上传仅包含单张清晰人脸的图片",
        )

    record_type = form.get("type") or settings.default_category
    try:
        record = await engine.register_from_image(
            name=name,
            img_source=contents,
            category=record_type,
            occupation=form.get("occupation") or None,
            type_=record_type,
            remarks=form.get("remarks") or None,
        )
        return _item_with_person(record)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"注册人脸失败: {exc}") from exc


@face_records_bp.put("/face_records/{record_id}")
async def update_face_record(record_id: str, request: Request):
    engine = get_face_engine()
    try:
        data = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON body") from exc

    metadata = {key: data[key] for key in ("occupation", "type", "remarks") if key in data}
    if not data.get("name") and not metadata:
        raise HTTPException(status_code=400, detail="no fields to update")
    try:
        updated = await engine.update_person_record(
            record_id,
            name=data.get("name"),
            metadata=metadata,
        )
        if not updated:
            raise HTTPException(status_code=404, detail="人脸记录不存在")
        return _item_with_person(updated)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@face_records_bp.delete("/face_records/{record_id}")
async def delete_face_record(record_id: str):
    engine = get_face_engine()
    try:
        deleted = await engine.delete_person_record(record_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="人脸记录不存在")
        return {"message": "人脸记录及人物信息删除成功"}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
