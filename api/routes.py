"""API routes for face recognition service."""

import io
import os
from pathlib import Path
from typing import Optional, Union

import cv2
import httpx
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from PIL import Image

from wcm_facerec import __version__
from wcm_facerec.config import settings
from wcm_facerec.face_engine import FaceEngine, get_face_engine

from .schemas import (
    DetectFaceResponse,
    FaceDetectionResult,
    FaceSearchResult,
    HealthResponse,
    RegisterFaceRequest,
    RegisterFaceResponse,
    SearchFaceRequest,
    SearchFaceResponse,
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model=settings.deepface_model,
        embedding_dim=settings.embedding_dim,
        version=__version__,
    )


@router.post("/detect", response_model=DetectFaceResponse)
async def detect_faces(
    file: UploadFile | None = File(None),
    url: str | None = Query(None),
):
    """Detect faces in an image.

    Accepts either an uploaded file or a URL.
    """
    engine = get_face_engine()
    img_source: Union[str, Path, bytes]

    if file:
        contents = await file.read()
        if len(contents) > settings.max_file_size_mb * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large")
        img_source = contents
        image_source = file.filename or "uploaded_file"
    elif url:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            img_source = response.content
            image_source = url
    else:
        raise HTTPException(status_code=400, detail="Either file or url must be provided")

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
            results.append(FaceDetectionResult(
                face_id=f"face_{i}",
                confidence=face.get("confidence", 0.0),
                facial_area=face.get("facial_area", {}),
            ))

        return DetectFaceResponse(
            faces=results,
            image_source=image_source,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
    finally:
        if isinstance(img_source, Path) and str(img_source).startswith("/tmp/facerec_"):
            img_source.unlink(missing_ok=True)


@router.post("/register", response_model=RegisterFaceResponse)
async def register_face(
    name: str = Form(...),
    file: UploadFile | None = File(None),
    url: str | None = Form(None),
):
    """Register a face in the database.

    Accepts either an uploaded file or a URL.
    """
    engine = get_face_engine()
    img_source: Union[str, Path, bytes]

    if file:
        contents = await file.read()
        if len(contents) > settings.max_file_size_mb * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large")
        img_source = contents
    elif url:
        img_source = url
    else:
        raise HTTPException(status_code=400, detail="Either file or url must be provided")

    try:
        record = await engine.register_from_image(
            name=name,
            img_source=img_source,
            file_url=url if file else None,
        )
        return RegisterFaceResponse(
            id=str(record.id),
            name=record.name,
            message="Face registered successfully",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")


@router.post("/search", response_model=SearchFaceResponse)
async def search_faces(request: SearchFaceRequest):
    """Search for similar faces in the database."""
    engine = get_face_engine()

    if not request.url:
        raise HTTPException(status_code=400, detail="url is required for search")

    try:
        # Generate embedding from URL
        embedding = await engine.generate_embedding_async(request.url)

        results = engine.search(
            embedding=embedding,
            name=request.name,
            top_k=request.top_k,
            threshold=request.threshold,
        )

        return SearchFaceResponse(
            results=[FaceSearchResult(**r) for r in results],
            query_embedding_dim=settings.embedding_dim,
        )
    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/video/register")
async def register_video_faces(
    name: str = Form(...),
    file: UploadFile | None = File(None),
    url: str | None = Form(None),
    sample_interval: float = Form(1.0, ge=0.1, le=10.0, description="Seconds between frame samples"),
):
    """Register all faces from a video file.

    Samples frames at the specified interval and registers all detected faces.
    """
    engine = get_face_engine()

    if file:
        contents = await file.read()
        if len(contents) > settings.max_file_size_mb * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large")
        video_path = Path(f"/tmp/facerec_video_{os.urandom(8).hex()}.mp4")
        with open(video_path, "wb") as f:
            f.write(contents)
    elif url:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            video_path = Path(f"/tmp/facerec_video_{os.urandom(8).hex()}.mp4")
            with open(video_path, "wb") as f:
                f.write(response.content)
    else:
        raise HTTPException(status_code=400, detail="Either file or url must be provided")

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")
    finally:
        video_path.unlink(missing_ok=True)
