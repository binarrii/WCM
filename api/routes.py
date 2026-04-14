"""API routes for face recognition service."""

import io
import os
from pathlib import Path
from typing import Optional, Union

import cv2
import httpx
import numpy as np
from flask import Blueprint, request, jsonify
from PIL import Image

from wcm_facerec import __version__
from wcm_facerec.config import settings
from wcm_facerec.face_engine import FaceEngine, get_face_engine

api_bp = Blueprint("api", __name__)


@api_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model": settings.deepface_model,
        "embedding_dim": settings.embedding_dim,
        "version": __version__,
    })


@api_bp.route("/detect", methods=["POST"])
def detect_faces():
    """Detect faces in an image.

    Accepts either an uploaded file or a URL via form data.
    """
    engine = get_face_engine()
    img_source: Union[str, Path, bytes]
    image_source = "unknown"

    # Get file upload or URL
    if "file" in request.files:
        file = request.files["file"]
        if file.filename:
            contents = file.read()
            if len(contents) > settings.max_file_size_mb * 1024 * 1024:
                return jsonify({"error": "File too large"}), 413
            img_source = contents
            image_source = file.filename
    elif request.form.get("url"):
        url = request.form.get("url")
        try:
            resp = httpx.get(url, timeout=60.0)
            resp.raise_for_status()
            img_source = resp.content
            image_source = url
        except Exception as e:
            return jsonify({"error": f"Failed to fetch image: {str(e)}"}), 400
    else:
        return jsonify({"error": "Either file or url must be provided"}), 400

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

        return jsonify({
            "faces": results,
            "image_source": image_source,
        })
    except Exception as e:
        return jsonify({"error": f"Detection failed: {str(e)}"}), 500
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()


@api_bp.route("/register", methods=["POST"])
def register_face():
    """Register a face in the database.

    Accepts form data with name and either file or url.
    """
    engine = get_face_engine()

    name = request.form.get("name")
    if not name:
        return jsonify({"error": "name is required"}), 400

    img_source: Union[str, Path, bytes]

    if "file" in request.files:
        file = request.files["file"]
        if file.filename:
            contents = file.read()
            if len(contents) > settings.max_file_size_mb * 1024 * 1024:
                return jsonify({"error": "File too large"}), 413
            img_source = contents
    elif request.form.get("url"):
        img_source = request.form.get("url")
    else:
        return jsonify({"error": "Either file or url must be provided"}), 400

    try:
        record = engine.register_from_image(
            name=name,
            img_source=img_source,
            file_url=request.form.get("url") if "file" in request.files else None,
        )
        return jsonify({
            "id": str(record.id),
            "name": record.name,
            "message": "Face registered successfully",
        })
    except Exception as e:
        return jsonify({"error": f"Registration failed: {str(e)}"}), 500


@api_bp.route("/search", methods=["POST"])
def search_faces():
    """Search for similar faces in the database."""
    engine = get_face_engine()

    data = request.get_json() or {}
    url = data.get("url")

    if not url:
        return jsonify({"error": "url is required for search"}), 400

    name = data.get("name")
    top_k = data.get("top_k", 10)
    threshold = data.get("threshold", 0.4)

    try:
        import asyncio
        embedding = asyncio.run(engine.generate_embedding_async(url))

        results = engine.search(
            embedding=embedding,
            name=name,
            top_k=top_k,
            threshold=threshold,
        )

        return jsonify({
            "results": results,
            "query_embedding_dim": settings.embedding_dim,
        })
    except httpx.HTTPError as e:
        return jsonify({"error": f"Failed to fetch image: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Search failed: {str(e)}"}), 500


@api_bp.route("/video/register", methods=["POST"])
def register_video_faces():
    """Register all faces from a video file.

    Samples frames at the specified interval and registers all detected faces.
    """
    engine = get_face_engine()

    name = request.form.get("name")
    if not name:
        return jsonify({"error": "name is required"}), 400

    sample_interval = float(request.form.get("sample_interval", 1.0))
    if sample_interval < 0.1 or sample_interval > 10.0:
        sample_interval = 1.0

    video_path = None
    try:
        if "file" in request.files:
            file = request.files["file"]
            if file.filename:
                contents = file.read()
                if len(contents) > settings.max_file_size_mb * 1024 * 1024:
                    return jsonify({"error": "File too large"}), 413
                video_path = Path(f"/tmp/facerec_video_{os.urandom(8).hex()}.mp4")
                with open(video_path, "wb") as f:
                    f.write(contents)
        elif request.form.get("url"):
            url = request.form.get("url")
            resp = httpx.get(url, timeout=120.0)
            resp.raise_for_status()
            video_path = Path(f"/tmp/facerec_video_{os.urandom(8).hex()}.mp4")
            with open(video_path, "wb") as f:
                f.write(resp.content)
        else:
            return jsonify({"error": "Either file or url must be provided"}), 400

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return jsonify({"error": "Could not open video file"}), 400

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
        return jsonify({
            "name": name,
            "total_frames_processed": frame_idx,
            "faces_registered": registered_count,
            "errors": errors if errors else None,
        })
    except Exception as e:
        return jsonify({"error": f"Video processing failed: {str(e)}"}), 500
    finally:
        if video_path and video_path.exists():
            video_path.unlink()
