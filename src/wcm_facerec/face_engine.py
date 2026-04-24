"""Core face recognition engine using DeepFace."""

from __future__ import annotations

import io
import os
import tempfile
import uuid
from pathlib import Path
from typing import Literal, Optional, Union

import httpx
import numpy as np
from deepface import DeepFace
from PIL import Image

from .config import settings
from .database import FaceRecord, Person, get_session, register_vector_type


class FaceEngine:
    """Face recognition engine using DeepFace."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        distance_metric: Optional[str] = None,
    ):
        """Initialize face engine.

        Args:
            model_name: DeepFace model to use (default from settings)
            distance_metric: Distance metric for comparison (default from settings)
        """
        self.model_name = model_name or settings.deepface_model
        self.distance_metric = distance_metric or settings.deepface_distance_metric
        self.embedding_dim = settings.embedding_dim

    def _load_image_from_path(self, path: str) -> np.ndarray:
        """Load image from disk path."""
        return np.array(Image.open(path))

    async def _load_image_from_url(self, url: str) -> np.ndarray:
        """Load image from URL."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))
            return np.array(image)

    def _extract_faces_from_video_frame(self, frame: np.ndarray, frame_idx: int, fps: float) -> list[dict]:
        """Extract faces from a video frame."""
        try:
            faces = DeepFace.extract_faces(
                img_path=frame,
                detector_backend="yolov8",
                enforce_detection=False,
                align=True,
            )
            results = []
            for i, face in enumerate(faces):
                results.append({
                    "face": face["face"],
                    "confidence": face["confidence"],
                    "face_id": f"frame_{frame_idx}_face_{i}",
                    "frame_time": frame_idx / fps if fps > 0 else 0,
                })
            return results
        except Exception:
            return []

    def generate_embedding(self, img_source: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Generate face embedding from image path, URL, or numpy array.

        Args:
            img_source: Path to image file, URL, or numpy array

        Returns:
            Face embedding as numpy array
        """
        # Pass numpy array directly to avoid file I/O
        if isinstance(img_source, np.ndarray):
            embedding = DeepFace.represent(
                img_path=img_source,
                model_name=self.model_name,
                enforce_detection=False,
                align=True,
            )
        else:
            embedding = DeepFace.represent(
                img_path=str(img_source),
                model_name=self.model_name,
                enforce_detection=False,
                align=True,
            )
        return np.array(embedding[0]["embedding"])

    async def generate_embedding_async(self, img_source: Union[str, Path, bytes, np.ndarray]) -> np.ndarray:
        """Generate face embedding asynchronously (supports URLs, bytes, and arrays).

        Args:
            img_source: Path, URL, image bytes, or numpy array

        Returns:
            Face embedding as numpy array
        """
        temp_path = None
        try:
            if isinstance(img_source, bytes):
                # Save bytes to temp file
                temp_path = Path(tempfile.gettempdir()) / f"wcm_emb_{uuid.uuid4().hex[:12]}.jpg"
                with open(temp_path, "wb") as f:
                    f.write(img_source)
            elif img_source.startswith(("http://", "https://")):
                # Download and save to temp file
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(img_source)
                    response.raise_for_status()
                ext = self._get_ext_from_content_type(response.headers.get("content-type", ""))
                temp_path = Path(tempfile.gettempdir()) / f"wcm_emb_{uuid.uuid4().hex[:12]}{ext}"
                with open(temp_path, "wb") as f:
                    f.write(response.content)
            else:
                # Local file - use as-is
                temp_path = img_source

            embedding = DeepFace.represent(
                img_path=str(temp_path),
                model_name=self.model_name,
                enforce_detection=False,
                align=True,
            )
            return np.array(embedding[0]["embedding"])
        finally:
            if temp_path and str(img_source).startswith(("http://", "https://")) and temp_path.exists():
                temp_path.unlink()

    def _get_ext_from_content_type(self, content_type: str) -> str:
        """Get file extension from content type."""
        ext_map = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
            "image/gif": ".gif",
            "image/bmp": ".bmp",
        }
        return ext_map.get(content_type.lower(), ".jpg")

    def detect_faces(self, img_source: Union[str, Path, np.ndarray]) -> list[dict]:
        """Detect faces in an image.

        Args:
            img_source: Path to image file, URL, or numpy array

        Returns:
            List of detected face dictionaries with 'face', 'confidence', 'facial_area'
        """
        try:
            # Pass numpy array directly to avoid file I/O
            if isinstance(img_source, np.ndarray):
                faces = DeepFace.extract_faces(
                    img_path=img_source,
                    detector_backend="yolov8",
                    enforce_detection=False,
                    align=True,
                )
            else:
                faces = DeepFace.extract_faces(
                    img_path=str(img_source),
                    detector_backend="yolov8",
                    enforce_detection=False,
                    align=True,
                )
            return faces
        except Exception as e:
            raise RuntimeError(f"Face detection failed: {e}")

    def search(
        self,
        embedding: np.ndarray,
        name: Optional[str] = None,
        top_k: int = 10,
        threshold: float = 0.4,
    ) -> list[dict]:
        """Search for similar faces in the database.

        Args:
            embedding: Query embedding
            name: Optional name filter
            top_k: Number of results to return
            threshold: Similarity threshold (lower = more similar)

        Returns:
            List of matching face records with distance
        """
        import json

        session = get_session()

        query = session.query(FaceRecord, Person).outerjoin(Person, FaceRecord.person_id == Person.id)
        if name:
            query = query.filter(FaceRecord.name == name)

        # Fetch all matching records (limit to 1000 for performance)
        results = query.limit(1000).all()

        # Calculate distances manually for JSON-stored embeddings
        distances = []
        for record, person in results:
            try:
                stored_embedding = json.loads(record.embedding) if isinstance(record.embedding, str) else record.embedding
                stored_vec = np.array(stored_embedding)

                if self.distance_metric == "cosine":
                    # Cosine distance
                    dot = np.dot(embedding, stored_vec)
                    norm_emb = np.linalg.norm(embedding)
                    norm_stored = np.linalg.norm(stored_vec)
                    dist = 1 - dot / (norm_emb * norm_stored)
                elif self.distance_metric == "euclidean":
                    # Euclidean distance
                    dist = np.linalg.norm(embedding - stored_vec)
                else:
                    # Euclidean L2 normalized
                    emb_norm = embedding / np.linalg.norm(embedding)
                    stored_norm = stored_vec / np.linalg.norm(stored_vec)
                    dist = np.linalg.norm(emb_norm - stored_norm)

                distances.append((dist, record, person))
            except Exception:
                continue

        # Sort by distance and take top_k
        distances.sort(key=lambda x: x[0])

        matches = []
        for dist, record, person in distances[:top_k]:
            if dist <= threshold:
                match = {
                    "id": str(record.id),
                    "name": record.name,
                    "file_path": record.file_path,
                    "file_url": record.file_url,
                    "distance": float(dist),
                    "confidence": record.confidence,
                    "person_id": str(record.person_id) if record.person_id else None,
                    "frame_time": record.frame_time,
                    "created_at": record.created_at.isoformat() if record.created_at else None,
                }
                if person:
                    match["person"] = {
                        "name": person.name,
                        "occupation": person.occupation,
                        "type": person.type_,
                        "remarks": person.remarks,
                    }
                matches.append(match)

        session.close()
        return matches

    def register_face(
        self,
        name: str,
        embedding: np.ndarray,
        file_path: Optional[str] = None,
        file_url: Optional[str] = None,
        confidence: Optional[float] = None,
        face_id: Optional[str] = None,
        frame_time: Optional[float] = None,
    ) -> FaceRecord:
        """Register a face in the database.

        Args:
            name: Person name
            embedding: Face embedding vector
            file_path: Optional local file path
            file_url: Optional URL
            confidence: Detection confidence
            face_id: For video: which face identifier
            frame_time: For video: timestamp in seconds

        Returns:
            Created FaceRecord
        """
        session = get_session()
        register_vector_type(session.connection())

        record = FaceRecord(
            id=uuid.uuid4(),
            name=name,
            embedding=embedding.tolist(),
            file_path=file_path,
            file_url=file_url,
            model=self.model_name,
            confidence=confidence,
            face_id=face_id,
            frame_time=frame_time,
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        session.close()
        return record

    def register_from_image(
        self,
        name: str,
        img_source: Union[str, Path],
        file_url: Optional[str] = None,
    ) -> FaceRecord:
        """Register a face from an image file or URL.

        Args:
            name: Person name
            img_source: Path or URL to image
            file_url: Optional URL (if img_source is a local path)

        Returns:
            Created FaceRecord
        """
        embedding = self.generate_embedding(img_source)
        return self.register_face(
            name=name,
            embedding=embedding,
            file_path=str(img_source) if not str(img_source).startswith(("http://", "https://")) else None,
            file_url=file_url,
        )


# Global engine instance
_engine: Optional[FaceEngine] = None


def get_face_engine() -> FaceEngine:
    """Get or create global FaceEngine instance."""
    global _engine
    if _engine is None:
        _engine = FaceEngine()
    return _engine
