"""Core face recognition engine using DeepFace."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Literal, Optional, Union

import cv2
import numpy as np
from deepface import DeepFace

from .config import settings
from .database import FaceRecord, Person, get_session, register_vector_type

# Minimum face area in pixels (128x128)
MIN_FACE_PIXELS = 128 * 128


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

    def _extract_faces_from_video_frame(self, frame: np.ndarray, frame_idx: int, fps: float) -> list[dict]:
        """Extract faces from a video frame."""
        try:
            faces = DeepFace.extract_faces(
                img_path=frame,
                enforce_detection=False,
                align=True,
            )
            if not faces:
                return []

            # Sort by area and take top 3
            def get_face_area(f):
                fa = f.get("facial_area", {})
                return (fa.get("w", 0) or 0) * (fa.get("h", 0) or 0)

            sorted_faces = sorted(faces, key=get_face_area, reverse=True)
            top_faces = sorted_faces[:3]

            results = []
            for i, face in enumerate(top_faces):
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
                detector_backend="skip",
                enforce_detection=False,
                align=True,
            )
        else:
            embedding = DeepFace.represent(
                img_path=str(img_source),
                model_name=self.model_name,
                detector_backend="skip",
                enforce_detection=False,
                align=True,
            )
        return np.array(embedding[0]["embedding"])

    async def generate_embedding_async(self, img_source: Union[str, Path, bytes, np.ndarray]) -> np.ndarray:
        """Generate face embedding asynchronously (supports bytes and arrays).

        Args:
            img_source: Path to local file, image bytes, or numpy array

        Returns:
            Face embedding as numpy array
        """
        if isinstance(img_source, np.ndarray):
            # Already a numpy array - use directly
            img_array = img_source
        elif isinstance(img_source, bytes):
            # Bytes - decode to numpy array via cv2
            nparr = np.frombuffer(img_source, np.uint8)
            img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            # Local file - load via cv2
            img_array = cv2.imread(str(img_source))

        embedding = DeepFace.represent(
            img_path=img_array,
            model_name=self.model_name,
            detector_backend="skip",
            enforce_detection=False,
            align=True,
        )
        return np.array(embedding[0]["embedding"])

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
                    enforce_detection=False,
                    align=True,
                )
            else:
                faces = DeepFace.extract_faces(
                    img_path=str(img_source),
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
        """Search for similar faces using pgvector's SQL operators.

        Args:
            embedding: Query embedding
            name: Optional name filter
            top_k: Number of results to return
            threshold: Similarity threshold (lower = more similar)

        Returns:
            List of matching face records with distance
        """
        from sqlalchemy import text

        session = get_session()
        register_vector_type(session.connection())

        # pgvector distance operators:
        # <-> = Euclidean distance
        # <=> = Cosine distance
        if self.distance_metric == "cosine":
            op = "<=>"
        elif self.distance_metric == "euclidean":
            op = "<->"
        else:
            # euclidean_l2 - vectors are already normalized in DeepFace
            op = "<=>"

        # Convert numpy array to pgvector format [x,y,z]
        embedding_str = "[" + ",".join(str(x) for x in embedding.tolist()) + "]"

        # Build parameterized SQL with optional name filter
        if name:
            sql = text(f"""
                SELECT
                    fr.id, fr.name, fr.file_path, fr.file_url, fr.confidence,
                    fr.person_id, fr.frame_time, fr.created_at,
                    fr.embedding {op} :embedding AS distance,
                    p.name as person_name, p.occupation, p.type, p.remarks
                FROM face_records fr
                LEFT JOIN persons p ON fr.person_id = p.id
                WHERE fr.name = :name
                  AND fr.embedding {op} :embedding <= :threshold
                ORDER BY fr.embedding {op} :embedding
                LIMIT :top_k
            """)
            result = session.execute(sql, {"embedding": embedding_str, "name": name, "threshold": threshold, "top_k": top_k})
        else:
            sql = text(f"""
                SELECT
                    fr.id, fr.name, fr.file_path, fr.file_url, fr.confidence,
                    fr.person_id, fr.frame_time, fr.created_at,
                    fr.embedding {op} :embedding AS distance,
                    p.name as person_name, p.occupation, p.type, p.remarks
                FROM face_records fr
                LEFT JOIN persons p ON fr.person_id = p.id
                WHERE fr.embedding {op} :embedding <= :threshold
                ORDER BY fr.embedding {op} :embedding
                LIMIT :top_k
            """)
            result = session.execute(sql, {"embedding": embedding_str, "threshold": threshold, "top_k": top_k})

        matches = []
        for row in result:
            match = {
                "id": str(row.id),
                "name": row.name,
                "file_path": row.file_path,
                "file_url": row.file_url,
                "distance": float(row.distance),
                "confidence": row.confidence,
                "person_id": str(row.person_id) if row.person_id else None,
                "frame_time": row.frame_time,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            if row.person_name:
                match["person"] = {
                    "name": row.person_name,
                    "occupation": row.occupation,
                    "type": row.type,
                    "remarks": row.remarks,
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
        img_source: Union[str, Path, bytes],
        file_url: Optional[str] = None,
    ) -> FaceRecord:
        """Register a face from an image file or bytes.

        Args:
            name: Person name
            img_source: Path to local image file, or image bytes
            file_url: Optional URL (stored but not used for loading)

        Returns:
            Created FaceRecord

        Raises:
            ValueError: If no face is detected in the image
        """
        # Convert to numpy array via cv2
        if isinstance(img_source, bytes):
            # Bytes - decode via cv2
            nparr = np.frombuffer(img_source, np.uint8)
            img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            # Local file path - load via cv2
            img_array = cv2.imread(str(img_source))

        # Detect faces from numpy array
        faces = self.detect_faces(img_array)
        if not faces:
            raise ValueError(f"No face detected in image")

        # Filter faces by minimum area and sort by area
        def get_face_area(f):
            fa = f.get("facial_area", {})
            return (fa.get("w", 0) or 0) * (fa.get("h", 0) or 0)

        valid_faces = [f for f in faces if get_face_area(f) >= MIN_FACE_PIXELS]
        if not valid_faces:
            raise ValueError(f"No face with area >= {MIN_FACE_PIXELS} detected in image")

        sorted_faces = sorted(valid_faces, key=get_face_area, reverse=True)
        best_face = sorted_faces[0]["face"]
        confidence = sorted_faces[0].get("confidence")

        # Generate embedding from cropped face (numpy array)
        embedding = self.generate_embedding(best_face)

        return self.register_face(
            name=name,
            embedding=embedding,
            file_path=str(img_source) if isinstance(img_source, (str, Path)) else None,
            file_url=file_url,
            confidence=confidence,
        )


# Global engine instance
_engine: Optional[FaceEngine] = None


def get_face_engine() -> FaceEngine:
    """Get or create global FaceEngine instance."""
    global _engine
    if _engine is None:
        _engine = FaceEngine()
    return _engine
