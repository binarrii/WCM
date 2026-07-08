"""Core face recognition engine using DeepFace."""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Optional, Union

import httpx
import cv2
import numpy as np
from sqlalchemy import text

from .config import settings
from .database import FaceRecord, get_session, register_vector_type

# Minimum face area in pixels (128x128)
MIN_FACE_PIXELS = 128 * 128


def _detect_image_ext(image_bytes: bytes) -> str:
    """Detect image file extension from magic bytes. Defaults to .jpg."""
    if len(image_bytes) >= 3 and image_bytes[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if len(image_bytes) >= 8 and image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if len(image_bytes) >= 6 and image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return ".webp"
    if len(image_bytes) >= 2 and image_bytes[:2] == b"BM":
        return ".bmp"
    return ".jpg"


def _persist_image(
    image_bytes: bytes,
    name: str,
    category: str,
    ext: Optional[str] = None,
) -> str:
    """Save image bytes under ``<data_root>/<category>/<name>_<md5><ext>``.

    Returns the absolute file path. Reuses the existing file if the hash
    already exists (idempotent), so re-registering the same image does
    not create duplicates.
    """
    safe_name = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in name) or "unknown"
    content_hash = hashlib.md5(image_bytes).hexdigest()
    final_ext = ext or _detect_image_ext(image_bytes)
    target_dir = Path("/tmp/wcm") / category
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{safe_name}_{content_hash}{final_ext}"
    if not target_path.exists():
        target_path.write_bytes(image_bytes)
    return str(target_path)


def _l2_normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """Return a L2-normalized copy of an embedding."""
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm


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
        self.api_url = settings.deepface_api_url

    def _prepare_image(self, img_source: Union[str, Path, bytes, np.ndarray]) -> str:
        import base64
        if isinstance(img_source, np.ndarray):
            _, buf = cv2.imencode('.jpg', img_source)
            b64_img = base64.b64encode(buf).decode('utf-8')
            return f"data:image/jpeg;base64,{b64_img}"
        elif isinstance(img_source, bytes):
            b64_img = base64.b64encode(img_source).decode('utf-8')
            return f"data:image/jpeg;base64,{b64_img}"
        elif isinstance(img_source, (str, Path)):
            with open(img_source, "rb") as f:
                b64_img = base64.b64encode(f.read()).decode('utf-8')
                return f"data:image/jpeg;base64,{b64_img}"
        return ""

    async def _extract_faces_from_video_frame(self, frame: np.ndarray, frame_idx: int, fps: float) -> list[dict]:
        """Extract faces from a video frame."""
        
        img_b64 = self._prepare_image(frame)
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{self.api_url}/represent",
                    json={
                        "img": img_b64,
                        "model_name": self.model_name,
                        "detector_backend": "retinaface",
                        "enforce_detection": False,
                        "align": True
                    }
                )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            
            faces = []
            for res in results:
                fa = res.get("facial_area", {})
                w = fa.get("w", 0) or 0
                h = fa.get("h", 0) or 0
                x = fa.get("x", 0) or 0
                y = fa.get("y", 0) or 0
                
                if min(w, h) < 80:
                    continue
                    
                face_crop = frame[max(0, y):y+h, max(0, x):x+w]
                if face_crop.size == 0:
                    continue
                    
                faces.append({
                    "face": face_crop,
                    "confidence": res.get("face_confidence", 1.0),
                    "facial_area": fa,
                    "area": w * h,
                    "embedding": np.array(res["embedding"]) if "embedding" in res else None
                })

            # Sort by area and take top 3
            sorted_faces = sorted(faces, key=lambda f: f["area"], reverse=True)
            top_faces = sorted_faces[:3]

            results_out = []
            for i, face in enumerate(top_faces):
                emb = face["embedding"]
                if emb is not None and self.distance_metric == "euclidean_l2":
                    emb = _l2_normalize_embedding(emb)
                    
                results_out.append({
                    "face": face["face"],
                    "confidence": face["confidence"],
                    "face_id": f"frame_{frame_idx}_face_{i}",
                    "frame_time": frame_idx / fps if fps > 0 else 0,
                    "embedding": emb,
                })
            return results_out
        except Exception as e:
            print(f"DeepFace API Error (_extract_faces_from_video_frame): {e}")
            return []

    async def generate_embedding(self, img_source: Union[str, Path, bytes, np.ndarray]) -> np.ndarray:
        """Generate face embedding asynchronously (supports bytes and arrays).

        Args:
            img_source: Path to local file, image bytes, or numpy array

        Returns:
            Face embedding as numpy array
        """
        
        img_b64 = self._prepare_image(img_source)
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{self.api_url}/represent",
                    json={
                        "img": img_b64,
                        "model_name": self.model_name,
                        "detector_backend": "retinaface",
                        "enforce_detection": False,
                        "align": True
                    }
                )
                resp.raise_for_status()
                data = resp.json()
                embedding_array = np.array(data["results"][0]["embedding"])
                if self.distance_metric == "euclidean_l2":
                    embedding_array = _l2_normalize_embedding(embedding_array)
                return embedding_array
        except Exception as e:
            print(f"DeepFace API Error (generate_embedding): {e}")
            raise

    async def detect_faces(self, img_source: Union[str, Path, np.ndarray]) -> list[dict]:
        """Detect faces in an image asynchronously.

        Args:
            img_source: Path to image file, URL, or numpy array

        Returns:
            List of detected face dictionaries with 'face', 'confidence', 'facial_area', 'embedding'
        """
        
        img_b64 = self._prepare_image(img_source)
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{self.api_url}/represent",
                    json={
                        "img": img_b64,
                        "model_name": self.model_name,
                        "detector_backend": "retinaface",
                        "enforce_detection": False,
                        "align": True
                    }
                )
                resp.raise_for_status()
                data = resp.json()
                results = data.get("results", [])
            
            # Load original image to crop faces manually since API only returns coordinates
            if isinstance(img_source, np.ndarray):
                img_array = img_source
            else:
                img_array = cv2.imread(str(img_source), cv2.IMREAD_COLOR)

            faces = []
            for res in results:
                fa = res.get("facial_area", {})
                w = fa.get("w", 0) or 0
                h = fa.get("h", 0) or 0
                x = fa.get("x", 0) or 0
                y = fa.get("y", 0) or 0
                
                # Filter: short side (min of w/h) must be >= 80 pixels
                if min(w, h) < 80:
                    continue
                    
                face_crop = img_array[max(0, y):y+h, max(0, x):x+w]
                if face_crop.size == 0:
                    continue
                
                faces.append({
                    "face": face_crop,
                    "confidence": res.get("face_confidence", 1.0),
                    "facial_area": fa,
                    "area": w * h,
                    "embedding": np.array(res["embedding"]) if "embedding" in res else None
                })
            
            # Sort by area descending, keep top 3
            faces = sorted(faces, key=lambda f: f["area"], reverse=True)[:3]
            
            # normalize embeddings
            for f in faces:
                if f["embedding"] is not None and self.distance_metric == "euclidean_l2":
                    f["embedding"] = _l2_normalize_embedding(f["embedding"])
                    
            return faces
        except Exception as e:
            print(f"DeepFace API Error (detect_faces): {e}")
            return []


    async def search(
        self,
        img_source: Union[str, Path, bytes, np.ndarray],
        name: Optional[str] = None,
        top_k: int = 10,
        threshold: float = 0.4,
    ) -> list[dict]:
        
        img_b64 = await __import__('asyncio').to_thread(self._prepare_image, img_source)
        
        matches = []
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{self.api_url}/search",
                    json={
                        "img": img_b64,
                        "model_name": self.model_name,
                        "detector_backend": "retinaface",
                        "align": True,
                        "enforce_detection": False,
                    }
                )
                resp.raise_for_status()
                data = resp.json()
                
            results = data.get("results", [])
            if not results:
                return []
                
            valid_matches = []
            valid_uuids = set()
            
            for face_results in results:
                for match_dict in face_results:
                    identity = match_dict.get("identity")
                    distance = match_dict.get("distance")
                    if distance is not None and distance > threshold:
                        continue
                    if not identity:
                        continue
                    try:
                        record_uuid = uuid.UUID(str(identity))
                        valid_uuids.add(str(record_uuid))
                        valid_matches.append(match_dict)
                    except ValueError:
                        continue
                        
            if not valid_uuids:
                return []

            def db_query():
                session = get_session()
                from sqlalchemy import text
                in_clause = ",".join(f"'{u}'" for u in valid_uuids)
                sql = text(f"""
                    SELECT
                        fr.id, fr.name, fr.file_path, fr.face_file_path, fr.file_url, fr.confidence,
                        fr.person_id, fr.frame_time, fr.created_at,
                        p.name as person_name, p.occupation, p."type", p.remarks
                    FROM face_records fr
                    LEFT JOIN persons p ON fr.person_id = p.id
                    WHERE fr.id IN ({in_clause})
                """)
                rows = session.execute(sql).fetchall()
                session.close()
                return {str(row.id): row for row in rows}

            db_records = await __import__('asyncio').to_thread(db_query)
            
            for match_dict in valid_matches:
                identity = str(uuid.UUID(str(match_dict.get("identity"))))
                row = db_records.get(identity)
                if not row:
                    continue
                    
                if name and row.name != name:
                    continue
                    
                distance = match_dict.get("distance")
                match = {
                    "id": str(row.id),
                    "name": row.name,
                    "file_path": row.file_path,
                    "face_file_path": row.face_file_path,
                    "file_url": row.file_url,
                    "distance": float(distance),
                    "confidence": row.confidence,
                    "person_id": str(row.person_id) if row.person_id else None,
                    "frame_time": row.frame_time,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                    "source_x": match_dict.get("source_x"),
                    "source_y": match_dict.get("source_y"),
                    "source_w": match_dict.get("source_w"),
                    "source_h": match_dict.get("source_h"),
                    "person_name": row.person_name,
                    "occupation": row.occupation,
                    "type": getattr(row, "type", getattr(row, "type_", None)),
                    "remarks": row.remarks,
                }
                matches.append(match)
                
            matches.sort(key=lambda x: x["distance"])
            return matches[:top_k]
            
        except Exception as e:
            print(f"DeepFace API Error (search): {e}")
            return []


    def register_face(
        self,
        name: str,
        embedding: np.ndarray,
        file_path: Optional[str] = None,
        face_file_path: Optional[str] = None,
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
            face_file_path: Optional cropped face file path
            file_url: Optional URL
            confidence: Detection confidence
            face_id: For video: which face identifier
            frame_time: For video: timestamp in seconds

        Returns:
            Created FaceRecord
        """
        session = get_session()
        register_vector_type(session.connection())

        if self.distance_metric == "euclidean_l2":
            embedding = _l2_normalize_embedding(embedding)

        record = FaceRecord(
            id=uuid.uuid4(),
            name=name,
            embedding=embedding.tolist(),
            file_path=file_path,
            face_file_path=face_file_path,
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

    async def register_from_image(
        self,
        name: str,
        img_source: Union[str, Path, bytes],
        file_url: Optional[str] = None,
        category: Optional[str] = None,
    ) -> FaceRecord:
        """Register a face from an image file or bytes.

        The image is persisted under ``<data_root>/<category>/`` with a
        content-hashed filename, and the resulting absolute path is stored
        in the database record as ``file_path``. Re-registering the same
        image is a no-op for the file (the existing file is reused).

        Args:
            name: Person name
            img_source: Path to local image file, or image bytes
            file_url: Optional URL (stored but not used for loading)
            category: Subdirectory under data_root. Defaults to
                ``settings.default_category``.

        Returns:
            Created FaceRecord

        Raises:
            ValueError: If no face is detected in the image
        """
        cat = category or settings.default_category
        source_path: Optional[Path] = None
        ext = None
        if isinstance(img_source, str) and (img_source.startswith("http://") or img_source.startswith("https://")):
            
            resp = httpx.get(img_source)
            resp.raise_for_status()
            image_bytes = resp.content
        elif isinstance(img_source, (str, Path)):
            source_path = Path(img_source)
            if not source_path.exists():
                raise ValueError(f"Image file not found: {source_path}")
            image_bytes = source_path.read_bytes()
            ext = source_path.suffix.lower() or None
        else:
            image_bytes = bytes(img_source)

        # Persist to /data/wcm/<category>/<name>_<md5><ext>
        persisted_path = _persist_image(image_bytes, name, cat, ext=ext)

        # Create FaceRecord first to get the ID
        # Since we use DeepFace's DB for embeddings, we don't save embedding here
        # (Assuming the DB schema allows embedding to be null, or we just pass a zero array)
        session = get_session()
        from wcm_facerec.database import register_vector_type
        register_vector_type(session.connection())
        
        record = FaceRecord(
            id=uuid.uuid4(),
            name=name,
            file_path=persisted_path,
            file_url=file_url,
            model=self.model_name,
            embedding=[0.0] * self.embedding_dim, # Dummy embedding to satisfy NOT NULL if not altered
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        session.close()

        
        img_b64 = self._prepare_image(image_bytes)
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{self.api_url}/register",
                    json={
                        "img": img_b64,
                        "img_name": str(record.id),
                        "model_name": self.model_name,
                        "detector_backend": "retinaface",
                        "align": True,
                        "enforce_detection": False,
                    }
                )
                resp.raise_for_status()
        except Exception as e:
            # Rollback record if API fails
            session = get_session()
            session.delete(record)
            session.commit()
            session.close()
            print(f"DeepFace API Error (register_from_image): {e}")
            raise ValueError(f"Failed to register face via official API: {e}")

        return record

    async def verify_faces(self, img1: Union[str, Path, np.ndarray], img2: Union[str, Path, np.ndarray]) -> bool:
        """Verify if two faces are the same using the DeepFace API asynchronously.

        Applies a tighter distance threshold than DeepFace's built-in one
        (``settings.verify_distance_threshold``) to reject borderline
        look-alikes that the default ~0.30 cutoff would let through.

        Args:
            img1: First image path, url or numpy array.
            img2: Second image path, url or numpy array.

        Returns:
            True if verified, False otherwise.
        """
        
        img1_b64 = self._prepare_image(img1)
        img2_b64 = self._prepare_image(img2)
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{self.api_url}/verify",
                    json={
                        "img1": img1_b64,
                        "img2": img2_b64,
                        "model_name": self.model_name,
                        "distance_metric": self.distance_metric,
                        "detector_backend": "retinaface",
                        "enforce_detection": False,
                        "align": True
                    }
                )
                resp.raise_for_status()
                data = resp.json()
                
                distance = data.get("distance")
                if distance is None:
                    return False
                return float(distance) <= settings.verify_distance_threshold
        except Exception as e:
            print(f"DeepFace API Error (verify_faces): {e}")
            return False


# Global engine instance
_engine: Optional[FaceEngine] = None


def get_face_engine() -> FaceEngine:
    """Get or create global FaceEngine instance."""
    global _engine
    if _engine is None:
        _engine = FaceEngine()
    return _engine
