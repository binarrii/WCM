"""Core face recognition engine, now backed by InsightFace Server.

Public API is preserved so that existing callers (api/routes.py,
api/handlers.py, main.py, scripts/*.py) keep compiling:

    - FaceEngine(model_name, distance_metric)  # legacy kwargs ignored
    - engine.detect_faces(img_source) -> list[dict]
    - engine.generate_embedding(img_source) -> np.ndarray
    - engine.search(img_source, name=None, top_k=10, threshold=0.3) -> list[dict]
    - engine.register_face(name, file_path=None) -> FaceRecord  (sync, DB-only)
    - engine.register_from_image(name, img_source, category=None) -> FaceRecord
    - engine.verify_faces(img1, img2) -> bool
    - get_face_engine() -> FaceEngine  (cached singleton)

Internally every method delegates to InsightFaceAdapter. The legacy
``img_source`` overload (path | bytes | np.ndarray) is normalized to bytes
at the boundary; the route layer is expected to already download URLs.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from pathlib import Path

import cv2
import numpy as np

from .config import settings, warn_deprecated
from .database import FaceRecord, get_session
from .ifs_adapter import InsightFaceAdapter

logger = logging.getLogger(__name__)


# Minimum face area in pixels (kept for back-compat with scripts).
MIN_FACE_PIXELS = 32 * 32


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
    ext: str | None = None,
) -> str:
    """Save image bytes under ``/tmp/wcm/<category>/<name>_<md5><ext>``.

    Returns the absolute file path. Reuses the existing file if the hash
    already exists (idempotent).
    """
    safe_name = (
        "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in name) or "unknown"
    )
    content_hash = hashlib.md5(image_bytes).hexdigest()
    final_ext = ext or _detect_image_ext(image_bytes)
    target_dir = Path("/tmp/wcm") / category
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{safe_name}_{content_hash}{final_ext}"
    if not target_path.exists():
        target_path.write_bytes(image_bytes)
    return str(target_path)


def _to_bytes(img_source: str | Path | bytes | np.ndarray) -> bytes:
    """Coerce the legacy ``img_source`` overload to raw bytes.

    Raises ValueError for URLs — the route layer owns URL downloads now.
    """
    if isinstance(img_source, np.ndarray):
        ok, buf = cv2.imencode(".jpg", img_source)
        if not ok:
            raise ValueError("failed to JPEG-encode ndarray")
        return buf.tobytes()
    if isinstance(img_source, (bytes, bytearray)):
        return bytes(img_source)
    if isinstance(img_source, (str, Path)):
        s = str(img_source)
        if s.startswith(("http://", "https://")):
            raise ValueError(
                "URLs must be downloaded by the route layer via _download_url_safe "
                "before calling FaceEngine — engine accepts bytes only."
            )
        return Path(s).read_bytes()
    raise TypeError(f"unsupported img_source type: {type(img_source)!r}")


class FaceEngine:
    """Face recognition engine backed by InsightFace Server."""

    def __init__(
        self,
        model_name: str | None = None,
        distance_metric: str | None = None,
    ):
        """Args are accepted for back-compat with the old DeepFace signature
        and are ignored — InsightFace Server ships a single fixed model
        (buffalo_m, 512-dim ArcFace R50)."""
        warn_deprecated()
        self.model_name = model_name or "Facenet512"
        self.distance_metric = distance_metric or "cosine"
        self.embedding_dim = settings.embedding_dim
        self.api_url = settings.insightface_base_url
        self._adapter = InsightFaceAdapter(
            base_url=settings.insightface_base_url,
            collection_id=settings.insightface_collection_id,
            timeout=settings.insightface_timeout_s,
            api_key=settings.insightface_api_key or None,
        )

    # ------------------------------------------------------------------
    # Read paths (legacy contract preserved)
    # ------------------------------------------------------------------
    async def detect_faces(self, img_source: str | Path | bytes | np.ndarray) -> list[dict]:
        """Detect faces in an image.

        Accepts the legacy overload (path | bytes | np.ndarray). Each
        result has the original ``{face, confidence, facial_area, area,
        embedding}`` dict. Sorted by area desc, top 3, ``min(w,h) >= 80``
        filter applied inside the adapter. Embeddings are populated by
        default — the legacy DeepFace path returned them inline.
        """
        try:
            image_bytes = _to_bytes(img_source)
        except (TypeError, ValueError):
            return []
        return await self._run(
            self._adapter.detect,
            image_bytes,
            max_keep=3,
            include_embeddings=True,
        )

    async def generate_embedding(self, img_source: str | Path | bytes | np.ndarray) -> np.ndarray:
        """Generate a 512-d float32 embedding for the most prominent face."""
        image_bytes = _to_bytes(img_source)
        return await self._run(self._adapter.embed, image_bytes)

    async def search(
        self,
        img_source: str | Path | bytes | np.ndarray,
        name: str | None = None,
        top_k: int = 10,
        threshold: float = 0.3,
    ) -> list[dict]:
        """Search the default collection for similar faces.

        ``threshold`` is on the legacy cosine-distance scale (lower = more
        similar). It is converted internally to a similarity floor
        ``1 - threshold`` before hitting InsightFace.
        """
        image_bytes = _to_bytes(img_source)
        min_similarity = max(0.0, 1.0 - float(threshold))
        matches = await self._run(
            self._adapter.search,
            image_bytes,
            top_k=top_k,
            min_similarity=min_similarity,
        )
        # Drop matches whose name doesn't match the requested filter, fetch
        # the bbox for each remaining hit in parallel, and synthesize the
        # ``category`` fallback from the file_path if metadata is empty
        # (legacy FaceEngine derived category from /tmp/wcm/<...>/<cat>/...).
        out: list[dict] = []
        for m in matches:
            if name and m.get("name") != name:
                continue
            bbox = await self._run(
                self._adapter.get_face_bbox,
                m["person_id"],
                m["matched_face_id"],
            )
            if bbox:
                m["source_x"] = bbox["x"]
                m["source_y"] = bbox["y"]
                m["source_w"] = bbox["w"]
                m["source_h"] = bbox["h"]
            # Legacy fallback: category parsed from file_path segments
            if not m.get("category") and m.get("file_path"):
                parts = m["file_path"].split("/", 4)
                if len(parts) > 3:
                    m["category"] = parts[3]
            out.append(m)
        out.sort(key=lambda x: x["distance"])
        return out[:top_k]

    async def verify_faces(
        self,
        img1: str | Path | bytes | np.ndarray,
        img2: str | Path | bytes | np.ndarray,
    ) -> bool:
        """Return True iff similarity ≥ ``insightface_verify_similarity_threshold``.

        Legacy callers passed URLs/paths/arrays; we coerce to bytes here.
        """
        a = _to_bytes(img1)
        b = _to_bytes(img2)
        sim = await self._run(self._adapter.compare, a, b)
        return sim >= settings.insightface_verify_similarity_threshold

    # ------------------------------------------------------------------
    # Write paths
    # ------------------------------------------------------------------
    def register_face(
        self,
        name: str,
        file_path: str | None = None,
    ) -> FaceRecord:
        """Sync DB-only insert (no remote call). Preserved for back-compat
        with scripts that want to record a row without enrolling."""
        session = get_session()
        try:
            record = FaceRecord(
                id=uuid.uuid4(),
                name=name,
                file_path=file_path,
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            return record
        finally:
            session.close()

    async def register_from_image(
        self,
        name: str,
        img_source: str | Path | bytes | np.ndarray,
        category: str | None = None,
        *,
        occupation: str | None = None,
        type_: str | None = None,
        remarks: str | None = None,
        external_id: str | None = None,
    ) -> FaceRecord:
        """Persist bytes, write a FaceRecord, enroll into InsightFace.

        Writes into both the aggregate collection (``insightface_collection_id``)
        and the per-category collection if ``category`` matches
        ``settings.insightface_category_collections``. Returns the local
        FaceRecord; InsightFace persons' ``external_id`` is set to this
        record's UUID so future ``search`` calls can recover ``id`` /
        ``created_at``.
        """
        try:
            image_bytes = _to_bytes(img_source)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc

        cat = category or settings.default_category
        persisted_path = _persist_image(image_bytes, name, cat)

        # Local FaceRecord (UUID PK)
        session = get_session()
        try:
            record = FaceRecord(
                id=uuid.uuid4(),
                name=name,
                file_path=persisted_path,
            )
            session.add(record)
            session.commit()
            session.refresh(record)
        finally:
            session.close()

        record_id_str = str(record.id)
        # Stash the path under metadata so search results can echo it back.
        metadata = {
            "category": cat,
            "occupation": occupation or "",
            "type": type_ or "",
            "remarks": remarks or "",
            "file_path": persisted_path,
        }

        # Always enroll into the configured aggregate collection.
        await self._run(
            self._adapter.register_person,
            name=name,
            image_bytes=image_bytes,
            metadata=metadata,
            external_id=record_id_str,
        )
        # Plus the per-category collection, when one is configured.
        category_cid = settings.insightface_category_collections.get(cat)
        if category_cid and category_cid != settings.insightface_collection_id:
            await self._run(
                self._adapter.register_person,
                name=name,
                image_bytes=image_bytes,
                metadata=metadata,
                external_id=record_id_str,
                collection_id=category_cid,
            )
        return record

    # ------------------------------------------------------------------
    # Internal helper: run sync SDK calls without blocking the event loop.
    # ------------------------------------------------------------------
    @staticmethod
    async def _run(func, *args, **kwargs):
        import asyncio

        return await asyncio.to_thread(func, *args, **kwargs)


# Global engine instance
_engine: FaceEngine | None = None


def get_face_engine() -> FaceEngine:
    """Get or create the global FaceEngine instance."""
    global _engine
    if _engine is None:
        _engine = FaceEngine()
    return _engine


# Re-export the helpers so existing imports keep working.
__all__ = [
    "FaceEngine",
    "MIN_FACE_PIXELS",
    "get_face_engine",
    "_persist_image",
]
