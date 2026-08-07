"""Core face recognition engine, now backed by InsightFace Server.

Public API is preserved so that existing callers (api/routes.py,
api/handlers.py, main.py, scripts/*.py) keep compiling:

    - FaceEngine(model_name, distance_metric)  # legacy kwargs ignored
    - engine.detect_faces(img_source) -> list[dict]
    - engine.generate_embedding(img_source) -> np.ndarray
    - engine.search(img_source, name=None, top_k=10, threshold=0.3) -> list[dict]
    - engine.search_multi_face(img_source, ...) -> dict  # new multi-face shape
    - engine.register_from_image(name, img_source, category=None, ...) -> dict
      (flat record-item dict from IFS; no SQL row written)
    - engine.verify_faces(img1, img2) -> bool
    - get_face_engine() -> FaceEngine  (cached singleton)

Internally every method delegates to InsightFaceAdapter. The legacy
``img_source`` overload (path | bytes | np.ndarray) is normalized to bytes
at the boundary; the route layer is expected to already download URLs.

Note: ``register_face`` (sync, DB-only) has been removed — there is no
local SQL table to write to anymore. IFS Person is the canonical record.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path

import cv2
import numpy as np

from .config import settings, warn_deprecated
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

        For backward compatibility this still returns a single flat list of
        matches sorted by distance. The implementation now scans every face
        in the query image (per the adapter) and concatenates the per-face
        match lists. The per-match ``face_index`` and ``query_face_bbox``
        fields identify which query face produced the match.

        ``threshold`` is on the legacy cosine-distance scale (lower = more
        similar). It is converted internally to a similarity floor
        ``1 - threshold`` before hitting InsightFace.
        """
        grouped = await self.search_multi_face(
            img_source,
            name=name,
            top_k=top_k,
            threshold=threshold,
        )
        return grouped["all_results"]

    async def search_multi_face(
        self,
        img_source: str | Path | bytes | np.ndarray,
        *,
        name: str | None = None,
        top_k: int = 10,
        threshold: float = 0.3,
        min_face_pixels: int = 80,
        max_faces: int = 10,
    ) -> dict:
        """Search the default collection for similar faces, every face in
        the query image at once.

        Returns a dict with:

            {
              "face_count": int,
              "faces": [
                {
                  "face_index": int,
                  "bbox": {"x", "y", "w", "h"},
                  "detection_score": float,
                  "matches": [<legacy match dict with face_index>],
                },
                ...
              ],
              "all_results": [<flat list, sorted by distance>],
            }
        """
        image_bytes = _to_bytes(img_source)
        # threshold on the legacy cosine-distance scale (lower=more similar).
        # 0.0 means "no filter" — pass through as no-op rather than a
        # min_similarity of 1.0 (which would silently return nothing).
        if float(threshold) <= 0.0:
            min_similarity = 0.0
        else:
            min_similarity = max(0.0, 1.0 - float(threshold))
        grouped = await asyncio.to_thread(
            self._adapter.search_multi_face,
            image_bytes,
            top_k=top_k,
            min_similarity=min_similarity,
            min_face_pixels=min_face_pixels,
            max_faces=max_faces,
        )

        # For each per-face match, fetch the matched face's bbox in IFS
        # (per face — keyed by matched_face_id) and synthesize a category
        # fallback from the file_path when metadata is empty.
        for face in grouped["faces"]:
            for m in face["matches"]:
                if name and m.get("name") != name:
                    m["_filtered"] = True
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
                if not m.get("category") and m.get("file_path"):
                    parts = m["file_path"].split("/", 4)
                    if len(parts) > 3:
                        m["category"] = parts[3]

        # Drop filtered matches from the structured view and rebuild the
        # flat list so the legacy shape is identical to the single-face
        # contract.
        for face in grouped["faces"]:
            face["matches"] = [m for m in face["matches"] if not m.pop("_filtered", False)]
        grouped["all_results"] = []
        for face in grouped["faces"]:
            grouped["all_results"].extend(face["matches"])
        grouped["all_results"].sort(key=lambda x: x["distance"])
        # Cap each face's matches to top_k (the adapter already does this
        # but the filter step may have removed some).
        for face in grouped["faces"]:
            face["matches"] = face["matches"][:top_k]
        return grouped

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
    async def register_from_image(
        self,
        name: str,
        img_source: str | Path | bytes | np.ndarray,
        category: str | None = None,
        *,
        occupation: str | None = None,
        type_: str | None = None,
        remarks: str | None = None,
    ) -> dict:
        """Persist bytes to ``/tmp/wcm`` and enroll into InsightFace Server.

        Writes into the aggregate collection (``insightface_collection_id``)
        and, when ``category`` is mapped in
        ``settings.insightface_category_collections``, also into the
        per-category collection. The per-category duplicate carries
        ``external_id=<aggregate_person_id>`` so an admin backfill can
        correlate them later.

        No local SQL row is written — IFS Person is the canonical record.
        Returns a flat record-item dict (see
        ``ifs_adapter._person_to_item``) keyed by the aggregate IFS
        ``Person.id`` (which becomes the webui's ``record.id``).
        """
        try:
            image_bytes = _to_bytes(img_source)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc

        cat = category or settings.default_category
        persisted_path = _persist_image(image_bytes, name, cat)

        # Metadata shared by both enrollments so list / search results
        # can echo the form fields back without a DB lookup.
        metadata = {
            "category": cat,
            "occupation": occupation or "",
            "type": type_ or "",
            "remarks": remarks or "",
            "file_path": persisted_path,
        }

        # Always enroll into the configured aggregate collection.
        person_id, _face_id = await self._run(
            self._adapter.register_person,
            name=name,
            image_bytes=image_bytes,
            metadata=metadata,
        )
        # Plus the per-category collection, when one is configured. Tag
        # the duplicate with external_id=<aggregate_id> for correlation.
        category_cid = settings.insightface_category_collections.get(cat)
        if category_cid and category_cid != settings.insightface_collection_id:
            await self._run(
                self._adapter.register_person,
                name=name,
                image_bytes=image_bytes,
                metadata=metadata,
                external_id=person_id,
                collection_id=category_cid,
            )

        # Read back the canonical aggregate record so we surface the
        # server-assigned `created_at` (and any server-side normalization
        # of name / metadata).
        record = await self._run(self._adapter.get_person, person_id)
        if not record:
            # Should not happen — we just created it — but be defensive.
            record = {"id": person_id, "name": name}
        return record

    # ------------------------------------------------------------------
    # Internal helper: run sync SDK calls without blocking the event loop.
    # ------------------------------------------------------------------
    @staticmethod
    async def _run(func, *args, **kwargs):
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
