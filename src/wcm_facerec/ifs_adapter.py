"""Thin wrapper around the vendored InsightFace Server SDK.

Owns every unit conversion between InsightFace's native shape and the legacy
FaceEngine dict contract that api/routes.py and api/handlers.py consume:

    InsightFace.search → matches[i].similarity (∈ [0,1], higher=better)
    FaceEngine.search  → matches[i]["distance"] (lower=better, cos-distance scale)

Plus the legacy fields that InsightFace carries in Person.metadata JSON:
occupation, type, remarks, category. The WCM `Person` row (Postgres) still
holds the canonical copy of those, but search results now need to surface
them from the metadata.
"""

from __future__ import annotations

import io
import json
import logging
from typing import Any

import numpy as np
from PIL import Image

from wcm_facerec.vendor.insightface_server import (  # type: ignore  # noqa: F401  (ImageInput re-export)
    Client,
    ImageInput,
)
from wcm_facerec.vendor.insightface_server.exceptions import NotFoundError

from .config import settings

logger = logging.getLogger(__name__)


class InsightFaceAdapter:
    """Synchronous adapter over the InsightFace Server SDK.

    All methods accept already-prepared bytes; URL/path/ndarray conversion is
    the caller's responsibility (it lives in api/utils.py and FaceEngine).
    """

    def __init__(
        self,
        base_url: str,
        collection_id: str,
        *,
        timeout: float = 60.0,
        api_key: str | None = None,
    ) -> None:
        self._collection_id = collection_id
        self._client = Client(
            base_url=base_url,
            api_key=api_key or None,
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # Health / readiness
    # ------------------------------------------------------------------
    def health(self) -> dict:
        """Returns the parsed JSON of GET /v1/health."""
        return dict(self._client.health().to_dict())

    # ------------------------------------------------------------------
    # Detection / embedding
    # ------------------------------------------------------------------
    def detect(
        self,
        image_bytes: bytes,
        *,
        min_face_pixels: int = 80,
        max_keep: int = 3,
        include_embeddings: bool = False,
    ) -> list[dict]:
        """Detect faces in an image.

        Mirrors the legacy FaceEngine contract: each result has
        ``{facial_area: {x, y, w, h}, confidence, area, embedding: np.ndarray | None}``.
        Returns at most ``max_keep`` faces, sorted by area desc, filtered by
        ``min(w, h) >= min_face_pixels``.

        ``include_embeddings=True`` makes a second call to ``/v1/embeddings`` to
        populate each face's ``embedding`` field. Off by default because the
        legacy ``/v1/detect`` endpoint returns no embeddings, and most callers
        that want an embedding call ``embed()`` directly. Pass ``True`` when
        you need both detection and per-face embeddings in one shot.
        """
        result = self._client.detect(image=image_bytes, max_faces=max_keep)
        faces = result.faces
        out: list[dict] = []
        np_img = _decode(image_bytes)

        embeddings_by_index: list[np.ndarray | None] = []
        if include_embeddings and faces:
            emb_result = self._client.embeddings(image=image_bytes)
            for ef in emb_result.faces or []:
                e = ef.get("embedding")
                embeddings_by_index.append(
                    np.asarray(e, dtype=np.float32) if e is not None else None
                )

        for i, f in enumerate(faces):
            bb = f.get("bbox", {}).get("pixels", {}) or {}
            x = int(bb.get("x", 0))
            y = int(bb.get("y", 0))
            w = int(bb.get("width", 0))
            h = int(bb.get("height", 0))
            if min(w, h) < min_face_pixels:
                continue
            confidence = float(f.get("detection_score") or 0.0)
            embedding_list = f.get("embedding")
            embedding: np.ndarray | None = None
            if embedding_list is not None:
                embedding = np.asarray(embedding_list, dtype=np.float32)
            elif i < len(embeddings_by_index):
                embedding = embeddings_by_index[i]
            face_crop = _crop_or_none(np_img, x, y, w, h)
            out.append(
                {
                    "face": face_crop,
                    "confidence": confidence,
                    "facial_area": {"x": x, "y": y, "w": w, "h": h},
                    "area": w * h,
                    "embedding": embedding,
                }
            )
        out.sort(key=lambda d: d["area"], reverse=True)
        return out[:max_keep]

    def embed(self, image_bytes: bytes) -> np.ndarray:
        """Return a 1-D float32 embedding (512-d) for the most prominent face.

        Raises IndexError if no face is detected.
        """
        result = self._client.embeddings(image=image_bytes)
        faces = result.faces
        if not faces:
            raise IndexError("no face detected")
        emb = faces[0].get("embedding")
        if emb is None:
            raise IndexError("embedding missing in response")
        return np.asarray(emb, dtype=np.float32)

    def compare(self, a_bytes: bytes, b_bytes: bytes) -> float:
        """Return the cosine similarity in [0,1] between two faces."""
        result = self._client.compare(source=a_bytes, target=b_bytes)
        return float(result.similarity)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(
        self,
        image_bytes: bytes,
        *,
        top_k: int,
        min_similarity: float,
    ) -> list[dict]:
        """Search the default collection for similar faces.

        Each returned dict has the legacy contract:

            id, name, distance, person_id, created_at,
            source_x, source_y, source_w, source_h,
            person_name, occupation, category, type, remarks,
            file_path, similarity, matched_face_id

        ``distance`` is computed as ``1 - similarity`` so legacy sort/filters
        keep working unchanged.
        """
        result = self._client.search(
            self._collection_id,
            image=image_bytes,
            limit=top_k,
            threshold=min_similarity,
        )
        matches = result.matches or []
        out: list[dict] = []
        for m in matches:
            person = m.get("person") or {}
            similarity = float(m.get("similarity") or 0.0)
            metadata = person.get("metadata") or {}
            if isinstance(metadata, str):
                # Defensive: server should already JSON-decode
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}
            out.append(
                {
                    "id": person.get("id"),
                    "name": person.get("name"),
                    "person_name": person.get("name"),
                    "person_id": person.get("id"),
                    "similarity": similarity,
                    "distance": 1.0 - similarity,
                    "matched_face_id": m.get("matched_face_id"),
                    "created_at": person.get("created_at"),
                    "file_path": metadata.get("file_path"),
                    "category": metadata.get("category"),
                    "occupation": metadata.get("occupation"),
                    "type": metadata.get("type"),
                    "remarks": metadata.get("remarks"),
                    # source_x/y/w/h populated lazily by FaceEngine.search after a
                    # per-match list_faces round-trip; default to query bbox here.
                    "source_x": None,
                    "source_y": None,
                    "source_w": None,
                    "source_h": None,
                }
            )
        return out

    # ------------------------------------------------------------------
    # Collection stats
    # ------------------------------------------------------------------
    def collection_stats(self, collection_id: str) -> dict:
        """Return person/face counts for a single IFS collection.

        Uses the server's ``get_collection`` (``person_count`` /
        ``face_count`` fields) — a single call, no pagination. Callers
        that need per-category counts should call this once per category
        collection and sum the ``person_count`` values.

        Returns ``{"person_count": int, "face_count": int}``.
        """
        result = self._client.get_collection(collection_id)
        coll = result.collection if hasattr(result, "collection") else result
        return {
            "person_count": int(coll.get("person_count") or 0),
            "face_count": int(coll.get("face_count") or 0),
        }

    # ------------------------------------------------------------------
    # Multi-face search
    # ------------------------------------------------------------------
    def search_multi_face(
        self,
        image_bytes: bytes,
        *,
        top_k: int = 5,
        min_similarity: float = 0.0,
        min_face_pixels: int = 80,
        max_faces: int = 10,
        quality_weight: float | None = None,
        norm_reference: float | None = None,
        adaptive_threshold_step: float | None = None,
    ) -> dict:
        """Detect every face in an image and search the collection for each one.

        InsightFace Server's /search endpoint takes one query image and
        returns matches for the *most prominent* face only — there is no
        server-side fan-out. To support multi-face queries we:

        1. POST /v1/detect (or /v1/embeddings) to enumerate every face in
           the image along with its bounding box.
        2. For each face, JPEG-encode a tight crop and POST
           /v1/collections/{cid}/search with the crop as the query image.
        3. Merge the per-face match lists, tagging each result with the
           originating face's index and bbox so the caller can render the
           original image with one frame per matched face.

        Optional scoring enhancements (all default to ``None`` = use the
        ``Settings`` value, which itself defaults to ``0.0`` = off):

        * ``quality_weight`` (#3): multiply similarity by
          ``(1 - w) + w * query_detection_score``. Penalizes matches
          against low-quality probe faces.
        * ``norm_reference`` (#2 MPS proxy): fetch the probe embedding
          via ``/v1/embeddings`` and weight matches by
          ``min(probe_norm / norm_reference, 1.0)``. MagFace-style
          quality signal. Pass ``0`` to disable even when the setting
          is non-zero.
        * ``adaptive_threshold_step`` (#1): for each matched Person,
          compute ``adaptive = base + step * min(face_count, 10)`` and
          drop matches below their per-Person threshold.

        Returns a dict with the shape:

            {
              "query_dim": 512,
              "face_count": int,
              "faces": [
                {
                  "face_index": int,
                  "bbox": {"x", "y", "w", "h"},
                  "detection_score": float,
                  "matches": [<same shape as single-face search>],
                },
                ...
              ],
              "all_results": [<flat list, sorted by effective_distance>],
            }
        """
        # Resolve scoring knobs against Settings when caller passed None.
        if quality_weight is None:
            quality_weight = settings.insightface_quality_weight
        if norm_reference is None:
            norm_reference = settings.insightface_norm_reference
        if adaptive_threshold_step is None:
            adaptive_threshold_step = settings.insightface_adaptive_threshold_step

        detected = self._client.detect(image=image_bytes, max_faces=max_faces)
        faces_raw = detected.faces or []
        np_img = _decode(image_bytes)

        # #2 norm-aware scoring: fetch probe embedding once if enabled.
        probe_norm: float | None = None
        if norm_reference and norm_reference > 0:
            try:
                emb_result = self._client.embeddings(image=image_bytes)
                if emb_result.faces:
                    raw_emb = emb_result.faces[0].get("embedding")
                    if raw_emb:
                        probe_norm = float(np.linalg.norm(raw_emb))
            except Exception as exc:  # noqa: BLE001
                logger.warning("IFS /embeddings failed; norm scoring disabled: %s", exc)
            # Auto-disable when the server returns L2-normalized
            # embeddings (norm ≈ 1.0 always). The MagFace-style quality
            # signal is only meaningful for *un-normalized* embeddings.
            if probe_norm is not None and probe_norm < 1.05:
                logger.debug(
                    "IFS returned L2-normalized embeddings (norm=%.3f); "
                    "disabling #2 norm-aware scoring",
                    probe_norm,
                )
                probe_norm = None

        out_faces: list[dict] = []
        all_results: list[dict] = []
        for idx, f in enumerate(faces_raw):
            bb = (f.get("bbox") or {}).get("pixels") or {}
            x = int(bb.get("x", 0))
            y = int(bb.get("y", 0))
            w = int(bb.get("width", 0))
            h = int(bb.get("height", 0))
            if min(w, h) < min_face_pixels:
                continue
            confidence = float(f.get("detection_score") or 0.0)
            # Crop the face and JPEG-encode it; the server expects JPEG/PNG
            # bytes, not raw ndarray.
            crop_bytes = _encode_crop(np_img, x, y, w, h) if np_img is not None else None
            if crop_bytes is None:
                # Decoder failed (unusual): fall back to the whole image,
                # which still produces one round of matches.
                crop_bytes = image_bytes
            # Over-fetch when adaptive threshold is on — the global filter
            # passes matches through, we re-filter per-Person below.
            server_limit = top_k * 3 if adaptive_threshold_step else top_k
            try:
                result = self._client.search(
                    self._collection_id,
                    image=crop_bytes,
                    limit=server_limit,
                    threshold=min_similarity,
                )
            except Exception as exc:  # noqa: BLE001 — surface a per-face error
                logger.warning("IFS search failed for face %s: %s", idx, exc)
                continue

            face_view = {
                "face_index": idx,
                "bbox": {"x": x, "y": y, "w": w, "h": h},
                "detection_score": confidence,
                "matches": [],
            }
            for m in result.matches or []:
                person = m.get("person") or {}
                similarity = float(m.get("similarity") or 0.0)
                metadata = person.get("metadata") or {}
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        metadata = {}
                match = {
                    "id": person.get("id"),
                    "name": person.get("name"),
                    "person_name": person.get("name"),
                    "person_id": person.get("id"),
                    "similarity": similarity,
                    "distance": 1.0 - similarity,
                    "matched_face_id": m.get("matched_face_id"),
                    "created_at": person.get("created_at"),
                    "file_path": metadata.get("file_path"),
                    "category": metadata.get("category"),
                    "occupation": metadata.get("occupation"),
                    "type": metadata.get("type"),
                    "remarks": metadata.get("remarks"),
                    "face_count": person.get("face_count"),
                    "face_index": idx,
                    "query_face_bbox": {"x": x, "y": y, "w": w, "h": h},
                }
                # Apply scoring enhancements in-place.
                _apply_match_scoring(
                    match,
                    quality_weight=quality_weight,
                    query_detection_score=confidence,
                    probe_norm=probe_norm,
                    norm_reference=norm_reference,
                    adaptive_threshold_step=adaptive_threshold_step,
                    base_similarity=settings.insightface_verify_similarity_threshold,
                )
                face_view["matches"].append(match)
                all_results.append(match)
            out_faces.append(face_view)

        # Sort by effective distance (legacy = raw distance when no weights).
        all_results.sort(key=lambda r: r.get("effective_distance", r["distance"]))
        return {
            "face_count": len(out_faces),
            "faces": out_faces,
            "all_results": all_results,
        }

    # ------------------------------------------------------------------
    # Register / delete
    # ------------------------------------------------------------------
    def register_person(
        self,
        name: str,
        image_bytes: bytes,
        *,
        metadata: dict[str, Any] | None = None,
        external_id: str | None = None,
        person_id: str | None = None,
        collection_id: str | None = None,
    ) -> tuple[str, str]:
        """Enroll one person with one face into a collection.

        Returns ``(person_id, face_id)`` from the server. The caller is
        responsible for persisting bytes via ``_persist_image`` (the engine
        does this before calling). No local SQL row is written — IFS Person
        is the canonical record.
        """
        cid = collection_id or self._collection_id
        result = self._client.create_person(
            cid,
            images=[image_bytes],
            name=name,
            metadata=metadata or {},
            external_id=external_id,
            person_id=person_id,
        )
        faces = result.faces or []
        if not faces:
            rejected = result.rejected_images or []
            raise RuntimeError(f"InsightFace did not enroll any face; rejected={rejected}")
        return str(result.person["id"]), str(faces[0]["id"])

    def delete_person(self, person_id: str, *, collection_id: str | None = None) -> None:
        cid = collection_id or self._collection_id
        self._client.delete_person(cid, person_id)

    # ------------------------------------------------------------------
    # Person CRUD (list / get / update)
    # ------------------------------------------------------------------
    def list_persons(
        self,
        *,
        limit: int = 50,
        cursor: str | None = None,
        search: str | None = None,
        collection_id: str | None = None,
    ) -> tuple[list[dict], str | None]:
        """List persons in a collection, returning flat record-item dicts.

        Calls ``GET /v1/collections/{cid}/persons`` (server-side cursor
        pagination). Returns ``(items, next_cursor)`` — ``next_cursor`` is
        ``None`` when there are no more pages. ``search=`` matches names
        server-side; IFS does not currently filter by metadata.
        """
        cid = collection_id or self._collection_id
        page = self._client.list_persons(
            cid, limit=limit, cursor=cursor, search=search
        )
        items = [_person_to_item(p) for p in (page.persons or [])]
        return items, page.next_cursor

    def get_person(
        self,
        person_id: str,
        *,
        collection_id: str | None = None,
    ) -> dict | None:
        """Read a single person by id. Returns ``None`` on 404.

        Other transport errors (5xx, auth) propagate.
        """
        cid = collection_id or self._collection_id
        try:
            result = self._client.get_person(cid, person_id)
        except NotFoundError:
            return None
        return _person_to_item(result.person)

    def update_person(
        self,
        person_id: str,
        *,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
        collection_id: str | None = None,
    ) -> dict:
        """PATCH /v1/collections/{cid}/persons/{id}.

        At least one of ``name`` / ``metadata`` must be supplied (the SDK
        raises ``ValueError`` otherwise). Returns the updated person as a
        flat record-item dict.

        Note: we forward kwargs conditionally so a ``None`` value is not
        serialized as JSON ``null``. IFS interprets ``null`` for these
        fields as "clear the value", which would wipe the name on a
        metadata-only update.
        """
        cid = collection_id or self._collection_id
        kwargs: dict[str, object] = {}
        if name is not None:
            kwargs["name"] = name
        if metadata is not None:
            kwargs["metadata"] = metadata
        if not kwargs:
            raise ValueError("at least one of name / metadata must be supplied")
        result = self._client.update_person(cid, person_id, **kwargs)
        return _person_to_item(result.person)

    # ------------------------------------------------------------------
    # Per-match enrichment (bbox for the matched face)
    # ------------------------------------------------------------------
    def get_face_bbox(
        self,
        person_id: str,
        face_id: str,
        *,
        collection_id: str | None = None,
    ) -> dict | None:
        """Return ``{x, y, w, h}`` for the matched face, or None on 404.

        Used by FaceEngine.search to populate ``source_x/y/w/h`` so handlers
        can crop thumbnails the same way the legacy SQL join did.
        """
        cid = collection_id or self._collection_id
        try:
            page = self._client.list_faces(cid, person_id, limit=100)
        except NotFoundError:
            return None
        for f in page.faces:
            if f.get("id") == face_id:
                bb = (f.get("bounding_box") or {}).get("pixels") or {}
                if not bb:
                    return None
                return {
                    "x": int(bb.get("x", 0)),
                    "y": int(bb.get("y", 0)),
                    "w": int(bb.get("width", 0)),
                    "h": int(bb.get("height", 0)),
                }
        return None


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _decode(image_bytes: bytes) -> np.ndarray | None:
    try:
        return np.asarray(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    except Exception:
        return None


def _crop_or_none(img: np.ndarray | None, x: int, y: int, w: int, h: int) -> np.ndarray | None:
    if img is None or w <= 0 or h <= 0:
        return None
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(img.shape[1], x + w), min(img.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    return img[y0:y1, x0:x1]


def _encode_crop(img: np.ndarray | None, x: int, y: int, w: int, h: int) -> bytes | None:
    """Crop a region of the decoded image and return JPEG bytes."""
    crop = _crop_or_none(img, x, y, w, h)
    if crop is None:
        return None
    pil = Image.fromarray(crop)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


# ----------------------------------------------------------------------
# Match scoring enhancements (#1, #2, #3)
# ----------------------------------------------------------------------
def _apply_match_scoring(
    match: dict[str, Any],
    *,
    quality_weight: float,
    query_detection_score: float,
    probe_norm: float | None,
    norm_reference: float | None,
    adaptive_threshold_step: float,
    base_similarity: float,
) -> None:
    """Mutate ``match`` in place to add the per-match scoring fields.

    Populates the following dict keys (all optional / conditional):

    * ``quality_factor`` (#3): ``(1 - quality_weight) + quality_weight * q``
      where ``q = query_detection_score``. Equals ``1.0`` when weight=0.
    * ``fused_similarity`` / ``fused_distance`` (#3): quality-weighted score.
    * ``probe_norm`` / ``norm_factor`` (#2): L2 norm of the probe embedding
      and the multiplicative weight. ``None`` when norm scoring disabled.
    * ``mps_similarity`` / ``mps_distance`` (#3 × #2 combined).
    * ``adaptive_threshold`` / ``passes_adaptive`` (#1): per-Person
      threshold derived from Person.face_count, and whether this match
      passes it.
    * ``effective_distance``: the value downstream code should sort by.
      Equals ``distance`` when no scoring is active; the combined quality
      + norm score otherwise.
    """
    similarity = float(match.get("similarity") or 0.0)

    # #3 quality-aware fusion
    if quality_weight and quality_weight > 0.0:
        q = float(query_detection_score or 0.0)
        quality_factor = (1.0 - quality_weight) + quality_weight * q
        fused_similarity = similarity * quality_factor
        fused_distance = 1.0 - fused_similarity
        match["quality_factor"] = quality_factor
        match["fused_similarity"] = fused_similarity
        match["fused_distance"] = fused_distance
        effective_similarity = fused_similarity
    else:
        match["quality_factor"] = 1.0
        effective_similarity = similarity

    # #2 norm-aware scoring (MPS proxy)
    if probe_norm is not None and norm_reference and norm_reference > 0:
        norm_factor = min(probe_norm / float(norm_reference), 1.0)
        mps_similarity = effective_similarity * norm_factor
        mps_distance = 1.0 - mps_similarity
        match["probe_norm"] = probe_norm
        match["norm_factor"] = norm_factor
        match["mps_similarity"] = mps_similarity
        match["mps_distance"] = mps_distance
        effective_similarity = mps_similarity
    # When #2 is disabled we don't add probe_norm/norm_factor — keeps the
    # legacy match shape clean for callers that don't opt in.

    match["effective_similarity"] = effective_similarity
    match["effective_distance"] = 1.0 - effective_similarity

    # #1 adaptive per-Person threshold (post-scoring filter, not a score)
    if adaptive_threshold_step and adaptive_threshold_step > 0.0:
        face_count = int(match.get("face_count") or 0)
        adaptive_threshold = base_similarity + adaptive_threshold_step * min(face_count, 10)
        match["adaptive_threshold"] = adaptive_threshold
        match["passes_adaptive"] = similarity >= adaptive_threshold
    else:
        match["passes_adaptive"] = True


# ----------------------------------------------------------------------
# Person flattening
# ----------------------------------------------------------------------
def _person_to_item(person: dict[str, Any] | None) -> dict[str, Any]:
    """Flatten an IFS ``Person`` (dict) into the legacy record-item shape.

    The IFS server returns ``metadata`` as a parsed dict for
    ``get_person`` / ``list_persons`` / ``update_person`` (see
    ``results.py:44-51``); the JSON-string fallback here is defensive for
    unexpected server payloads.

    The returned dict keys mirror what ``api/routes.py`` writes to the
    webui: ``id`` is the IFS aggregate ``Person.id`` (used as the webui's
    record identifier), with the WCM-specific fields
    (``category`` / ``occupation`` / ``type`` / ``remarks`` / ``file_path``)
    lifted from ``metadata``.
    """
    if not person:
        return {}
    metadata = person.get("metadata") or {}
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {}
    pid = person.get("id")
    return {
        "id": str(pid) if pid is not None else None,
        "name": person.get("name"),
        "external_id": person.get("external_id"),
        "face_count": person.get("face_count"),
        "created_at": person.get("created_at"),
        "updated_at": person.get("updated_at"),
        # WCM-specific fields lifted from metadata
        "category": metadata.get("category"),
        "occupation": metadata.get("occupation"),
        "type": metadata.get("type"),
        "remarks": metadata.get("remarks"),
        "file_path": metadata.get("file_path"),
    }
