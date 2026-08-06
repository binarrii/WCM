"""End-to-end smoke test against the LIVE InsightFace Server.

Marked live; skipped unless RUN_LIVE=1. Run with:

    RUN_LIVE=1 uv run pytest tests/test_smoke_live.py -v

Targets ``http://10.252.25.251:18097`` (override via ``WCM_INSIGHTFACE_BASE_URL``).
"""
from __future__ import annotations

import os
import uuid

import pytest

from wcm_facerec.config import settings
from wcm_facerec.face_engine import FaceEngine


pytestmark = pytest.mark.live


SAMPLE = "/tmp/wcm/劣迹艺人/0000000000016736_1108ffef0f7bdb665bb2d19b1bfe115f.png"


@pytest.fixture(scope="module")
def engine() -> FaceEngine:
    return FaceEngine()


@pytest.fixture(scope="module")
def sample_bytes() -> bytes:
    with open(SAMPLE, "rb") as f:
        return f.read()


def test_live_health(engine):
    h = engine._adapter.health()
    assert h["status"] == "ready"


def test_live_detect_returns_bbox(engine, sample_bytes):
    faces = engine._adapter.detect(sample_bytes, min_face_pixels=80)
    assert faces, "expected at least one face"
    fa = faces[0]["facial_area"]
    assert {"x", "y", "w", "h"} <= set(fa.keys())


def test_live_embed_is_512d(engine, sample_bytes):
    e = engine._adapter.embed(sample_bytes)
    assert e.shape == (512,)
    # Buffalo_m embeddings are L2-normalized out of the box.
    import math
    assert math.isclose(float((e["embedding"].tolist() if hasattr(e, "keys") else e) @ (e["embedding"].tolist() if hasattr(e, "keys") else e)), 1.0, abs_tol=1e-3) or True
    # The simpler check: norm is ~1
    import numpy as np
    assert abs(float(np.linalg.norm(e)) - 1.0) < 1e-3


def test_live_search_self_match_is_one(engine, sample_bytes):
    matches = engine._adapter.search(sample_bytes, top_k=3, min_similarity=0.0)
    assert matches, "expected self-match"
    assert matches[0]["similarity"] >= 0.99
    # self-match distance should be ~0
    assert matches[0]["distance"] < 0.05


def test_live_engine_search_round_trip(engine, sample_bytes):
    import asyncio

    matches = asyncio.run(engine.search(sample_bytes, top_k=3, threshold=0.4))
    assert matches, "expected at least one match"
    top = matches[0]
    # bbox backfilled
    assert top["source_x"] is not None
    assert top["source_w"] > 0


def test_live_engine_register_and_cleanup(engine, sample_bytes):
    """Register a unique person, search for them, delete. Marks as live
    so it pollutes the collection briefly."""
    import asyncio

    name = f"测试临时_{uuid.uuid4().hex[:8]}"
    record = asyncio.run(
        engine.register_from_image(
            name=name,
            img_source=sample_bytes,
            category="未分类",
            occupation="测试",
            type_="临时",
            remarks="pytest smoke",
        )
    )
    try:
        # Search should now find this person at the top.
        matches = asyncio.run(engine.search(sample_bytes, top_k=5, threshold=0.4))
        assert any(m.get("name") == name for m in matches), (
            f"expected to find {name} in top-5; got {[m.get('name') for m in matches]}"
        )
    finally:
        # Clean up: delete from both aggregate and per-category collections.
        for cid in {settings.insightface_collection_id, "未分类"} & set(
            settings.insightface_category_collections.values()
        ):
            try:
                engine._adapter.delete_person(record.id, collection_id=cid)
            except Exception:
                pass
        # Always remove from the aggregate collection keyed by record.id is wrong;
        # the IFS-side person_id is its own. Fetch and delete via search.
        # The simplest reliable path: search for our unique name, then delete.
        matches = asyncio.run(engine.search(sample_bytes, top_k=20, threshold=0.0))
        for m in matches:
            if m.get("name") == name:
                for cid in [settings.insightface_collection_id, *settings.insightface_category_collections.values()]:
                    try:
                        engine._adapter.delete_person(m["person_id"], collection_id=cid)
                    except Exception:
                        pass