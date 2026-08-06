"""Shared pytest fixtures.

Goals:
- Keep the live InsightFace Server out of the default test loop.
- Provide a mocked httpx transport that the vendored SDK can hit so the
  adapter exercises real serialization/deserialization paths.
- Generate a tiny in-memory JPEG fixture for image bytes.
"""
from __future__ import annotations

import io
import os
from typing import Any

import httpx
import pytest
from PIL import Image as PILImage


# ----------------------------------------------------------------------
# Live-vs-unit gate
# ----------------------------------------------------------------------
def pytest_collection_modifyitems(config, items):
    marker = pytest.mark.skipif(
        not os.environ.get("RUN_LIVE"),
        reason="set RUN_LIVE=1 to run tests against the live InsightFace Server",
    )
    for item in items:
        if "live" in item.keywords:
            item.add_marker(marker)


# ----------------------------------------------------------------------
# Image fixture
# ----------------------------------------------------------------------
@pytest.fixture(scope="session")
def sample_image_bytes() -> bytes:
    """A small white JPEG (1x1). Not a face — fine for shape-only tests."""
    buf = io.BytesIO()
    PILImage.new("RGB", (16, 16), "white").save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture(scope="session")
def face_like_image_bytes() -> bytes:
    """A larger JPEG. Used for detect/embed tests against the live server."""
    buf = io.BytesIO()
    PILImage.new("RGB", (640, 480), "white").save(buf, format="JPEG")
    return buf.getvalue()


# ----------------------------------------------------------------------
# Mocked httpx transport
# ----------------------------------------------------------------------
class FakeTransport(httpx.BaseTransport):
    """Routes httpx requests to canned responses keyed by (method, path)."""

    def __init__(self):
        self.responses: dict[tuple[str, str], dict[str, Any]] = {}
        self.calls: list[tuple[str, str, dict | None]] = []

    def register(self, method: str, path: str, body: dict[str, Any]) -> None:
        self.responses[(method.upper(), path)] = body

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        key = (request.method.upper(), request.url.path)
        self.calls.append((request.method, str(request.url), None))
        if key in self.responses:
            return httpx.Response(200, json=self.responses[key])
        return httpx.Response(404, json={"error": {"code": "not_found", "message": f"unmocked {key}"}})


@pytest.fixture
def fake_transport() -> FakeTransport:
    return FakeTransport()