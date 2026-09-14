"""Stable request fingerprints independent of multipart boundary randomness."""

import hashlib
import json
from contextlib import suppress

from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartParser


async def person_request_fingerprint(request):
    body = await request.body()  # Preserve the body for the downstream handler.
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    payload = hashlib.sha256(body).hexdigest()
    if content_type == "multipart/form-data":
        fields = {}
        form = await MultiPartParser(request.headers, request.stream()).parse()
        try:
            for name, value in form.multi_items():
                if isinstance(value, UploadFile):
                    data = await value.read()
                    value = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
                fields.setdefault(name, []).append(value)
        finally:
            await form.close()
        payload = fields
    elif content_type == "application/json":
        with suppress(ValueError, UnicodeDecodeError):
            payload = json.loads(body)
    intent = [request.method, request.url.path, request.url.query, content_type, payload]
    return hashlib.sha256(
        json.dumps(intent, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
