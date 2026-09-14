"""Serve historical image URLs from whichever shared backend is configured."""

import mimetypes
from pathlib import Path

from botocore.exceptions import ClientError
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from wcm_facerec import image_store
from wcm_facerec.cluster import run_sync
from wcm_facerec.config import settings

images_bp = APIRouter()


@images_bp.api_route("/images/{key:path}", methods=["GET", "HEAD"])
async def get_image(key: str, request: Request):
    try:
        path = Path("/tmp/wcm") / key
        object_key = image_store.object_key(path)
        metadata = await run_sync(image_store.stat, path)
    except (ValueError, FileNotFoundError):
        raise HTTPException(404, "图片不存在") from None
    headers = {"Accept-Ranges": "bytes", "Cache-Control": "public, max-age=3600"}
    etag = metadata.get("ETag")
    if etag:
        headers["ETag"] = etag
        if request.headers.get("if-none-match") == etag:
            return Response(status_code=304, headers=headers)
    if request.method == "HEAD":
        headers["Content-Length"] = str(metadata["ContentLength"])
        return Response(
            headers=headers, media_type=metadata.get("ContentType") or mimetypes.guess_type(key)[0]
        )
    parameters = {"Bucket": settings.s3_bucket, "Key": object_key}
    if request.headers.get("range"):
        parameters["Range"] = request.headers["range"]
    try:
        result = await run_sync(image_store.client().get_object, **parameters)
    except ClientError as exc:
        status = exc.response["ResponseMetadata"]["HTTPStatusCode"]
        if status in (404, 416):
            raise HTTPException(status, "图片不存在或 Range 无效") from exc
        raise HTTPException(503, "图片存储暂不可用") from exc
    headers["Content-Length"] = str(result["ContentLength"])
    if result.get("ContentRange"):
        headers["Content-Range"] = result["ContentRange"]

    def chunks():
        body = result["Body"]
        try:
            yield from body.iter_chunks(64 * 1024)
        finally:
            body.close()

    return StreamingResponse(
        chunks(),
        status_code=206 if result.get("ContentRange") else 200,
        headers=headers,
        media_type=result.get("ContentType", "application/octet-stream"),
        background=BackgroundTask(result["Body"].close),
    )
