"""Stable /tmp/wcm metadata paths backed by local files or S3 object keys."""

import hashlib
import mimetypes
from functools import lru_cache
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from .config import settings


@lru_cache(maxsize=4)
def _client(endpoint, region, access_key, secret_key):
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
            connect_timeout=5,
            read_timeout=30,
            retries={"max_attempts": 2},
        ),
    )


def client():
    return _client(
        settings.s3_endpoint, settings.s3_region, settings.s3_access_key, settings.s3_secret_key
    )


def relative_path(value, root=Path("/tmp/wcm")):
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    relative = path.resolve().relative_to(root.resolve())
    if not relative.parts or any(part.startswith(".") for part in relative.parts):
        raise ValueError("Invalid public image path")
    return relative


def object_key(value, root=Path("/tmp/wcm")):
    return f"{settings.s3_prefix.strip('/')}/{relative_path(value, root).as_posix()}"


def stat(value, root=Path("/tmp/wcm")):
    relative_path(value, root)
    if settings.image_storage == "local":
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(str(value))
        return {"ContentLength": path.stat().st_size}
    try:
        return client().head_object(Bucket=settings.s3_bucket, Key=object_key(value, root))
    except ClientError as exc:
        if str(exc.response["Error"]["Code"]) in {"404", "NoSuchKey", "NotFound"}:
            raise FileNotFoundError(str(value)) from exc
        raise


def exists(value, root=Path("/tmp/wcm")):
    try:
        stat(value, root)
        return True
    except FileNotFoundError:
        return False


def read_bytes(value, root=Path("/tmp/wcm")):
    relative_path(value, root)
    if settings.image_storage == "local":
        return Path(value).read_bytes()
    try:
        result = client().get_object(Bucket=settings.s3_bucket, Key=object_key(value, root))
    except ClientError as exc:
        if str(exc.response["Error"]["Code"]) in {"404", "NoSuchKey", "NotFound"}:
            raise FileNotFoundError(str(value)) from exc
        raise
    with result["Body"] as body:
        if result["ContentLength"] > settings.max_file_size_mb * 1024 * 1024:
            raise ValueError("Image exceeds the configured size limit")
        return body.read()


def write_bytes(value, data, root=Path("/tmp/wcm")):
    relative_path(value, root)
    if settings.image_storage == "local":
        path = Path(value)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    else:
        client().put_object(
            Bucket=settings.s3_bucket,
            Key=object_key(value, root),
            Body=data,
            ContentType=mimetypes.guess_type(str(value))[0] or "application/octet-stream",
            Metadata={"sha256": hashlib.sha256(data).hexdigest()},
        )


def delete(value, root=Path("/tmp/wcm")):
    relative_path(value, root)
    if settings.image_storage == "local":
        Path(value).unlink(missing_ok=True)
    else:
        client().delete_object(Bucket=settings.s3_bucket, Key=object_key(value, root))
