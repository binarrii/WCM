"""Object-key image storage with compatibility for historical /tmp/wcm paths."""

import hashlib
import mimetypes
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from .config import settings
from .model_budget import remaining_request_time


@lru_cache(maxsize=4)
def _client(endpoint, region, access_key, secret_key):
    result = boto3.client(
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
    # Runs before every HTTP attempt, including botocore's internal retries.
    result.meta.events.register("before-send.s3", _before_send)
    return result


def _before_send(**kwargs):
    remaining_request_time()


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
    value = str(value)
    if value.startswith(settings.s3_prefix.strip("/") + "/"):
        return validate_key(value)
    return validate_key(f"{settings.s3_prefix.strip('/')}/{relative_path(value, root).as_posix()}")


def validate_key(value):
    """Admit only image objects in the configured public namespace."""
    prefix = settings.s3_prefix.strip("/") + "/"
    if (
        not isinstance(value, str)
        or not value.startswith(prefix)
        or not value[len(prefix) :]
        or "\\" in value
        or any(ord(char) < 32 for char in value)
        or any(not part or part.startswith(".") for part in value.split("/"))
    ):
        raise ValueError("Invalid image object key")
    return value


def reference(value, root=Path("/tmp/wcm")):
    return object_key(value, root) if settings.image_storage == "s3" else str(value)


def image_refs(item):
    """New fields are authoritative; read legacy fields only when absent."""
    item = item or {}
    metadata = item.get("metadata") or {}
    source = item if any(k in item for k in ("image_key", "image_keys")) else metadata
    if source.get("image_key") is not None or source.get("image_keys") is not None:
        values = source.get("image_keys")
        values = [source.get("image_key"), *(values if isinstance(values, list) else [])]
        return list(dict.fromkeys(validate_key(v) for v in values if v))
    source = item if any(k in item for k in ("file_path", "image_paths")) else metadata
    values = source.get("image_paths")
    values = [source.get("file_path"), *(values if isinstance(values, list) else [])]
    return list(dict.fromkeys(reference(v) for v in values if isinstance(v, str) and v))


def with_images(metadata, references):
    result = dict(metadata)
    refs = list(dict.fromkeys(reference(value) for value in references))
    if settings.image_storage == "s3":
        result.pop("file_path", None)
        result.pop("image_paths", None)
        result.update(image_key=refs[0] if refs else None, image_keys=refs)
    else:
        result.update(file_path=refs[0] if refs else None, image_paths=refs)
    return result


def public_url(value):
    key = object_key(value)
    relative = key[len(settings.s3_prefix.strip("/")) + 1 :]
    return "/images/" + quote(relative, safe="/")


def snapshot_ref(image):
    return validate_key(image["key"]) if "key" in image else reference(image["path"])


def canonical_images(images):
    if settings.image_storage != "s3":
        return images
    return [{"key": snapshot_ref(image), "sha256": image["sha256"]} for image in images]


def stat(value, root=Path("/tmp/wcm")):
    if settings.image_storage == "local":
        relative_path(value, root)
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
    if settings.image_storage == "local":
        relative_path(value, root)
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
    if settings.image_storage == "local":
        relative_path(value, root)
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
    if settings.image_storage == "local":
        relative_path(value, root)
        Path(value).unlink(missing_ok=True)
    else:
        client().delete_object(Bucket=settings.s3_bucket, Key=object_key(value, root))
