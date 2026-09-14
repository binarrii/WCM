"""Keep embedded evidence in object storage and persist stable image references."""

import base64
import hashlib
from pathlib import Path

from wcm_facerec import image_store
from wcm_facerec.config import settings


def archive_evidence(task_id, value):
    if settings.image_storage != "s3":
        return value
    if isinstance(value, list):
        return [archive_evidence(task_id, item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if key in {"face_image_b64", "source_face_b64"} and isinstance(item, str):
            data = base64.b64decode(item, validate=True)
            digest = hashlib.sha256(data).hexdigest()
            path = Path("/tmp/wcm/evidence") / task_id / f"{digest}.jpg"
            image_store.write_bytes(path, data)
            result[key.removesuffix("_b64") + "_url"] = (
                "/images/" + path.relative_to("/tmp/wcm").as_posix()
            )
        else:
            result[key] = archive_evidence(task_id, item)
    return result
