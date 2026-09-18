"""Content-addressed catalog for immutable review files and task references.

Physical keys remain unique per upload attempt. A lost database lock can leave
an abandoned file, but a late writer can never overwrite a newer shared object.
"""

import hashlib
import re
import uuid
from contextlib import contextmanager
from pathlib import Path

import pymysql
from botocore.exceptions import ClientError

from wcm_facerec import image_store
from wcm_facerec.config import settings
from wcm_facerec.model_budget import remaining_request_time

from . import review_task_store as store


def initialize_sync(cursor):
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS review_media_objects (
            id CHAR(32) PRIMARY KEY,
            storage_scope CHAR(64) NOT NULL,
            sha256 CHAR(64) NOT NULL,
            storage VARCHAR(10) NOT NULL,
            object_key VARCHAR(512) NOT NULL,
            size_bytes BIGINT UNSIGNED NOT NULL,
            state VARCHAR(12) NOT NULL,
            created_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
            UNIQUE KEY idx_media_object_key (storage_scope, object_key),
            INDEX idx_media_object_content (storage_scope, sha256, state)
        ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
    """)
    cursor.execute("SHOW COLUMNS FROM review_media LIKE 'object_id'")
    if not cursor.fetchone():
        try:
            cursor.execute("""ALTER TABLE review_media ADD COLUMN object_id CHAR(32) NULL,
                ADD INDEX idx_media_object_reference (object_id)""")
        except pymysql.err.OperationalError as exc:
            if exc.args[0] != 1060:
                raise


def storage_scope(storage):
    location = (
        [settings.s3_endpoint, settings.s3_bucket, settings.review_media_prefix.strip("/")]
        if storage == "s3"
        else [str(Path(settings.review_media_dir).resolve())]
    )
    return hashlib.sha256(store._json_dump([storage, *location]).encode()).hexdigest()


@contextmanager
def locked(scope, digest):
    # This lock also applies in single-worker mode. All SQL under the lock uses
    # this same connection: it must never reconnect and silently lose ownership.
    name = (
        "wcm-media:"
        + hashlib.sha256(f"{settings.review_tasks_db_name}:{scope}:{digest}".encode()).hexdigest()[
            :48
        ]
    )
    with store._connect() as connection, connection.cursor() as cursor:
        while True:
            remaining_request_time()
            cursor.execute("SELECT GET_LOCK(%s, 1) AS acquired", (name,))
            if cursor.fetchone()["acquired"] == 1:
                break
        # The connection context closes (and releases named locks) on every exit.
        yield connection, cursor


def local_path(media):
    name = media["object_key"].rsplit("/", 1)[-1]
    if not re.fullmatch(r"[0-9a-f]{32}\.mp4", name):
        raise ValueError("Invalid review media object key")
    return Path(settings.review_media_dir) / name


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(1024 * 1024):
            remaining_request_time()
            digest.update(chunk)
    return digest.hexdigest()


def inspect_object(media):
    if media["storage"] == "s3":
        try:
            result = image_store.client().head_object(
                Bucket=settings.s3_bucket, Key=media["object_key"]
            )
        except ClientError as exc:
            if str(exc.response["Error"]["Code"]) in {"404", "NoSuchKey", "NotFound"}:
                return None
            raise
        return result["ContentLength"], result.get("Metadata", {}).get("sha256")
    path = local_path(media)
    return (path.stat().st_size, fingerprint(path)) if path.is_file() else None


def reusable(cursor, scope, digest, size):
    cursor.execute(
        """SELECT * FROM review_media_objects
        WHERE storage_scope=%s AND sha256=%s AND state='ready' ORDER BY created_at, id""",
        (scope, digest),
    )
    for item in cursor.fetchall():
        if item["size_bytes"] == size and inspect_object(item) == (size, digest):
            return item
        cursor.execute("UPDATE review_media_objects SET state='missing' WHERE id=%s", (item["id"],))
    return None


def create(cursor, scope, digest, storage, key, size, state="uploading"):
    item = {
        "id": uuid.uuid4().hex,
        "storage_scope": scope,
        "sha256": digest,
        "storage": storage,
        "object_key": key,
        "size_bytes": size,
        "state": state,
    }
    cursor.execute(
        """INSERT INTO review_media_objects
        (id,storage_scope,sha256,storage,object_key,size_bytes,state)
        VALUES (%s,%s,%s,%s,%s,%s,%s)""",
        tuple(item.values()),
    )
    return item


def delete_file(item):
    if item["storage"] == "s3":
        image_store.client().delete_object(Bucket=settings.s3_bucket, Key=item["object_key"])
    else:
        local_path(item).unlink(missing_ok=True)


def delete_unreferenced(item):
    if item["storage_scope"] != storage_scope(item["storage"]):
        return
    with locked(item["storage_scope"], item["sha256"]) as (_, cursor):
        cursor.execute("SELECT * FROM review_media_objects WHERE id=%s", (item["id"],))
        current = cursor.fetchone()
        if not current:
            return
        cursor.execute(
            """SELECT id FROM review_media WHERE object_id=%s OR
            (object_id IS NULL AND storage=%s AND object_key=%s) LIMIT 1""",
            (item["id"], item["storage"], item["object_key"]),
        )
        if cursor.fetchone():
            return
        # Retire durably before deleting storage. If this lock/connection is lost
        # during the SDK request, another uploader must allocate a different key.
        cursor.execute(
            "UPDATE review_media_objects SET state='deleting' WHERE id=%s", (item["id"],)
        )
        delete_file(current)
        cursor.execute("DELETE FROM review_media_objects WHERE id=%s", (item["id"],))


def unreferenced():
    with store._connect() as connection, connection.cursor() as cursor:
        cursor.execute("""SELECT o.* FROM review_media_objects o
            WHERE NOT EXISTS (SELECT 1 FROM review_media m WHERE m.object_id=o.id OR
                (m.object_id IS NULL AND m.storage=o.storage AND m.object_key=o.object_key))
            AND (o.state IN ('ready','deleting') OR
                 o.created_at < TIMESTAMPADD(DAY,-1,UTC_TIMESTAMP(3)))
            ORDER BY o.created_at LIMIT 20""")
        return cursor.fetchall()


def legacy_candidates():
    with store._connect() as connection, connection.cursor() as cursor:
        cursor.execute("""SELECT m.* FROM review_media m
            LEFT JOIN review_tasks t ON t.id=m.task_id
            WHERE m.object_id IS NULL
            AND (t.id IS NULL OR t.status NOT IN ('processing','cancelling'))
            ORDER BY m.created_at LIMIT 20""")
        return cursor.fetchall()


def migrate_legacy(media):
    # Legacy GC and migration share this outer lock, then take the content lock.
    # Retired duplicate files stay registered for 24h for in-flight readers and
    # rollback; task IDs, media IDs and each reference's expiry are unchanged.
    with locked("legacy", "") as (_, legacy_cursor):
        legacy_cursor.execute("SELECT * FROM review_media WHERE id=%s", (media["id"],))
        media = legacy_cursor.fetchone()
        if not media or media["object_id"]:
            return
        inspected = inspect_object(media)
        if not inspected or not re.fullmatch(r"[0-9a-f]{64}", inspected[1] or ""):
            return
        size, digest = inspected
        scope = storage_scope(media["storage"])
        with locked(scope, digest) as (connection, cursor):
            obj = reusable(cursor, scope, digest, size)
            connection.begin()
            try:
                cursor.execute(
                    "SELECT status FROM review_tasks WHERE id=%s FOR UPDATE", (media["task_id"],)
                )
                task = cursor.fetchone()
                if task and task["status"] in {"processing", "cancelling"}:
                    connection.rollback()
                    return
                if obj is None:
                    cursor.execute(
                        "SELECT * FROM review_media_objects WHERE storage_scope=%s AND object_key=%s",
                        (scope, media["object_key"]),
                    )
                    obj = cursor.fetchone()
                    if obj:
                        assert obj["sha256"] == digest and obj["size_bytes"] == size
                        cursor.execute(
                            "UPDATE review_media_objects SET state='ready' WHERE id=%s",
                            (obj["id"],),
                        )
                    else:
                        obj = create(
                            cursor,
                            scope,
                            digest,
                            media["storage"],
                            media["object_key"],
                            size,
                            "ready",
                        )
                elif obj["object_key"] != media["object_key"]:
                    cursor.execute(
                        "SELECT id FROM review_media_objects WHERE storage_scope=%s AND object_key=%s",
                        (scope, media["object_key"]),
                    )
                    if not cursor.fetchone():
                        create(
                            cursor,
                            scope,
                            digest,
                            media["storage"],
                            media["object_key"],
                            size,
                            "retired",
                        )
                cursor.execute(
                    "UPDATE review_media SET object_id=%s, object_key=%s WHERE id=%s AND object_id IS NULL",
                    (obj["id"], obj["object_key"], media["id"]),
                )
                cursor.execute(
                    """UPDATE review_tasks SET media=JSON_SET(media,
                    '$.object_id',%s,'$.object_key',%s)
                    WHERE id=%s AND JSON_UNQUOTE(JSON_EXTRACT(media,'$.id'))=%s""",
                    (obj["id"], obj["object_key"], media["task_id"], media["id"]),
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
