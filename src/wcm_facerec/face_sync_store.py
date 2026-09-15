"""WCM outbox, replica checkpoints and read admission. Never accesses SQLite."""

import hashlib
import json
import random
from contextvars import ContextVar
from datetime import timezone
from urllib.parse import urlsplit
from uuid import uuid4

import pymysql

from . import image_store
from .cluster import connect
from .config import settings

read_guard = ContextVar("face_replica_read_guard", default=None)


class ReplicationUnavailable(RuntimeError):
    pass


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


def identity(cid, pid):
    return digest([cid, pid])


def source():
    return settings.insightface_base_url.rstrip("/")


def collections():
    return sorted(
        {settings.insightface_collection_id, *settings.insightface_category_collections.values()}
    )


def initialize():
    if not settings.insightface_replication_enabled:
        return
    if not settings.cluster_enabled or settings.image_storage != "s3":
        raise ReplicationUnavailable("人脸同步需要 MySQL 集群模式及 S3 图片存储")
    urls = set()
    for name, url in settings.insightface_replicas.items():
        parsed = urlsplit(url)
        if (
            not name
            or name.startswith("__")
            or len(name) > 64
            or parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Invalid replica name/URL")
        if url.rstrip("/") in urls | {source()}:
            raise ValueError("Replica URLs must be distinct from each other and the primary")
        urls.add(url.rstrip("/"))
    with connect() as db, db.cursor() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS face_sync_control (
            id INT PRIMARY KEY, initialized BOOLEAN NOT NULL DEFAULT FALSE,
            head BIGINT NOT NULL DEFAULT 0, source_url TEXT NOT NULL, collections JSON NOT NULL,
            primary_recovering BOOLEAN NOT NULL DEFAULT FALSE
        )""")
        c.execute("SHOW COLUMNS FROM face_sync_control LIKE 'primary_recovering'")
        if not c.fetchone():
            try:
                c.execute(
                    "ALTER TABLE face_sync_control ADD COLUMN primary_recovering BOOLEAN NOT NULL DEFAULT FALSE"
                )
            except pymysql.err.OperationalError as exc:
                if exc.args[0] != 1060:
                    raise
        c.execute("""CREATE TABLE IF NOT EXISTS face_sync_changes (
            seq BIGINT AUTO_INCREMENT PRIMARY KEY, operation_id VARCHAR(64) NOT NULL UNIQUE,
            payload JSON NOT NULL, created_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3)
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS face_sync_people (
            identity CHAR(64) PRIMARY KEY, seq BIGINT NOT NULL, payload JSON NOT NULL
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS face_sync_nodes (
            id VARCHAR(64) PRIMARY KEY, url TEXT NOT NULL, state VARCHAR(24) NOT NULL DEFAULT 'new',
            applied_seq BIGINT NOT NULL DEFAULT 0, owner VARCHAR(64) NULL,
            heartbeat DATETIME(3) NULL, attempts INT NOT NULL DEFAULT 0,
            next_retry DATETIME(3) NULL, last_error VARCHAR(500) NULL,
            updated_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3)
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS face_sync_reads (
            id CHAR(32) PRIMARY KEY, node_id VARCHAR(64) NOT NULL, expires DATETIME(3) NOT NULL,
            INDEX node_expiry (node_id, expires)
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS face_sync_applied (
            node_id VARCHAR(64) NOT NULL, identity CHAR(64) NOT NULL,
            manifest_hash CHAR(64) NOT NULL, images_hash CHAR(64) NOT NULL,
            PRIMARY KEY (node_id, identity)
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS face_sync_requests (
            node_id VARCHAR(64) PRIMARY KEY, request_id CHAR(32) NOT NULL,
            target_seq BIGINT NOT NULL,
            requested_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
            completed_at DATETIME(3) NULL
        )""")
        c.execute(
            "INSERT IGNORE INTO face_sync_control (id, source_url, collections) VALUES (1,%s,%s)",
            (source(), encode(collections())),
        )
        for name, url in settings.insightface_replicas.items():
            c.execute(
                "INSERT IGNORE INTO face_sync_nodes (id,url) VALUES (%s,%s)",
                (name, url.rstrip("/")),
            )
            c.execute("SELECT url FROM face_sync_nodes WHERE id=%s", (name,))
            if c.fetchone()["url"] != url.rstrip("/"):
                raise ReplicationUnavailable(f"副本 {name} 地址已改变；请使用新副本名重新初始化")


def control(cursor, *, lock=False, require_initialized=True):
    cursor.execute("SELECT * FROM face_sync_control WHERE id=1" + (" FOR UPDATE" if lock else ""))
    row = cursor.fetchone()
    if not row or (require_initialized and not row["initialized"]):
        raise ReplicationUnavailable("人脸同步基线尚未初始化")
    if row["source_url"] != source() or json.loads(row["collections"]) != collections():
        raise ReplicationUnavailable("人脸主节点或集合配置已改变，需要重新建立同步基线")
    return row


def before_write():
    if settings.insightface_replication_enabled:
        with connect() as db, db.cursor() as c:
            if control(c)["primary_recovering"]:
                raise ReplicationUnavailable("主节点正在从备份恢复，暂不接受人物写入")


def assert_primary_clean():
    with connect() as db, db.cursor() as c:
        if control(c)["primary_recovering"]:
            raise ReplicationUnavailable("主节点恢复尚未完成，暂不接受主节点读取")
        c.execute(
            "SELECT id FROM person_operations WHERE status IN ('running','recovering','uncertain') AND JSON_UNQUOTE(JSON_EXTRACT(payload,'$.kind'))='transaction' LIMIT 1"
        )
        if c.fetchone():
            raise ReplicationUnavailable("主节点有待恢复的人物操作，请等待恢复后重试")


def append(cursor, operation_id, snapshots):
    """Caller commits this outbox and the successful person operation together."""
    control(cursor, lock=True)
    cursor.execute(
        "INSERT INTO face_sync_changes (operation_id,payload) VALUES (%s,%s)",
        (operation_id, encode(snapshots)),
    )
    seq = cursor.lastrowid
    for snapshot in snapshots:
        cursor.execute(
            "INSERT INTO face_sync_people (identity,seq,payload) VALUES (%s,%s,%s) ON DUPLICATE KEY UPDATE seq=VALUES(seq),payload=VALUES(payload)",
            (identity(snapshot["collection"], snapshot["person_id"]), seq, encode(snapshot)),
        )
    cursor.execute("UPDATE face_sync_control SET head=%s WHERE id=1", (seq,))
    return seq


def status():
    with connect() as db, db.cursor() as c:
        db.begin()
        root = control(c, require_initialized=False)
        c.execute(
            "SELECT n.id,n.url,n.state,n.applied_seq,n.heartbeat,n.attempts,n.next_retry,"
            "n.last_error,n.updated_at,UTC_TIMESTAMP(3) AS observed_at,"
            "(n.heartbeat > DATE_SUB(UTC_TIMESTAMP(3), INTERVAL %s SECOND)) AS heartbeat_fresh,"
            "r.request_id,r.target_seq,r.requested_at,r.completed_at "
            "FROM face_sync_nodes n LEFT JOIN face_sync_requests r ON r.node_id=n.id ORDER BY n.id",
            (settings.insightface_replica_health_ttl_s,),
        )
        nodes = []
        for row in c.fetchall():
            if row["id"] not in settings.insightface_replicas:
                continue
            for key, value in row.items():
                if value is not None and (
                    key.endswith("_at") or key in {"heartbeat", "next_retry"}
                ):
                    row[key] = value.replace(tzinfo=timezone.utc)
            row["lag"] = max(0, root["head"] - row["applied_seq"])
            row["heartbeat_fresh"] = bool(row["heartbeat_fresh"])
            row["read_eligible"] = bool(
                root["initialized"]
                and row["state"] == "ready"
                and row["applied_seq"] == root["head"]
                and row["heartbeat_fresh"]
            )
            request = {
                "id": row.pop("request_id"),
                "target_sequence": row.pop("target_seq"),
                "requested_at": row.pop("requested_at"),
                "completed_at": row.pop("completed_at"),
            }
            row["manual_request"] = request if request["id"] else None
            nodes.append(row)
        c.execute(
            "SELECT COUNT(*) AS n FROM person_operations WHERE status IN ('running','recovering','uncertain') AND JSON_UNQUOTE(JSON_EXTRACT(payload,'$.kind'))='transaction'"
        )
        pending = c.fetchone()["n"]
        db.commit()
    return {
        "enabled": True,
        "initialized": bool(root["initialized"]),
        "committed_sequence": root["head"],
        "write_policy": "primary_and_durable_outbox",
        "primary_recovering": bool(root["primary_recovering"]),
        "pending_primary_operations": pending,
        "replicas": nodes,
    }


def request_sync(node_id=None):
    """Persist a coalesced request; only the existing replica owner may execute it."""
    if not settings.insightface_replication_enabled:
        raise ReplicationUnavailable("InsightFace 副本同步未启用")
    if node_id is not None and node_id not in settings.insightface_replicas:
        raise ValueError("副本不存在")
    names = [node_id] if node_id is not None else sorted(settings.insightface_replicas)
    requested, skipped = [], []
    with connect() as db:
        db.begin()
        try:
            with db.cursor() as c:
                root = control(c, lock=True)
                if root["primary_recovering"]:
                    raise ReplicationUnavailable("主节点正在恢复，请完成恢复后再触发同步")
                for name in names:
                    c.execute("SELECT state FROM face_sync_nodes WHERE id=%s FOR UPDATE", (name,))
                    node = c.fetchone()
                    state = node["state"] if node else "new"
                    if state not in {"ready", "retry"}:
                        skipped.append({"id": name, "state": state})
                        continue
                    c.execute(
                        "SELECT * FROM face_sync_requests WHERE node_id=%s FOR UPDATE", (name,)
                    )
                    previous = c.fetchone()
                    reused = bool(previous and previous["completed_at"] is None)
                    request_id = previous["request_id"] if reused else uuid4().hex
                    if not reused:
                        c.execute(
                            "INSERT INTO face_sync_requests (node_id,request_id,target_seq) VALUES (%s,%s,%s) "
                            "ON DUPLICATE KEY UPDATE request_id=VALUES(request_id),target_seq=VALUES(target_seq),"
                            "requested_at=UTC_TIMESTAMP(3),completed_at=NULL",
                            (name, request_id, root["head"]),
                        )
                    # Do not change state, owner, checkpoint, or quarantine reason.
                    c.execute("UPDATE face_sync_nodes SET next_retry=NULL WHERE id=%s", (name,))
                    requested.append(
                        {
                            "id": name,
                            "request_id": request_id,
                            "target_sequence": previous["target_seq"] if reused else root["head"],
                            "reused": reused,
                        }
                    )
                if not requested:
                    raise ReplicationUnavailable(
                        "没有可触发的副本；请检查同步中、未初始化或隔离状态"
                    )
            db.commit()
        except BaseException:
            db.rollback()
            raise
    return {"requested": requested, "skipped": skipped}


def complete_sync_request(name, request_id):
    """Acknowledge only the request observed before this worker's health check."""
    if not request_id:
        return
    with connect() as db, db.cursor() as c:
        db.begin()
        try:
            c.execute(
                "SELECT state,applied_seq FROM face_sync_nodes WHERE id=%s FOR UPDATE", (name,)
            )
            node = c.fetchone()
            if node and node["state"] == "ready":
                c.execute(
                    "UPDATE face_sync_requests SET completed_at=UTC_TIMESTAMP(3) "
                    "WHERE node_id=%s AND request_id=%s AND completed_at IS NULL AND target_seq<=%s",
                    (name, request_id, node["applied_seq"]),
                )
            db.commit()
        except BaseException:
            db.rollback()
            raise


def acquire_read():
    """Admit against a committed version, atomically with draining admission."""
    with connect() as db:
        db.begin()
        try:
            with db.cursor() as c:
                root = control(c)
                candidates = list(settings.insightface_replicas)
                random.shuffle(candidates)
                for name in candidates:
                    c.execute(
                        "SELECT id,url,applied_seq FROM face_sync_nodes WHERE id=%s AND state='ready' AND applied_seq=%s AND heartbeat > DATE_SUB(UTC_TIMESTAMP(3), INTERVAL %s SECOND) FOR UPDATE",
                        (name, root["head"], settings.insightface_replica_health_ttl_s),
                    )
                    node = c.fetchone()
                    if node:
                        lease = uuid4().hex
                        c.execute(
                            "INSERT INTO face_sync_reads (id,node_id,expires) VALUES (%s,%s,DATE_ADD(UTC_TIMESTAMP(3), INTERVAL %s SECOND))",
                            (lease, name, settings.insightface_read_lease_s),
                        )
                        db.commit()
                        return dict(node, lease=lease)
            db.commit()
        except BaseException:
            db.rollback()
            raise
    return None


def renew_read(lease):
    with connect() as db, db.cursor() as c:
        changed = c.execute(
            "UPDATE face_sync_reads SET expires=DATE_ADD(UTC_TIMESTAMP(3), INTERVAL %s SECOND) WHERE id=%s AND expires > UTC_TIMESTAMP(3)",
            (settings.insightface_read_lease_s, lease),
        )
        if not changed:
            raise ReplicationUnavailable("副本读取租约已失效")


def release_read(lease):
    with connect() as db, db.cursor() as c:
        c.execute("DELETE FROM face_sync_reads WHERE id=%s", (lease,))


def quarantine(name, reason):
    with connect() as db, db.cursor() as c:
        c.execute(
            "UPDATE face_sync_nodes SET state='quarantined',last_error=%s WHERE id=%s",
            (reason[:500], name),
        )


def node(name):
    with connect() as db, db.cursor() as c:
        root = control(c)
        c.execute(
            "SELECT n.*, (n.next_retry IS NULL OR n.next_retry <= UTC_TIMESTAMP(3)) AS due,"
            "r.request_id AS sync_request_id FROM face_sync_nodes n "
            "LEFT JOIN face_sync_requests r ON r.node_id=n.id AND r.completed_at IS NULL WHERE n.id=%s",
            (name,),
        )
        return root, c.fetchone()


def begin_sync(name, owner):
    with connect() as db, db.cursor() as c:
        # Admission locks this same row. Readers already admitted retain leases.
        return (
            c.execute(
                "UPDATE face_sync_nodes SET state='draining',owner=%s WHERE id=%s AND state IN ('ready','retry')",
                (owner, name),
            )
            == 1
        )


def drained(name):
    with connect() as db, db.cursor() as c:
        c.execute("DELETE FROM face_sync_reads WHERE expires <= UTC_TIMESTAMP(3)")
        c.execute("SELECT COUNT(*) AS n FROM face_sync_reads WHERE node_id=%s", (name,))
        return c.fetchone()["n"] == 0


def owned_update(name, owner, sql, args=()):
    with connect() as db, db.cursor() as c:
        if (
            c.execute(
                "UPDATE face_sync_nodes SET "
                + sql
                + " WHERE id=%s AND owner=%s AND state IN ('draining','syncing')",
                (*args, name, owner),
            )
            != 1
        ):
            raise ReplicationUnavailable("副本同步执行权已失效")


def changes(after):
    with connect() as db, db.cursor() as c:
        c.execute(
            "SELECT seq,payload FROM face_sync_changes WHERE seq>%s ORDER BY seq LIMIT %s",
            (after, settings.insightface_sync_batch_size),
        )
        return [dict(row, payload=json.loads(row["payload"])) for row in c.fetchall()]


def checkpoint(name, owner, seq):
    owned_update(
        name, owner, "applied_seq=%s,heartbeat=UTC_TIMESTAMP(3),attempts=0,last_error=NULL", (seq,)
    )


def applied(name, snapshot):
    with connect() as db, db.cursor() as c:
        c.execute(
            "SELECT * FROM face_sync_applied WHERE node_id=%s AND identity=%s",
            (name, identity(snapshot["collection"], snapshot["person_id"])),
        )
        return c.fetchone()


def record_applied(name, owner, snapshot):
    with connect() as db:
        db.begin()
        with db.cursor() as c:
            c.execute("SELECT state,owner FROM face_sync_nodes WHERE id=%s FOR UPDATE", (name,))
            row = c.fetchone()
            if row["owner"] != owner or row["state"] != "syncing":
                raise ReplicationUnavailable("副本同步执行权已失效")
            c.execute(
                "INSERT INTO face_sync_applied (node_id,identity,manifest_hash,images_hash) VALUES (%s,%s,%s,%s) ON DUPLICATE KEY UPDATE manifest_hash=VALUES(manifest_hash),images_hash=VALUES(images_hash)",
                (
                    name,
                    identity(snapshot["collection"], snapshot["person_id"]),
                    digest(snapshot),
                    digest(image_store.canonical_images(snapshot["images"])),
                ),
            )
        db.commit()


def ready(name, owner):
    owned_update(
        name,
        owner,
        "state='ready',owner=NULL,heartbeat=UTC_TIMESTAMP(3),last_error=NULL,next_retry=NULL,attempts=0",
    )


def heartbeat(name):
    with connect() as db, db.cursor() as c:
        c.execute("UPDATE face_sync_nodes SET heartbeat=UTC_TIMESTAMP(3) WHERE id=%s", (name,))


def retry(name, owner, error):
    owned_update(
        name,
        owner,
        "state='retry',owner=NULL,attempts=attempts+1,next_retry=DATE_ADD(UTC_TIMESTAMP(3),INTERVAL LEAST(300,POW(2,LEAST(attempts,8))) SECOND),last_error=%s",
        (error[:500],),
    )
