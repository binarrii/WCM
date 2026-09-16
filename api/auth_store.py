"""Shared authentication state. MySQL in the cluster, SQLite for local development."""

import hashlib
import json
import os
import secrets
import time
from contextlib import contextmanager
from functools import lru_cache
from ipaddress import ip_network
from pathlib import Path

from cryptography.fernet import Fernet
from fastapi import HTTPException
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Double,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    delete,
    insert,
    select,
    update,
)
from sqlalchemy.engine import URL

from wcm_facerec.config import settings


class AuthSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="WCM_AUTH_", env_file=".env", extra="ignore")
    database_url: str = ""
    secret_key: str = ""
    origins: list[str] = ["http://localhost:5173", "http://localhost:8000"]
    rp_id: str = "localhost"
    cookie_secure: bool = False
    trusted_proxies: list[str] = []
    session_hours: int = Field(default=12, ge=1, le=168)

    @field_validator("trusted_proxies")
    @classmethod
    def valid_proxy_networks(cls, values):
        return [str(ip_network(value, strict=False)) for value in values]


config = AuthSettings()
metadata = MetaData()
users = Table(
    "wcm_users",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("username", String(64), unique=True, nullable=False),
    Column("display_name", String(80), nullable=False),
    Column("password_hash", Text, nullable=False),
    Column("role", String(24), nullable=False),
    Column("active", Boolean, nullable=False, default=True),
    Column("totp_secret", Text),
    Column("totp_last_step", Integer, default=-1),
    Column("recovery_hashes", Text, default="[]"),
    Column("created_at", Double, nullable=False),
)
avatars = Table(
    "wcm_user_avatars",
    metadata,
    Column("user_id", String(36), primary_key=True),
    Column("content", LargeBinary, nullable=False),
    Column("version", String(64), nullable=False),
    Column("updated_at", Double, nullable=False),
)
sessions = Table(
    "wcm_sessions",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("user_id", String(36), index=True),
    Column("csrf", String(64), nullable=False),
    Column("expires_at", Double, index=True),
    Column("verified_at", Double, nullable=False),
)
credentials = Table(
    "wcm_passkeys",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("user_id", String(36), index=True),
    Column("credential_id", Text, nullable=False),
    Column("public_key", Text, nullable=False),
    Column("sign_count", BigInteger, nullable=False),
    Column("name", String(80), nullable=False),
    Column("created_at", Double, nullable=False),
)
passkey_details = Table(
    "wcm_passkey_details",
    metadata,
    Column("key_id", String(64), primary_key=True),
    Column("aaguid", String(36), nullable=False),
    Column("client_ip", String(45)),
)
challenges = Table(
    "wcm_auth_challenges",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("kind", String(32)),
    Column("user_id", String(36)),
    Column("binding", String(64)),
    Column("payload", Text),
    Column("expires_at", Double, index=True),
)
policies = Table(
    "wcm_role_permissions",
    metadata,
    Column("role", String(24), primary_key=True),
    Column("permissions", Text, nullable=False),
)
limits = Table(
    "wcm_auth_limits",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("count", Integer),
    Column("expires_at", Double, index=True),
)
mutex = Table("wcm_auth_mutex", metadata, Column("id", Integer, primary_key=True))
audit = Table(
    "wcm_auth_audit",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("actor_id", String(36)),
    Column("action", String(64)),
    Column("target_id", String(64)),
    Column("created_at", Double),
)
PERMISSIONS = {
    "people.read": "查看人物与人脸检索",
    "people.write": "新增、修改和删除人物",
    "review.read": "查看和下载审核结果",
    "review.run": "提交内容审核",
    "review.manage": "取消和删除审核任务",
    "parameters.manage": "参数配置",
    "system.manage": "系统管理",
}
ADMIN_ONLY = {"parameters.manage", "system.manage"}
DEFAULTS = {
    "user": ["people.read", "review.read", "review.run"],
    "admin": list(PERMISSIONS),
    "superadmin": list(PERMISSIONS),
}
COOKIE = "wcm_session"


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


@lru_cache(maxsize=1)
def engine():
    url = config.database_url
    if not url and settings.review_tasks_db_enabled:
        url = URL.create(
            "mysql+pymysql",
            username=settings.review_tasks_db_user,
            password=settings.review_tasks_db_password,
            host=settings.review_tasks_db_host,
            port=settings.review_tasks_db_port,
            database=settings.review_tasks_db_name,
            query={"charset": "utf8mb4"},
        )
    if not url:
        if settings.cluster_enabled:
            raise RuntimeError("Cluster authentication requires a shared database")
        Path("data").mkdir(exist_ok=True)
        url = "sqlite:///data/auth.sqlite3"
    sqlite = str(url).startswith("sqlite")
    if settings.cluster_enabled and sqlite:
        raise RuntimeError("Cluster authentication cannot use SQLite")
    return create_engine(
        url,
        pool_pre_ping=True,
        pool_recycle=300,
        connect_args={"check_same_thread": False, "timeout": 30} if sqlite else {},
    )


@lru_cache(maxsize=1)
def cipher():
    key = config.secret_key
    if not key:
        if settings.cluster_enabled:
            raise RuntimeError("Set WCM_AUTH_SECRET_KEY to a shared Fernet key")
        path = Path("data/auth.key")
        path.parent.mkdir(exist_ok=True)
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(fd, "wb") as stream:
                stream.write(Fernet.generate_key())
        key = path.read_text().strip()
    return Fernet(key.encode())


def initialize():
    cipher()
    db = engine()
    # DDL/startup is serialized across API replicas on the same database.
    with db.connect() as connection:
        if (
            db.dialect.name == "mysql"
            and connection.exec_driver_sql("SELECT GET_LOCK('wcm-auth-schema', 30)").scalar() != 1
        ):
            raise RuntimeError("Authentication schema initialization timed out")
        try:
            metadata.create_all(connection)
            connection.commit()
            if not connection.execute(select(mutex.c.id)).first():
                connection.execute(insert(mutex).values(id=1))
            for role, permissions in DEFAULTS.items():
                if not connection.execute(select(policies).where(policies.c.role == role)).first():
                    connection.execute(
                        insert(policies).values(role=role, permissions=json.dumps(permissions))
                    )
            connection.commit()
        finally:
            if db.dialect.name == "mysql":
                connection.exec_driver_sql("SELECT RELEASE_LOCK('wcm-auth-schema')")


@contextmanager
def transaction():
    """Serialize short auth mutations, including first signup and one-time challenges.

    Expected HTTP errors commit consumed attempts/challenges; other errors roll back.
    Password hashes are computed before acquiring this lock where possible.
    """
    with engine().connect() as connection:
        if engine().dialect.name == "sqlite":
            connection.exec_driver_sql("BEGIN IMMEDIATE")
        else:
            connection.begin()
            connection.execute(select(mutex).where(mutex.c.id == 1).with_for_update())
        try:
            yield connection
        except HTTPException:
            connection.commit()
            raise
        except BaseException:
            connection.rollback()
            raise
        else:
            connection.commit()


def row(connection, table, condition):
    value = connection.execute(select(table).where(condition)).mappings().first()
    return dict(value) if value else None


def permissions(connection, role):
    if role == "superadmin":
        return list(PERMISSIONS) + ["users.manage"]
    policy = row(connection, policies, policies.c.role == role)
    allowed = set(json.loads(policy["permissions"])) if policy else set()
    if role == "user":
        allowed -= ADMIN_ONLY
    return sorted(allowed & PERMISSIONS.keys())


def public_user(connection, user):
    avatar_version = connection.execute(
        select(avatars.c.version).where(avatars.c.user_id == user["id"])
    ).scalar()
    return {
        key: user[key] for key in ("id", "username", "display_name", "role", "active", "created_at")
    } | {
        "totp_enabled": bool(user["totp_secret"]),
        "permissions": permissions(connection, user["role"]),
        "avatar_version": avatar_version,
    }


def identity(token):
    if not token or len(token) > 128:
        return None
    with engine().connect() as connection:
        session = row(connection, sessions, sessions.c.id == digest(token))
        if not session or session["expires_at"] <= time.time():
            return None
        user = row(connection, users, users.c.id == session["user_id"])
        if not user or not user["active"]:
            return None
        return {"user": public_user(connection, user), "session": session}


def limit(key, maximum=10, seconds=300):
    with transaction() as connection:
        now = time.time()
        connection.execute(delete(limits).where(limits.c.expires_at < now))
        item = row(connection, limits, limits.c.id == digest(key))
        if item and item["count"] >= maximum:
            raise HTTPException(
                429, "尝试次数过多，请稍后再试", headers={"Retry-After": str(seconds)}
            )
        if item:
            connection.execute(
                update(limits).where(limits.c.id == item["id"]).values(count=item["count"] + 1)
            )
        else:
            connection.execute(
                insert(limits).values(id=digest(key), count=1, expires_at=now + seconds)
            )
        connection.execute(delete(challenges).where(challenges.c.expires_at < now))
        connection.execute(delete(sessions).where(sessions.c.expires_at < now))


def challenge(connection, kind, user_id=None, binding=None, payload=None):
    token = secrets.token_urlsafe(32)
    connection.execute(
        insert(challenges).values(
            id=digest(token),
            kind=kind,
            user_id=user_id,
            binding=binding,
            payload=json.dumps(payload or {}),
            expires_at=time.time() + 300,
        )
    )
    return token


def consume(connection, token, kind, binding=None):
    item = row(connection, challenges, challenges.c.id == digest(token))
    if (
        not item
        or item["kind"] != kind
        or item["binding"] != binding
        or item["expires_at"] <= time.time()
    ):
        raise HTTPException(400, "验证已失效，请重新开始")
    connection.execute(delete(challenges).where(challenges.c.id == item["id"]))
    item["payload"] = json.loads(item["payload"])
    return item
