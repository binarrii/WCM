"""MySQL-backed, process-local cache for general application parameters."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import logging
import math
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Literal

import pymysql
from pymysql.cursors import DictCursor

from wcm_facerec import runtime_parameters
from wcm_facerec.config import (
    BUSINESS_PARAMETER_SPECS,
    normalize_business_parameter,
    settings,
)

logger = logging.getLogger(__name__)

ParameterType = Literal["string", "number", "boolean", "enum", "json"]
EnumScalar = str | int | float
PARAMETER_TYPES = frozenset({"string", "number", "boolean", "enum", "json"})
SYNC_INTERVAL_SECONDS = 2


class ParameterStoreUnavailable(RuntimeError):
    """Raised when parameter persistence is disabled or unreachable."""


class ParameterAlreadyExists(RuntimeError):
    """Raised when a parameter key is created twice."""


class ParameterNotFound(RuntimeError):
    """Raised when a parameter key does not exist."""


class ParameterProtected(RuntimeError):
    """Raised when a built-in runtime parameter is deleted."""


@dataclass(frozen=True)
class _CachedParameter:
    key: str
    value: Any
    value_type: ParameterType
    group: str
    enum_values: tuple[EnumScalar, ...] | None
    created_at: str | None
    updated_at: str | None

    def public(self) -> dict[str, Any]:
        spec = BUSINESS_PARAMETER_SPECS.get(self.key)
        secret = bool(spec and spec.secret)
        return {
            "key": self.key,
            "value": None if secret else copy.deepcopy(self.value),
            "type": self.value_type,
            "group": self.group,
            "options": list(self.enum_values) if self.enum_values is not None else None,
            "built_in": spec is not None,
            "secret": secret,
            "has_value": bool(self.value) if secret else True,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


_cache_lock = threading.RLock()
_database_sync_lock = threading.RLock()
_snapshot: Mapping[str, _CachedParameter] = MappingProxyType({})
_loaded_at: str | None = None
_version = 0
_sync_task: asyncio.Task | None = None


def is_enabled() -> bool:
    return settings.review_tasks_db_enabled


def _connect():
    if not is_enabled():
        raise ParameterStoreUnavailable("参数配置数据库未启用")
    try:
        return pymysql.connect(
            host=settings.review_tasks_db_host,
            port=settings.review_tasks_db_port,
            user=settings.review_tasks_db_user,
            password=settings.review_tasks_db_password,
            database=settings.review_tasks_db_name,
            charset="utf8mb4",
            cursorclass=DictCursor,
            autocommit=True,
            connect_timeout=settings.review_tasks_db_connect_timeout_s,
            read_timeout=10,
            write_timeout=10,
            init_command="SET time_zone = '+00:00'",
        )
    except pymysql.MySQLError as exc:
        raise ParameterStoreUnavailable(f"参数配置数据库不可用：{exc}") from exc


async def _run(function, *args):
    try:
        return await asyncio.to_thread(function, *args)
    except (
        ParameterAlreadyExists,
        ParameterNotFound,
        ParameterProtected,
        ParameterStoreUnavailable,
    ):
        raise
    except pymysql.MySQLError as exc:
        raise ParameterStoreUnavailable(f"参数配置数据库操作失败：{exc}") from exc


def _enum_scalar_identity(value: Any) -> tuple[str, Any]:
    if isinstance(value, str):
        return ("string", value)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("enum 类型的可选值只能是 string 或 number")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("enum 类型的 number 可选值必须是有限数字")
    return ("number", value)


def validate_enum_values(enum_values: Any) -> tuple[EnumScalar, ...]:
    """Return validated enum metadata with type-aware uniqueness."""
    if not isinstance(enum_values, (list, tuple)) or not enum_values:
        raise ValueError("enum 类型必须提供非空的可选值列表")
    validated: list[EnumScalar] = []
    identities: set[tuple[str, Any]] = set()
    for option in enum_values:
        identity = _enum_scalar_identity(option)
        if identity in identities:
            raise ValueError("enum 类型的可选值不能重复")
        identities.add(identity)
        validated.append(option)
    return tuple(validated)


def _encode_enum_values(enum_values: Any) -> str:
    return json.dumps(
        validate_enum_values(enum_values),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _decode_enum_values(raw_value: str | None) -> tuple[EnumScalar, ...]:
    if raw_value is None:
        raise ParameterStoreUnavailable("数据库中的 enum 参数缺少可选值")
    try:
        value = json.loads(raw_value)
        return validate_enum_values(value)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ParameterStoreUnavailable("数据库中的 enum 可选值无效") from exc


def _json_loads(raw_value: str) -> Any:
    def reject_constant(constant: str) -> None:
        raise ValueError(f"无效 JSON 常量：{constant}")

    return json.loads(raw_value, parse_constant=reject_constant)


def encode_value(
    value: Any,
    value_type: ParameterType,
    enum_values: Any = None,
) -> str:
    """Validate and encode a typed value for text storage."""
    if value_type != "enum" and enum_values is not None:
        raise ValueError("只有 enum 类型可以设置可选值")
    if value_type == "string":
        if not isinstance(value, str):
            raise ValueError("string 类型的值必须是字符串")
        return value
    if value_type == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("number 类型的值必须是数字")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("number 类型的值必须是有限数字")
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    if value_type == "boolean":
        if not isinstance(value, bool):
            raise ValueError("boolean 类型的值必须是 true 或 false")
        return json.dumps(value)
    if value_type == "enum":
        options = validate_enum_values(enum_values)
        identity = _enum_scalar_identity(value)
        if identity not in {_enum_scalar_identity(option) for option in options}:
            raise ValueError("enum 类型的值必须来自可选值列表")
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    if value_type == "json":
        try:
            return json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("json 类型的值必须是有效 JSON") from exc
    raise ValueError(f"不支持的参数类型：{value_type}")


def decode_value(
    raw_value: str,
    value_type: ParameterType,
    enum_values: Any = None,
) -> Any:
    """Decode a persisted value and reject invalid manually edited rows."""
    if value_type == "string":
        return raw_value
    try:
        value = _json_loads(raw_value)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ParameterStoreUnavailable("数据库包含无法解析的参数值") from exc
    if value_type == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ParameterStoreUnavailable("数据库中的 number 参数不是数字")
        if isinstance(value, float) and not math.isfinite(value):
            raise ParameterStoreUnavailable("数据库中的 number 参数不是有限数字")
        return value
    if value_type == "boolean":
        if not isinstance(value, bool):
            raise ParameterStoreUnavailable("数据库中的 boolean 参数不是布尔值")
        return value
    if value_type == "enum":
        try:
            options = validate_enum_values(enum_values)
            identity = _enum_scalar_identity(value)
        except ValueError as exc:
            raise ParameterStoreUnavailable("数据库中的 enum 参数无效") from exc
        if identity not in {_enum_scalar_identity(option) for option in options}:
            raise ParameterStoreUnavailable("数据库中的 enum 参数值不在可选值列表中")
        return value
    if value_type == "json":
        return value
    raise ParameterStoreUnavailable(f"数据库包含不支持的参数类型：{value_type}")


def _iso(value: datetime | str | None) -> str | None:
    if value is None or isinstance(value, str):
        return value
    return value.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def _cached_parameter(row: dict[str, Any]) -> _CachedParameter:
    key = row["config_key"]
    value_type = row["value_type"]
    enum_values = _decode_enum_values(row.get("enum_values")) if value_type == "enum" else None
    if value_type != "enum" and row.get("enum_values") is not None:
        raise ParameterStoreUnavailable(f"非 enum 参数 {key} 不能包含可选值")
    value = decode_value(row["config_value"], value_type, enum_values)
    spec = BUSINESS_PARAMETER_SPECS.get(key)
    if spec and value_type != spec.value_type:
        raise ParameterStoreUnavailable(
            f"内置参数 {key} 的类型必须是 {spec.value_type}，当前为 {value_type}"
        )
    if spec and row["group_name"] != spec.group:
        raise ParameterStoreUnavailable(
            f"内置参数 {key} 的分组必须是 {spec.group}，当前为 {row['group_name']}"
        )
    if spec and spec.enum_values != enum_values:
        raise ParameterStoreUnavailable(f"内置参数 {key} 的枚举可选值与定义不一致")
    try:
        value = normalize_business_parameter(key, value)
    except ValueError as exc:
        raise ParameterStoreUnavailable(f"内置参数 {key} 的值无效：{exc}") from exc
    return _CachedParameter(
        key=key,
        value=value,
        value_type=value_type,
        group=row["group_name"],
        enum_values=enum_values,
        created_at=_iso(row.get("created_at")),
        updated_at=_iso(row.get("updated_at")),
    )


def _install_snapshot(rows: list[dict[str, Any]]) -> None:
    global _loaded_at, _snapshot, _version
    cached = [_cached_parameter(row) for row in rows]
    entries = {entry.key: entry for entry in cached}
    loaded_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    with _cache_lock:
        # Readers see either the complete old mapping or the complete new mapping.
        _snapshot = MappingProxyType(entries)
        runtime_parameters.install(
            {key: entry.value for key, entry in entries.items() if key in BUSINESS_PARAMETER_SPECS}
        )
        _loaded_at = loaded_at
        _version += 1


def _initialize_sync() -> None:
    with _connect() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT GET_LOCK('wcm_system_parameters_initialize', 30) AS acquired")
        lock_result = cursor.fetchone()
        if not lock_result or lock_result.get("acquired") != 1:
            raise ParameterStoreUnavailable("等待参数配置表初始化锁超时")
        # The advisory lock is connection-scoped and is released when this
        # short-lived initialization connection closes.
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS system_parameters (
                `key` VARCHAR(191) NOT NULL PRIMARY KEY,
                `value` LONGTEXT NOT NULL,
                `type` VARCHAR(16) NOT NULL,
                `group` VARCHAR(100) NOT NULL DEFAULT 'default',
                enum_values LONGTEXT NULL,
                created_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
                updated_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3)
                    ON UPDATE CURRENT_TIMESTAMP(3),
                INDEX idx_system_parameters_group_key (`group`, `key`),
                CONSTRAINT chk_system_parameters_type
                    CHECK (`type` IN ('string', 'number', 'boolean', 'enum', 'json'))
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
            """
        )
        cursor.execute("SHOW COLUMNS FROM system_parameters LIKE 'enum_values'")
        if cursor.fetchone() is None:
            cursor.execute(
                "ALTER TABLE system_parameters ADD COLUMN enum_values LONGTEXT NULL AFTER `group`"
            )
        cursor.execute(
            """
            SELECT CHECK_CLAUSE
            FROM information_schema.CHECK_CONSTRAINTS
            WHERE CONSTRAINT_SCHEMA = DATABASE()
              AND CONSTRAINT_NAME = 'chk_system_parameters_type'
            """
        )
        type_constraint = cursor.fetchone()
        check_clause = (type_constraint or {}).get("CHECK_CLAUSE", "").lower()
        if "boolean" not in check_clause or "enum" not in check_clause:
            if type_constraint is not None:
                cursor.execute(
                    "ALTER TABLE system_parameters DROP CHECK chk_system_parameters_type"
                )
            cursor.execute(
                """
                ALTER TABLE system_parameters
                ADD CONSTRAINT chk_system_parameters_type
                CHECK (`type` IN ('string', 'number', 'boolean', 'enum', 'json'))
                """
            )

        # Existing values always win. Metadata follows the built-in definition,
        # allowing old string/json values to be converted without losing edits.
        for key, spec in BUSINESS_PARAMETER_SPECS.items():
            cursor.execute(
                """
                SELECT `value` AS config_value, `type` AS value_type,
                       enum_values
                FROM system_parameters
                WHERE `key` = %s
                """,
                (key,),
            )
            existing = cursor.fetchone()
            if existing is None:
                value = settings.seed_value(key)
            else:
                previous_options = (
                    _decode_enum_values(existing.get("enum_values"))
                    if existing["value_type"] == "enum"
                    else None
                )
                value = decode_value(
                    existing["config_value"],
                    existing["value_type"],
                    previous_options,
                )
            value = normalize_business_parameter(key, value)
            encoded_value = encode_value(value, spec.value_type, spec.enum_values)
            encoded_options = (
                _encode_enum_values(spec.enum_values) if spec.value_type == "enum" else None
            )
            if existing is None:
                cursor.execute(
                    """
                    INSERT INTO system_parameters
                        (`key`, `value`, `type`, `group`, enum_values)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (key, encoded_value, spec.value_type, spec.group, encoded_options),
                )
            else:
                cursor.execute(
                    """
                    UPDATE system_parameters
                    SET `value` = %s, `type` = %s, `group` = %s,
                        enum_values = %s
                    WHERE `key` = %s
                    """,
                    (encoded_value, spec.value_type, spec.group, encoded_options, key),
                )


def _load_sync() -> list[dict[str, Any]]:
    with _connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT `key` AS config_key, `value` AS config_value,
                   `type` AS value_type, `group` AS group_name,
                   enum_values, created_at, updated_at
            FROM system_parameters
            ORDER BY `group`, `key`
            """
        )
        return list(cursor.fetchall())


def _refresh_sync() -> int:
    with _database_sync_lock:
        rows = _load_sync()
        _install_snapshot(rows)
        return len(rows)


async def refresh() -> int:
    """Reload all rows and atomically publish a new in-memory snapshot."""
    return await _run(_refresh_sync)


async def _sync_loop() -> None:
    while True:
        await asyncio.sleep(SYNC_INTERVAL_SECONDS)
        try:
            await refresh()
        except ParameterStoreUnavailable as exc:
            logger.warning("Parameter cache refresh unavailable: %s", exc)


async def initialize() -> None:
    """Create storage, load the first snapshot, and start cross-worker syncing."""
    global _sync_task
    if not is_enabled():
        _install_snapshot([])
        return
    await _run(_initialize_sync)
    await refresh()
    if _sync_task is None or _sync_task.done():
        _sync_task = asyncio.create_task(_sync_loop(), name="system-parameter-sync")


async def close() -> None:
    global _sync_task
    task = _sync_task
    _sync_task = None
    if task is None:
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


def list_parameters() -> dict[str, Any]:
    """Return a defensive copy of the current snapshot without database I/O."""
    with _cache_lock:
        items = [entry.public() for entry in _snapshot.values()]
        return {
            "items": items,
            "total": len(items),
            "loaded_at": _loaded_at,
            "version": _version,
        }


_MISSING = object()


def get(key: str, default: Any = _MISSING) -> Any:
    """Read a typed parameter from memory for latency-sensitive call sites."""
    with _cache_lock:
        entry = _snapshot.get(key)
        if entry is not None:
            return copy.deepcopy(entry.value)
    if default is _MISSING:
        raise KeyError(key)
    return default


def _encoded_value(
    key: str,
    value: Any,
    value_type: ParameterType,
    group: str,
    enum_values: Any = None,
) -> tuple[str, str | None]:
    spec = BUSINESS_PARAMETER_SPECS.get(key)
    if spec:
        if value_type != spec.value_type:
            raise ValueError(f"内置参数 {key} 的类型必须是 {spec.value_type}")
        if group != spec.group:
            raise ValueError(f"内置参数 {key} 的分组必须是 {spec.group}")
        if spec.enum_values != (
            validate_enum_values(enum_values) if value_type == "enum" else None
        ):
            raise ValueError(f"内置参数 {key} 的枚举可选值不能修改")
        value = normalize_business_parameter(key, value)
    encoded_value = encode_value(value, value_type, enum_values)
    encoded_options = _encode_enum_values(enum_values) if value_type == "enum" else None
    return encoded_value, encoded_options


def _create_and_refresh_sync(
    key: str,
    value: Any,
    value_type: ParameterType,
    group: str,
    enum_values: Any = None,
) -> dict[str, Any]:
    encoded, encoded_options = _encoded_value(key, value, value_type, group, enum_values)
    with _database_sync_lock:
        try:
            with _connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO system_parameters
                        (`key`, `value`, `type`, `group`, enum_values)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (key, encoded, value_type, group, encoded_options),
                )
        except pymysql.err.IntegrityError as exc:
            if exc.args and exc.args[0] == 1062:
                raise ParameterAlreadyExists(f"参数 {key} 已存在") from exc
            raise
        rows = _load_sync()
        _install_snapshot(rows)
        return _snapshot[key].public()


async def create(
    key: str,
    value: Any,
    value_type: ParameterType,
    group: str,
    enum_values: Any = None,
) -> dict[str, Any]:
    return await _run(_create_and_refresh_sync, key, value, value_type, group, enum_values)


def _update_and_refresh_sync(
    key: str,
    value: Any,
    value_type: ParameterType,
    group: str,
    enum_values: Any = None,
) -> dict[str, Any]:
    encoded, encoded_options = _encoded_value(key, value, value_type, group, enum_values)
    with _database_sync_lock:
        with _connect() as connection, connection.cursor() as cursor:
            affected = cursor.execute(
                """
                UPDATE system_parameters
                SET `value` = %s, `type` = %s, `group` = %s,
                    enum_values = %s
                WHERE `key` = %s
                """,
                (encoded, value_type, group, encoded_options, key),
            )
            if not affected:
                cursor.execute("SELECT 1 FROM system_parameters WHERE `key` = %s", (key,))
                if cursor.fetchone() is None:
                    raise ParameterNotFound(f"参数 {key} 不存在")
        rows = _load_sync()
        _install_snapshot(rows)
        return _snapshot[key].public()


async def update(
    key: str,
    value: Any,
    value_type: ParameterType,
    group: str,
    enum_values: Any = None,
) -> dict[str, Any]:
    return await _run(_update_and_refresh_sync, key, value, value_type, group, enum_values)


def _delete_and_refresh_sync(key: str) -> None:
    if key in BUSINESS_PARAMETER_SPECS:
        raise ParameterProtected(f"内置参数 {key} 不能删除")
    with _database_sync_lock:
        with _connect() as connection, connection.cursor() as cursor:
            if not cursor.execute("DELETE FROM system_parameters WHERE `key` = %s", (key,)):
                raise ParameterNotFound(f"参数 {key} 不存在")
        _install_snapshot(_load_sync())


async def delete(key: str) -> None:
    await _run(_delete_and_refresh_sync, key)
