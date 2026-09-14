"""MySQL-owned cluster locks and cancellation-safe blocking calls."""

import asyncio
import hashlib
from contextlib import asynccontextmanager
from contextvars import ContextVar

import pymysql
from pymysql.cursors import DictCursor

from . import runtime_parameters
from .config import settings


def connect():
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


async def run_sync(function, *args, **kwargs):
    """A cancelled coroutine must not release a lock while its SDK thread runs."""
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await asyncio.gather(task, return_exceptions=True)
        raise


_held: ContextVar[tuple] = ContextVar("cluster_locks", default=())


class LockLost(RuntimeError):
    pass


def check_locks():
    if any(handle["lost"] for handle in _held.get()):
        raise LockLost("集群协调连接已断开，停止继续写入")


def _acquire(connection, resource, limit):
    with connection.cursor() as cursor:
        for index in range(limit):
            name = f"wcm:{resource}:{index}"
            cursor.execute("SELECT GET_LOCK(%s, 0) AS acquired", (name,))
            if cursor.fetchone()["acquired"] == 1:
                return name
    return None


def _check(connection, name):
    with connection.cursor() as cursor:
        cursor.execute("SELECT IS_USED_LOCK(%s) = CONNECTION_ID() AS owned", (name,))
        if cursor.fetchone()["owned"] != 1:
            raise LockLost("集群锁已失效")


def _try_slot(resource, limit):
    connection = connect()
    acquired = None
    try:
        acquired = _acquire(connection, resource, limit)
        if acquired:
            return connection, acquired
    finally:
        if not acquired:
            connection.close()
    return None


@asynccontextmanager
async def cluster_slot(resource, limit=1):
    if not settings.cluster_enabled:
        yield
        return
    # Global named locks survive commits but are released when the connection dies.
    scope = hashlib.sha256(f"{settings.cluster_namespace}:{resource}".encode()).hexdigest()[:40]
    connection = None
    monitor = None
    handle = {"lost": False}
    token = None
    owner = asyncio.current_task()
    try:
        acquired = None
        while acquired is None:
            acquired = await run_sync(_try_slot, scope, limit)
            if acquired is None:
                await asyncio.sleep(0.25)
        connection, name = acquired
        token = _held.set((*_held.get(), handle))

        async def watch():
            try:
                while True:
                    await asyncio.sleep(2)
                    await run_sync(_check, connection, name)
            except asyncio.CancelledError:
                raise
            except Exception:
                handle["lost"] = True
                owner.cancel()

        monitor = asyncio.create_task(watch())
        try:
            yield
            check_locks()
        except asyncio.CancelledError:
            if handle["lost"]:
                raise LockLost("集群协调连接已断开") from None
            raise
    finally:
        if monitor:
            monitor.cancel()
            await asyncio.gather(monitor, return_exceptions=True)
        if token is not None:
            _held.reset(token)
        if connection:
            connection.close()


@asynccontextmanager
async def model_slot(model):
    if not settings.cluster_enabled:
        yield
        return
    key = f"{model}_concurrency"
    limit = runtime_parameters.get_live(key, settings.seed_value(key))
    async with cluster_slot(f"model:{model}", int(limit)):
        yield
