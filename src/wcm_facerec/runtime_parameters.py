"""Thread-visible runtime overrides populated from the parameter database."""

from __future__ import annotations

import copy
import threading
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from types import MappingProxyType
from typing import Any

_lock = threading.RLock()
_snapshot: Mapping[str, Any] = MappingProxyType({})
_MISSING = object()
_task_snapshot: ContextVar[dict | None] = ContextVar("task_parameters", default=None)


def install(values: Mapping[str, Any]) -> None:
    """Atomically replace every runtime override with defensive copies."""
    global _snapshot
    replacement = MappingProxyType(copy.deepcopy(dict(values)))
    with _lock:
        _snapshot = replacement


def get(key: str, default: Any = _MISSING) -> Any:
    """Return a defensive copy so callers cannot mutate the shared snapshot."""
    frozen = _task_snapshot.get()
    if frozen is not None and key in frozen:
        return copy.deepcopy(frozen[key])
    return get_live(key, default)


def get_live(key: str, default: Any = _MISSING) -> Any:
    with _lock:
        if key in _snapshot:
            return copy.deepcopy(_snapshot[key])
    if default is _MISSING:
        raise KeyError(key)
    return default


def snapshot() -> dict[str, Any]:
    with _lock:
        return copy.deepcopy(dict(_snapshot))


@contextmanager
def frozen(values):
    token = _task_snapshot.set(copy.deepcopy(values))
    try:
        yield
    finally:
        _task_snapshot.reset(token)
