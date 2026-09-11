"""Thread-visible runtime overrides populated from the parameter database."""

from __future__ import annotations

import copy
import threading
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

_lock = threading.RLock()
_snapshot: Mapping[str, Any] = MappingProxyType({})
_MISSING = object()


def install(values: Mapping[str, Any]) -> None:
    """Atomically replace every runtime override with defensive copies."""
    global _snapshot
    replacement = MappingProxyType(copy.deepcopy(dict(values)))
    with _lock:
        _snapshot = replacement


def get(key: str, default: Any = _MISSING) -> Any:
    """Return a defensive copy so callers cannot mutate the shared snapshot."""
    with _lock:
        if key in _snapshot:
            return copy.deepcopy(_snapshot[key])
    if default is _MISSING:
        raise KeyError(key)
    return default


def snapshot() -> dict[str, Any]:
    with _lock:
        return copy.deepcopy(dict(_snapshot))
