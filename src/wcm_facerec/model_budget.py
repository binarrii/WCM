"""Share one model attempt's deadline/cancellation with synchronous SDK threads."""

import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Event

import httpx


@dataclass
class _Budget:
    deadline: float | None
    cancelled: Event = field(default_factory=Event)


_current_budget: ContextVar[_Budget | None] = ContextVar("model_request_budget", default=None)


@contextmanager
def model_request_budget(timeout):
    budget = _Budget(time.monotonic() + timeout if timeout is not None else None)
    token = _current_budget.set(budget)
    try:
        yield
    finally:
        budget.cancelled.set()
        _current_budget.reset(token)


@contextmanager
def request_scope():
    """Also make non-model SDK threads cooperatively cancellable."""
    if _current_budget.get() is not None:
        yield
    else:
        with model_request_budget(None):
            yield


def cancel_model_requests():
    budget = _current_budget.get()
    if budget is not None:
        budget.cancelled.set()


def remaining_request_time():
    budget = _current_budget.get()
    if budget is None:
        return None
    remaining = budget.deadline - time.monotonic() if budget.deadline is not None else None
    if budget.cancelled.is_set() or (remaining is not None and remaining <= 0):
        raise httpx.ReadTimeout("Model attempt ended; no further SDK requests may start")
    return remaining
