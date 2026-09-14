"""Context propagated to child coroutines and SDK threads, never process globals."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class Execution:
    task_id: str
    token: str
    attempt: int


current_execution: ContextVar[Execution | None] = ContextVar("review_execution", default=None)


@contextmanager
def execution_scope(task_id, token, attempt):
    previous = current_execution.set(Execution(task_id, token, attempt))
    try:
        yield
    finally:
        current_execution.reset(previous)
