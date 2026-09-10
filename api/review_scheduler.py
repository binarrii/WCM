"""Shared task admission for all API workers in one container.

Each flock represents one running review. The OS releases it on process exit;
waiting reviews hold no slot. Keep the lock files in place (never unlink them).
"""

import asyncio
import fcntl
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from wcm_facerec.config import settings


@asynccontextmanager
async def review_task_slot(*, directory=None, limit=None):
    limit = settings.review_task_concurrency if limit is None else limit
    if limit < 1:
        raise ValueError("review task concurrency must be positive")
    directory = Path(directory or Path(tempfile.gettempdir()) / "wcm-review-slots")
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    slot = None
    try:
        while slot is None:
            for index in range(limit):
                candidate = (directory / f"{index}.lock").open("a")
                try:
                    fcntl.flock(candidate.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    candidate.close()
                except BaseException:
                    candidate.close()
                    raise
                else:
                    slot = candidate
                    break
            if slot is None:
                await asyncio.sleep(0.25)
        yield
    finally:
        if slot is not None:
            slot.close()
