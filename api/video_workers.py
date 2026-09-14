"""Run decoder iteration and cleanup on one dedicated thread per video pass."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager


@asynccontextmanager
async def threaded_iterator(factory):
    """Pull one item at a time: caller queues provide bounded backpressure.

    Open/next/close all run on the same thread. Cancellation waits for an
    in-progress decoder read before closing it, so its file can safely be deleted.
    """
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="wcm-video")
    iterator = None
    end = object()

    def advance():
        nonlocal iterator
        if iterator is None:
            iterator = iter(factory())
        return next(iterator, end)

    def close():
        if iterator is not None and hasattr(iterator, "close"):
            iterator.close()

    async def items():
        while True:
            pending = loop.run_in_executor(executor, advance)
            try:
                item = await asyncio.shield(pending)
            except asyncio.CancelledError:
                # The queued close runs after this read on the same executor.
                # Retrieve exceptions without replacing the cancellation.
                await asyncio.gather(pending, return_exceptions=True)
                raise
            if item is end:
                break
            yield item

    stream = items()
    try:
        yield stream
    finally:
        try:
            await stream.aclose()
            await loop.run_in_executor(executor, close)
        finally:
            executor.shutdown(wait=False)
