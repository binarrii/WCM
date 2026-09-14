"""Stop a review in its owning worker after a durable cancellation request."""

import asyncio
import logging

from . import review_task_store
from .review_events import review_events

logger = logging.getLogger(__name__)


class ReviewTaskCancelled(Exception):
    """The user cancelled this review; it is not a model failure."""


async def run_cancellable_review(task_id, operation, *, poll_interval=2.0):
    if not task_id or not review_task_store.is_enabled():
        return await operation()

    async with review_events.subscribe() as events:
        # Subscribe before checking storage: cancellation may precede execution.
        if await review_task_store.cancellation_requested(task_id):
            raise ReviewTaskCancelled("审核任务已取消")

        async def watch():
            loop = asyncio.get_running_loop()
            deadline = loop.time() + poll_interval
            while True:
                try:
                    event = await asyncio.wait_for(
                        events.get(), timeout=max(0, deadline - loop.time())
                    )
                    if event.get("type") != "resync" and not (
                        event.get("reason") == "cancelling" and task_id in event.get("task_ids", [])
                    ):
                        continue
                except asyncio.TimeoutError:
                    pass
                deadline = loop.time() + poll_interval
                try:
                    if await review_task_store.cancellation_requested(task_id):
                        return
                except review_task_store.ReviewTaskStoreUnavailable:
                    # The persisted request remains retryable if IPC/DB is unavailable.
                    logger.warning("Cancellation check unavailable: task=%s", task_id)

        review = asyncio.create_task(operation())
        cancelled = asyncio.create_task(watch())
        try:
            await asyncio.wait((review, cancelled), return_when=asyncio.FIRST_COMPLETED)
            if cancelled.done():
                await cancelled
                raise ReviewTaskCancelled("审核任务已取消")
            return await review
        finally:
            # Wait for model calls, decoder and download cleanup before reporting stopped.
            for task in (review, cancelled):
                if not task.done():
                    task.cancel()
            await asyncio.gather(review, cancelled, return_exceptions=True)
