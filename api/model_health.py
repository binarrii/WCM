"""Bounded model retries and per-video fail-fast protection."""

import asyncio
import logging
from collections import deque
from contextvars import ContextVar
from functools import wraps

import httpx

from wcm_facerec.config import settings

from .utils import VIDEO_EXTENSIONS

logger = logging.getLogger(__name__)


class ModelServiceUnavailable(RuntimeError):
    """A video's model failure budget was exhausted."""


class ModelHealth:
    def __init__(self):
        self.outcomes = {}
        self.stopped = asyncio.Event()
        self.error = None

    def record(self, model, failed):
        if self.error is not None:
            raise self.error
        outcomes = self.outcomes.setdefault(model, deque(maxlen=10))
        outcomes.append(failed)
        failures = sum(outcomes)
        if failures >= 5:
            logger.warning(
                "Model failure protection tripped: model=%s completed_calls=%d failures=%d retry_limit=1",
                model,
                len(outcomes),
                failures,
            )
            self.error = ModelServiceUnavailable(
                f"{MODEL_LABELS[model]}模型调用频繁错误或超时，已提前终止该审核任务。"
            )
            self.stopped.set()
            raise self.error


_current_health = ContextVar("model_health", default=None)


MODEL_LABELS = {
    "visual": "visual 视觉",
    "ocr": "OCR 文字",
    "guard": "guard 安全判定",
    "face": "face 人脸",
}
TIMEOUT_SETTINGS = {
    "visual": "visual_timeout_s",
    "ocr": "ocr_timeout_s",
    "guard": "guard_timeout_s",
    "face": "insightface_timeout_s",
}


async def call_model(model, operation):
    """Retry once with a fresh deadline, then count one final outcome.

    Successful compatibility fallbacks count once. Cancellations and another
    model's open circuit never count as service failures for this model.
    """
    timeout = getattr(settings, TIMEOUT_SETTINGS[model])
    health = _current_health.get()
    for attempt in range(2):
        if health is not None and health.error is not None:
            raise health.error
        try:
            try:
                result = await asyncio.wait_for(operation(), timeout=timeout)
            except asyncio.TimeoutError as exc:
                raise httpx.ReadTimeout(f"{model} model exceeded {timeout:g}s deadline") from exc
        except ModelServiceUnavailable:
            raise
        except Exception as exc:
            if attempt == 0:
                logger.warning(
                    "Model call failed; retrying once: model=%s error=%s timeout_s=%g",
                    model,
                    type(exc).__name__,
                    timeout,
                )
                continue
            if health is not None:
                health.record(model, True)
            raise
        if health is not None:
            health.record(model, False)
        return result


def model_call(model):
    """Reusable decorator for an async operation that invokes one model."""

    def decorate(operation):
        @wraps(operation)
        async def protected(*args, **kwargs):
            return await call_model(model, lambda: operation(*args, **kwargs))

        return protected

    return decorate


async def gather_stages(*operations):
    """Cancel and await siblings when any stage aborts its window."""
    tasks = [asyncio.create_task(operation) for operation in operations]
    try:
        return await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def protect_video_review(operation):
    @wraps(operation)
    async def protected(url, *args, **kwargs):
        # Nested window review shares the parent video's budgets; images and
        # independent video tasks must not accumulate each other's failures.
        if _current_health.get() is not None or not any(
            url.lower().endswith(ext) for ext in VIDEO_EXTENSIONS
        ):
            return await operation(url, *args, **kwargs)
        health = ModelHealth()
        token = _current_health.set(health)
        review = asyncio.create_task(operation(url, *args, **kwargs))
        stopped = asyncio.create_task(health.stopped.wait())
        try:
            # A consumer can fail while its producer is blocked on a full queue
            # or queue.join(). Wake the owner in either case and run cleanup.
            await asyncio.wait((review, stopped), return_when=asyncio.FIRST_COMPLETED)
            if health.error is not None:
                raise health.error
            return await review
        finally:
            review.cancel()
            stopped.cancel()
            await asyncio.gather(review, stopped, return_exceptions=True)
            _current_health.reset(token)

    return protected
