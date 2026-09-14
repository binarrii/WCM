"""Run independently: python -m api.worker. SIGTERM drains active reviews."""

import asyncio
import contextlib
import logging
import os
import signal
import socket
import time
import uuid
from pathlib import Path

from wcm_facerec import runtime_parameters
from wcm_facerec.config import settings
from wcm_facerec.execution import execution_scope

from . import parameter_store, review_task_store, task_queue
from .model_clients import model_client_pool
from .review_events import review_events
from .routes import _run_review_task

logger = logging.getLogger(__name__)


async def execute(task):
    async def heartbeat():
        while True:
            await asyncio.sleep(settings.review_heartbeat_seconds)
            if not await task_queue.renew(task["id"], task["lease_token"]):
                raise RuntimeError("Execution lease lost")

    with (
        execution_scope(task["id"], task["lease_token"], task["attempts"]),
        runtime_parameters.frozen(task["runtime_parameters"]),
    ):
        parameters = task["parameters"]
        work = asyncio.create_task(
            _run_review_task(
                task["id"],
                task["video_url"],
                parameters["sample_interval"],
                parameters["top_k"],
                parameters["threshold"],
            )
        )
        lease = asyncio.create_task(heartbeat())
        try:
            ready, _ = await asyncio.wait((work, lease), return_when=asyncio.FIRST_COMPLETED)
            if work in ready:
                await work
            else:
                await lease  # A DB failure stops the old execution before its lease expires.
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Review attempt ended: task=%s attempt=%s error=%s",
                task["id"],
                task["attempts"],
                type(exc).__name__,
            )
        finally:
            work.cancel()
            lease.cancel()
            await asyncio.gather(work, lease, return_exceptions=True)


async def serve():
    if not settings.cluster_enabled:
        raise RuntimeError("WCM_CLUSTER_ENABLED=true is required for the standalone worker")
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    await review_task_store.initialize()
    await task_queue.initialize()
    await parameter_store.initialize()
    await review_events.start()
    worker_id = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
    tasks = set()
    try:
        async with model_client_pool():
            while not stop.is_set():
                Path("/tmp/wcm-worker-health").write_text(str(time.time()))
                tasks = {task for task in tasks if not task.done()}
                if len(tasks) < settings.worker_concurrency:
                    try:
                        task = await task_queue.claim(worker_id)
                    except Exception as exc:
                        logger.warning("Queue temporarily unavailable: %s", type(exc).__name__)
                        task = None
                    if task:
                        tasks.add(asyncio.create_task(execute(task)))
                        continue
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(stop.wait(), timeout=0.5)
            logger.info("Worker draining %s review(s)", len(tasks))
            if tasks:
                _, pending = await asyncio.wait(tasks, timeout=settings.worker_shutdown_seconds)
                for task in pending:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await review_events.close()
        await parameter_store.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    asyncio.run(serve())
