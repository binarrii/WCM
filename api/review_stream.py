"""Push persisted review progress to task-list and review-page subscribers."""

import asyncio
import contextlib
import re

from fastapi import WebSocketDisconnect

from . import review_task_store
from .review_events import review_events

HEARTBEAT_SECONDS = 20


def subscription_ids(payload):
    ids = payload.get("task_ids", [])
    if (
        payload.get("type") != "subscribe"
        or not isinstance(ids, list)
        or len(ids) > 100
        or any(not isinstance(item, str) or not re.fullmatch(r"[\w-]{1,64}", item) for item in ids)
        or not isinstance(payload.get("watch_list", False), bool)
    ):
        raise ValueError("无效的审核进度订阅")
    return list(dict.fromkeys(ids))


async def send(websocket, payload):
    await asyncio.wait_for(websocket.send_json(payload), timeout=5)


async def stream_review_tasks(websocket, payload):
    pending = set()
    try:
        if not isinstance(payload, dict):
            raise ValueError("无效的审核进度订阅")
        ids = subscription_ids(payload)
        watch_list = payload.get("watch_list", False)
        async with review_events.subscribe() as queue:
            # Register first: updates during this read stay queued, without a snapshot gap.
            tasks = await review_task_store.get_summaries(ids)
            await send(
                websocket,
                {
                    "type": "snapshot",
                    "tasks": tasks,
                    "missing_ids": sorted(set(ids) - {task["id"] for task in tasks}),
                },
            )
            incoming = asyncio.create_task(websocket.receive_text())
            update = asyncio.create_task(queue.get())
            pending = {incoming, update}
            heartbeat_at = asyncio.get_running_loop().time() + HEARTBEAT_SECONDS
            while True:
                ready, _ = await asyncio.wait(
                    pending,
                    timeout=max(0, heartbeat_at - asyncio.get_running_loop().time()),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if asyncio.get_running_loop().time() >= heartbeat_at:
                    await send(websocket, {"type": "heartbeat"})
                    heartbeat_at = asyncio.get_running_loop().time() + HEARTBEAT_SECONDS
                if incoming in ready:
                    incoming.result()  # Detect a disconnect promptly, including while idle.
                    pending.discard(incoming)
                    incoming = asyncio.create_task(websocket.receive_text())
                    pending.add(incoming)
                if update in ready:
                    event = update.result()
                    if event["type"] == "resync":
                        await websocket.close(code=1013)
                        break
                    interested = (event["type"] == "progress" and event["task_id"] in ids) or (
                        event["type"] == "changed"
                        and (watch_list or set(ids).intersection(event["task_ids"]))
                    )
                    if interested:
                        await send(websocket, event)
                        heartbeat_at = asyncio.get_running_loop().time() + HEARTBEAT_SECONDS
                    pending.discard(update)
                    update = asyncio.create_task(queue.get())
                    pending.add(update)
    except (WebSocketDisconnect, RuntimeError, asyncio.CancelledError):
        pass
    except ValueError as exc:
        with contextlib.suppress(Exception):
            await send(websocket, {"type": "error", "error": str(exc)})
            await websocket.close(code=1008)
    except Exception:
        with contextlib.suppress(Exception):
            await websocket.close(code=1013)
    finally:
        for task in pending:
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.gather(*pending, return_exceptions=True)


@contextlib.asynccontextmanager
async def push_review_progress(websocket, task_id):
    """Share the submission socket without coupling audit lifetime to a browser."""
    async with review_events.subscribe() as queue:
        closing = False

        async def forward():
            heartbeat_at = asyncio.get_running_loop().time() + HEARTBEAT_SECONDS
            while not closing:
                try:
                    event = await asyncio.wait_for(
                        queue.get(),
                        timeout=max(0.001, heartbeat_at - asyncio.get_running_loop().time()),
                    )
                except asyncio.TimeoutError:
                    event = {"type": "heartbeat"}
                if event["type"] == "resync":
                    await websocket.close(code=1013)
                    return
                if event["type"] == "heartbeat" or (
                    event["type"] == "progress" and event["task_id"] == task_id
                ):
                    await send(websocket, event)
                    heartbeat_at = asyncio.get_running_loop().time() + HEARTBEAT_SECONDS

        sender = asyncio.create_task(forward())
        try:
            yield
        finally:
            closing = True
            # Wake the reader even if cancellation races with a completed wait_for on Python 3.10.
            if not queue.full():
                queue.put_nowait({"type": "stop"})
            sender.cancel()
            # Network errors must never cancel an audit or change its persisted outcome.
            await asyncio.gather(sender, return_exceptions=True)
