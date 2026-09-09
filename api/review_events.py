"""Local event fan-out across API workers; no broker or database polling."""

import asyncio
import contextlib
import json
import logging
import os
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)


class ReviewEventBus:
    def __init__(self, directory="/tmp/wcm-review-events"):
        self.directory = Path(directory)
        self.path = None
        self.server = None
        self.listeners = set()

    async def start(self):
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.path = self.directory / f"worker-{os.getpid()}-{uuid.uuid4().hex[:12]}.sock"
        self.server = await asyncio.start_unix_server(
            self._receive, path=str(self.path), limit=262144
        )

    async def close(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.server = None
        if self.path:
            self.path.unlink(missing_ok=True)
            self.path = None
        self._deliver({"type": "resync"})

    @contextlib.asynccontextmanager
    async def subscribe(self):
        queue = asyncio.Queue(maxsize=128)
        self.listeners.add(queue)
        try:
            yield queue
        finally:
            self.listeners.discard(queue)

    def _deliver(self, event):
        for queue in self.listeners:
            if queue.full():
                # A slow browser must reconnect and recover a fresh database snapshot.
                while not queue.empty():
                    queue.get_nowait()
                queue.put_nowait({"type": "resync"})
            else:
                queue.put_nowait(event)

    async def _receive(self, reader, writer):
        try:
            raw = await asyncio.wait_for(reader.readline(), timeout=1)
            event = json.loads(raw)
            if isinstance(event, dict) and event.get("type") in {"progress", "changed"}:
                self._deliver(event)
                writer.write(b"ok\n")
                await writer.drain()
        except (ValueError, OSError, asyncio.TimeoutError):
            pass
        finally:
            writer.close()
            with contextlib.suppress(OSError):
                await writer.wait_closed()

    async def _send(self, path, raw):
        writer = None
        try:
            reader, writer = await asyncio.open_unix_connection(str(path))
            writer.write(raw)
            await writer.drain()
            if await reader.readline() != b"ok\n":
                raise OSError("Event acknowledgement missing")
        except (FileNotFoundError, ConnectionRefusedError):
            # Unique per-worker paths cannot belong to a replacement worker.
            path.unlink(missing_ok=True)
        finally:
            if writer:
                writer.close()
                with contextlib.suppress(OSError):
                    await writer.wait_closed()

    async def publish(self, event):
        self._deliver(event)
        if not self.server:
            return  # Single-process scripts/tests can use the local fan-out.
        try:
            raw = (json.dumps(event, ensure_ascii=False) + "\n").encode()
            peers = [path for path in self.directory.glob("worker-*.sock") if path != self.path]
            results = await asyncio.gather(
                *(asyncio.wait_for(self._send(path, raw), timeout=0.5) for path in peers),
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, Exception):
                    logger.warning("Review event delivery unavailable: %s", type(result).__name__)
        except Exception as exc:
            # Event delivery is independent of the persisted audit outcome.
            logger.warning("Review event publication unavailable: %s", type(exc).__name__)


review_events = ReviewEventBus()
