"""Worker/event-loop owned model connection pools, closed with the API lifespan."""

import asyncio
from contextlib import asynccontextmanager

import httpx

from wcm_facerec.cluster import model_slot
from wcm_facerec.model_budget import remaining_request_time

_pools = {}


async def _request_budget(request):
    remaining = remaining_request_time()
    if remaining is not None:
        request.extensions["timeout"] = {
            key: min(value, remaining) if value is not None else remaining
            for key, value in request.extensions["timeout"].items()
        }


@asynccontextmanager
async def model_client_pool():
    loop = asyncio.get_running_loop()
    if loop in _pools:
        raise RuntimeError("Model client pool already started for this event loop")
    clients = _pools[loop] = {}
    try:
        yield
    finally:
        _pools.pop(loop, None)
        await asyncio.gather(*(client.aclose() for client in clients.values()))


@asynccontextmanager
async def model_client(model, timeout):
    async with model_slot(model):
        async with _model_client(model, timeout) as client:
            yield client


@asynccontextmanager
async def _model_client(model, timeout):
    clients = _pools.get(asyncio.get_running_loop())
    if clients is None:
        # CLI calls and tests without an application lifespan still own/close
        # their connections; no client is shared across separate asyncio.run calls.
        async with httpx.AsyncClient(
            timeout=timeout, event_hooks={"request": [_request_budget]}
        ) as client:
            yield client
        return
    if model not in clients:
        clients[model] = httpx.AsyncClient(
            timeout=timeout,
            event_hooks={"request": [_request_budget]},
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=32),
        )
    yield clients[model]
