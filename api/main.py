"""FastAPI application for face recognition and media review."""

import hashlib
import os
import socket
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from wcm_facerec import __version__, face_sync_store, image_store, person_operations
from wcm_facerec.cluster import cluster_slot, run_sync
from wcm_facerec.config import settings

from . import parameter_store, task_queue
from .face_records import face_records_bp
from .images import images_bp
from .model_clients import model_client_pool
from .parameters import parameters_bp
from .request_identity import person_request_fingerprint
from .review_events import review_events
from .review_task_store import initialize as initialize_review_tasks
from .review_tasks import review_tasks_bp
from .routes import api_bp


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize optional service-owned persistence before accepting traffic."""
    await initialize_review_tasks()
    await task_queue.initialize()
    if settings.cluster_enabled:
        await run_sync(person_operations.initialize)
        if settings.image_storage != "s3" or not settings.s3_endpoint:
            raise RuntimeError("Cluster mode requires configured S3 image storage")
        await run_sync(image_store.client().head_bucket, Bucket=settings.s3_bucket)
    await parameter_store.initialize()
    await run_sync(face_sync_store.initialize)
    try:
        await review_events.start()
        try:
            async with model_client_pool():
                yield
        finally:
            await review_events.close()
    finally:
        await parameter_store.close()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="WCM Face Recognition API",
        version=__version__,
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def person_write_context(request, call_next):
        key = request.headers.get("idempotency-key")
        if key and len(key) > 128:
            return JSONResponse({"detail": "幂等键过长"}, status_code=400)
        identity = None
        if key and settings.cluster_enabled:
            fingerprint = await person_request_fingerprint(request)
            identity = (hashlib.sha256(key.encode()).hexdigest(), fingerprint)
        token = person_operations.request_key.set(identity)
        revision = person_operations.expected_revision.set(
            request.headers.get("if-match", "").strip('"') or None
        )
        try:
            if (
                settings.insightface_replication_enabled
                and request.method == "GET"
                and request.url.path.startswith("/api/v1/face_records")
            ):
                async with cluster_slot("person-library"):
                    await run_sync(face_sync_store.assert_primary_clean)
                    response = await call_next(request)
            else:
                response = await call_next(request)
            if settings.cluster_enabled:
                response.headers["X-WCM-Instance"] = socket.gethostname()
            return response
        except face_sync_store.ReplicationUnavailable as exc:
            return JSONResponse({"detail": str(exc)}, status_code=503)
        finally:
            person_operations.request_key.reset(token)
            person_operations.expected_revision.reset(revision)

    # Enable CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    # Register blueprints
    app.include_router(api_bp, prefix="/api/v1")
    app.include_router(face_records_bp, prefix="/api/v1")
    app.include_router(review_tasks_bp, prefix="/api/v1")
    app.include_router(parameters_bp, prefix="/api/v1")

    @app.get("/api/v1/insightface/replication")
    async def replication_status():
        if not settings.insightface_replication_enabled:
            return {"enabled": False}
        return await run_sync(face_sync_store.status)

    # Mount persisted face images before the SPA catch-all.
    if settings.image_storage == "s3":
        app.include_router(images_bp)
    else:
        os.makedirs("/tmp/wcm", exist_ok=True)
        app.mount("/images", StaticFiles(directory="/tmp/wcm"), name="images")

    # The Docker image includes the built Vue dashboard at /www. Local API
    # development still works without that directory.
    if os.path.isdir("/www"):
        app.mount("/", StaticFiles(directory="/www", html=True), name="webui")

    return app


app = create_app()


def main():
    """Run the application."""
    import uvicorn

    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
    )


if __name__ == "__main__":
    main()
