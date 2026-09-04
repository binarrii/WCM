"""FastAPI application for face recognition and media review."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from wcm_facerec import __version__
from wcm_facerec.config import settings

from .face_records import face_records_bp
from .review_task_store import initialize as initialize_review_tasks
from .review_tasks import review_tasks_bp
from .routes import api_bp


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize optional service-owned persistence before accepting traffic."""
    await initialize_review_tasks()
    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="WCM Face Recognition API",
        version=__version__,
        lifespan=lifespan,
    )

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

    # Mount persisted face images before the SPA catch-all.
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
