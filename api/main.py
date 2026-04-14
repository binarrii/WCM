"""FastAPI application for face recognition service."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from wcm_facerec import __version__
from wcm_facerec.config import settings
from wcm_facerec.database import init_db

from .routes import api_bp


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    try:
        init_db()
    except Exception as e:
        print(f"Warning: Database initialization failed: {e}")
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
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # Register blueprints
    app.include_router(api_bp, prefix="/api/v1")

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
