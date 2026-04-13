"""FastAPI application for face recognition service."""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from wcm_facerec import __version__
from wcm_facerec.config import settings
from wcm_facerec.database import init_db

from .routes import router

app = FastAPI(
    title="WCM Face Recognition API",
    description="Large Scale Face Recognition with DeepFace + pgvector",
    version=__version__,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api/v1", tags=["face-recognition"])


@app.on_event("startup")
async def startup_event():
    """Initialize database on startup."""
    try:
        init_db()
    except Exception as e:
        print(f"Warning: Database initialization failed: {e}")


def main():
    """Run the application."""
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )


if __name__ == "__main__":
    main()
