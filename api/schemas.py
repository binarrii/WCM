"""Pydantic schemas for API requests and responses."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class FaceDetectionResult(BaseModel):
    """Face detection result."""

    face_id: str
    confidence: float
    facial_area: dict


class FaceSearchResult(BaseModel):
    """Face search result."""

    id: str
    name: str
    file_path: Optional[str] = None
    file_url: Optional[str] = None
    distance: float
    confidence: Optional[float] = None
    face_id: Optional[str] = None
    frame_time: Optional[float] = None
    created_at: Optional[str] = None


class RegisterFaceRequest(BaseModel):
    """Request to register a face."""

    name: str = Field(..., min_length=1, max_length=255)
    url: Optional[str] = Field(None, description="URL to image")


class RegisterFaceResponse(BaseModel):
    """Response after registering a face."""

    id: str
    name: str
    message: str


class SearchFaceRequest(BaseModel):
    """Request to search for faces."""

    url: Optional[str] = Field(None, description="URL to image")
    name: Optional[str] = Field(None, description="Filter by name")
    top_k: int = Field(10, ge=1, le=100)
    threshold: float = Field(0.4, ge=0.0, le=1.0)


class SearchFaceResponse(BaseModel):
    """Response with search results."""

    results: list[FaceSearchResult]
    query_embedding_dim: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model: str
    embedding_dim: int
    version: str


class DetectFaceResponse(BaseModel):
    """Response with detected faces."""

    faces: list[FaceDetectionResult]
    image_source: str
