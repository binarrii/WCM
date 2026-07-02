"""Configuration management for face recognition service."""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


_EMBEDDING_DIMS = {
    "VGG-Face": 4096,
    "Facenet": 128,
    "Facenet512": 512,
    "OpenFace": 128,
    "DeepFace": 4096,
    "DeepID": 2048,
    "ArcFace": 512,
    "Dlib": 128,
    "SFace": 128,
}


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(env_prefix="WCM_", env_file=".env", extra="ignore")

    # Database
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "facerec"
    db_user: str = "postgres"
    db_password: str = "postgres"

    # DeepFace
    deepface_model: Literal["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"] = "Facenet512"
    deepface_distance_metric: Literal["cosine", "euclidean", "euclidean_l2"] = "cosine"
    # Max distance for the verify step to accept a candidate as the same person.
    # Tighter than DeepFace's built-in threshold (~0.30 for Facenet512+cosine) to
    # reject borderline look-alikes.
    verify_distance_threshold: float = 0.10

    # Engine Config
    face_engine_mode: Literal["thread_pool", "process_pool", "triton"] = "triton"
    triton_server_url: str = "http://127.0.0.1:8001"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    max_file_size_mb: int = 100
    model_api_url: str = "https://models.ai.wtvdev.com/v1/chat/completions"
    model_api_key: str = "sk-o8EGlzXqMQi8Ba06E2B1BcF8217c45B6Bb70Ce5765B70c42"
    nsfw_api_url: str = "http://127.0.0.1:3005/predict"

    # Filesystem
    data_root: str = "/data/wcm"
    default_category: str = "未分类"

    @property
    def database_url(self) -> str:
        """Get database connection URL."""
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension based on model."""
        return _EMBEDDING_DIMS.get(self.deepface_model, 512)


settings = Settings()
