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

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    max_file_size_mb: int = 100

    @property
    def database_url(self) -> str:
        """Get database connection URL."""
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension based on model."""
        return _EMBEDDING_DIMS.get(self.deepface_model, 4096)


settings = Settings()
