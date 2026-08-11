"""Configuration management for face recognition service."""

import warnings
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# InsightFace Server is a single-model service (currently buffalo_m v0.7,
# 512-dim ArcFace R50). The legacy per-model dim lookup is kept only as a
# reference; the engine now always uses 512.
INSIGHTFACE_EMBEDDING_DIM = 512


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(env_prefix="WCM_", env_file=".env", extra="ignore")

    # Database
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "facerec"
    db_user: str = "postgres"
    db_password: str = "postgres"

    # ---- InsightFace Server (replaces DeepFace) ----
    insightface_base_url: str = "http://10.252.25.251:18097"
    insightface_collection_id: str = "all-persons"
    insightface_timeout_s: float = 60.0
    # InsightFace's /compare returns similarity in [0,1] (higher = better).
    # The legacy verify_distance_threshold was on a cosine-distance scale
    # (lower = better). For buffalo_m + ArcFace R50, similarity ≥ 0.55 is a
    # reasonable starting point; tune against your data.
    insightface_verify_similarity_threshold: float = 0.55
    # Optional bearer token. Empty when auth is disabled on the server.
    insightface_api_key: str = ""

    # ---- Search-quality enhancements (all default to 0.0 = off) ----
    # #3 quality-aware fusion: weight each match's similarity by the query
    # face's detection_score. factor = (1 - w) + w * q, where q is in
    # [0, 1]. w=0 leaves matches unchanged; w=0.3 means a low-quality probe
    # (q=0.5) penalizes a 0.9-similarity match down to ~0.75.
    insightface_quality_weight: float = 0.0
    # #1 adaptive per-Person threshold: for Persons with more enrolled
    # faces (face_count), raise the acceptance bar by `step` per face,
    # capped at 10. adaptive = base + step * min(face_count, 10).
    # step=0 disables; step=0.005 means a 10-face Person requires ~0.605
    # similarity to match, vs. 0.55 for a 1-face Person.
    insightface_adaptive_threshold_step: float = 0.0
    # #2 norm-aware scoring (MPS proxy): the probe embedding's L2 norm is
    # a quality signal (MagFace). factor = min(norm / ref, 1.0).
    # ref=30 is typical for ArcFace R50; tune against your embeddings.
    # Set to 0 to disable the /embeddings round-trip entirely.
    insightface_norm_reference: float = 30.0

    # Map a WCM Person category (Chinese strings) to the per-category
    # InsightFace collection. Each register writes twice: once into
    # `insightface_collection_id` (aggregated) and once into the mapped
    # collection. Categories not in the map only land in the aggregate.
    insightface_category_collections: dict[str, str] = {
        "劣迹艺人": "bad-artists",
        "时政敏感": "political",
        "落马官员": "corrupt-officials",
    }

    # ---- Deprecated DeepFace settings (no-op; kept for one release) ----
    # These are NOT read by the new FaceEngine. They exist only so that an
    # operator who reverts the branch sees the old config still works.
    deepface_model: Literal[
        "VGG-Face",
        "Facenet",
        "Facenet512",
        "OpenFace",
        "DeepFace",
        "DeepID",
        "ArcFace",
        "Dlib",
        "SFace",
    ] = "Facenet512"  # noqa: F841
    deepface_distance_metric: Literal["cosine", "euclidean", "euclidean_l2"] = "cosine"  # noqa: F841
    deepface_api_url: str = "http://127.0.0.1:5000"  # noqa: F841
    # Cosine-distance threshold kept for callers that still consult it.
    # Emits a DeprecationWarning on first read; plan to remove next release.
    verify_distance_threshold: float = 0.3  # noqa: F841

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    max_file_size_mb: int = 100
    model_api_url: str = "https://models.ai.wtvdev.com/v1/chat/completions"
    model_api_key: str = "sk-o8EGlzXqMQi8Ba06E2B1BcF8217c45B6Bb70Ce5765B70c42"

    # Filesystem
    data_root: str = "/data/wcm"
    default_category: str = "未分类"

    @property
    def database_url(self) -> str:
        """Get database connection URL."""
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    @property
    def embedding_dim(self) -> int:
        """InsightFace Server ships a single model (buffalo_m)."""
        return INSIGHTFACE_EMBEDDING_DIM


settings = Settings()


def warn_deprecated() -> None:
    """Emit a one-shot warning when DeepFace-style settings are still on disk.

    Called from FaceEngine.__init__ so we surface the deprecation in the
    process log without forcing every operator to grep their .env files.
    """
    if not hasattr(warn_deprecated, "_emitted"):
        if settings.deepface_api_url != "http://127.0.0.1:5000":
            warnings.warn(
                "WCM_DEEPFACE_API_URL is set but no longer used — FaceEngine now "
                "talks to InsightFace Server. Set WCM_INSIGHTFACE_BASE_URL instead.",
                DeprecationWarning,
                stacklevel=3,
            )
        warn_deprecated._emitted = True  # type: ignore[attr-defined]
