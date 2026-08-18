"""Configuration management for face recognition service."""

from pydantic_settings import BaseSettings, SettingsConfigDict

# InsightFace Server is a single-model service (currently buffalo_m v0.7,
# 512-dim ArcFace R50). The legacy per-model dim lookup is kept only as a
# reference; the engine now always uses 512.
INSIGHTFACE_EMBEDDING_DIM = 512


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(env_prefix="WCM_", env_file=".env", extra="ignore")

    # ---- InsightFace Server ----
    insightface_base_url: str = "http://10.252.25.251:18097"
    insightface_model_name: str = "buffalo_m"
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
    #
    # NOTE: default is 0.0 because the upstream IFS Server 0.2.0 returns
    # L2-normalized embeddings (norm=1.0 always), so there is no norm
    # signal to extract. Opt in only when self-hosting an IFS variant
    # that exposes raw (un-normalized) embeddings. The adapter also
    # auto-disables this path if probe_norm comes back ≈ 1.0.
    insightface_norm_reference: float = 0.0

    # Map a WCM Person category (Chinese strings) to the per-category
    # InsightFace collection. Each register writes twice: once into
    # `insightface_collection_id` (aggregated) and once into the mapped
    # collection. Categories not in the map only land in the aggregate.
    insightface_category_collections: dict[str, str] = {
        "劣迹艺人": "bad-artists",
        "时政敏感": "political",
        "落马官员": "corrupt-officials",
    }

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
    def embedding_dim(self) -> int:
        """InsightFace Server ships a single model (buffalo_m)."""
        return INSIGHTFACE_EMBEDDING_DIM


settings = Settings()
