"""Bootstrap settings and database-managed business parameter definitions."""

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from . import runtime_parameters

# InsightFace Server is a single-model service (currently buffalo_m v0.7,
# 512-dim ArcFace R50). The legacy per-model dim lookup is kept only as a
# reference; the engine now always uses 512.
INSIGHTFACE_EMBEDDING_DIM = 512

# Keep omitted matching thresholds consistent across HTTP, WebSocket,
# the engine and the adapter. Distance is the complement of similarity.
DEFAULT_SIMILARITY_THRESHOLD = 0.5
DEFAULT_DISTANCE_THRESHOLD = 1.0 - DEFAULT_SIMILARITY_THRESHOLD


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(env_prefix="WCM_", env_file=".env", extra="ignore")

    # ---- InsightFace Server ----
    insightface_base_url: str = "http://10.252.25.251:18097"
    insightface_model_name: str = "buffalo_m"
    insightface_collection_id: str = "all-persons"
    insightface_timeout_s: float = Field(default=10.0, gt=0)
    # InsightFace's /compare returns similarity in [0,1] (higher = better).
    # The legacy verify_distance_threshold was on a cosine-distance scale
    # (lower = better). Both now default to 0.5; explicit configuration
    # can still override the verification threshold.
    insightface_verify_similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
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
    # step=0 disables; at the default base of 0.5, step=0.005 means a
    # 10-face Person requires 0.55 similarity, vs. 0.505 for 1 face.
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

    # ---- Profile / low-quality video face optimization ----
    # The rollout is guarded so the complete decision path can be disabled
    # without changing the public request contract.
    face_profile_optimization: bool = True
    # Expand detector boxes before the crop is sent back through ArcFace. A
    # square padded crop retains forehead/chin/nose context on profile faces.
    face_crop_padding: float = Field(default=0.25, ge=0.0, le=0.5)
    # Video review may retrieve weaker candidates internally, but only the
    # caller's confirmation threshold is allowed to produce a final finding.
    face_candidate_similarity: float = Field(default=0.35, ge=0.0, le=1.0)
    face_high_similarity: float = Field(default=0.65, ge=0.0, le=1.0)
    face_min_candidate_margin: float = Field(default=0.05, ge=0.0, le=1.0)
    face_min_confirming_frames: int = Field(default=2, ge=1, le=10)
    # IFS exposes a 0..1 pose quality signal (1 is frontal), rather than raw
    # yaw/pitch/roll. Values below this floor take the difficult-face path.
    face_profile_pose_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    face_low_sharpness_threshold: float = Field(default=0.15, ge=0.0, le=1.0)
    face_track_max_gap_s: float = Field(default=2.5, gt=0.0, le=30.0)
    # Targeted neighbour review stays bounded on the CPU-only IFS deployment.
    face_neighbor_offsets_s: tuple[float, ...] = (-0.4, -0.2, 0.2, 0.4)
    face_max_extra_frames_per_window: int = Field(default=3, ge=0, le=12)
    face_max_extra_call_ratio: float = Field(default=0.30, ge=0.0, le=2.0)
    face_neighbor_concurrency: int = Field(default=2, ge=1, le=8)
    # P4 gallery audit target; the IFS collection itself permits up to 20.
    face_gallery_target_samples: int = Field(default=5, ge=1, le=20)

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
    review_task_concurrency: int = Field(default=4, ge=1)
    review_window_concurrency: int = Field(default=4, ge=1)
    jpeg_quality: int = Field(default=95, ge=1, le=100)
    visual_timeout_s: float = Field(default=50.0, gt=0)
    ocr_timeout_s: float = Field(default=10.0, gt=0)
    guard_timeout_s: float = Field(default=10.0, gt=0)
    model_api_url: str = "https://models.ai.wtvdev.com/v1/chat/completions"
    model_api_key: str = ""

    # Multi-image Qwen input, with contact-sheet compatibility fallback.
    nsfw_image_mode: Literal["auto", "montage"] = "auto"
    nsfw_verify_target: bool = False
    nsfw_sampling_mode: Literal["fixed", "scene"] = "scene"
    nsfw_scene_max_stride: int = Field(default=3, ge=1, le=10)
    nsfw_scene_cut_threshold: float = Field(default=27.0, gt=0, le=255)
    nsfw_review_mode: Literal["target", "window"] = "window"
    nsfw_window_max_seconds: float = Field(default=10.0, gt=0, le=60)

    # ---- Review task storage ----
    # Disabled by default so local API-only development remains lightweight.
    # Docker Compose enables it and points the API at the bundled MySQL service.
    review_tasks_db_enabled: bool = False
    review_tasks_db_host: str = "127.0.0.1"
    review_tasks_db_port: int = 3306
    review_tasks_db_name: str = "wcm"
    review_tasks_db_user: str = "wcm"
    review_tasks_db_password: str = "wcm"
    review_tasks_db_connect_timeout_s: int = 5

    # Filesystem
    data_root: str = "/data/wcm"
    default_category: str = "未分类"

    @property
    def embedding_dim(self) -> int:
        """InsightFace Server ships a single model (buffalo_m)."""
        return INSIGHTFACE_EMBEDDING_DIM


@dataclass(frozen=True)
class BusinessParameterSpec:
    value_type: Literal["string", "number", "boolean", "enum", "json"]
    group: str
    secret: bool = False
    enum_values: tuple[str | int | float, ...] | None = None


# Network addresses, ports, database bootstrap credentials and filesystem paths
# intentionally stay in Settings/environment variables: the database-backed
# snapshot cannot be loaded until those values are already known.
BUSINESS_PARAMETER_SPECS = {
    # InsightFace library behavior and credentials.
    "insightface_base_url": BusinessParameterSpec("string", "人脸服务"),
    "insightface_model_name": BusinessParameterSpec("string", "人脸服务"),
    "insightface_collection_id": BusinessParameterSpec("string", "人脸服务"),
    "insightface_api_key": BusinessParameterSpec("string", "人脸服务", secret=True),
    "insightface_timeout_s": BusinessParameterSpec("number", "人脸服务"),
    "insightface_verify_similarity_threshold": BusinessParameterSpec("number", "人脸服务"),
    "insightface_quality_weight": BusinessParameterSpec("number", "人脸服务"),
    "insightface_adaptive_threshold_step": BusinessParameterSpec("number", "人脸服务"),
    "insightface_norm_reference": BusinessParameterSpec("number", "人脸服务"),
    "insightface_category_collections": BusinessParameterSpec("json", "人物库"),
    "default_category": BusinessParameterSpec("string", "人物库"),
    # Profile and low-quality video face optimization.
    "face_profile_optimization": BusinessParameterSpec("boolean", "人脸优化"),
    "face_crop_padding": BusinessParameterSpec("number", "人脸优化"),
    "face_candidate_similarity": BusinessParameterSpec("number", "人脸优化"),
    "face_high_similarity": BusinessParameterSpec("number", "人脸优化"),
    "face_min_candidate_margin": BusinessParameterSpec("number", "人脸优化"),
    "face_min_confirming_frames": BusinessParameterSpec("number", "人脸优化"),
    "face_profile_pose_threshold": BusinessParameterSpec("number", "人脸优化"),
    "face_low_sharpness_threshold": BusinessParameterSpec("number", "人脸优化"),
    "face_track_max_gap_s": BusinessParameterSpec("number", "人脸优化"),
    "face_neighbor_offsets_s": BusinessParameterSpec("json", "人脸优化"),
    "face_max_extra_frames_per_window": BusinessParameterSpec("number", "人脸优化"),
    "face_max_extra_call_ratio": BusinessParameterSpec("number", "人脸优化"),
    "face_neighbor_concurrency": BusinessParameterSpec("number", "人脸优化"),
    "face_gallery_target_samples": BusinessParameterSpec("number", "人脸优化"),
    # API limits and review scheduling.
    "max_file_size_mb": BusinessParameterSpec("number", "审核调度"),
    "review_task_concurrency": BusinessParameterSpec("number", "审核调度"),
    "review_window_concurrency": BusinessParameterSpec("number", "审核调度"),
    "jpeg_quality": BusinessParameterSpec("number", "审核调度"),
    "visual_timeout_s": BusinessParameterSpec("number", "审核调度"),
    "ocr_timeout_s": BusinessParameterSpec("number", "审核调度"),
    "guard_timeout_s": BusinessParameterSpec("number", "审核调度"),
    # Model gateway and behavior.
    "model_api_url": BusinessParameterSpec("string", "模型服务"),
    "model_api_key": BusinessParameterSpec("string", "模型服务", secret=True),
    "nsfw_image_mode": BusinessParameterSpec("enum", "内容审核", enum_values=("auto", "montage")),
    "nsfw_verify_target": BusinessParameterSpec("boolean", "内容审核"),
    "nsfw_sampling_mode": BusinessParameterSpec("enum", "内容审核", enum_values=("fixed", "scene")),
    "nsfw_scene_max_stride": BusinessParameterSpec("number", "内容审核"),
    "nsfw_scene_cut_threshold": BusinessParameterSpec("number", "内容审核"),
    "nsfw_review_mode": BusinessParameterSpec("enum", "内容审核", enum_values=("target", "window")),
    "nsfw_window_max_seconds": BusinessParameterSpec("number", "内容审核"),
}


def normalize_business_parameter(key: str, value: Any) -> Any:
    """Apply the same enum/range/coercion validation as bootstrap settings."""
    if key not in BUSINESS_PARAMETER_SPECS:
        return value
    validated = Settings(_env_file=None, **{key: value})
    return object.__getattribute__(validated, key)


class RuntimeSettings:
    """Compatibility proxy that overlays live business values on Settings."""

    def __init__(self, bootstrap: Settings):
        object.__setattr__(self, "_bootstrap", bootstrap)

    def __getattr__(self, name: str) -> Any:
        bootstrap = object.__getattribute__(self, "_bootstrap")
        fallback = getattr(bootstrap, name)
        if name in BUSINESS_PARAMETER_SPECS:
            return runtime_parameters.get(name, fallback)
        return fallback

    def __setattr__(self, name: str, value: Any) -> None:
        # Keep existing tests and operational scripts that monkeypatch settings
        # compatible while the production read path remains database-backed.
        setattr(object.__getattribute__(self, "_bootstrap"), name, value)

    def seed_value(self, name: str) -> Any:
        """Return the pre-database value used only for first-run seeding."""
        return getattr(object.__getattribute__(self, "_bootstrap"), name)


settings = RuntimeSettings(Settings())
