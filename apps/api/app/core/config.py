from functools import lru_cache
from pathlib import Path
import sys
from typing import Annotated

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

_CONFIG_FILE = Path(__file__).resolve()
PROJECT_ROOT = next(
    (
        parent
        for parent in _CONFIG_FILE.parents
        if (parent / "docker-compose.yml").exists() or (parent / "storage").exists()
    ),
    _CONFIG_FILE.parents[min(2, len(_CONFIG_FILE.parents) - 1)],
)


class Settings(BaseSettings):
    project_name: str = "AegisPro"
    environment: str = "development"
    database_url: str = "postgresql+psycopg://aegispro:change-me@localhost:5432/aegispro"
    redis_url: str = "redis://localhost:6379/0"
    ai_service_url: str = "http://127.0.0.1:8100"
    secret_key: str = Field(min_length=24, default="replace-with-a-long-random-secret")
    service_callback_token: str | None = Field(
        default=None,
        validation_alias=AliasChoices("SERVICE_CALLBACK_TOKEN", "API_SERVICE_CALLBACK_TOKEN"),
    )
    jwt_issuer: str = Field(default="aegispro-api", validation_alias=AliasChoices("JWT_ISSUER", "API_JWT_ISSUER"))
    jwt_audience: str = Field(default="aegispro-web", validation_alias=AliasChoices("JWT_AUDIENCE", "API_JWT_AUDIENCE"))
    auth_cookie_secure: bool = Field(
        default=False,
        validation_alias=AliasChoices("AUTH_COOKIE_SECURE", "API_AUTH_COOKIE_SECURE"),
    )
    auth_cookie_samesite: str = Field(
        default="lax",
        validation_alias=AliasChoices("AUTH_COOKIE_SAMESITE", "API_AUTH_COOKIE_SAMESITE"),
    )
    access_token_minutes: int = 30
    refresh_token_days: int = 7
    cors_origins: Annotated[list[str], NoDecode] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3010",
        "http://127.0.0.1:3010",
    ]
    auto_create_tables: bool = True
    bootstrap_admin_email: str = "admin@aegispro.local"
    bootstrap_admin_password: str = "ChangeMe123!"
    sendgrid_api_key: str | None = Field(default=None, validation_alias=AliasChoices("SENDGRID_API_KEY"))
    password_reset_from_email: str | None = Field(
        default=None,
        validation_alias=AliasChoices("PASSWORD_RESET_FROM_EMAIL", "API_PASSWORD_RESET_FROM_EMAIL"),
    )
    password_reset_from_name: str = Field(
        default="AegisPro Security",
        validation_alias=AliasChoices("PASSWORD_RESET_FROM_NAME", "API_PASSWORD_RESET_FROM_NAME"),
    )
    web_app_url: str = Field(
        default="http://localhost:3000",
        validation_alias=AliasChoices("WEB_APP_URL", "API_WEB_APP_URL"),
    )
    password_reset_token_minutes: int = Field(
        default=30,
        ge=5,
        le=1440,
        validation_alias=AliasChoices(
            "PASSWORD_RESET_TOKEN_MINUTES", "API_PASSWORD_RESET_TOKEN_MINUTES"
        ),
    )
    recognition_backend: str = Field(default="hash", validation_alias=AliasChoices("API_RECOGNITION_BACKEND"))
    recognition_embedding_model: str = Field(
        default="image-hash-v1",
        validation_alias=AliasChoices("API_RECOGNITION_EMBEDDING_MODEL"),
    )
    recognition_embedding_dimensions: int = Field(
        default=16,
        ge=4,
        le=512,
        validation_alias=AliasChoices("API_RECOGNITION_EMBEDDING_DIMENSIONS"),
    )
    recognition_allow_fallback: bool = Field(
        default=True,
        validation_alias=AliasChoices("API_RECOGNITION_ALLOW_FALLBACK"),
    )
    recognition_insightface_model: str = Field(
        default="buffalo_m",
        validation_alias=AliasChoices("API_RECOGNITION_INSIGHTFACE_MODEL"),
    )
    recognition_insightface_providers: Annotated[list[str], NoDecode] = Field(
        default=["CPUExecutionProvider"],
        validation_alias=AliasChoices("API_RECOGNITION_INSIGHTFACE_PROVIDERS"),
    )
    recognition_insightface_ctx_id: int = Field(
        default=-1,
        validation_alias=AliasChoices("API_RECOGNITION_INSIGHTFACE_CTX_ID"),
    )
    recognition_insightface_det_size: Annotated[tuple[int, int], NoDecode] = Field(
        default=(640, 640),
        validation_alias=AliasChoices("API_RECOGNITION_INSIGHTFACE_DET_SIZE"),
    )
    recognition_enrollment_min_det_score: float = Field(
        default=0.60,
        ge=0,
        le=1,
        validation_alias=AliasChoices("API_RECOGNITION_ENROLLMENT_MIN_DET_SCORE"),
    )
    recognition_enrollment_min_face_size: int = Field(
        default=48,
        ge=16,
        le=1024,
        validation_alias=AliasChoices("API_RECOGNITION_ENROLLMENT_MIN_FACE_SIZE"),
    )
    recognition_enrollment_require_single_face: bool = Field(
        default=True,
        validation_alias=AliasChoices("API_RECOGNITION_ENROLLMENT_REQUIRE_SINGLE_FACE"),
    )
    recognition_runtime_max_templates_per_person: int = Field(
        default=12,
        ge=1,
        le=100,
        validation_alias=AliasChoices("API_RECOGNITION_RUNTIME_MAX_TEMPLATES_PER_PERSON"),
    )
    recognition_runtime_template_min_det_score: float = Field(
        default=0.60,
        ge=0,
        le=1,
        validation_alias=AliasChoices("API_RECOGNITION_RUNTIME_TEMPLATE_MIN_DET_SCORE"),
    )
    recognition_runtime_template_duplicate_similarity: float = Field(
        default=0.9995,
        ge=0.9,
        le=1,
        validation_alias=AliasChoices("API_RECOGNITION_RUNTIME_TEMPLATE_DUPLICATE_SIMILARITY"),
    )
    storage_root: Path = PROJECT_ROOT / "storage"
    camera_secret_keys: Annotated[list[str], NoDecode] = Field(
        default_factory=list,
        validation_alias=AliasChoices("CAMERA_SECRET_KEYS", "API_CAMERA_SECRET_KEYS"),
    )
    camera_allowed_protocols: Annotated[list[str], NoDecode] = Field(
        default=["http", "https", "rtsp"],
        validation_alias=AliasChoices("CAMERA_ALLOWED_PROTOCOLS", "API_CAMERA_ALLOWED_PROTOCOLS"),
    )
    camera_allowed_ports: Annotated[list[int], NoDecode] = Field(
        default=[80, 443, 554, 8080, 8443],
        validation_alias=AliasChoices("CAMERA_ALLOWED_PORTS", "API_CAMERA_ALLOWED_PORTS"),
    )
    camera_allowed_networks: Annotated[list[str], NoDecode] = Field(
        default=["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "fc00::/7"],
        validation_alias=AliasChoices("CAMERA_ALLOWED_NETWORKS", "API_CAMERA_ALLOWED_NETWORKS"),
    )
    camera_allowed_hostnames: Annotated[list[str], NoDecode] = Field(
        default_factory=list,
        validation_alias=AliasChoices("CAMERA_ALLOWED_HOSTNAMES", "API_CAMERA_ALLOWED_HOSTNAMES"),
    )
    camera_blocked_networks: Annotated[list[str], NoDecode] = Field(
        default=[
            "0.0.0.0/8",
            "127.0.0.0/8",
            "169.254.0.0/16",
            "224.0.0.0/4",
            "240.0.0.0/4",
            "::/128",
            "::1/128",
            "fe80::/10",
            "ff00::/8",
        ],
        validation_alias=AliasChoices("CAMERA_BLOCKED_NETWORKS", "API_CAMERA_BLOCKED_NETWORKS"),
    )
    camera_max_redirects: int = Field(
        default=3,
        ge=0,
        le=10,
        validation_alias=AliasChoices("CAMERA_MAX_REDIRECTS", "API_CAMERA_MAX_REDIRECTS"),
    )
    camera_media_agent_python: str = Field(
        default=sys.executable,
        validation_alias=AliasChoices("CAMERA_MEDIA_AGENT_PYTHON", "API_CAMERA_MEDIA_AGENT_PYTHON"),
    )
    camera_media_agent_timeout_seconds: int = Field(
        default=15,
        ge=1,
        le=120,
        validation_alias=AliasChoices(
            "CAMERA_MEDIA_AGENT_TIMEOUT_SECONDS", "API_CAMERA_MEDIA_AGENT_TIMEOUT_SECONDS"
        ),
    )
    database_pool_size: int = Field(
        default=10,
        ge=1,
        le=100,
        validation_alias=AliasChoices("DATABASE_POOL_SIZE", "API_DATABASE_POOL_SIZE"),
    )
    database_max_overflow: int = Field(
        default=20,
        ge=0,
        le=100,
        validation_alias=AliasChoices("DATABASE_MAX_OVERFLOW", "API_DATABASE_MAX_OVERFLOW"),
    )
    database_pool_recycle_seconds: int = Field(
        default=1800,
        ge=30,
        le=86400,
        validation_alias=AliasChoices(
            "DATABASE_POOL_RECYCLE_SECONDS", "API_DATABASE_POOL_RECYCLE_SECONDS"
        ),
    )
    continuous_detection_enabled: bool = Field(
        default=False, validation_alias=AliasChoices("API_CONTINUOUS_DETECTION_ENABLED")
    )
    continuous_detection_batch_size: int = Field(
        default=4,
        ge=1,
        le=32,
        validation_alias=AliasChoices("API_CONTINUOUS_DETECTION_BATCH_SIZE"),
    )
    continuous_detection_max_pending_per_camera: int = Field(
        default=1,
        ge=1,
        le=8,
        validation_alias=AliasChoices("API_CONTINUOUS_DETECTION_MAX_PENDING_PER_CAMERA"),
    )
    continuous_detection_scheduler_interval_ms: int = Field(
        default=250,
        ge=50,
        le=5000,
        validation_alias=AliasChoices("API_CONTINUOUS_DETECTION_SCHEDULER_INTERVAL_MS"),
    )
    continuous_detection_hazard_interval_seconds: float = Field(
        default=0.5,
        ge=0.25,
        le=30.0,
        validation_alias=AliasChoices("API_CONTINUOUS_DETECTION_HAZARD_INTERVAL_SECONDS"),
    )
    continuous_detection_recognition_interval_seconds: float = Field(
        default=4.0,
        ge=0.5,
        le=30.0,
        validation_alias=AliasChoices("API_CONTINUOUS_DETECTION_RECOGNITION_INTERVAL_SECONDS"),
    )
    detection_duplicate_window_seconds: int = Field(
        default=15,
        ge=1,
        le=3600,
        validation_alias=AliasChoices("API_DETECTION_DUPLICATE_WINDOW_SECONDS"),
    )
    sound_alert_unknown_scan_threshold: int = Field(
        default=3,
        ge=2,
        le=10,
        validation_alias=AliasChoices("API_SOUND_ALERT_UNKNOWN_SCAN_THRESHOLD"),
    )
    sound_alert_unknown_cooldown_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=3600.0,
        validation_alias=AliasChoices("API_SOUND_ALERT_UNKNOWN_COOLDOWN_SECONDS"),
    )
    sound_alert_hazard_cooldown_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=3600.0,
        validation_alias=AliasChoices("API_SOUND_ALERT_HAZARD_COOLDOWN_SECONDS"),
    )
    manual_camera_scan_cooldown_seconds: int = Field(
        default=30,
        ge=1,
        le=3600,
        validation_alias=AliasChoices("API_MANUAL_CAMERA_SCAN_COOLDOWN_SECONDS"),
    )
    file_video_scan_step_seconds: float = Field(
        default=0.5,
        ge=0.05,
        le=30.0,
        validation_alias=AliasChoices("API_FILE_VIDEO_SCAN_STEP_SECONDS"),
    )
    camera_overlay_ttl_seconds: int = Field(
        default=5,
        ge=2,
        le=60,
        validation_alias=AliasChoices("API_CAMERA_OVERLAY_TTL_SECONDS"),
    )
    camera_overlay_person_grace_seconds: float = Field(
        default=2.5,
        ge=0,
        le=10,
        validation_alias=AliasChoices("API_CAMERA_OVERLAY_PERSON_GRACE_SECONDS"),
    )
    event_clip_before_seconds: int = Field(
        default=5,
        ge=1,
        le=60,
        validation_alias=AliasChoices("API_EVENT_CLIP_BEFORE_SECONDS"),
    )
    event_clip_after_seconds: int = Field(
        default=5,
        ge=1,
        le=60,
        validation_alias=AliasChoices("API_EVENT_CLIP_AFTER_SECONDS"),
    )
    event_clip_fps: int = Field(
        default=10,
        ge=1,
        le=30,
        validation_alias=AliasChoices("API_EVENT_CLIP_FPS"),
    )
    incident_retention_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        validation_alias=AliasChoices("API_INCIDENT_RETENTION_HOURS"),
    )
    incident_extended_retention_hours: int = Field(
        default=72,
        ge=24,
        le=720,
        validation_alias=AliasChoices("API_INCIDENT_EXTENDED_RETENTION_HOURS"),
    )
    incident_compliance_retention_hours: int = Field(
        default=168,
        ge=24,
        le=2160,
        validation_alias=AliasChoices("API_INCIDENT_COMPLIANCE_RETENTION_HOURS"),
    )
    incident_cleanup_interval_seconds: int = Field(
        default=60,
        ge=10,
        le=3600,
        validation_alias=AliasChoices("API_INCIDENT_CLEANUP_INTERVAL_SECONDS"),
    )
    video_rag_enabled: bool = Field(
        default=False, validation_alias=AliasChoices("VIDEO_RAG_ENABLED", "API_VIDEO_RAG_ENABLED")
    )
    video_rag_ollama_url: str = Field(
        default="http://127.0.0.1:11434",
        validation_alias=AliasChoices("VIDEO_RAG_OLLAMA_URL", "API_VIDEO_RAG_OLLAMA_URL"),
    )
    video_rag_vision_model: str = Field(
        default="gemma3:4b", validation_alias=AliasChoices("VIDEO_RAG_VISION_MODEL")
    )
    video_rag_embedding_model: str = Field(
        default="embeddinggemma", validation_alias=AliasChoices("VIDEO_RAG_EMBEDDING_MODEL")
    )
    video_rag_embedding_dimensions: int = Field(
        default=768, ge=1, le=2000, validation_alias=AliasChoices("VIDEO_RAG_EMBEDDING_DIMENSIONS")
    )
    video_rag_visual_indexing_enabled: bool = Field(
        default=True, validation_alias=AliasChoices("VIDEO_RAG_VISUAL_INDEXING_ENABLED")
    )
    video_rag_request_timeout_seconds: int = Field(
        default=120, ge=5, le=900, validation_alias=AliasChoices("VIDEO_RAG_REQUEST_TIMEOUT_SECONDS")
    )
    video_rag_query_timeout_seconds: int = Field(
        default=30, ge=5, le=180, validation_alias=AliasChoices("VIDEO_RAG_QUERY_TIMEOUT_SECONDS")
    )
    video_rag_worker_interval_seconds: int = Field(
        default=15, ge=2, le=3600, validation_alias=AliasChoices("VIDEO_RAG_WORKER_INTERVAL_SECONDS")
    )
    video_rag_lease_seconds: int = Field(
        default=600, ge=30, le=3600, validation_alias=AliasChoices("VIDEO_RAG_LEASE_SECONDS")
    )
    video_rag_max_attempts: int = Field(
        default=3, ge=1, le=10, validation_alias=AliasChoices("VIDEO_RAG_MAX_ATTEMPTS")
    )
    video_rag_max_frames: int = Field(
        default=6, ge=1, le=12, validation_alias=AliasChoices("VIDEO_RAG_MAX_FRAMES")
    )
    video_rag_evidence_read_timeout_seconds: int = Field(
        default=15,
        ge=3,
        le=120,
        validation_alias=AliasChoices("VIDEO_RAG_EVIDENCE_READ_TIMEOUT_SECONDS"),
    )
    video_rag_default_limit: int = Field(
        default=5, ge=1, le=20, validation_alias=AliasChoices("VIDEO_RAG_DEFAULT_LIMIT")
    )
    video_rag_min_relevance: float = Field(
        default=0.25, ge=0, le=1, validation_alias=AliasChoices("VIDEO_RAG_MIN_RELEVANCE")
    )

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def split_origins(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @field_validator("storage_root", mode="before")
    @classmethod
    def resolve_storage_root(cls, value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return (PROJECT_ROOT / path).resolve()

    @field_validator("camera_secret_keys", mode="before")
    @classmethod
    def split_camera_secret_keys(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator(
        "camera_allowed_protocols",
        "camera_allowed_networks",
        "camera_allowed_hostnames",
        "camera_blocked_networks",
        mode="before",
    )
    @classmethod
    def split_string_lists(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("camera_allowed_ports", mode="before")
    @classmethod
    def split_int_lists(cls, value: str | list[int]) -> list[int]:
        if isinstance(value, str):
            return [int(item.strip()) for item in value.split(",") if item.strip()]
        return value

    @field_validator("camera_allowed_protocols", mode="after")
    @classmethod
    def normalize_camera_protocols(cls, value: list[str]) -> list[str]:
        return [item.strip().lower() for item in value if item.strip()]

    @field_validator("recognition_backend", mode="before")
    @classmethod
    def normalize_backend_name(cls, value: str) -> str:
        return value.strip().lower().replace("-", "_")

    @field_validator("recognition_insightface_providers", mode="before")
    @classmethod
    def split_providers(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [provider.strip() for provider in value.split(",") if provider.strip()]
        return value

    @field_validator("recognition_insightface_det_size", mode="before")
    @classmethod
    def parse_det_size(cls, value: str | tuple[int, int]) -> tuple[int, int]:
        if isinstance(value, str):
            parts = [part.strip() for part in value.lower().split("x") if part.strip()]
            if len(parts) == 2:
                return int(parts[0]), int(parts[1])
        return value

    @model_validator(mode="after")
    def validate_production_runtime(self) -> "Settings":
        if self.environment.strip().lower() != "production":
            return self

        if "sqlite" in self.database_url.lower():
            raise ValueError("Production mode requires a PostgreSQL DATABASE_URL, not SQLite.")
        if self.secret_key == "replace-with-a-long-random-secret":
            raise ValueError("Production mode requires a non-default SECRET_KEY.")
        if not self.service_callback_token:
            raise ValueError("Production mode requires SERVICE_CALLBACK_TOKEN.")
        if not self.auth_cookie_secure:
            raise ValueError("Production mode requires AUTH_COOKIE_SECURE=true.")
        if self.bootstrap_admin_password == "ChangeMe123!":
            raise ValueError("Production mode requires a non-default BOOTSTRAP_ADMIN_PASSWORD.")
        if not self.sendgrid_api_key:
            raise ValueError("Production mode requires SENDGRID_API_KEY for password recovery.")
        if not self.password_reset_from_email:
            raise ValueError("Production mode requires PASSWORD_RESET_FROM_EMAIL.")
        if not self.web_app_url.strip():
            raise ValueError("Production mode requires WEB_APP_URL.")
        if self.recognition_backend != "insightface":
            raise ValueError("Production mode requires API_RECOGNITION_BACKEND=insightface.")
        if self.recognition_allow_fallback:
            raise ValueError("Production mode requires API_RECOGNITION_ALLOW_FALLBACK=false.")

        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
