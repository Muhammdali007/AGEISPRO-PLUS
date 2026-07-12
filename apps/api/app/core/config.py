from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[4]


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
        default="buffalo_l",
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
    storage_root: Path = PROJECT_ROOT / "storage"
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
    detection_duplicate_window_seconds: int = Field(
        default=15,
        ge=1,
        le=3600,
        validation_alias=AliasChoices("API_DETECTION_DUPLICATE_WINDOW_SECONDS"),
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
        if self.bootstrap_admin_password == "ChangeMe123!":
            raise ValueError("Production mode requires a non-default BOOTSTRAP_ADMIN_PASSWORD.")
        if self.recognition_backend != "insightface":
            raise ValueError("Production mode requires API_RECOGNITION_BACKEND=insightface.")
        if self.recognition_allow_fallback:
            raise ValueError("Production mode requires API_RECOGNITION_ALLOW_FALLBACK=false.")

        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
