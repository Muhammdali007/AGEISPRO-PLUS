from functools import lru_cache
from typing import Annotated

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict
from pydantic import field_validator


class Settings(BaseSettings):
    project_name: str = Field(
        default="AegisPro AI Service",
        validation_alias=AliasChoices("AI_PROJECT_NAME", "PROJECT_NAME"),
    )
    environment: str = Field(default="development", validation_alias=AliasChoices("AI_ENVIRONMENT", "ENVIRONMENT"))
    model_backend: str = Field(default="simulated", validation_alias=AliasChoices("AI_MODEL_BACKEND"))
    model_fallback_backend: str = Field(default="simulated", validation_alias=AliasChoices("AI_MODEL_FALLBACK_BACKEND"))
    allow_backend_fallback: bool = Field(default=True, validation_alias=AliasChoices("AI_ALLOW_BACKEND_FALLBACK"))
    model_name: str = Field(default="yolo11", validation_alias=AliasChoices("AI_MODEL_NAME"))
    model_version: str = Field(default="phase5-sim", validation_alias=AliasChoices("AI_MODEL_VERSION"))
    model_weights_path: str = Field(default="yolo11n.pt", validation_alias=AliasChoices("AI_MODEL_WEIGHTS_PATH"))
    model_device: str | None = Field(default=None, validation_alias=AliasChoices("AI_MODEL_DEVICE"))
    model_image_size: int = Field(default=640, ge=64, validation_alias=AliasChoices("AI_MODEL_IMAGE_SIZE"))
    model_half_precision: bool = Field(default=False, validation_alias=AliasChoices("AI_MODEL_HALF_PRECISION"))
    model_tracker_config: str = Field(
        default="bytetrack.yaml",
        validation_alias=AliasChoices("AI_MODEL_TRACKER_CONFIG"),
    )
    model_track_persist: bool = Field(default=True, validation_alias=AliasChoices("AI_MODEL_TRACK_PERSIST"))
    model_label_aliases: Annotated[dict[str, list[str]], NoDecode] = Field(
        default_factory=lambda: {
            "weapon": ["weapon", "gun", "knife", "pistol", "rifle", "firearm", "handgun"],
            "fire": ["fire", "flame"],
            "smoke": ["smoke"],
            "person": ["person"],
        },
        validation_alias=AliasChoices("AI_MODEL_LABEL_ALIASES"),
    )
    recognition_backend: str = Field(default="hash", validation_alias=AliasChoices("AI_RECOGNITION_BACKEND"))
    recognition_embedding_model: str = Field(
        default="image-hash-v1",
        validation_alias=AliasChoices("AI_RECOGNITION_EMBEDDING_MODEL"),
    )
    inference_fps: float = Field(default=5.0, validation_alias=AliasChoices("AI_INFERENCE_FPS"))
    confidence_threshold: float = Field(default=0.55, ge=0, le=1, validation_alias=AliasChoices("AI_CONFIDENCE_THRESHOLD"))
    recognition_match_threshold: float = Field(
        default=0.82,
        ge=0,
        le=1,
        validation_alias=AliasChoices("AI_RECOGNITION_MATCH_THRESHOLD"),
    )
    recognition_embedding_dimensions: int = Field(
        default=16,
        ge=4,
        le=512,
        validation_alias=AliasChoices("AI_RECOGNITION_EMBEDDING_DIMENSIONS"),
    )
    recognition_allow_fallback: bool = Field(
        default=True,
        validation_alias=AliasChoices("AI_RECOGNITION_ALLOW_FALLBACK"),
    )
    recognition_insightface_model: str = Field(
        default="buffalo_l",
        validation_alias=AliasChoices("AI_RECOGNITION_INSIGHTFACE_MODEL"),
    )
    recognition_insightface_providers: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["CPUExecutionProvider"],
        validation_alias=AliasChoices("AI_RECOGNITION_INSIGHTFACE_PROVIDERS"),
    )
    recognition_insightface_ctx_id: int = Field(
        default=-1,
        validation_alias=AliasChoices("AI_RECOGNITION_INSIGHTFACE_CTX_ID"),
    )
    recognition_insightface_det_size: Annotated[tuple[int, int], NoDecode] = Field(
        default=(640, 640),
        validation_alias=AliasChoices("AI_RECOGNITION_INSIGHTFACE_DET_SIZE"),
    )
    api_event_callback_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("AI_API_EVENT_CALLBACK_URL"),
    )
    api_event_callback_token: str | None = Field(
        default=None,
        validation_alias=AliasChoices("AI_API_EVENT_CALLBACK_TOKEN"),
    )
    enable_event_callback: bool = Field(default=False, validation_alias=AliasChoices("AI_ENABLE_EVENT_CALLBACK"))

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @field_validator("model_backend", "model_fallback_backend", "recognition_backend", mode="before")
    @classmethod
    def normalize_backend_name(cls, value: str) -> str:
        return value.strip().lower().replace("-", "_")

    @field_validator("model_label_aliases", mode="before")
    @classmethod
    def normalize_label_aliases(cls, value: dict[str, list[str]]) -> dict[str, list[str]]:
        return {
            detector.strip().lower(): [alias.strip().lower() for alias in aliases if alias.strip()]
            for detector, aliases in value.items()
        }

    @field_validator("recognition_insightface_providers", mode="before")
    @classmethod
    def normalize_provider_list(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [provider.strip() for provider in value.split(",") if provider.strip()]
        return value

    @field_validator("recognition_insightface_det_size", mode="before")
    @classmethod
    def normalize_det_size(cls, value: str | tuple[int, int]) -> tuple[int, int]:
        if isinstance(value, str):
            parts = [part.strip() for part in value.lower().split("x") if part.strip()]
            if len(parts) == 2:
                return int(parts[0]), int(parts[1])
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
