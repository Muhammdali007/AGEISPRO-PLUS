from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[4]


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
    model_person_weapon_weights_path: str | None = Field(
        default="storage/models/yolo11n.pt",
        validation_alias=AliasChoices("AI_MODEL_PERSON_WEAPON_WEIGHTS_PATH"),
    )
    model_weapon_weights_path: str | None = Field(
        default=None,
        validation_alias=AliasChoices("AI_MODEL_WEAPON_WEIGHTS_PATH"),
    )
    model_fire_smoke_weights_path: str | None = Field(
        default=None,
        validation_alias=AliasChoices("AI_MODEL_FIRE_SMOKE_WEIGHTS_PATH"),
    )
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
            "weapon": [
                "weapon",
                "gun",
                "knife",
                "scissor",
                "scissors",
                "pistol",
                "rifle",
                "firearm",
                "handgun",
                "kitchen_knife",
                "shotgun",
            ],
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
    person_confidence_threshold: float = Field(
        default=0.35, ge=0, le=1, validation_alias=AliasChoices("AI_PERSON_CONFIDENCE_THRESHOLD")
    )
    weapon_confidence_threshold: float = Field(
        default=0.25, ge=0, le=1, validation_alias=AliasChoices("AI_WEAPON_CONFIDENCE_THRESHOLD")
    )
    fire_confidence_threshold: float = Field(
        default=0.25, ge=0, le=1, validation_alias=AliasChoices("AI_FIRE_CONFIDENCE_THRESHOLD")
    )
    smoke_confidence_threshold: float = Field(
        default=0.05, ge=0, le=1, validation_alias=AliasChoices("AI_SMOKE_CONFIDENCE_THRESHOLD")
    )
    recognition_match_threshold: float = Field(
        default=0.82,
        ge=0,
        le=1,
        validation_alias=AliasChoices("AI_RECOGNITION_MATCH_THRESHOLD"),
    )
    recognition_min_margin: float = Field(
        default=0.08,
        ge=0,
        le=1,
        validation_alias=AliasChoices("AI_RECOGNITION_MIN_MARGIN"),
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

    @field_validator(
        "model_weights_path",
        "model_person_weapon_weights_path",
        "model_weapon_weights_path",
        "model_fire_smoke_weights_path",
        mode="before",
    )
    @classmethod
    def normalize_optional_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            return None
        candidate = Path(normalized)
        if candidate.is_absolute():
            return str(candidate)
        if (PROJECT_ROOT / candidate).exists():
            return str((PROJECT_ROOT / candidate).resolve())
        if (PROJECT_ROOT / "storage" / "models" / candidate).exists():
            return str((PROJECT_ROOT / "storage" / "models" / candidate).resolve())
        if candidate.parent == Path("."):
            for search_dir in (PROJECT_ROOT, PROJECT_ROOT / "storage" / "models"):
                if (search_dir / candidate).exists():
                    return str((search_dir / candidate).resolve())
        return normalized

    @field_validator("model_label_aliases", mode="before")
    @classmethod
    def normalize_label_aliases(cls, value: dict[str, list[str]]) -> dict[str, list[str]]:
        return {
            detector.strip().lower().replace("-", "_").replace(" ", "_"): [
                alias.strip().lower().replace("-", "_").replace(" ", "_")
                for alias in aliases
                if alias.strip()
            ]
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

    @model_validator(mode="after")
    def validate_production_runtime(self) -> "Settings":
        if self.environment.strip().lower() != "production":
            return self

        if self.model_backend == "simulated":
            raise ValueError("Production mode does not allow the simulated inference backend.")
        if self.allow_backend_fallback:
            raise ValueError("Production mode requires AI_ALLOW_BACKEND_FALLBACK=false.")
        if self.recognition_backend != "insightface":
            raise ValueError("Production mode requires AI_RECOGNITION_BACKEND=insightface.")
        if self.recognition_allow_fallback:
            raise ValueError("Production mode requires AI_RECOGNITION_ALLOW_FALLBACK=false.")

        required_paths = {
            "AI_MODEL_WEIGHTS_PATH": self.model_weights_path,
            "AI_MODEL_WEAPON_WEIGHTS_PATH": self.model_weapon_weights_path,
            "AI_MODEL_FIRE_SMOKE_WEIGHTS_PATH": self.model_fire_smoke_weights_path,
        }
        for env_name, candidate in required_paths.items():
            if not candidate:
                raise ValueError(f"Production mode requires {env_name} to point to a trained checkpoint.")
            if not Path(candidate).exists():
                raise ValueError(f"{env_name} points to a missing file: {candidate}")

        optional_paths = {
            "AI_MODEL_PERSON_WEAPON_WEIGHTS_PATH": self.model_person_weapon_weights_path,
        }
        for env_name, candidate in optional_paths.items():
            if candidate and not Path(candidate).exists():
                raise ValueError(f"{env_name} points to a missing file: {candidate}")

        if not self.api_event_callback_url:
            raise ValueError("Production mode requires AI_API_EVENT_CALLBACK_URL.")
        if not self.api_event_callback_token:
            raise ValueError("Production mode requires AI_API_EVENT_CALLBACK_TOKEN.")

        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
