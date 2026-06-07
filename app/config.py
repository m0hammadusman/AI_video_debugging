from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "AI Video Dubbing Backend"
    app_version: str = "1.0.0"
    debug: bool = False

    api_prefix: str = "/api"
    allowed_origins: str = "*"
    max_upload_mb: int = Field(default=1024, ge=1)
    worker_count: int = Field(default=1, ge=1, le=4)
    job_retention_days: int = Field(default=7, ge=1)

    data_dir: Path = PROJECT_ROOT / "data"
    models_dir: Path = PROJECT_ROOT / "models"
    uploads_dir: Path = PROJECT_ROOT / "data" / "uploads"
    outputs_dir: Path = PROJECT_ROOT / "data" / "outputs"
    work_dir: Path = PROJECT_ROOT / "data" / "work"
    transcripts_dir: Path = PROJECT_ROOT / "data" / "transcripts"
    translations_dir: Path = PROJECT_ROOT / "data" / "translations"
    logs_dir: Path = PROJECT_ROOT / "data" / "logs"
    database_path: Path = PROJECT_ROOT / "data" / "jobs.sqlite3"

    ffmpeg_binary: str = "ffmpeg"
    ffprobe_binary: str = "ffprobe"
    audio_sample_rate: int = Field(default=16000, ge=8000, le=48000)

    whisper_model: str = "large-v3"
    whisper_device: str = "auto"
    whisper_compute_type: str = "auto"
    whisper_beam_size: int = Field(default=5, ge=1, le=10)
    whisper_vad_filter: bool = True

    translation_model: str = "Helsinki-NLP/opus-mt-en-hi"
    translation_batch_size: int = Field(default=8, ge=1, le=64)
    translation_max_input_tokens: int = Field(default=384, ge=32, le=512)

    tts_model: str = "facebook/mms-tts-hin"
    tts_device: str = "auto"
    tts_noise_scale: float = Field(default=0.667, ge=0.0, le=2.0)
    tts_noise_scale_duration: float = Field(default=0.8, ge=0.0, le=2.0)
    max_tts_speedup: float = Field(default=1.30, ge=1.0, le=2.0)
    max_video_duration_seconds: int = Field(default=7200, ge=1)
    keep_work_files: bool = False
    offline_mode: bool = False

    @property
    def cors_origins(self) -> list[str]:
        value = self.allowed_origins.strip()
        if value == "*":
            return ["*"]
        return [item.strip() for item in value.split(",") if item.strip()]

    @property
    def allowed_extensions(self) -> set[str]:
        return {".mp4", ".mov", ".avi", ".mkv"}

    def ensure_directories(self) -> None:
        for path in (
            self.data_dir,
            self.models_dir,
            self.uploads_dir,
            self.outputs_dir,
            self.work_dir,
            self.transcripts_dir,
            self.translations_dir,
            self.logs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
