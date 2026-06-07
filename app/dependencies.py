from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.database import Database
from app.services.audio_sync_service import AudioSyncService
from app.services.ffmpeg_service import FFmpegService
from app.services.job_manager import JobManager
from app.services.model_registry import ModelRegistry
from app.services.pipeline import DubbingPipeline
from app.services.transcription_service import TranscriptionService
from app.services.translation_service import TranslationService
from app.services.tts_service import TTSService


@lru_cache
def get_database() -> Database:
    settings = get_settings()
    return Database(settings.database_path)


@lru_cache
def get_ffmpeg() -> FFmpegService:
    return FFmpegService(get_settings())


@lru_cache
def get_model_registry() -> ModelRegistry:
    return ModelRegistry(get_settings())


@lru_cache
def get_pipeline() -> DubbingPipeline:
    settings = get_settings()
    database = get_database()
    ffmpeg = get_ffmpeg()
    models = get_model_registry()
    transcription = TranscriptionService(settings, models)
    translation = TranslationService(settings, models)
    tts = TTSService(settings, models)
    audio_sync = AudioSyncService(settings, ffmpeg, tts)
    return DubbingPipeline(
        settings,
        database,
        ffmpeg,
        transcription,
        translation,
        audio_sync,
    )


@lru_cache
def get_job_manager() -> JobManager:
    return JobManager(get_settings(), get_pipeline())
