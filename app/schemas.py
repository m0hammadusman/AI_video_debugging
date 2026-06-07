from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


JobStatus = Literal["queued", "processing", "completed", "failed"]


class UploadResponse(BaseModel):
    job_id: str
    status: JobStatus
    status_url: str


class JobSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    original_filename: str
    status: JobStatus
    progress: int = Field(ge=0, le=100)
    current_step: str
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    video_duration: float | None = None


class JobDetail(JobSummary):
    download_url: str | None = None
    preview_url: str | None = None
    transcript_url: str | None = None
    translation_url: str | None = None


class JobListResponse(BaseModel):
    items: list[JobSummary]
    total: int
    limit: int
    offset: int


class Segment(BaseModel):
    index: int
    start: float
    end: float
    text: str


class TranscriptResponse(BaseModel):
    job_id: str
    language: str
    segments: list[Segment]


class TranslationSegment(Segment):
    source_text: str


class TranslationResponse(BaseModel):
    job_id: str
    source_language: str
    target_language: str
    segments: list[TranslationSegment]


class JobEvent(BaseModel):
    timestamp: datetime
    level: str
    message: str


class JobLogsResponse(BaseModel):
    job_id: str
    events: list[JobEvent]


class HealthResponse(BaseModel):
    status: str
    ffmpeg: bool
    ffprobe: bool
    database: bool
    version: str
