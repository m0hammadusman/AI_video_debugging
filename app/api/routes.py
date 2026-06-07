from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import FileResponse

from app.config import get_settings
from app.database import Database
from app.dependencies import get_database, get_ffmpeg, get_job_manager
from app.schemas import (
    HealthResponse,
    JobDetail,
    JobEvent,
    JobListResponse,
    JobLogsResponse,
    JobSummary,
    TranscriptResponse,
    TranslationResponse,
    UploadResponse,
)
from app.utils.files import remove_file, remove_tree, safe_filename


router = APIRouter()
settings = get_settings()


def _require_job(database: Database, job_id: str) -> dict:
    job = database.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


def _summary(job: dict) -> JobSummary:
    return JobSummary(**job)


def _detail(job: dict, request: Request) -> JobDetail:
    base = str(request.base_url).rstrip("/")
    prefix = settings.api_prefix
    complete = job["status"] == "completed"
    return JobDetail(
        **job,
        download_url=f"{base}{prefix}/download/{job['id']}" if complete else None,
        preview_url=f"{base}{prefix}/jobs/{job['id']}/preview" if complete else None,
        transcript_url=(
            f"{base}{prefix}/jobs/{job['id']}/transcript"
            if job.get("transcript_path")
            else None
        ),
        translation_url=(
            f"{base}{prefix}/jobs/{job['id']}/translation"
            if job.get("translation_path")
            else None
        ),
    )


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    database = get_database()
    ffmpeg_ok, ffprobe_ok = get_ffmpeg().available()
    return HealthResponse(
        status="ok" if database.ping() and ffmpeg_ok and ffprobe_ok else "degraded",
        ffmpeg=ffmpeg_ok,
        ffprobe=ffprobe_ok,
        database=database.ping(),
        version=settings.app_version,
    )


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_video(
    request: Request,
    file: Annotated[UploadFile, File(description="MP4, MOV, AVI, or MKV video")],
) -> UploadResponse:
    database = get_database()
    ffmpeg = get_ffmpeg()
    manager = get_job_manager()

    original_name = safe_filename(file.filename or "video")
    extension = Path(original_name).suffix.lower()
    if extension not in settings.allowed_extensions:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported format. Allowed formats: {sorted(settings.allowed_extensions)}",
        )

    job_id = uuid.uuid4().hex
    upload_dir = settings.uploads_dir / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / original_name
    max_bytes = settings.max_upload_mb * 1024 * 1024
    written = 0

    try:
        with target.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload exceeds the {settings.max_upload_mb} MB limit.",
                    )
                output.write(chunk)
        await file.close()

        if written == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        try:
            media = ffmpeg.probe(target)
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid or corrupt media file: {exc}",
            ) from exc

        if not media["has_video"]:
            raise HTTPException(status_code=400, detail="The file contains no video stream.")
        if not media["has_audio"]:
            raise HTTPException(status_code=400, detail="The file contains no audio stream.")
        if media["duration"] > settings.max_video_duration_seconds:
            raise HTTPException(
                status_code=413,
                detail=(
                    "Video duration exceeds the configured limit of "
                    f"{settings.max_video_duration_seconds} seconds."
                ),
            )

        database.create_job(
            job_id=job_id,
            original_filename=original_name,
            stored_path=str(target),
        )
        manager.submit(job_id)
    except HTTPException:
        remove_tree(upload_dir)
        raise
    except Exception as exc:
        remove_tree(upload_dir)
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc

    base = str(request.base_url).rstrip("/")
    return UploadResponse(
        job_id=job_id,
        status="queued",
        status_url=f"{base}{settings.api_prefix}/status/{job_id}",
    )


@router.get("/status/{job_id}", response_model=JobDetail)
def job_status(job_id: str, request: Request) -> JobDetail:
    job = _require_job(get_database(), job_id)
    return _detail(job, request)


@router.get("/jobs/{job_id}", response_model=JobDetail)
def job_detail(job_id: str, request: Request) -> JobDetail:
    job = _require_job(get_database(), job_id)
    return _detail(job, request)


@router.get("/jobs", response_model=JobListResponse)
def list_jobs(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> JobListResponse:
    jobs, total = get_database().list_jobs(limit, offset)
    return JobListResponse(
        items=[_summary(job) for job in jobs],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/jobs/{job_id}/logs", response_model=JobLogsResponse)
def job_logs(job_id: str) -> JobLogsResponse:
    database = get_database()
    _require_job(database, job_id)
    events = [JobEvent(**item) for item in database.get_events(job_id)]
    return JobLogsResponse(job_id=job_id, events=events)


@router.get("/jobs/{job_id}/transcript", response_model=TranscriptResponse)
def get_transcript(job_id: str) -> TranscriptResponse:
    job = _require_job(get_database(), job_id)
    path_value = job.get("transcript_path")
    if not path_value or not Path(path_value).is_file():
        raise HTTPException(status_code=404, detail="Transcript is not available.")
    return TranscriptResponse(**json.loads(Path(path_value).read_text(encoding="utf-8")))


@router.get("/jobs/{job_id}/translation", response_model=TranslationResponse)
def get_translation(job_id: str) -> TranslationResponse:
    job = _require_job(get_database(), job_id)
    path_value = job.get("translation_path")
    if not path_value or not Path(path_value).is_file():
        raise HTTPException(status_code=404, detail="Translation is not available.")
    return TranslationResponse(**json.loads(Path(path_value).read_text(encoding="utf-8")))


@router.get("/jobs/{job_id}/preview")
def preview_video(job_id: str) -> FileResponse:
    job = _require_job(get_database(), job_id)
    if job["status"] != "completed":
        raise HTTPException(status_code=409, detail="The job is not completed.")
    output_path = Path(job["output_path"] or "")
    if not output_path.is_file():
        raise HTTPException(status_code=404, detail="Output video is missing.")
    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"{Path(job['original_filename']).stem}_hindi.mp4",
        content_disposition_type="inline",
    )


@router.get("/download/{job_id}")
def download_video(job_id: str) -> FileResponse:
    job = _require_job(get_database(), job_id)
    if job["status"] != "completed":
        raise HTTPException(status_code=409, detail="The job is not completed.")
    output_path = Path(job["output_path"] or "")
    if not output_path.is_file():
        raise HTTPException(status_code=404, detail="Output video is missing.")
    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"{Path(job['original_filename']).stem}_hindi.mp4",
    )


@router.delete("/jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_job(job_id: str) -> None:
    database = get_database()
    job = _require_job(database, job_id)
    if job["status"] in {"queued", "processing"}:
        raise HTTPException(
            status_code=409,
            detail="A queued or processing job cannot be deleted.",
        )

    paths = [
        settings.uploads_dir / job_id,
        settings.outputs_dir / job_id,
        settings.work_dir / job_id,
        settings.transcripts_dir / job_id,
        settings.translations_dir / job_id,
    ]
    if not database.delete_job(job_id):
        raise HTTPException(status_code=404, detail="Job not found.")
    for path in paths:
        remove_tree(path)
