from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import get_settings
from app.dependencies import get_database, get_job_manager


settings = get_settings()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    database = get_database()
    database.initialize()
    database.mark_interrupted_jobs_failed()
    yield
    get_job_manager().shutdown()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Fully self-hosted English-to-Hindi video dubbing API using "
        "FFmpeg, Faster-Whisper, MarianMT, and MMS/VITS."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix=settings.api_prefix, tags=["dubbing"])


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": f"{settings.api_prefix}/health",
    }
