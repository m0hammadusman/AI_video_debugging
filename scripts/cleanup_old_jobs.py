from __future__ import annotations

import argparse
import shutil
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.config import get_settings
from app.database import Database


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Delete old completed or failed dubbing jobs.")
    parser.add_argument("--days", type=int, default=None)
    args = parser.parse_args()

    settings = get_settings()
    days = args.days or settings.job_retention_days
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    database = Database(settings.database_path)
    database.initialize()

    jobs, _ = database.list_jobs(limit=100000, offset=0)
    deleted = 0
    for job in jobs:
        if job["status"] not in {"completed", "failed"}:
            continue
        completed = parse_time(job.get("completed_at")) or parse_time(job.get("created_at"))
        if not completed or completed >= cutoff:
            continue

        job_id = job["id"]
        if database.delete_job(job_id):
            for base in (
                settings.uploads_dir,
                settings.outputs_dir,
                settings.work_dir,
                settings.transcripts_dir,
                settings.translations_dir,
            ):
                shutil.rmtree(base / job_id, ignore_errors=True)
            deleted += 1

    print(f"Deleted {deleted} job(s) older than {days} day(s).")


if __name__ == "__main__":
    main()
