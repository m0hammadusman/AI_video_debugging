from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    """Small thread-safe SQLite repository for jobs and progress events."""

    def __init__(self, path: Path):
        self.path = path
        self._write_lock = threading.RLock()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._write_lock, self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    original_filename TEXT NOT NULL,
                    stored_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress INTEGER NOT NULL DEFAULT 0,
                    current_step TEXT NOT NULL DEFAULT 'Queued',
                    error TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    output_path TEXT,
                    transcript_path TEXT,
                    translation_path TEXT,
                    video_duration REAL,
                    source_language TEXT NOT NULL DEFAULT 'en',
                    target_language TEXT NOT NULL DEFAULT 'hi'
                );

                CREATE TABLE IF NOT EXISTS job_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_jobs_created_at
                    ON jobs(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_events_job_id
                    ON job_events(job_id, id);
                """
            )
            conn.commit()

    def ping(self) -> bool:
        try:
            with self.connect() as conn:
                conn.execute("SELECT 1").fetchone()
            return True
        except sqlite3.Error:
            return False

    def create_job(
        self,
        *,
        job_id: str,
        original_filename: str,
        stored_path: str,
    ) -> dict[str, Any]:
        now = utc_now()
        with self._write_lock, self.connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    id, original_filename, stored_path, status, progress,
                    current_step, created_at
                ) VALUES (?, ?, ?, 'queued', 0, 'Queued', ?)
                """,
                (job_id, original_filename, stored_path, now),
            )
            conn.execute(
                """
                INSERT INTO job_events (job_id, timestamp, level, message)
                VALUES (?, ?, 'INFO', 'Job created and queued')
                """,
                (job_id, now),
            )
            conn.commit()
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return dict(row) if row else None

    def list_jobs(self, limit: int, offset: int) -> tuple[list[dict[str, Any]], int]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM jobs
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
            total = conn.execute("SELECT COUNT(*) AS count FROM jobs").fetchone()["count"]
        return [dict(row) for row in rows], int(total)

    def update_job(self, job_id: str, **fields: Any) -> None:
        if not fields:
            return
        allowed = {
            "status",
            "progress",
            "current_step",
            "error",
            "started_at",
            "completed_at",
            "output_path",
            "transcript_path",
            "translation_path",
            "video_duration",
        }
        invalid = set(fields) - allowed
        if invalid:
            raise ValueError(f"Unsupported job fields: {sorted(invalid)}")

        assignments = ", ".join(f"{name} = ?" for name in fields)
        values = list(fields.values()) + [job_id]
        with self._write_lock, self.connect() as conn:
            conn.execute(f"UPDATE jobs SET {assignments} WHERE id = ?", values)
            conn.commit()

    def add_event(self, job_id: str, message: str, level: str = "INFO") -> None:
        with self._write_lock, self.connect() as conn:
            conn.execute(
                """
                INSERT INTO job_events (job_id, timestamp, level, message)
                VALUES (?, ?, ?, ?)
                """,
                (job_id, utc_now(), level.upper(), message),
            )
            conn.commit()

    def get_events(self, job_id: str, limit: int = 500) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT timestamp, level, message
                FROM job_events
                WHERE job_id = ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (job_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def mark_interrupted_jobs_failed(self) -> int:
        now = utc_now()
        with self._write_lock, self.connect() as conn:
            rows = conn.execute(
                "SELECT id FROM jobs WHERE status IN ('queued', 'processing')"
            ).fetchall()
            job_ids = [row["id"] for row in rows]
            if job_ids:
                conn.execute(
                    """
                    UPDATE jobs
                    SET status = 'failed',
                        current_step = 'Interrupted',
                        error = 'The server stopped before processing completed.',
                        completed_at = ?
                    WHERE status IN ('queued', 'processing')
                    """,
                    (now,),
                )
                conn.executemany(
                    """
                    INSERT INTO job_events (job_id, timestamp, level, message)
                    VALUES (?, ?, 'ERROR', 'Job interrupted by server restart')
                    """,
                    [(job_id, now) for job_id in job_ids],
                )
            conn.commit()
        return len(job_ids)

    def delete_job(self, job_id: str) -> bool:
        with self._write_lock, self.connect() as conn:
            cursor = conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            conn.commit()
        return cursor.rowcount > 0
