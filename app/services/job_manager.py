from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor

from app.config import Settings
from app.services.pipeline import DubbingPipeline


class JobManager:
    """Runs blocking AI pipelines outside FastAPI request threads."""

    def __init__(self, settings: Settings, pipeline: DubbingPipeline):
        self.executor = ThreadPoolExecutor(
            max_workers=settings.worker_count,
            thread_name_prefix="dubbing-worker",
        )
        self.pipeline = pipeline
        self._futures: dict[str, Future] = {}
        self._lock = threading.RLock()

    def submit(self, job_id: str) -> None:
        with self._lock:
            future = self.executor.submit(self.pipeline.run, job_id)
            self._futures[job_id] = future
            future.add_done_callback(lambda _future: self._forget(job_id))

    def _forget(self, job_id: str) -> None:
        with self._lock:
            self._futures.pop(job_id, None)

    def is_running(self, job_id: str) -> bool:
        with self._lock:
            future = self._futures.get(job_id)
            return bool(future and not future.done())

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=False)
