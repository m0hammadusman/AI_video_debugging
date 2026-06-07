from pathlib import Path

from fastapi.testclient import TestClient

from app.config import get_settings
from app.dependencies import get_database
from app.main import app


def test_root_and_health(tmp_path: Path, monkeypatch) -> None:
    settings = get_settings()
    settings.database_path = tmp_path / "test.sqlite3"
    get_database.cache_clear()

    with TestClient(app) as client:
        root = client.get("/")
        assert root.status_code == 200
        assert root.json()["docs"] == "/docs"

        health = client.get("/api/health")
        assert health.status_code == 200
        payload = health.json()
        assert "database" in payload
        assert "ffmpeg" in payload
