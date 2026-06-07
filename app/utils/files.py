from __future__ import annotations

import re
import shutil
from pathlib import Path


_FILENAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(filename: str) -> str:
    name = Path(filename).name.strip()
    name = _FILENAME_PATTERN.sub("_", name)
    name = name.strip("._")
    return name or "upload.bin"


def remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def remove_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
