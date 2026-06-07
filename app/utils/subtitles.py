from __future__ import annotations

from pathlib import Path
from typing import Iterable


def _srt_timestamp(seconds: float) -> str:
    milliseconds = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def write_srt(segments: Iterable[dict], path: Path, text_key: str = "text") -> None:
    blocks: list[str] = []
    for position, segment in enumerate(segments, start=1):
        blocks.append(
            "\n".join(
                [
                    str(position),
                    f"{_srt_timestamp(float(segment['start']))} --> "
                    f"{_srt_timestamp(float(segment['end']))}",
                    str(segment[text_key]).strip(),
                ]
            )
        )
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
