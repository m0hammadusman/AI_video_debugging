from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.services.model_registry import ModelRegistry


class TranscriptionService:
    def __init__(self, settings: Settings, models: ModelRegistry):
        self.settings = settings
        self.models = models

    def transcribe(self, audio_path: Path) -> list[dict]:
        model = self.models.get_whisper()
        segments_iter, _info = model.transcribe(
            str(audio_path),
            language="en",
            task="transcribe",
            beam_size=self.settings.whisper_beam_size,
            vad_filter=self.settings.whisper_vad_filter,
            word_timestamps=False,
            condition_on_previous_text=True,
        )

        result: list[dict] = []
        for index, segment in enumerate(segments_iter):
            text = segment.text.strip()
            start = max(0.0, float(segment.start))
            end = max(start + 0.05, float(segment.end))
            if not text:
                continue
            result.append(
                {
                    "index": index,
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "text": text,
                }
            )

        if not result:
            raise RuntimeError("No English speech could be transcribed from the uploaded video.")
        return result
