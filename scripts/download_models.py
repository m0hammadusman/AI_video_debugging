from __future__ import annotations

from app.config import get_settings
from app.services.model_registry import ModelRegistry


def main() -> None:
    settings = get_settings()
    registry = ModelRegistry(settings)

    print(f"Downloading Faster-Whisper model: {settings.whisper_model}")
    registry.get_whisper()

    print(f"Downloading translation model: {settings.translation_model}")
    registry.get_translation()

    print(f"Downloading Hindi TTS model: {settings.tts_model}")
    registry.get_tts()

    print(f"Models are cached under: {settings.models_dir}")


if __name__ == "__main__":
    main()
