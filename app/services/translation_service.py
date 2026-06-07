from __future__ import annotations

from typing import Sequence

import torch

from app.config import Settings
from app.services.model_registry import ModelRegistry


class TranslationService:
    def __init__(self, settings: Settings, models: ModelRegistry):
        self.settings = settings
        self.models = models

    def translate_segments(self, segments: Sequence[dict]) -> list[dict]:
        tokenizer, model, device = self.models.get_translation()
        translated: list[dict] = []

        batch_size = self.settings.translation_batch_size
        for start_index in range(0, len(segments), batch_size):
            batch = segments[start_index : start_index + batch_size]
            source_texts = [item["text"].strip() for item in batch]
            encoded = tokenizer(
                source_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.settings.translation_max_input_tokens,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}

            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    num_beams=4,
                    max_new_tokens=256,
                    early_stopping=True,
                )
            outputs = tokenizer.batch_decode(generated, skip_special_tokens=True)

            for item, output in zip(batch, outputs, strict=True):
                hindi = output.strip()
                if not hindi:
                    hindi = item["text"].strip()
                translated.append(
                    {
                        "index": int(item["index"]),
                        "start": float(item["start"]),
                        "end": float(item["end"]),
                        "source_text": item["text"],
                        "text": hindi,
                    }
                )

        return translated
