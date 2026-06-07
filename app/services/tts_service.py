from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from app.config import Settings
from app.services.model_registry import ModelRegistry


class TTSService:
    def __init__(self, settings: Settings, models: ModelRegistry):
        self.settings = settings
        self.models = models

    def synthesize(self, text: str, output_wav: Path) -> int:
        tokenizer, model, device = self.models.get_tts()

        cleaned = " ".join(text.strip().split())
        if not cleaned:
            raise RuntimeError("Cannot synthesize an empty translation segment.")

        inputs = tokenizer(cleaned, return_tensors="pt")
        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        model.noise_scale = self.settings.tts_noise_scale
        model.noise_scale_duration = self.settings.tts_noise_scale_duration

        with torch.inference_mode():
            waveform = model(**inputs).waveform

        audio = waveform.squeeze().detach().float().cpu().numpy()
        audio = np.nan_to_num(
            audio,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 0.98:
            audio = audio * (0.98 / peak)

        sample_rate = int(model.config.sampling_rate)

        output_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(
            output_wav,
            audio,
            sample_rate,
            subtype="PCM_16",
        )

        return sample_rate