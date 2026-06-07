from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import soundfile as sf

from app.config import Settings
from app.services.ffmpeg_service import FFmpegService
from app.services.tts_service import TTSService


ProgressCallback = Callable[[int, str], None]


class AudioSyncService:
    def __init__(
        self,
        settings: Settings,
        ffmpeg: FFmpegService,
        tts: TTSService,
    ):
        self.settings = settings
        self.ffmpeg = ffmpeg
        self.tts = tts

    def build_timeline(
        self,
        segments: Sequence[dict],
        video_duration: float,
        work_dir: Path,
        output_wav: Path,
        progress: ProgressCallback,
    ) -> None:
        segment_dir = work_dir / "tts_segments"
        fitted_dir = work_dir / "fitted_segments"
        segment_dir.mkdir(parents=True, exist_ok=True)
        fitted_dir.mkdir(parents=True, exist_ok=True)

        sample_rate = self.settings.audio_sample_rate
        fitted_segments: list[tuple[dict, Path]] = []
        total = max(1, len(segments))

        for position, segment in enumerate(segments):
            raw_path = segment_dir / f"{position:06d}.wav"
            fitted_path = fitted_dir / f"{position:06d}.wav"

            generated_rate = self.tts.synthesize(segment["text"], raw_path)
            if generated_rate <= 0:
                raise RuntimeError("The TTS model returned an invalid sample rate.")

            target_duration = max(0.08, float(segment["end"]) - float(segment["start"]))
            self.ffmpeg.fit_audio_to_duration(
                raw_path,
                fitted_path,
                target_duration,
                sample_rate,
            )
            fitted_segments.append((segment, fitted_path))

            percentage = 60 + int(((position + 1) / total) * 25)
            progress(min(85, percentage), f"Generating Hindi speech {position + 1}/{total}")

        total_samples = max(1, int(round(video_duration * sample_rate)))
        timeline_path = work_dir / "timeline.float32"
        timeline = np.memmap(
            timeline_path,
            dtype=np.float32,
            mode="w+",
            shape=(total_samples,),
        )
        timeline[:] = 0.0

        for segment, fitted_path in fitted_segments:
            audio, rate = sf.read(fitted_path, dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if int(rate) != sample_rate:
                raise RuntimeError(
                    f"Unexpected fitted segment sample rate: {rate}; expected {sample_rate}"
                )

            start_sample = max(0, int(round(float(segment["start"]) * sample_rate)))
            if start_sample >= total_samples:
                continue
            end_sample = min(total_samples, start_sample + len(audio))
            usable = end_sample - start_sample
            if usable <= 0:
                continue
            timeline[start_sample:end_sample] += audio[:usable]

        peak = float(np.max(np.abs(timeline))) if total_samples else 0.0
        if peak > 0.98:
            timeline[:] *= 0.98 / peak

        output_wav.parent.mkdir(parents=True, exist_ok=True)
        with sf.SoundFile(
            output_wav,
            mode="w",
            samplerate=sample_rate,
            channels=1,
            subtype="PCM_16",
        ) as output:
            block_size = sample_rate * 30
            for offset in range(0, total_samples, block_size):
                output.write(np.asarray(timeline[offset : offset + block_size]))

        timeline.flush()
        del timeline
        timeline_path.unlink(missing_ok=True)
