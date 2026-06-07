from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

from app.config import Settings


class FFmpegError(RuntimeError):
    pass


class FFmpegService:
    def __init__(self, settings: Settings):
        self.settings = settings

    def available(self) -> tuple[bool, bool]:
        return (
            shutil.which(self.settings.ffmpeg_binary) is not None,
            shutil.which(self.settings.ffprobe_binary) is not None,
        )

    def _run(self, command: list[str], *, timeout: int | None = None) -> subprocess.CompletedProcess:
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=timeout,
            )
        except FileNotFoundError as exc:
            raise FFmpegError(f"Required executable is missing: {command[0]}") from exc
        except subprocess.TimeoutExpired as exc:
            raise FFmpegError(f"Command timed out: {' '.join(command[:4])}") from exc

        if result.returncode != 0:
            error = result.stderr.strip()[-4000:]
            raise FFmpegError(error or f"Command failed with exit code {result.returncode}")
        return result

    def probe(self, media_path: Path) -> dict[str, Any]:
        result = self._run(
            [
                self.settings.ffprobe_binary,
                "-v",
                "error",
                "-show_entries",
                "format=duration,format_name:stream=index,codec_type,codec_name",
                "-of",
                "json",
                str(media_path),
            ],
            timeout=60,
        )
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise FFmpegError("ffprobe returned invalid JSON") from exc

        streams = payload.get("streams", [])
        has_video = any(stream.get("codec_type") == "video" for stream in streams)
        has_audio = any(stream.get("codec_type") == "audio" for stream in streams)
        try:
            duration = float(payload.get("format", {}).get("duration", 0))
        except (TypeError, ValueError):
            duration = 0.0

        return {
            "duration": duration,
            "has_video": has_video,
            "has_audio": has_audio,
            "streams": streams,
            "format_name": payload.get("format", {}).get("format_name"),
        }

    def extract_audio(self, video_path: Path, output_wav: Path) -> None:
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        self._run(
            [
                self.settings.ffmpeg_binary,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(video_path),
                "-map",
                "0:a:0",
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(self.settings.audio_sample_rate),
                "-c:a",
                "pcm_s16le",
                str(output_wav),
            ]
        )

    @staticmethod
    def _atempo_chain(factor: float) -> str:
        """Build a quality-preserving chain with factors in FFmpeg's safest range."""
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError("Tempo factor must be a positive finite number")

        factors: list[float] = []
        while factor > 2.0:
            factors.append(2.0)
            factor /= 2.0
        while factor < 0.5:
            factors.append(0.5)
            factor /= 0.5
        factors.append(factor)
        return ",".join(f"atempo={item:.8f}" for item in factors)

    def fit_audio_to_duration(
        self,
        source_wav: Path,
        output_wav: Path,
        target_duration: float,
        sample_rate: int,
    ) -> None:
        if target_duration <= 0:
            raise FFmpegError("Target audio duration must be greater than zero")

        probe = self.probe(source_wav)
        source_duration = max(float(probe["duration"]), 0.001)
        tempo_factor = source_duration / target_duration
        tempo_filter = self._atempo_chain(tempo_factor)
        audio_filter = (
            f"{tempo_filter},"
            f"apad=pad_dur={target_duration:.6f},"
            f"atrim=duration={target_duration:.6f}"
        )
        self._run(
            [
                self.settings.ffmpeg_binary,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source_wav),
                "-filter:a",
                audio_filter,
                "-ac",
                "1",
                "-ar",
                str(sample_rate),
                "-c:a",
                "pcm_s16le",
                str(output_wav),
            ]
        )

    def merge_video_and_audio(
        self,
        source_video: Path,
        dubbed_audio: Path,
        output_video: Path,
    ) -> None:
        output_video.parent.mkdir(parents=True, exist_ok=True)
        copy_command = [
            self.settings.ffmpeg_binary,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source_video),
            "-i",
            str(dubbed_audio),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output_video),
        ]
        try:
            self._run(copy_command)
            return
        except FFmpegError:
            output_video.unlink(missing_ok=True)

        # Some source codecs cannot be copied into MP4. Re-encode only as fallback.
        reencode_command = [
            self.settings.ffmpeg_binary,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source_video),
            "-i",
            str(dubbed_audio),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output_video),
        ]
        self._run(reencode_command)
