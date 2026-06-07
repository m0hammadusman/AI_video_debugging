from __future__ import annotations

import json
import logging
import shutil
import traceback
from pathlib import Path

from app.config import Settings
from app.database import Database, utc_now
from app.services.audio_sync_service import AudioSyncService
from app.services.ffmpeg_service import FFmpegService
from app.services.transcription_service import TranscriptionService
from app.services.translation_service import TranslationService
from app.utils.files import remove_tree
from app.utils.subtitles import write_srt


class DubbingPipeline:
    def __init__(
        self,
        settings: Settings,
        database: Database,
        ffmpeg: FFmpegService,
        transcription: TranscriptionService,
        translation: TranslationService,
        audio_sync: AudioSyncService,
    ):
        self.settings = settings
        self.database = database
        self.ffmpeg = ffmpeg
        self.transcription = transcription
        self.translation = translation
        self.audio_sync = audio_sync
        self.logger = logging.getLogger("dubbing.pipeline")

    def _update(self, job_id: str, progress: int, step: str) -> None:
        self.database.update_job(
            job_id,
            status="processing",
            progress=max(0, min(99, int(progress))),
            current_step=step,
        )
        self.database.add_event(job_id, step)

    def run(self, job_id: str) -> None:
        job = self.database.get_job(job_id)
        if not job:
            return

        source_video = Path(job["stored_path"])
        work_dir = self.settings.work_dir / job_id
        output_dir = self.settings.outputs_dir / job_id
        transcript_dir = self.settings.transcripts_dir / job_id
        translation_dir = self.settings.translations_dir / job_id

        work_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        transcript_dir.mkdir(parents=True, exist_ok=True)
        translation_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.database.update_job(
                job_id,
                status="processing",
                progress=1,
                current_step="Starting processing",
                started_at=utc_now(),
                error=None,
            )
            self.database.add_event(job_id, "Processing started")

            self._update(job_id, 5, "Validating media")
            media = self.ffmpeg.probe(source_video)
            duration = float(media["duration"])
            if not media["has_video"]:
                raise RuntimeError("The uploaded file does not contain a video stream.")
            if not media["has_audio"]:
                raise RuntimeError("The uploaded file does not contain an audio stream.")
            if duration <= 0:
                raise RuntimeError("The video duration could not be determined.")
            if duration > self.settings.max_video_duration_seconds:
                raise RuntimeError(
                    f"Video exceeds the configured maximum duration of "
                    f"{self.settings.max_video_duration_seconds} seconds."
                )
            self.database.update_job(job_id, video_duration=duration)

            extracted_audio = work_dir / "english_audio.wav"
            self._update(job_id, 10, "Extracting English audio")
            self.ffmpeg.extract_audio(source_video, extracted_audio)

            self._update(job_id, 20, "Transcribing English speech")
            transcript = self.transcription.transcribe(extracted_audio)
            transcript_json = transcript_dir / "transcript.json"
            transcript_srt = transcript_dir / "transcript.srt"
            transcript_json.write_text(
                json.dumps(
                    {
                        "job_id": job_id,
                        "language": "en",
                        "segments": transcript,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            write_srt(transcript, transcript_srt)

            self._update(job_id, 45, "Translating English to Hindi")
            translations = self.translation.translate_segments(transcript)
            translation_json = translation_dir / "translation.json"
            translation_srt = translation_dir / "translation.srt"
            translation_json.write_text(
                json.dumps(
                    {
                        "job_id": job_id,
                        "source_language": "en",
                        "target_language": "hi",
                        "segments": translations,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            write_srt(translations, translation_srt)

            dubbed_audio = work_dir / "hindi_timeline.wav"
            self._update(job_id, 60, "Generating synchronized Hindi speech")
            self.audio_sync.build_timeline(
                translations,
                duration,
                work_dir,
                dubbed_audio,
                lambda value, step: self._update(job_id, value, step),
            )

            output_video = output_dir / "dubbed_hindi.mp4"
            self._update(job_id, 92, "Merging Hindi audio with video")
            self.ffmpeg.merge_video_and_audio(source_video, dubbed_audio, output_video)

            self._update(job_id, 98, "Verifying final output")
            output_media = self.ffmpeg.probe(output_video)
            if not output_media["has_video"] or not output_media["has_audio"]:
                raise RuntimeError("The final output did not contain valid video and audio streams.")

            self.database.update_job(
                job_id,
                status="completed",
                progress=100,
                current_step="Completed",
                completed_at=utc_now(),
                output_path=str(output_video),
                transcript_path=str(transcript_json),
                translation_path=str(translation_json),
                error=None,
            )
            self.database.add_event(job_id, "Hindi-dubbed video completed successfully")
        except Exception as exc:
            self.logger.error("Job %s failed: %s\n%s", job_id, exc, traceback.format_exc())
            self.database.update_job(
                job_id,
                status="failed",
                current_step="Failed",
                completed_at=utc_now(),
                error=str(exc)[:4000],
            )
            self.database.add_event(job_id, str(exc)[:1000], level="ERROR")
        finally:
            if not self.settings.keep_work_files:
                remove_tree(work_dir)
