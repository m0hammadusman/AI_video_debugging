from __future__ import annotations

import os
import threading
from typing import Any

import torch
from faster_whisper import WhisperModel
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    VitsModel,
)

from app.config import Settings


class ModelRegistry:
    """Lazy, process-local model cache. One worker is recommended for GPU use."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.RLock()
        self._whisper: WhisperModel | None = None
        self._translation_tokenizer: Any | None = None
        self._translation_model: Any | None = None
        self._tts_tokenizer: Any | None = None
        self._tts_model: VitsModel | None = None

        if settings.offline_mode:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    @staticmethod
    def torch_device(requested: str) -> torch.device:
        if requested == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(requested)

    def whisper_device(self) -> str:
        requested = self.settings.whisper_device
        if requested != "auto":
            return requested
        return "cuda" if torch.cuda.is_available() else "cpu"

    def whisper_compute_type(self) -> str:
        requested = self.settings.whisper_compute_type
        if requested != "auto":
            return requested
        return "float16" if self.whisper_device() == "cuda" else "int8"

    def get_whisper(self) -> WhisperModel:
        with self._lock:
            if self._whisper is None:
                self._whisper = WhisperModel(
                    self.settings.whisper_model,
                    device=self.whisper_device(),
                    compute_type=self.whisper_compute_type(),
                    download_root=str(self.settings.models_dir / "whisper"),
                )
            return self._whisper

    def get_translation(self) -> tuple[Any, Any, torch.device]:
        with self._lock:
            if self._translation_model is None or self._translation_tokenizer is None:
                cache = str(self.settings.models_dir / "huggingface")
                self._translation_tokenizer = AutoTokenizer.from_pretrained(
                    self.settings.translation_model,
                    cache_dir=cache,
                    local_files_only=self.settings.offline_mode,
                )
                self._translation_model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.settings.translation_model,
                    cache_dir=cache,
                    local_files_only=self.settings.offline_mode,
                )
                device = self.torch_device("auto")
                self._translation_model.to(device)
                self._translation_model.eval()
            device = next(self._translation_model.parameters()).device
            return self._translation_tokenizer, self._translation_model, device

    def get_tts(self) -> tuple[Any, VitsModel, torch.device]:
        with self._lock:
            if self._tts_model is None or self._tts_tokenizer is None:
                cache = str(self.settings.models_dir / "huggingface")
                self._tts_tokenizer = AutoTokenizer.from_pretrained(
                    self.settings.tts_model,
                    cache_dir=cache,
                    local_files_only=self.settings.offline_mode,
                )
                self._tts_model = VitsModel.from_pretrained(
                    self.settings.tts_model,
                    cache_dir=cache,
                    local_files_only=self.settings.offline_mode,
                )
                device = self.torch_device(self.settings.tts_device)
                self._tts_model.to(device)
                self._tts_model.eval()
            device = next(self._tts_model.parameters()).device
            return self._tts_tokenizer, self._tts_model, device
