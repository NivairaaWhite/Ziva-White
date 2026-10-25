"""
voice/stt.py - speech-to-text, fully offline.

  vosk            small (50 MB-1.8 GB) Kaldi models, fast on a CPU, ideal for always-on listening.
                  Download a model from https://alphacephei.com/vosk/models, unpack it, and point
                  voice.stt.model at the folder (default models/vosk/model).
  faster_whisper  much more accurate, heavier. voice.stt.model is a model name ("base.en", "small")
                  -- downloaded once on first use -- or a folder with a converted model.

Neither is a dependency of the core project; a missing one raises VoiceUnavailable with the
install command instead of a stack trace.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np

from .errors import VoiceUnavailable


class SpeechToText:
    name = "stt"

    def transcribe(self, pcm: bytes, sample_rate: int = 16000) -> str:
        raise NotImplementedError


class VoskSTT(SpeechToText):
    name = "vosk"

    def __init__(self, model_dir: str, sample_rate: int = 16000):
        try:
            import vosk
        except ImportError as e:
            raise VoiceUnavailable("Speech recognition needs `pip install vosk`.") from e
        if not os.path.isdir(model_dir):
            raise VoiceUnavailable(f"No Vosk model at {model_dir}. Download one from https://alphacephei.com/vosk/models "
                                   f"(e.g. vosk-model-small-en-us), unpack it there, or change voice.stt.model.")
        vosk.SetLogLevel(-1)
        self._vosk = vosk
        try:
            self._model = vosk.Model(model_dir)
        except Exception as e:  # noqa: BLE001
            raise VoiceUnavailable(f"Vosk could not load the model in {model_dir}: {e}") from e
        self.sample_rate = sample_rate

    def transcribe(self, pcm: bytes, sample_rate: int = 16000) -> str:
        rec = self._vosk.KaldiRecognizer(self._model, sample_rate)
        rec.AcceptWaveform(pcm)
        return json.loads(rec.FinalResult()).get("text", "").strip()


class FasterWhisperSTT(SpeechToText):
    name = "faster_whisper"

    def __init__(self, model: str, language: Optional[str] = "en", compute_type: str = "int8"):
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise VoiceUnavailable("This speech engine needs `pip install faster-whisper`.") from e
        try:
            self._model = WhisperModel(model, device="auto", compute_type=compute_type)
        except Exception as e:  # noqa: BLE001 - includes "cannot download the model" when offline
            raise VoiceUnavailable(f"faster-whisper could not load model {model!r}: {e}") from e
        self.language = language or None

    def transcribe(self, pcm: bytes, sample_rate: int = 16000) -> str:
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = self._model.transcribe(audio, language=self.language, beam_size=1, vad_filter=False,
                                             condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segments).strip()


def build_stt(cfg) -> SpeechToText:
    stt = cfg.section("voice").get("stt", {}) or {}
    engine = stt.get("engine", "vosk")
    rate = int(stt.get("sample_rate", 16000))
    model = str(stt.get("model", ""))
    if engine == "vosk":
        return VoskSTT(str(cfg.resolve(model)), rate)
    if engine == "faster_whisper":
        local = cfg.resolve(model)
        return FasterWhisperSTT(str(local) if local.exists() else model, stt.get("language", "en"))
    raise VoiceUnavailable(f"Unknown speech engine {engine!r}")
