"""
voice/vad.py - voice-activity detection: cut a continuous microphone stream into utterances.

Energy-based on purpose: no model download, negligible CPU (it runs on every 30 ms frame, all day),
and it adapts to the room. Speech is "RMS energy well above a slowly-tracked noise floor". Only
the utterances it finds are ever handed to the (much heavier) speech-to-text engine, which is what
keeps an always-on microphone cheap.

Behaviour worth knowing:
  * the first `calibration_frames` frames only measure the room, so start it in a quiet moment;
  * a short pre-roll is kept so the first syllable is not clipped;
  * speech shorter than `min_speech_ms` (a cough, a click, a door) is discarded;
  * an utterance ends after `silence_ms` of quiet, or is cut at `max_utterance_s`.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Optional

import numpy as np


class UtteranceSegmenter:
    def __init__(self, sample_rate: int = 16000, frame_ms: int = 30, silence_ms: int = 800,
                 max_utterance_s: float = 15.0, min_speech_ms: int = 200, threshold_factor: float = 3.0,
                 min_threshold: float = 150.0, preroll_ms: int = 300, calibration_frames: int = 10,
                 floor_alpha: float = 0.05):
        self.sample_rate = sample_rate
        self.frame_samples = sample_rate * frame_ms // 1000
        self.frame_ms = frame_ms
        self.silence_frames = max(1, silence_ms // frame_ms)
        self.max_frames = int(max_utterance_s * 1000 // frame_ms)
        self.min_speech_frames = max(1, min_speech_ms // frame_ms)
        self.factor = float(threshold_factor)
        self.min_threshold = float(min_threshold)
        self.calibration_frames = calibration_frames
        self.alpha = floor_alpha
        self._preroll: Deque[np.ndarray] = deque(maxlen=max(1, preroll_ms // frame_ms))
        self._seen = 0
        self.noise_floor = 0.0
        self.reset_utterance()

    def reset_utterance(self) -> None:
        self._frames: list = []
        self._speech_frames = 0
        self._quiet_run = 0
        self.in_speech = False

    @staticmethod
    def rms(frame: np.ndarray) -> float:
        if frame.size == 0:
            return 0.0
        f = frame.astype(np.float64)
        return float(np.sqrt(np.mean(f * f)))

    @property
    def threshold(self) -> float:
        return max(self.noise_floor * self.factor, self.min_threshold)

    def feed(self, frame: np.ndarray) -> Optional[bytes]:
        """Feed one frame of int16 mono audio. Returns the finished utterance (raw int16 bytes) or None."""
        level = self.rms(frame)
        self._seen += 1
        if self._seen <= self.calibration_frames:                  # measuring the room, not listening yet
            self.noise_floor = level if self._seen == 1 else (self.noise_floor * 0.7 + level * 0.3)
            self._preroll.append(frame)
            return None

        speaking = level > self.threshold
        if not self.in_speech:
            if speaking:
                self.in_speech = True
                self._frames = list(self._preroll) + [frame]
                self._speech_frames = 1
                self._quiet_run = 0
            else:
                self.noise_floor += self.alpha * (level - self.noise_floor)   # track the room only while it is quiet
                self._preroll.append(frame)
            return None

        self._frames.append(frame)
        if speaking:
            self._speech_frames += 1
            self._quiet_run = 0
        else:
            self._quiet_run += 1
        if self._quiet_run >= self.silence_frames or len(self._frames) >= self.max_frames:
            return self._finish()
        return None

    def _finish(self) -> Optional[bytes]:
        keep = self._speech_frames >= self.min_speech_frames
        audio = np.concatenate(self._frames).astype(np.int16).tobytes() if keep and self._frames else None
        self._preroll.clear()
        self.reset_utterance()
        return audio
