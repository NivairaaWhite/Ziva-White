"""
voice/audio.py - the microphone.

Wraps `sounddevice` (PortAudio). Frames arrive on a driver thread, go through a bounded queue and
are handed out as fixed-size int16 mono numpy arrays. If the consumer falls behind, the oldest
audio is dropped rather than letting memory grow: for a live assistant, stale audio is worthless.
"""
from __future__ import annotations

import queue
import threading
from typing import Iterator, List, Optional, Union

import numpy as np

from .errors import VoiceUnavailable


def _sounddevice():
    try:
        import sounddevice as sd
    except ImportError as e:
        raise VoiceUnavailable("Microphone input needs `pip install sounddevice` (and PortAudio: "
                               "`sudo apt install libportaudio2` on Linux).") from e
    except OSError as e:     # sounddevice imports fine but the PortAudio library is missing
        raise VoiceUnavailable(f"PortAudio is not installed ({e}). On Linux: `sudo apt install libportaudio2`.") from e
    return sd


def list_input_devices() -> List[dict]:
    sd = _sounddevice()
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) > 0:
            out.append({"index": i, "name": d["name"], "channels": d["max_input_channels"],
                        "default_rate": d.get("default_samplerate")})
    return out


class MicStream:
    def __init__(self, sample_rate: int = 16000, frame_ms: int = 30, device: Union[int, str, None] = None,
                 max_queue_frames: int = 400):
        self.sample_rate = sample_rate
        self.frame_samples = sample_rate * frame_ms // 1000
        self.device = device
        self._q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=max_queue_frames)
        self._stream = None
        self.dropped = 0

    def _callback(self, indata, frames, time_info, status) -> None:      # driver thread: keep it tiny
        frame = np.frombuffer(bytes(indata), dtype=np.int16).copy()
        try:
            self._q.put_nowait(frame)
        except queue.Full:
            try:
                self._q.get_nowait()               # drop the oldest, keep the newest
                self._q.put_nowait(frame)
            except queue.Empty:
                pass
            self.dropped += 1

    def __enter__(self) -> "MicStream":
        sd = _sounddevice()
        try:
            self._stream = sd.RawInputStream(samplerate=self.sample_rate, blocksize=self.frame_samples, dtype="int16",
                                             channels=1, device=self.device, callback=self._callback)
            self._stream.start()
        except Exception as e:  # noqa: BLE001 - PortAudioError and friends
            raise VoiceUnavailable(f"Could not open the microphone ({e}). Check voice.stt.input_device "
                                   f"(`python -m zivawhite.app voice --devices` lists them) and OS microphone permission.") from e
        return self

    def __exit__(self, *exc) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None

    def drain(self) -> None:
        """Throw away buffered audio (used after speaking, so the assistant never hears itself)."""
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass

    def frames(self, stop: threading.Event, poll_s: float = 0.25) -> Iterator[np.ndarray]:
        while not stop.is_set():
            try:
                yield self._q.get(timeout=poll_s)
            except queue.Empty:
                continue
