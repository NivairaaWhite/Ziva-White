"""
voice/loop.py - the always-listening loop.

    microphone -> VAD (cut into utterances) -> offline STT -> wake word? -> assistant -> speak

Conversation model (all times from `voice:` in the config):
  * idle: every utterance is transcribed, but only one that BEGINS with a wake word does anything;
    everything else is discarded and never leaves this function;
  * "hey ziva what time is it" -> handled at once;  "hey ziva" alone -> `ack_phrase`, then the very
    next utterance (within `listen_timeout_s`) is the command;
  * after a reply you can keep talking without the wake word for `follow_up_s` seconds;
  * a stop phrase ("stop", "be quiet") silences speech immediately -- it is checked before the wake
    word, but it only interrupts while the assistant is awake or speaking.

Half-duplex by design: while it speaks, it does not listen, and buffered audio is dropped afterwards,
so it can never wake itself with its own voice. Interrupting speech is therefore done with the Stop
control in the web UI (or by it finishing a sentence), not by shouting over it.

`process_utterance` holds all the logic and takes plain bytes, so the whole state machine is tested
with fake STT/TTS and no audio hardware.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional

from .audio import MicStream
from .errors import VoiceUnavailable
from .stt import SpeechToText, build_stt
from .tts import TextToSpeech, build_tts
from .vad import UtteranceSegmenter
from .wake import WakeWordMatcher


class VoiceAssistant:
    def __init__(self, cfg, handler: Callable[[str], object], stt: Optional[SpeechToText] = None,
                 tts: Optional[TextToSpeech] = None, matcher: Optional[WakeWordMatcher] = None,
                 mic: Optional[MicStream] = None, clock: Callable[[], float] = time.monotonic,
                 log: Callable[[str], None] = print):
        self.cfg = cfg
        self.v = cfg.section("voice")
        self.handler = handler
        self.stt, self.tts = stt, tts
        wake = self.v.get("wake", {}) or {}
        self.matcher = matcher or WakeWordMatcher(wake.get("words", []), float(wake.get("sensitivity", 0.88)),
                                                  self.v.get("stop_phrases", []))
        self.mic = mic
        self._clock = clock
        self.log = log
        self._awake_until = 0.0
        self.speak_replies = bool(self.v.get("speak_replies", True))
        self._lock = threading.Lock()
        self.status = {"state": "starting", "last_heard": None, "last_reply": None, "utterances": 0, "woke": 0, "error": None}

    # -- state ------------------------------------------------------------------------------------
    def _set(self, **kw) -> None:
        with self._lock:
            self.status.update(kw)

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self.status)

    def _speak(self, text: str) -> None:
        if not (self.speak_replies and self.tts and text):
            return
        self._set(state="speaking")
        try:
            self.tts.speak(text)
        except VoiceUnavailable as e:
            self.log(f"[voice] cannot speak: {e}")
            self.speak_replies = False
        finally:
            if self.mic is not None:
                self.mic.drain()          # do not hear our own voice
            self._set(state="awake" if self._clock() < self._awake_until else "listening")

    def stop_speaking(self) -> None:
        if self.tts:
            self.tts.stop()

    # -- the whole conversation state machine ----------------------------------------------------------
    def process_utterance(self, pcm: bytes) -> Optional[str]:
        """Handle one finished utterance. Returns the reply text if one was produced (for tests/logs)."""
        if self.stt is None:
            raise VoiceUnavailable("no speech-to-text engine configured")
        try:
            text = self.stt.transcribe(pcm, int((self.v.get("stt", {}) or {}).get("sample_rate", 16000)))
        except Exception as e:  # noqa: BLE001 - one bad utterance must not end the loop
            self.log(f"[voice] speech recognition failed on an utterance: {type(e).__name__}: {e}")
            return None
        text = text.strip()
        self._set(utterances=self.status["utterances"] + 1)
        if not text:
            return None

        now = self._clock()
        awake = now < self._awake_until

        if (awake or (self.tts and self.tts.is_speaking)) and self.matcher.is_stop(text):
            self.stop_speaking()
            self._awake_until = 0.0
            self._set(state="listening", last_heard=text)
            return None

        wake = self.matcher.detect(text)
        if wake is not None:
            self._set(woke=self.status["woke"] + 1, last_heard=text)
            if not wake.remainder:
                self._awake_until = now + float((self.v.get("wake", {}) or {}).get("listen_timeout_s", 8))
                self._set(state="awake")
                ack = str((self.v.get("wake", {}) or {}).get("ack_phrase", "") or "")
                self.log(f"[voice] woke on {wake.word!r} (score {wake.score}); waiting for a command")
                self._speak(ack)
                self._awake_until = self._clock() + float((self.v.get("wake", {}) or {}).get("listen_timeout_s", 8))
                return ack or None
            return self._dispatch(wake.remainder)
        if awake:
            return self._dispatch(text)
        return None                       # not for us: dropped

    def _dispatch(self, command: str) -> Optional[str]:
        self._set(state="thinking", last_heard=command)
        self.log(f"[voice] heard: {command!r}")
        try:
            reply = self.handler(command)
            text = getattr(reply, "text", str(reply)) if reply is not None else ""
        except Exception as e:  # noqa: BLE001
            self.log(f"[voice] handler failed: {type(e).__name__}: {e}")
            text = "Sorry, something went wrong."
        self._set(last_reply=text)
        self._speak(text)
        self._awake_until = self._clock() + float((self.v.get("wake", {}) or {}).get("follow_up_s", 6))
        self._set(state="awake")
        return text

    # -- running against real hardware -------------------------------------------------------------------
    def run(self, stop: threading.Event) -> None:
        stt_cfg = self.v.get("stt", {}) or {}
        if self.stt is None:
            self.stt = build_stt(self.cfg)               # raises VoiceUnavailable with install steps
        if self.tts is None and self.speak_replies:
            try:
                self.tts = build_tts(self.cfg)
            except VoiceUnavailable as e:
                self.log(f"[voice] replies will be text only: {e}")
                self.speak_replies = False
        vad = self.v.get("vad", {}) or {}
        rate = int(stt_cfg.get("sample_rate", 16000))
        seg = UtteranceSegmenter(sample_rate=rate, silence_ms=int(vad.get("silence_ms", 800)),
                                 max_utterance_s=float(vad.get("max_utterance_s", 15)),
                                 min_speech_ms=int(vad.get("min_speech_ms", 200)),
                                 threshold_factor=float(vad.get("threshold_factor", 3.0)))
        mic = self.mic or MicStream(rate, 30, stt_cfg.get("input_device"))
        self.mic = mic
        words = ", ".join(repr(w) for w in (self.v.get("wake", {}) or {}).get("words", []))
        with mic:
            self.log(f"[voice] listening ({self.stt.name}); say one of: {words}")
            self._set(state="listening", error=None)
            for frame in mic.frames(stop):
                pcm = seg.feed(frame)
                if pcm is not None:
                    self.process_utterance(pcm)
        self._set(state="stopped")


def test_microphone(cfg, seconds: float = 30.0, log: Callable[[str], None] = print,
                    stop: Optional[threading.Event] = None) -> None:
    """`voice --test-mic`: listen for a while and print what the recogniser wrote down for each
    utterance and whether it counts as a wake word. Acts on nothing and speaks nothing, so it is
    the way to (a) check the microphone and speech model work and (b) discover how your engine
    spells the wake word, then add that spelling to voice.wake.words."""
    stt_cfg = cfg.section("voice").get("stt", {}) or {}
    wake = cfg.section("voice").get("wake", {}) or {}
    stt = build_stt(cfg)
    matcher = WakeWordMatcher(wake.get("words", []), float(wake.get("sensitivity", 0.88)),
                              cfg.section("voice").get("stop_phrases", []))
    vad = cfg.section("voice").get("vad", {}) or {}
    rate = int(stt_cfg.get("sample_rate", 16000))
    seg = UtteranceSegmenter(sample_rate=rate, silence_ms=int(vad.get("silence_ms", 800)),
                             max_utterance_s=float(vad.get("max_utterance_s", 15)),
                             min_speech_ms=int(vad.get("min_speech_ms", 200)),
                             threshold_factor=float(vad.get("threshold_factor", 3.0)))
    stop = stop or threading.Event()
    deadline = time.monotonic() + seconds
    log(f"[voice] listening for {seconds:.0f}s with {stt.name}. Stay quiet for the first second (it measures the room), "
        f"then speak. Wake words: {', '.join(wake.get('words', []))}")
    with MicStream(rate, 30, stt_cfg.get("input_device")) as mic:
        for frame in mic.frames(stop):
            if time.monotonic() > deadline:
                break
            pcm = seg.feed(frame)
            if pcm is None:
                continue
            text = stt.transcribe(pcm, rate).strip()
            if not text:
                log("[voice] (speech detected, nothing recognised)")
                continue
            w = matcher.detect(text)
            verdict = (f"WAKE ({w.word!r}, score {w.score}) then: {w.remainder!r}" if w
                       else "not a wake word" + (" (stop phrase)" if matcher.is_stop(text) else ""))
            log(f"[voice] heard {text!r} -> {verdict}")
    log(f"[voice] done. Room noise floor was about {seg.noise_floor:.0f}; speech threshold {seg.threshold:.0f}.")
