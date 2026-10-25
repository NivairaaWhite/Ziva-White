"""
voice/tts.py - text-to-speech.

Engines (voice.tts.engine):
  system    the OS's own voice, driven as a child process so it can be interrupted instantly:
              Windows  SAPI via PowerShell      macOS  `say`
              Linux    spd-say / espeak-ng / espeak / festival, whichever is installed
  pyttsx3   cross-platform Python wrapper (pip install pyttsx3)
  custom    your own command, e.g. a neural voice: ["piper", "--model", "m.onnx", "--output-raw"] -- the
            text replaces a "{text}" element in the argument list (never a shell string)
  auto      system if one exists, else pyttsx3

Text is sanitised before it is spoken (markdown, code blocks and URLs read aloud are noise) and
spoken sentence by sentence, so `stop()` takes effect at the next sentence boundary at worst and
immediately for the system engines.
"""
from __future__ import annotations

import re
import subprocess
import threading
from typing import List, Optional, Sequence, Tuple

from ..control.system import System
from .errors import VoiceUnavailable

_CODE_BLOCK = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`]*)`")
_URL = re.compile(r"https?://\S+")
_MD_MARKS = re.compile(r"[*_#>~|]+")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def speakable(text: str, max_chars: int = 700) -> str:
    t = _CODE_BLOCK.sub(" (code omitted) ", text)
    t = _LINK.sub(r"\1", t)
    t = _URL.sub("a link", t)
    t = _INLINE_CODE.sub(r"\1", t)
    t = _MD_MARKS.sub(" ", t)
    t = _EMOJI.sub("", t)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > max_chars:                       # long answers: speak the start, the screen has the rest
        cut = t[:max_chars]
        end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
        t = (cut[:end + 1] if end > max_chars // 2 else cut) + " ..."
    return t


def sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s and s.strip()]


SPEAK_TIMEOUT_S = 120.0     # no single sentence takes longer than this to say; a command still running is hung


def _ps_quote(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


class Engine:
    name = "engine"

    def speak(self, text: str) -> None:        # blocking
        raise NotImplementedError

    def stop(self) -> None:
        pass


class SystemEngine(Engine):
    name = "system"

    def __init__(self, system: System, rate_wpm: int = 175, voice: Optional[str] = None):
        self.sys, self.rate, self.voice = system, int(rate_wpm), voice
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def command(self) -> Optional[Tuple[List[str], str]]:
        """(argv, how) where text is sent on stdin, or None if this OS has no usable voice."""
        if self.sys.os == "windows":
            sapi_rate = max(-10, min(10, round((self.rate - 175) / 25)))
            script = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                      f"$s.Rate = {sapi_rate}; ")
            if self.voice:
                script += f"$s.SelectVoice({_ps_quote(self.voice)}); "
            script += "$s.Speak([Console]::In.ReadToEnd())"
            return ["powershell", "-NoProfile", "-NonInteractive", "-Command", script], "stdin"
        if self.sys.os == "macos":
            return ["say", "-r", str(self.rate), *(["-v", self.voice] if self.voice else [])], "stdin"
        if self.sys.which("spd-say"):
            return ["spd-say", "-e", "-w", "-r", str(max(-100, min(100, round((self.rate - 175) / 1.5))))], "stdin"
        for exe in ("espeak-ng", "espeak"):
            if self.sys.which(exe):
                return [exe, "-s", str(self.rate), *(["-v", self.voice] if self.voice else []), "--stdin"], "stdin"
        if self.sys.which("festival"):
            return ["festival", "--tts"], "stdin"
        return None

    def available(self) -> bool:
        return self.command() is not None

    def speak(self, text: str) -> None:
        cmd = self.command()
        if cmd is None:
            raise VoiceUnavailable("No system voice found. On Linux install one: `sudo apt install espeak-ng`.")
        argv, _ = cmd
        with self._lock:
            self._proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            proc = self._proc
        try:
            proc.communicate(text.encode("utf-8"), timeout=SPEAK_TIMEOUT_S)
        except (subprocess.TimeoutExpired, BrokenPipeError, OSError):
            proc.kill()
        finally:
            with self._lock:
                self._proc = None

    def stop(self) -> None:
        with self._lock:
            proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()


class Pyttsx3Engine(Engine):
    name = "pyttsx3"

    def __init__(self, rate_wpm: int = 175, voice: Optional[str] = None):
        try:
            import pyttsx3
        except ImportError as e:
            raise VoiceUnavailable("This voice engine needs `pip install pyttsx3`.") from e
        try:
            self._engine = pyttsx3.init()
        except Exception as e:  # noqa: BLE001 - driver init failures are platform specific
            raise VoiceUnavailable(f"pyttsx3 could not start a speech driver: {e}") from e
        self._engine.setProperty("rate", int(rate_wpm))
        if voice:
            for v in self._engine.getProperty("voices"):
                if voice.lower() in (v.name or "").lower() or voice == v.id:
                    self._engine.setProperty("voice", v.id)
                    break

    def speak(self, text: str) -> None:
        self._engine.say(text)
        self._engine.runAndWait()

    def stop(self) -> None:
        try:
            self._engine.stop()
        except Exception:  # noqa: BLE001
            pass


class CustomEngine(Engine):
    name = "custom"

    def __init__(self, template: Sequence[str], timeout_s: float = SPEAK_TIMEOUT_S):
        if not template or not any("{text}" in str(a) for a in template):
            raise VoiceUnavailable('voice.tts.command must be an argument list containing "{text}", e.g. ["my-tts", "{text}"].')
        self.template = [str(a) for a in template]
        self.timeout_s = timeout_s
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def argv_for(self, text: str) -> List[str]:
        """The command line for `text`. An argument that STARTS with the text must not start with "-": the program
        would read spoken words like "-5 degrees" as one of its own options (and could be made to do something
        else by text it was handed). A leading space is inaudible and removes the problem for every program, which
        `--` would not (not all of them understand it)."""
        safe = f" {text}" if text.startswith("-") else text
        return [a.replace("{text}", safe if a.startswith("{text}") else text) for a in self.template]

    def speak(self, text: str) -> None:
        argv = self.argv_for(text)
        try:
            with self._lock:
                self._proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                              stderr=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                proc = self._proc
        except OSError as e:
            raise VoiceUnavailable(f"voice.tts.command could not run: {e}") from e
        try:
            proc.wait(timeout=self.timeout_s)
        except subprocess.TimeoutExpired:
            # a hung command must not hold the speaking lock (and so every later reply) forever
            proc.kill()
            proc.wait()
        finally:
            with self._lock:
                self._proc = None

    def stop(self) -> None:
        with self._lock:
            p = self._proc
        if p is not None and p.poll() is None:
            p.terminate()


class TextToSpeech:
    """The facade the rest of the project uses: sanitise, split, speak, interrupt."""

    def __init__(self, engine: Engine):
        self.engine = engine
        self._stop = threading.Event()
        self._speaking = threading.Lock()
        self.is_speaking = False

    def speak(self, text: str) -> bool:
        """Speak `text` (blocking). Returns False if it was interrupted. Calls from different
        threads queue up behind each other instead of talking over one another."""
        clean = speakable(text)
        if not clean:
            return True
        with self._speaking:
            self._stop.clear()
            self.is_speaking = True
            try:
                for sentence in sentences(clean):
                    if self._stop.is_set():
                        return False
                    self.engine.speak(sentence)
                return not self._stop.is_set()
            finally:
                self.is_speaking = False

    def stop(self) -> None:
        self._stop.set()
        self.engine.stop()


def build_tts(cfg, system: Optional[System] = None) -> TextToSpeech:
    tts = cfg.section("voice").get("tts", {}) or {}
    engine_name = tts.get("engine", "auto")
    rate, voice = int(tts.get("rate", 175)), tts.get("voice")
    system = system or System()
    if engine_name == "custom":
        return TextToSpeech(CustomEngine(tts.get("command") or [], timeout_s=float(tts.get("timeout_s", SPEAK_TIMEOUT_S))))
    if engine_name in ("auto", "system"):
        eng = SystemEngine(system, rate, voice)
        if eng.available():
            return TextToSpeech(eng)
        if engine_name == "system":
            raise VoiceUnavailable("No system voice found. On Linux install one: `sudo apt install espeak-ng`.")
    return TextToSpeech(Pyttsx3Engine(rate, voice))
