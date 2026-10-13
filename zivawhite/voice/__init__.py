"""Voice: microphone -> voice-activity detection -> offline speech-to-text -> wake word ->
assistant -> text-to-speech. Every heavy dependency (sounddevice, vosk, faster-whisper, pyttsx3)
is imported lazily and optional; missing ones produce a VoiceUnavailable with install steps."""
from .errors import VoiceUnavailable
from .wake import WakeResult, WakeWordMatcher

__all__ = ["VoiceUnavailable", "WakeResult", "WakeWordMatcher"]
