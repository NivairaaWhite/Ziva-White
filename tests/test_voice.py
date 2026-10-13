"""Voice: wake words, voice-activity detection, speech sanitising, TTS command construction and the
whole conversation state machine -- with fake STT/TTS and synthetic audio, so no microphone or
speaker is needed."""
import threading

import numpy as np
import pytest
from helpers import make_cfg

from zivawhite.voice.loop import VoiceAssistant
from zivawhite.voice.tts import (CustomEngine, SystemEngine, TextToSpeech, build_tts, sentences, speakable)
from zivawhite.voice.vad import UtteranceSegmenter
from zivawhite.voice.wake import WakeWordMatcher, phonetic_key, similarity
from zivawhite.voice.errors import VoiceUnavailable

WORDS = ["hey ziva", "ziva", "hey snow white", "snow white"]


def matcher(sens=0.88):
    return WakeWordMatcher(WORDS, sens, ["stop", "never mind", "be quiet"])


# ------------------------------------------------------------------------------------- wake
@pytest.mark.parametrize("heard,word,rest", [
    ("hey ziva what time is it", "hey ziva", "what time is it"),
    ("Ziva", "ziva", ""),
    ("hey, Ziva!", "hey ziva", ""),
    ("hey siva turn up the volume", "hey ziva", "turn up the volume"),      # phonetic variants an offline recogniser writes
    ("zeva open calculator", "ziva", "open calculator"),
    ("hey shiva what time is it", "hey ziva", "what time is it"),
    ("heyziva play music", "hey ziva", "play music"),
    ("hey snow white, who created you?", "hey snow white", "who created you"),
    ("snow white", "snow white", ""),
    ("okay hey ziva lock the screen", "hey ziva", "lock the screen"),     # up to two filler words before it
])
def test_wake_word_detected_with_the_command_that_follows(heard, word, rest):
    r = matcher().detect(heard)
    assert r is not None and r.word == word and r.remainder == rest


@pytest.mark.parametrize("heard", [
    "what time is it", "I was talking to Ziva yesterday about the plan", "hey diva", "hey zebra", "the viva exam",
    "give a", "hey eva", "", "   ", "um uh er",
])
def test_things_that_are_not_wake_words_are_ignored(heard):
    assert matcher().detect(heard) is None


def test_sensitivity_trades_recall_for_false_wakes():
    assert matcher(0.80).detect("hey diva") is not None        # sloppy: wakes on near-misses
    assert matcher(0.88).detect("hey diva") is None


def test_wake_words_come_from_the_caller_not_the_code():
    m = WakeWordMatcher(["computer"], 0.88)
    assert m.detect("computer lights") is not None and m.detect("hey ziva") is None
    with pytest.raises(ValueError):
        WakeWordMatcher([], 0.9)
    with pytest.raises(ValueError):
        WakeWordMatcher(["   "], 0.9)


def test_stop_phrases_and_phonetics():
    m = matcher()
    assert m.is_stop("Stop!") and m.is_stop("never mind") and not m.is_stop("stop the music please")
    assert phonetic_key("ziva") == phonetic_key("siva") == phonetic_key("zeva")
    assert similarity("ziva", "banana") < 0.5


# --------------------------------------------------------------------------------------- VAD
def tone(n, amp, rate=16000, freq=220.0):
    t = np.arange(n) / rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.int16)


def noise(n, amp, seed=0):
    return (np.random.default_rng(seed).standard_normal(n) * amp).astype(np.int16)


def feed_all(seg, frames):
    out = []
    for f in frames:
        r = seg.feed(f)
        if r is not None:
            out.append(r)
    return out


def test_vad_cuts_one_utterance_out_of_a_noisy_room():
    seg = UtteranceSegmenter()
    n = seg.frame_samples
    room = [noise(n, 30, i) for i in range(30)]                    # ~0.9 s of quiet to calibrate
    speech = [tone(n, 3000) for _ in range(25)]                    # 750 ms of speech
    quiet = [noise(n, 30, 100 + i) for i in range(40)]             # > 800 ms of silence ends it
    got = feed_all(seg, room + speech + quiet)
    assert len(got) == 1
    frames_out = len(got[0]) // 2 / n
    # 25 frames of speech + 10 of pre-roll + 26 of trailing silence (800 ms / 30 ms) -- not the 95-frame stream
    assert 25 <= frames_out <= 25 + 10 + 26 + 1


def test_vad_keeps_the_start_of_the_first_word():
    seg = UtteranceSegmenter(preroll_ms=300)
    n = seg.frame_samples
    quiet = [noise(n, 30, i) for i in range(30)]
    marker = [np.full(n, 90, dtype=np.int16)]                       # below threshold: lands in the pre-roll
    got = feed_all(seg, quiet + marker + [tone(n, 3000)] * 20 + [noise(n, 30, 7)] * 40)
    pcm = np.frombuffer(got[0], dtype=np.int16)
    assert (pcm[:n] != 0).any() and 90 in pcm[:10 * n]


def test_vad_ignores_clicks_shorter_than_min_speech():
    seg = UtteranceSegmenter(min_speech_ms=200)
    n = seg.frame_samples
    got = feed_all(seg, [noise(n, 30, i) for i in range(30)] + [tone(n, 4000)] * 2 + [noise(n, 30, 9)] * 40)
    assert got == []


def test_vad_two_utterances_and_max_length_cut():
    seg = UtteranceSegmenter(max_utterance_s=1.0)
    n = seg.frame_samples
    room = [noise(n, 30, i) for i in range(30)]
    got = feed_all(seg, room + [tone(n, 3000)] * 100 + [noise(n, 30, 5)] * 40)     # 3 s of speech -> cut at 1 s
    assert len(got) >= 2
    assert all(len(g) // 2 <= int(1.0 * 16000) + n for g in got)


def test_vad_adapts_to_a_louder_room_but_not_to_speech():
    seg = UtteranceSegmenter()
    n = seg.frame_samples
    feed_all(seg, [noise(n, 30, i) for i in range(30)])
    quiet_floor = seg.noise_floor
    feed_all(seg, [noise(n, 120, i) for i in range(60)])                              # a fan turns on
    assert seg.noise_floor > quiet_floor
    floor_before_speech = seg.noise_floor
    feed_all(seg, [tone(n, 8000)] * 10)
    assert seg.noise_floor == floor_before_speech                                      # speech never raises the floor


# --------------------------------------------------------------------------------- speaking
def test_speech_sanitising():
    assert speakable("**Bold** and `code` and https://example.com/x?y=1 here") == "Bold and code and a link here"
    assert speakable("See [the docs](https://example.com) now") == "See the docs now"
    assert "code omitted" in speakable("before ```python\nprint(1)\n``` after")
    assert speakable("   ") == ""
    long = "Sentence number one is here. " * 60
    out = speakable(long, max_chars=100)
    assert len(out) <= 110 and out.endswith("...")


def test_sentence_splitting():
    assert sentences("One. Two! Three?\nFour") == ["One.", "Two!", "Three?", "Four"]


class FakeSys:
    def __init__(self, os_name, have=()): self.os, self.have = os_name, set(have)
    def which(self, n): return n if n in self.have else None


def test_system_engine_command_per_os():
    win = SystemEngine(FakeSys("windows"), 175, "Zira").command()[0]
    assert win[0] == "powershell" and "$s.Rate = 0" in win[-1] and "SelectVoice('Zira')" in win[-1]
    assert "ReadToEnd" in win[-1]                                                      # text arrives on stdin, never in the script
    assert SystemEngine(FakeSys("macos"), 200, "Samantha").command()[0] == ["say", "-r", "200", "-v", "Samantha"]
    assert SystemEngine(FakeSys("linux", ["espeak-ng", "espeak"]), 150).command()[0] == ["espeak-ng", "-s", "150", "--stdin"]
    assert SystemEngine(FakeSys("linux", ["spd-say", "espeak-ng"])).command()[0][0] == "spd-say"
    assert SystemEngine(FakeSys("linux", [])).command() is None
    assert not SystemEngine(FakeSys("linux", [])).available()


def test_quote_in_voice_name_cannot_break_out_of_the_powershell_script():
    script = SystemEngine(FakeSys("windows"), 175, "x'; calc; '").command()[0][-1]
    assert "SelectVoice('x''; calc; ''')" in script


def test_custom_engine_validation():
    with pytest.raises(VoiceUnavailable):
        CustomEngine([])
    with pytest.raises(VoiceUnavailable):
        CustomEngine(["my-tts"])
    CustomEngine(["my-tts", "{text}"])


def test_build_tts_reports_a_missing_voice_instead_of_crashing(tmp_path):
    cfg = make_cfg(tmp_path, voice={"tts": {"engine": "system"}})
    with pytest.raises(VoiceUnavailable):
        build_tts(cfg, system=FakeSys("linux", []))


class FakeEngine:
    def __init__(self): self.said, self.stopped = [], 0
    def speak(self, text): self.said.append(text)
    def stop(self): self.stopped += 1


def test_tts_speaks_sentence_by_sentence_and_can_be_interrupted():
    eng = FakeEngine()
    tts = TextToSpeech(eng)
    assert tts.speak("**One.** Two. Three.") is True and eng.said == ["One.", "Two.", "Three."]

    class StoppingEngine(FakeEngine):
        def speak(self, text):
            super().speak(text)
            if len(self.said) == 1:          # the person interrupts during the first sentence only
                tts2.stop()
    eng2 = StoppingEngine()
    tts2 = TextToSpeech(eng2)
    assert tts2.speak("One. Two. Three.") is False and eng2.said == ["One."]          # stopped at the boundary
    assert tts2.speak("Again.") is True                                                 # and usable afterwards


# ------------------------------------------------------------------------ the whole loop
class FakeSTT:
    name = "fake"
    def __init__(self, script): self.script = list(script)
    def transcribe(self, pcm, rate=16000):
        item = self.script.pop(0)
        if isinstance(item, Exception): raise item
        return item


class FakeTTS:
    def __init__(self): self.said, self.is_speaking, self.stops = [], False, 0
    def speak(self, text): self.said.append(text); return True
    def stop(self): self.stops += 1


class Clock:
    def __init__(self): self.t = 100.0
    def __call__(self): return self.t


def build(tmp_path, heard, handler=None, **voice):
    cfg = make_cfg(tmp_path, voice=voice or {})
    replies = []
    def default(cmd):
        replies.append(cmd)
        return type("R", (), {"text": f"did: {cmd}"})()
    clock, tts = Clock(), FakeTTS()
    va = VoiceAssistant(cfg, handler or default, stt=FakeSTT(heard), tts=tts, clock=clock, log=lambda *_: None)
    return va, tts, clock, replies


def say(va):
    return va.process_utterance(b"\x00\x00")


def test_not_awake_means_ignored(tmp_path):
    va, tts, _, replies = build(tmp_path, ["what time is it", "i love this song"])
    assert say(va) is None and say(va) is None
    assert replies == [] and tts.said == []                     # nothing was acted on or spoken


def test_wake_word_plus_command_in_one_breath(tmp_path):
    va, tts, _, replies = build(tmp_path, ["hey ziva what time is it"])
    assert say(va) == "did: what time is it"
    assert replies == ["what time is it"] and tts.said == ["did: what time is it"]


def test_wake_word_alone_acknowledges_then_takes_the_next_utterance(tmp_path):
    va, tts, clock, replies = build(tmp_path, ["hey ziva", "turn up the volume"])
    say(va)
    assert tts.said == ["Yes?"] and replies == []
    clock.t += 3
    say(va)
    assert replies == ["turn up the volume"] and tts.said[-1] == "did: turn up the volume"


def test_the_listening_window_closes(tmp_path):
    va, tts, clock, replies = build(tmp_path, ["hey ziva", "turn up the volume"])
    say(va)
    clock.t += 60                                                # never answered the "Yes?"
    assert say(va) is None and replies == []


def test_follow_up_without_the_wake_word_then_it_expires(tmp_path):
    va, _, clock, replies = build(tmp_path, ["hey ziva hello", "and another thing", "too late"])
    say(va)
    clock.t += 3
    say(va)                                                      # inside follow_up_s: no wake word needed
    clock.t += 30
    assert say(va) is None
    assert replies == ["hello", "and another thing"]


def test_silent_ack_when_configured_empty(tmp_path):
    va, tts, _, _ = build(tmp_path, ["ziva"], wake={"ack_phrase": ""})
    assert say(va) is None and tts.said == []


def test_stop_phrase_silences_and_ends_the_conversation(tmp_path):
    va, tts, clock, replies = build(tmp_path, ["hey ziva", "stop", "play music"])
    say(va)
    say(va)
    assert tts.stops == 1
    assert say(va) is None and replies == []                     # no longer awake


def test_stop_does_nothing_when_idle(tmp_path):
    va, tts, _, _ = build(tmp_path, ["stop"])
    say(va)
    assert tts.stops == 0


def test_speech_can_be_switched_off(tmp_path):
    va, tts, _, replies = build(tmp_path, ["hey ziva hello"], speak_replies=False)
    say(va)
    assert replies == ["hello"] and tts.said == []


def test_a_crashing_handler_or_recogniser_does_not_end_the_loop(tmp_path):
    def boom(cmd): raise RuntimeError("model exploded")
    va, tts, _, _ = build(tmp_path, ["hey ziva hello"], handler=boom)
    assert say(va) == "Sorry, something went wrong."

    va, _, _, replies = build(tmp_path, [RuntimeError("bad audio"), "hey ziva hi"])
    assert say(va) is None
    assert say(va) == "did: hi"


def test_status_reflects_what_happened(tmp_path):
    va, _, _, _ = build(tmp_path, ["hey ziva what time is it"])
    say(va)
    st = va.snapshot()
    assert st["utterances"] == 1 and st["woke"] == 1 and st["last_heard"] == "what time is it" and st["last_reply"] == "did: what time is it"


def test_run_with_fake_mic_cuts_utterances_and_stops(tmp_path):
    class FakeMic:
        def __init__(self, frames): self._frames, self.drained = frames, 0
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def drain(self): self.drained += 1
        def frames(self, stop):
            for f in self._frames:
                yield f
            stop.set()
    cfg = make_cfg(tmp_path)
    n = 480
    frames = [noise(n, 30, i) for i in range(30)] + [tone(n, 3000)] * 25 + [noise(n, 30, 99)] * 40
    heard = []
    va = VoiceAssistant(cfg, lambda c: heard.append(c) or type("R", (), {"text": "ok"})(), stt=FakeSTT(["hey ziva volume up"]),
                        tts=FakeTTS(), mic=FakeMic(frames), log=lambda *_: None)
    stop = threading.Event()
    va.run(stop)
    assert heard == ["volume up"] and va.snapshot()["state"] == "stopped" and va.mic.drained == 1
