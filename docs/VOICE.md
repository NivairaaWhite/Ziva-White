<!-- Moved verbatim out of the top-level README, which had grown past 45 KB. Back to the [README](../README.md). -->
Offline voice: microphone, wake words, speech.

# Voice, wake words

Pipeline: microphone → voice-activity detection → **offline** speech-to-text → wake word → assistant →
text-to-speech. Only utterances that *begin* with a wake word are acted on; everything else is dropped
without leaving the process.

**Setup (one manual step: a speech model):**

```bash
pip install -r requirements-voice.txt
# Linux also: sudo apt install libportaudio2 espeak-ng
# download a Vosk model from https://alphacephei.com/vosk/models (vosk-model-small-en-us is ~40 MB),
# unpack it so that models/vosk/model/ contains its files, then:
python -m zivawhite.app voice --devices       # pick your microphone (set voice.stt.input_device)
python -m zivawhite.app voice --say "hello"   # test the speaker
python -m zivawhite.app voice --test-mic      # see what it hears and which words count as a wake word
python -m zivawhite.app voice                 # talk to it; Ctrl+C to stop
```

To have it listen all the time, set `voice.enabled: true`; the daemon then keeps the voice loop alive.

- **Wake words** are `voice.wake.words` (defaults: "hey ziva", "ziva", "hey snow white", "snow white").
  Offline recognisers have never heard a made-up name and write down what it sounds like, so matching
  scores both spelling and a rough phonetic key ("siva", "zeva" and "shiva" wake it; "diva" and "eva" don't).
  If yours keeps writing something else, run `--test-mic`, see the spelling, and add it to the list.
  `voice.wake.sensitivity` is the minimum match score.
- **Conversation:** "hey Ziva, what time is it" works in one breath; "hey Ziva" alone gets the `ack_phrase`
  and the next utterance is the command; after a reply you can keep talking without the wake word for
  `follow_up_s`. "stop" / "be quiet" silences speech.
- **Half-duplex by design:** it doesn't listen while it speaks (so it can't wake itself), which means you
  can't interrupt by talking over it; use the Stop control in the web UI.
- **Engines:** speech-to-text `vosk` (small, always-on friendly) or `faster_whisper` (more accurate,
  heavier). Speech output uses your OS voice (Windows SAPI, macOS `say`, Linux spd-say/espeak), `pyttsx3`, or
  your own command (`voice.tts.engine: custom`, e.g. a neural voice).
- **In the browser** the web UI has a mic button, a hands-free mode with the same wake words, and spoken
  replies. That uses your browser's speech recognition, which in Chrome/Edge sends audio to the vendor's
  cloud. It is convenient, not private or offline; use the local voice loop for that.
