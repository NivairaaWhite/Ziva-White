"""
assistant.py - what happens to one thing the person says or types.

    1. A pending "are you sure?" gets its yes/no first.
    2. Questions about the assistant itself ("who created you?") are answered from the config.
    3. Device-control commands ("volume up", "open calculator", "set a timer") are matched by the
       fixed grammar in control/intents.py -- never by the language model.
    4. Everything else goes to the model (router.ask: local model and/or allow-listed web).

Voice and text take exactly the same path (voice just adds a microphone in front and a speaker
behind), so anything you can say you can also type, and the tests exercise one code path.

Safety properties that live here:
  * a command that needs confirmation (lock / sleep / restart / shutdown) is parked per session
    and only runs after a "yes" within `device_control.confirm_ttl_s`; any other message cancels it.
    The "yes" is checked AGAIN against `allow_control`: the confirmation guards against mishearing, it is
    not an access check, so whoever says "yes" must independently be allowed to control the device;
  * `allow_control=False` (used by the web server for connections that are not local or
    authenticated) means the grammar is recognised but refused, rather than silently handed to the
    model, so the person is told why nothing happened.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Tuple

from .config import Config
from .control import DeviceController, Intent, parse_intent
from .control.timers import Timer
from .errors import ModelBusy
from .identity import Identity


@dataclass
class Reply:
    text: str
    kind: str                              # identity | control | confirm | model | error | ignored
    ok: bool = True
    action: Optional[str] = None           # the control action, when kind is control/confirm
    route: Optional[str] = None            # local | web, when kind is model
    sources: List[str] = field(default_factory=list)
    reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v not in (None, [])} | {"text": self.text}


class Assistant:
    def __init__(self, cfg: Config, ask: Optional[Callable[[str, str], object]] = None, identity: Optional[Identity] = None,
                 controller: Optional[DeviceController] = None, speaker=None, clock: Callable[[], float] = time.monotonic):
        self.cfg = cfg
        self._ask = ask
        self.identity = identity or Identity.from_config(cfg)
        self.controller = controller
        self.speaker = speaker                          # a voice.tts.TextToSpeech, or None
        self._clock = clock
        self._pending: Dict[str, Tuple[Intent, float]] = {}
        self._lock = threading.Lock()
        self._events: Deque[dict] = deque(maxlen=100)
        self._event_id = 0
        self._speaker_failed = False

    # -- main entry ----------------------------------------------------------------------------
    def handle(self, text: str, session: str = "default", allow_control: bool = True, route: str = "auto") -> Reply:
        text = (text or "").strip()
        if not text:
            return Reply("", "ignored")

        pending = self._take_pending(session)
        if pending is not None:
            intent = parse_intent(text)
            if intent and intent.action == "confirm":
                if not allow_control:       # the pending command was parked by someone else, or under looser rules
                    return Reply("I can only control this device from a local or authenticated connection.", "control",
                                 ok=False, action=pending.action)
                return self._run(pending, confirmed=True)
            if intent and intent.action == "deny":
                return Reply("Okay, cancelled.", "control", action="cancelled")
            # anything else: the question lapses and the new message is handled normally

        answer = self.identity.answer(text)
        if answer is not None:
            return Reply(answer, "identity")

        intent = parse_intent(text) if self.controller is not None else None
        if intent is not None and intent.action not in ("confirm", "deny"):
            if not allow_control:
                return Reply("I can only control this device from a local or authenticated connection.", "control",
                             ok=False, action=intent.action)
            why = self.controller.refusal(intent)
            if why:
                return Reply(why, "control", ok=False, action=intent.action)
            if self.controller.needs_confirmation(intent):
                ttl = float(self.cfg.get("device_control.confirm_ttl_s", 20))
                with self._lock:
                    self._pending[session] = (intent, self._clock() + ttl)
                return Reply(f"Do you want me to {self.controller.describe(intent)}? Say yes to confirm.", "confirm",
                             action=intent.action)
            return self._run(intent)

        return self._model(text, route)

    # -- parts -------------------------------------------------------------------------------------
    def _take_pending(self, session: str) -> Optional[Intent]:
        with self._lock:
            item = self._pending.pop(session, None)
        if item is None:
            return None
        intent, expires = item
        return intent if self._clock() <= expires else None

    def _run(self, intent: Intent, confirmed: bool = False) -> Reply:
        result = self.controller.execute(intent, confirmed=confirmed)
        return Reply(result.message, "control", ok=result.ok, action=intent.action)

    def _model(self, text: str, route: str) -> Reply:
        if self._ask is None:
            return Reply("No language model is available right now.", "error", ok=False)
        try:
            ans = self._ask(text, route)
        except ModelBusy:
            return Reply("I'm still working on another request. Ask me again in a moment.", "error", ok=False)
        except Exception as e:  # noqa: BLE001 - the model failing must not take the assistant down
            return Reply(f"Something went wrong while thinking about that ({type(e).__name__}).", "error", ok=False)
        return Reply(getattr(ans, "text", str(ans)), "model", route=getattr(ans, "route", None),
                     sources=list(getattr(ans, "sources", []) or []), reason=getattr(ans, "reason", None))

    # -- notifications / background events -------------------------------------------------------------
    def add_event(self, kind: str, text: str) -> dict:
        with self._lock:
            self._event_id += 1
            ev = {"id": self._event_id, "ts": time.time(), "kind": kind, "text": text}
            self._events.append(ev)
        return ev

    def events_since(self, last_id: int = 0) -> List[dict]:
        with self._lock:
            return [e for e in self._events if e["id"] > last_id]

    def get_speaker(self):
        """The one shared TextToSpeech (built on first use), so a timer announcement can never talk
        over a voice reply. None when this machine has no usable voice."""
        if self.speaker is not None or self._speaker_failed:
            return self.speaker
        try:
            from .voice.tts import build_tts

            self.speaker = build_tts(self.cfg)
        except Exception:  # noqa: BLE001 - no voice available: notifications stay on screen only
            self._speaker_failed = True
        return self.speaker

    def notify(self, text: str, kind: str = "notification", speak: Optional[bool] = None) -> None:
        """Surface something that happened in the background (a timer firing): it goes in the event
        feed the web UI polls, and is spoken if speech is enabled."""
        self.add_event(kind, text)
        if speak is None:
            speak = bool(self.cfg.get("voice.speak_replies", True))
        if speak:
            sp = self.get_speaker()
            if sp is not None:
                try:
                    sp.speak(text)
                except Exception:  # noqa: BLE001
                    pass

    def on_timer(self, timer: Timer, late: bool) -> None:
        self.notify(timer.message(late), kind="timer")

    # -- facts for UIs ----------------------------------------------------------------------------------
    def info(self) -> dict:
        v = self.cfg.section("voice")
        return {
            "assistant_name": self.identity.assistant_name,
            "creator": {"name": self.identity.creator_name, "alias": self.identity.creator_alias},
            "wake_words": list((v.get("wake", {}) or {}).get("words", [])),
            "stop_phrases": list(v.get("stop_phrases", [])),
            "wake_sensitivity": (v.get("wake", {}) or {}).get("sensitivity"),
            "device_control": bool(self.cfg.get("device_control.enabled", True)) and self.controller is not None,
            "voice_loop_enabled": bool(v.get("enabled", False)),
        }
