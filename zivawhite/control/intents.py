"""
control/intents.py - the command grammar.

Spoken or typed text is turned into an Intent by fixed patterns, never by the language model and
never from fetched web text. That is the safety property: a hallucination or a malicious page
cannot make the assistant lock your screen, because only the person talking to it can produce a
string that matches these patterns.

Patterns are anchored (^...$) and deliberately conservative: "what is the volume of a sphere"
must not change your volume. Numbers may be digits or words ("fifty").
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Optional

_UNITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
          "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
          "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_NUM_WORDS = "|".join(sorted(list(_UNITS) + list(_TENS) + ["hundred"], key=len, reverse=True))
_NUM = rf"(?:\d{{1,3}}|(?:{_NUM_WORDS})(?:[ -](?:{_NUM_WORDS}))?)"


def words_to_number(text: str) -> Optional[int]:
    """'45' -> 45, 'fifty' -> 50, 'twenty five' -> 25, 'one hundred' -> 100."""
    t = text.strip().lower().replace("-", " ")
    if t.isdigit():
        return int(t)
    parts = t.split()
    if parts == ["hundred"] or parts == ["one", "hundred"] or parts == ["a", "hundred"]:
        return 100
    total = 0
    for w in parts:
        if w in _TENS:
            total += _TENS[w]
        elif w in _UNITS:
            total += _UNITS[w]
        else:
            return None
    return total if parts else None


@dataclass(frozen=True)
class Intent:
    action: str
    args: Dict[str, object] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.action + (f"({', '.join(f'{k}={v}' for k, v in self.args.items())})" if self.args else "")


_PUNCT = re.compile(r"[^\w\s%:.-]")
_SPACES = re.compile(r"\s+")
_LEAD = re.compile(r"^(?:(?:hey|ok|okay|please|can you|could you|would you|will you|i want you to|go ahead and|just)\s+)+")
_TRAIL = re.compile(r"\s+(?:please|now|for me|thanks|thank you)$")


def normalize_command(text: str) -> str:
    t = _PUNCT.sub(" ", text.lower().replace("'", ""))
    t = _SPACES.sub(" ", t).strip()
    t = _LEAD.sub("", t)
    t = _TRAIL.sub("", t)
    return t.strip(" .:-")      # typed input arrives with a trailing "." or "?"


def _n(m: re.Match, group: str = "n") -> Optional[int]:
    return words_to_number(m.group(group))


_DEVICE = r"(?:the |my )?(?:sound|volume|audio)"
_SCREEN = r"(?:the |my )?(?:screen|display|monitor|brightness)"
_TIME_UNIT = r"(second|seconds|sec|secs|minute|minutes|min|mins|hour|hours|hr|hrs)"
_UNIT_S = {"second": 1, "seconds": 1, "sec": 1, "secs": 1, "minute": 60, "minutes": 60, "min": 60, "mins": 60,
           "hour": 3600, "hours": 3600, "hr": 3600, "hrs": 3600}


def _duration(text: str) -> Optional[int]:
    """'5 minutes', 'an hour', 'ten seconds', 'one hour thirty minutes' -> seconds."""
    total, found = 0, False
    for m in re.finditer(rf"(?P<n>{_NUM}|an?|half an?) {_TIME_UNIT}", text):
        raw = m.group("n")
        n = 1 if raw in ("a", "an") else (0.5 if raw.startswith("half") else words_to_number(raw))
        if n is None:
            return None
        total += n * _UNIT_S[m.group(2)]
        found = True
    return int(total) if found else None


def parse_intent(text: str) -> Optional[Intent]:
    t = normalize_command(text)
    if not t or len(t) > 140:
        return None

    # -- stop / cancel is handled by the voice loop, not here ----------------------------------
    # -- volume ----------------------------------------------------------------------------------
    if re.fullmatch(rf"(?:turn )?(?:up|raise|increase) {_DEVICE}|{_DEVICE} up|louder|(?:turn it|make it) (?:up|louder)", t):
        return Intent("volume_up")
    if re.fullmatch(rf"(?:turn )?(?:down|lower|decrease|reduce) {_DEVICE}|{_DEVICE} down|quieter|softer|(?:turn it|make it) (?:down|quieter)", t):
        return Intent("volume_down")
    m = re.fullmatch(rf"(?:set|change|put) {_DEVICE} (?:to|at) (?P<n>{_NUM})(?: percent|%)?|{_DEVICE} (?:to|at) (?P<n2>{_NUM})(?: percent|%)?", t)
    if m:
        n = words_to_number(m.group("n") or m.group("n2"))
        if n is not None and 0 <= n <= 100:
            return Intent("volume_set", {"percent": n})
    if re.fullmatch(rf"mute(?: {_DEVICE}| the computer| everything)?|{_DEVICE} off", t):
        return Intent("mute")
    if re.fullmatch(rf"unmute(?: {_DEVICE}| the computer)?|{_DEVICE} on", t):
        return Intent("unmute")

    # -- brightness --------------------------------------------------------------------------------
    if re.fullmatch(rf"(?:turn )?(?:up|raise|increase) {_SCREEN}|{_SCREEN} up|brighter|(?:make it|make the screen) brighter", t):
        return Intent("brightness_up")
    if re.fullmatch(rf"(?:turn )?(?:down|lower|decrease|reduce|dim) {_SCREEN}|{_SCREEN} down|dimmer|dim the screen|(?:make it|make the screen) dimmer", t):
        return Intent("brightness_down")
    m = re.fullmatch(rf"(?:set|change|put) {_SCREEN} (?:to|at) (?P<n>{_NUM})(?: percent|%)?|{_SCREEN} (?:to|at) (?P<n2>{_NUM})(?: percent|%)?", t)
    if m:
        n = words_to_number(m.group("n") or m.group("n2"))
        if n is not None and 0 <= n <= 100:
            return Intent("brightness_set", {"percent": n})

    # -- media -----------------------------------------------------------------------------------------
    if re.fullmatch(r"(?:play|resume|pause|play pause|toggle)(?: the)?(?: music| media| song| track| video| playback)?", t) and t not in ("play",):
        return Intent("media_toggle")
    if t == "play":
        return Intent("media_toggle")
    if re.fullmatch(r"(?:next|skip)(?: the)?(?: song| track| one)?|skip this(?: song| track)?|play (?:the )?next(?: song| track)?", t):
        return Intent("media_next")
    if re.fullmatch(r"(?:previous|last|go back)(?: song| track| one)?|play (?:the )?previous(?: song| track)?", t):
        return Intent("media_previous")
    if re.fullmatch(r"stop(?: the)? (?:music|media|song|track|playback)", t):
        return Intent("media_stop")

    # -- screen / power ----------------------------------------------------------------------------------
    if re.fullmatch(r"(?:take |grab )?(?:a )?screenshot|(?:take|capture) (?:a )?(?:picture|photo|snapshot) of (?:the |my )?screen|capture (?:the |my )?screen", t):
        return Intent("screenshot")
    if re.fullmatch(r"lock(?: the| my)?(?: screen| computer| pc| laptop| device)?", t):
        return Intent("lock")
    if re.fullmatch(r"(?:go to |put (?:the |my )?(?:computer|pc|laptop|device) (?:to |in)?)?sleep(?: mode)?|suspend(?: the computer)?", t):
        return Intent("sleep")
    if re.fullmatch(r"(?:restart|reboot)(?: the| my)?(?: computer| pc| laptop| device| system)?", t):
        return Intent("restart")
    if re.fullmatch(r"(?:shut ?down|power off|turn off)(?: the| my)?(?: computer| pc| laptop| device| system)?", t) \
            and t not in ("turn off",):
        return Intent("shutdown")

    # -- apps / web ----------------------------------------------------------------------------------------
    m = re.fullmatch(r"(?:open|launch|start|run) (?:the )?(?P<app>[a-z0-9 ._-]{1,40}?)(?: app| application| program)?", t)
    if m:
        target = m.group("app").strip()
        if re.fullmatch(r"(?:website|site|page|url) .+", target):
            return Intent("open_url", {"url": re.sub(r"^(?:website|site|page|url) ", "", target)})
        return Intent("open_app", {"name": target})
    m = re.fullmatch(r"(?:go to|open|visit|browse to) (?:the )?(?:website |site |page )?(?P<host>[a-z0-9-]+(?: dot [a-z0-9-]+)+|[a-z0-9-]+(?:\.[a-z0-9-]+)+)(?P<path>/\S*)?", t)
    if m:
        return Intent("open_url", {"url": m.group("host").replace(" dot ", ".") + (m.group("path") or "")})

    # -- smart home ----------------------------------------------------------------------------------------------
    m = re.fullmatch(r"turn (?P<state>on|off) (?:the )?(?P<dev>[a-z0-9 ]{2,40})|turn (?:the )?(?P<dev2>[a-z0-9 ]{2,40}?) (?P<state2>on|off)", t)
    if m:
        dev = (m.group("dev") or m.group("dev2")).strip()
        state = m.group("state") or m.group("state2")
        if dev not in ("computer", "pc", "laptop", "device", "system", "screen", "display", "monitor", "it", "sound", "volume", "audio", "everything"):
            return Intent("smart_home", {"device": dev, "state": state})

    # -- device information ---------------------------------------------------------------------------------------
    if re.fullmatch(r"what(?:s| is) the time|what time is it|tell me the time|current time|time", t):
        return Intent("time")
    if re.fullmatch(r"what(?:s| is) (?:the )?(?:date|day)(?: today)?|what day is (?:it|today)|todays date|tell me the date", t):
        return Intent("date")
    if re.fullmatch(r"(?:what(?:s| is) (?:the |my )?)?battery(?: level| status| percentage| percent)?|how much battery(?: do i have| is left| is there)?(?: left)?|battery left", t):
        return Intent("battery")
    if re.fullmatch(r"(?:what(?:s| is) (?:the |my )?)?(?:cpu|processor) (?:usage|load)|how busy is (?:the |my )?(?:computer|cpu|pc)|system (?:status|usage)|(?:memory|ram) usage|how much (?:memory|ram) (?:is )?(?:used|left|free)", t):
        return Intent("system_status")

    # -- timers / reminders ------------------------------------------------------------------------------------------
    m = re.fullmatch(r"(?:set|start|create) (?:a |an )?(?:(?P<dur>.+?) )?timer(?: for (?P<dur2>.+))?", t)
    if m:
        secs = _duration(m.group("dur") or m.group("dur2") or "")
        if secs:
            return Intent("timer_set", {"seconds": secs, "label": ""})
    m = re.fullmatch(r"remind me (?:in (?P<dur>.+?) )?to (?P<what>.+?)(?: in (?P<dur2>.+))?", t)
    if m:
        secs = _duration(m.group("dur") or m.group("dur2") or "")
        if secs:
            return Intent("timer_set", {"seconds": secs, "label": m.group("what").strip()})
    if re.fullmatch(r"(?:cancel|stop|clear|delete) (?:all |my |the )?(?:timers?|reminders?|alarms?)", t):
        return Intent("timer_cancel")
    if re.fullmatch(r"(?:what|which|list|show)(?: are| is)?(?: my| the)? (?:timers?|reminders?)(?: (?:do i have|are set|are running))?|how much time (?:is )?left", t):
        return Intent("timer_list")

    # -- confirmation words (consumed by the Assistant when something is pending) -----------------------------------
    if re.fullmatch(r"yes|yeah|yep|yup|confirm|confirmed|do it|go ahead|sure|affirmative|i confirm", t):
        return Intent("confirm")
    if re.fullmatch(r"no|nope|nah|cancel|never mind|dont|do not|abort|negative|stop", t):
        return Intent("deny")
    return None
