"""
identity.py - who the assistant is and who made it.

The name, the creator and the exact wording of the answers all come from
configs/ziva.yaml (`identity:`), never from this file. Two things use them:

  1. Deterministic answers. "Who created you?" is answered straight from the
     config, before any language model is consulted. A small model can hallucinate
     a creator; a config lookup cannot. (assistant.py calls Identity.answer first.)

  2. Training conversations. Identity.training_records() renders the same facts as
     chat examples that are mixed into the conversation fine-tune
     (data/folder.py), so the model itself also learns them.

Matching is anchored on "you / your" so that "who made the Eiffel Tower" is never
mistaken for a question about the assistant.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

_APOSTROPHES = re.compile(r"[\u2018\u2019'`]")
_NON_WORD = re.compile(r"[^a-z0-9 ]+")
_SPACES = re.compile(r"\s+")
_FILLER_PREFIX = re.compile(r"^(?:(?:hey|hi|hello|ok|okay|please|so|um|uh|and|just|i want to know|i was wondering)\s+)+")
# "tell me who made you" / "do you know who created you" -> "who made you". Only when a wh-question
# follows, so "tell me about <creator>" keeps its own shape.
_ASK_PREFIX = re.compile(r"^(?:(?:can|could|would) you )?(?:tell me|do you know) (?=(?:who|what)\b)")
# "who is snow white" is far more likely to be about the fairy tale than the creator's alias;
# only treat the alias as the creator when none of these appear.
_FAIRY_TALE = re.compile(r"\b(fairy|tale|story|disney|dwarfs?|seven|apple|queen|prince|film|movie|cartoon)\b")

_CREATOR_PATTERNS = [
    r"who (?:created|made|built|designed|developed|programmed|invented|trained|wrote|coded|founded) (?:you|u)",
    r"who (?:is|s) (?:the one who |the person who |the guy who )?(?:created|made|built|designed|developed|programmed|trained) (?:you|u)",
    r"who (?:is|s) (?:your|ur) (?:creator|maker|developer|author|builder|inventor|programmer|designer|owner|father|boss)",
    r"who (?:are|re) (?:your|ur) (?:creator|maker|developer|author|builder|inventor|programmer|designer|owner)",
    r"(?:your|ur) (?:creator|maker|developer|author|builder|inventor|programmer) (?:is )?(?:who|name)",
    r"who do you (?:belong to|work for)",
    r"who is behind you",
    r"who gave you life",
    r"(?:what|whats) (?:is )?(?:your|ur) (?:creator|maker|developer)s? name",
]
_WHO_ARE_YOU_PATTERNS = [
    r"who are you", r"what are you", r"who is this", r"what is your name", r"whats your name",
    r"what do they call you", r"what should i call you", r"introduce yourself", r"tell me about yourself",
    r"describe yourself",
]

_CREATOR_RE = re.compile(r"^(?:" + "|".join(_CREATOR_PATTERNS) + r")$")
_WHO_ARE_YOU_RE = re.compile(r"^(?:" + "|".join(_WHO_ARE_YOU_PATTERNS) + r")$")


def normalize(text: str) -> str:
    """Lower-case, drop punctuation and apostrophes ("who's" -> "whos"), collapse spaces and
    strip conversational filler so patterns can be anchored."""
    t = _APOSTROPHES.sub("", text.lower())
    t = _NON_WORD.sub(" ", t)
    t = _SPACES.sub(" ", t).strip()
    # "whos" / "whats" normalise to the two-word form the patterns use
    t = re.sub(r"\bwhos\b", "who is", t)
    t = re.sub(r"\bwhats\b", "what is", t)
    t = _FILLER_PREFIX.sub("", t).strip()
    t = _ASK_PREFIX.sub("", t).strip()
    return t


@dataclass(frozen=True)
class IdentityMatch:
    kind: str       # "creator" | "who_are_you" | "about_creator"
    text: str       # the answer to give


class Identity:
    def __init__(self, assistant_name: str, creator_name: str, creator_alias: str,
                 answers: Dict[str, str], training_questions: Optional[Dict[str, List[str]]] = None):
        self.assistant_name = assistant_name.strip()
        self.creator_name = creator_name.strip()
        self.creator_alias = (creator_alias or "").strip()
        self._answers = dict(answers)
        self._training_questions = {k: list(v) for k, v in (training_questions or {}).items()}

        first = normalize(self.creator_name).split(" ")[0] if self.creator_name else ""
        self._creator_first = first
        alias = normalize(self.creator_alias)
        self._alias = alias
        who = r"(?:who is|who was|tell me about|what do you know about|do you know|have you heard of)"
        name_re = re.escape(first) + r"(?:\s+[a-z]{1,2})*" if first else None
        self._about_name_re = re.compile(rf"^{who} (?:the )?{name_re}(?: also known as .*| aka .*)?$") if name_re else None
        self._about_alias_re = re.compile(rf"^{who} (?:the )?{re.escape(alias)}$") if alias else None
        self._bare_name_re = re.compile(rf"^{name_re}$") if name_re else None

    # -- construction --------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg) -> "Identity":
        ident = cfg.section("identity")
        creator = ident.get("creator", {}) if isinstance(ident.get("creator"), dict) else {}
        training = ident.get("training", {}) if isinstance(ident.get("training"), dict) else {}
        questions = training.get("questions", {}) if isinstance(training.get("questions"), dict) else {}
        return cls(
            assistant_name=str(ident.get("assistant_name", "Assistant")),
            creator_name=str(creator.get("name", "")),
            creator_alias=str(creator.get("alias", "") or ""),
            answers=ident.get("answers", {}) if isinstance(ident.get("answers"), dict) else {},
            training_questions=questions,
        )

    # -- facts -----------------------------------------------------------------------
    def _fill(self, template: str) -> str:
        return (template.replace("{assistant_name}", self.assistant_name)
                        .replace("{creator_name}", self.creator_name)
                        .replace("{creator_alias}", self.creator_alias or self.creator_name))

    def answer_for(self, kind: str) -> str:
        template = self._answers.get(kind)
        if not template:
            raise KeyError(f"identity.answers.{kind} is not set in the config")
        return self._fill(template)

    # -- matching ----------------------------------------------------------------------
    def match(self, text: str) -> Optional[IdentityMatch]:
        if not self.creator_name:
            return None
        t = normalize(text)
        if not t or len(t) > 120:    # long messages are not a simple identity question
            return None
        if _CREATOR_RE.match(t):
            return IdentityMatch("creator", self.answer_for("creator"))
        if _WHO_ARE_YOU_RE.match(t):
            return IdentityMatch("who_are_you", self.answer_for("who_are_you"))
        if self._about_name_re and self._about_name_re.match(t):
            return IdentityMatch("about_creator", self.answer_for("about_creator"))
        if self._about_alias_re and self._about_alias_re.match(t) and not _FAIRY_TALE.search(t):
            return IdentityMatch("about_creator", self.answer_for("about_creator"))
        return None

    def answer(self, text: str) -> Optional[str]:
        m = self.match(text)
        return m.text if m else None

    # -- training data --------------------------------------------------------------------
    def training_records(self) -> List[dict]:
        """Chat examples teaching the model the same facts it is answered with above.
        Built entirely from the config: edit `identity:` and the next training run learns it."""
        records: List[dict] = []
        for kind, questions in self._training_questions.items():
            if kind not in self._answers:
                continue
            answer = self.answer_for(kind)
            for q in questions:
                q = self._fill(str(q)).strip()
                if q:
                    records.append({"messages": [{"role": "user", "content": q},
                                                 {"role": "assistant", "content": answer}]})
        return records
