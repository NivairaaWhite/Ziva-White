"""
voice/wake.py - wake words.

Works on the speech-to-text transcript of each utterance, so it is as private as the STT engine
(offline with Vosk / faster-whisper). An utterance wakes the assistant if it BEGINS with one of
`voice.wake.words` (give or take two filler words: "okay hey ziva ..."), which keeps "I was telling
Ziva about it yesterday" from waking anything.

Offline recognisers have never heard a made-up name like "Ziva", so they write down what it sounds
like ("siva", "zeva"). Matching therefore scores both spelling similarity and a rough phonetic key
and takes the better of the two. `sensitivity` is the minimum score to accept: lower accepts
sloppier matches (more false wakes). If your recogniser keeps mishearing a word, run
`python -m zivawhite.app voice --test-mic`, see what it wrote, and add that spelling to
`voice.wake.words`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List, Optional, Sequence, Tuple

_TOKEN_STRIP = re.compile(r"[^a-z0-9']+")
MAX_LEAD_TOKENS = 2


def _norm_token(tok: str) -> str:
    return _TOKEN_STRIP.sub("", tok.lower().replace("\u2019", "'")).replace("'", "")


def phonetic_key(word: str) -> str:
    """A deliberately crude sound-alike key (z~s, v~f, c/k/q~k, ph~f, i/y/e~i, doubled letters
    collapsed). Not a linguistics project: just enough that 'ziva', 'siva' and 'zeva' agree."""
    w = re.sub(r"[^a-z]", "", word.lower())
    for a, b in (("ph", "f"), ("sh", "s"), ("ch", "k"), ("ck", "k"), ("qu", "kw"), ("wh", "w"), ("kn", "n"), ("wr", "r")):
        w = w.replace(a, b)
    w = w.translate(str.maketrans({"z": "s", "v": "f", "c": "k", "q": "k", "x": "k", "y": "i", "e": "i", "o": "u", "w": "u"}))
    w = re.sub(r"(.)\1+", r"\1", w)
    w = re.sub(r"(?<=.)h", "", w)
    return w


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    spelling = SequenceMatcher(None, a, b).ratio()
    sound = SequenceMatcher(None, phonetic_key(a), phonetic_key(b)).ratio()
    return max(spelling, sound)


@dataclass(frozen=True)
class WakeResult:
    word: str            # which configured wake phrase matched
    score: float
    remainder: str       # what was said after it (may be empty)


class WakeWordMatcher:
    def __init__(self, words: Sequence[str], sensitivity: float = 0.88, stop_phrases: Sequence[str] = ()):
        phrases = []
        for w in words:
            toks = [t for t in (_norm_token(x) for x in str(w).split()) if t]
            if toks:
                phrases.append((str(w).strip(), toks))
        if not phrases:
            raise ValueError("at least one wake word is required")
        # longest phrase first so "hey snow white" wins over "snow white"
        self.phrases: List[Tuple[str, List[str]]] = sorted(phrases, key=lambda p: (-len(p[1]), -len("".join(p[1]))))
        self.sensitivity = float(sensitivity)
        self.stop_phrases = {" ".join(t for t in (_norm_token(x) for x in str(p).split()) if t) for p in stop_phrases}

    def is_stop(self, text: str) -> bool:
        norm = " ".join(t for t in (_norm_token(x) for x in text.split()) if t)
        return norm in self.stop_phrases

    def detect(self, transcript: str) -> Optional[WakeResult]:
        original = transcript.split()
        norm = [_norm_token(t) for t in original]
        idx = [i for i, t in enumerate(norm) if t]                 # skip tokens that were pure punctuation
        if not idx:
            return None
        best: Optional[Tuple[float, int, str, int]] = None        # (score, phrase length, phrase, end index in idx)
        for phrase, ptoks in self.phrases:
            target = "".join(ptoks)
            for start in range(0, min(MAX_LEAD_TOKENS, len(idx) - 1) + 1):
                for size in {max(1, len(ptoks) - 1), len(ptoks), len(ptoks) + 1}:     # "hey ziva" heard as "heyziva" or "a ziva"
                    if start + size > len(idx):
                        continue
                    heard = "".join(norm[i] for i in idx[start:start + size])
                    score = similarity(heard, target)
                    if score >= self.sensitivity:
                        cand = (score, len(ptoks), phrase, start + size)
                        if best is None or (cand[1], cand[0]) > (best[1], best[0]):
                            best = cand
        if best is None:
            return None
        score, _, phrase, end = best
        remainder = " ".join(original[idx[end]:]) if end < len(idx) else ""
        return WakeResult(phrase, round(score, 3), remainder.strip(" ,.;:!?-"))
