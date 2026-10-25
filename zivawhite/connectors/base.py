"""
Every backend the router can call - the local model, a web search, a
page fetch - implements this same tiny interface. That's what lets
router.py treat them interchangeably.

This project deliberately has no connector for a third-party AI model
API (no Anthropic/OpenAI/Gemini/generic-REST wrapper). ZivaWhite
answers from its own trained model; the only outside calls it makes
are read-only web search/fetch, used to ground an answer in current
information and to gather text for the model to train on -- never to
have another company's model write the answer for it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ConnectorResult:
    text: str
    source: str                       # connector name, for provenance
    ok: bool = True
    meta: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class Connector(ABC):
    name: str = "connector"

    @abstractmethod
    def available(self) -> bool:
        """True if this connector is configured and usable right now
        (API key present, endpoint reachable, etc.) -- checked before
        the router routes to it, so a missing key just means that
        backend is skipped, not a crash."""

    @abstractmethod
    def call(self, prompt: str, **kwargs) -> ConnectorResult:
        ...
