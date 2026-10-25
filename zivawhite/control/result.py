from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ActionResult:
    ok: bool
    message: str                       # what to say / show to the person
    data: dict = field(default_factory=dict)
