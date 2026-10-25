"""
control/timers.py - timers and reminders that keep working in the background.

"Set a timer for ten minutes" / "remind me in an hour to call mom". Timers survive restarts
(runs/timers.json). The scheduler thread checks them once a second and calls `on_fire`, which the
assistant wires to speech and to the web UI's notification feed. A timer that came due while the
assistant was off fires as soon as it starts, flagged as late.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, List, Optional


def format_duration(seconds: float) -> str:
    s = int(round(seconds))
    parts = []
    for unit, size in (("hour", 3600), ("minute", 60), ("second", 1)):
        n, s = divmod(s, size)
        if n:
            parts.append(f"{n} {unit}{'s' if n != 1 else ''}")
    return " ".join(parts) if parts else "0 seconds"


@dataclass
class Timer:
    id: int
    due: float
    seconds: int
    label: str
    created: float

    def message(self, late: bool = False) -> str:
        base = f"Time to {self.label}." if self.label else f"Your {format_duration(self.seconds)} timer is done."
        return base + (" (This came due while I was off.)" if late else "")


class TimerScheduler:
    def __init__(self, path: Optional[Path] = None, on_fire: Optional[Callable[[Timer, bool], None]] = None,
                 clock: Callable[[], float] = time.time):
        self.path = Path(path) if path else None
        self.on_fire = on_fire or (lambda t, late: None)
        self._clock = clock
        self._lock = threading.RLock()
        self._timers: List[Timer] = []
        self._load()

    # -- persistence -------------------------------------------------------------------------
    def _load(self) -> None:
        if not self.path or not self.path.exists():
            return
        try:
            self._timers = [Timer(**t) for t in json.loads(self.path.read_text(encoding="utf-8"))]
        except (OSError, ValueError, TypeError):
            self._timers = []

    def _save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), prefix=".timers_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump([asdict(t) for t in self._timers], f)
        os.replace(tmp, self.path)

    # -- api ---------------------------------------------------------------------------------------
    def add(self, seconds: int, label: str = "") -> Timer:
        with self._lock:
            now = self._clock()
            t = Timer(id=max((x.id for x in self._timers), default=0) + 1, due=now + seconds,
                      seconds=int(seconds), label=label.strip(), created=now)
            self._timers.append(t)
            self._save()
            return t

    def cancel_all(self) -> int:
        with self._lock:
            n = len(self._timers)
            self._timers.clear()
            self._save()
            return n

    def list(self) -> List[Timer]:
        with self._lock:
            return sorted(self._timers, key=lambda t: t.due)

    def remaining(self, t: Timer) -> float:
        return max(0.0, t.due - self._clock())

    def fire_due(self) -> List[Timer]:
        with self._lock:
            now = self._clock()
            due = [t for t in self._timers if t.due <= now]
            if due:
                self._timers = [t for t in self._timers if t.due > now]
                self._save()
        for t in due:
            late = (now - t.due) > 30
            try:
                self.on_fire(t, late)
            except Exception:  # noqa: BLE001 - a failing announcement must not stop other timers
                pass
        return due

    def run(self, stop: threading.Event, poll_s: float = 1.0) -> None:
        while not stop.is_set():
            self.fire_due()
            stop.wait(poll_s)
