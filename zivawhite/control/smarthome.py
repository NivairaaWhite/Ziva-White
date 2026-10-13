"""
control/smarthome.py - optional Home Assistant bridge ("turn off the living room light").

Off by default. Credentials come from the environment (.env), never from the YAML, so the
config file is safe to share: HOME_ASSISTANT_URL=http://homeassistant.local:8123 and
HOME_ASSISTANT_TOKEN=<long-lived access token>. The names you say map to entity ids in
configs/ziva.local.yaml (device_control.smart_home.devices). Only mapped devices can be
addressed -- there is no "turn off anything Home Assistant knows about".
"""
from __future__ import annotations

import difflib
import os
import re
from typing import Dict, Optional

from .result import ActionResult

_ENTITY_RE = re.compile(r"^[a-z_]+\.[a-z0-9_]+$")


def _norm(name: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"^(?:the|my)\s+", "", name.lower().strip())).strip()


class SmartHome:
    def __init__(self, devices: Dict[str, str], env: Optional[dict] = None, session=None, timeout: float = 5.0):
        self.devices = {_norm(k): v for k, v in (devices or {}).items() if isinstance(v, str) and _ENTITY_RE.match(v)}
        self.env = os.environ if env is None else env
        self._session = session
        self.timeout = timeout

    def resolve(self, spoken: str) -> Optional[str]:
        key = _norm(spoken)
        if key in self.devices:
            return self.devices[key]
        close = difflib.get_close_matches(key, list(self.devices), n=1, cutoff=0.75)
        return self.devices[close[0]] if close else None

    def set_state(self, spoken: str, on: bool) -> ActionResult:
        if not self.devices:
            return ActionResult(False, "No smart home devices are set up. Map names to entity ids under "
                                       "device_control.smart_home.devices.")
        entity = self.resolve(spoken)
        if entity is None:
            return ActionResult(False, f"I don't know a device called {spoken}.")
        base = self.env.get("HOME_ASSISTANT_URL", "").strip().rstrip("/")
        token = self.env.get("HOME_ASSISTANT_TOKEN", "").strip()
        if not base.startswith(("http://", "https://")) or not token:
            return ActionResult(False, "Home Assistant isn't configured: set HOME_ASSISTANT_URL and HOME_ASSISTANT_TOKEN in .env.")
        session = self._session
        if session is None:
            import requests

            session = requests
        try:
            r = session.post(f"{base}/api/services/homeassistant/{'turn_on' if on else 'turn_off'}",
                             json={"entity_id": entity}, headers={"Authorization": f"Bearer {token}"}, timeout=self.timeout)
        except Exception as e:  # noqa: BLE001 - network errors of any kind become a spoken failure
            return ActionResult(False, f"I couldn't reach Home Assistant ({type(e).__name__}).")
        if getattr(r, "status_code", 500) >= 300:
            return ActionResult(False, f"Home Assistant refused that (HTTP {r.status_code}).")
        return ActionResult(True, f"Turned {'on' if on else 'off'} the {_norm(spoken)}.")
