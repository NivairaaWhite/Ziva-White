"""
control/controller.py - runs an Intent, with the policy in configs/ziva.yaml enforced here.

Policy, all from `device_control:` in the config:
  * enabled: false      -> only harmless information and timers still work
  * apps                -> "open X" launches only apps listed there; anything else is refused
  * allow_power_actions -> shutdown / restart refused unless true
  * confirm_actions     -> these are never executed without `confirmed=True`, which only the
                           assistant sets after the person said yes. Enforced here rather than left
                           to callers, so no code path can skip it.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

from .backends import Backend, get_backend
from .intents import Intent
from .result import ActionResult
from .smarthome import SmartHome
from .timers import TimerScheduler, format_duration

ALWAYS_ALLOWED = {"time", "date", "battery", "system_status", "timer_set", "timer_cancel", "timer_list"}
POWER_ACTIONS = {"shutdown", "restart"}
_HOST_RE = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24}$")

SPOKEN = {"lock": "lock the screen", "sleep": "put the computer to sleep", "restart": "restart the computer",
          "shutdown": "shut down the computer"}


def normalize_url(raw: str) -> Optional[str]:
    """http(s) URLs with a real hostname only: no other schemes, no credentials, no IP/port tricks."""
    raw = raw.strip()
    if "://" not in raw:
        raw = "https://" + raw
    try:
        u = urlparse(raw)
        if u.scheme not in ("http", "https") or u.username or u.password or u.port:
            return None
        host = (u.hostname or "").lower()
    except ValueError:      # e.g. "javascript:alert(1)" -> "port could not be cast to integer"
        return None
    return raw if _HOST_RE.match(host) else None


class DeviceController:
    def __init__(self, cfg, backend: Optional[Backend] = None, smart_home: Optional[SmartHome] = None,
                 timers: Optional[TimerScheduler] = None, psutil_module=None,
                 now: Callable[[], datetime] = datetime.now):
        self.cfg = cfg
        self.dc = cfg.section("device_control")
        self.backend = backend or get_backend()
        sh = self.dc.get("smart_home", {}) or {}
        self.smart_home = smart_home if smart_home is not None else (
            SmartHome(sh.get("devices", {})) if sh.get("enabled") else None)
        self.timers = timers
        self._psutil = psutil_module
        self._now = now

    # -- policy --------------------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return bool(self.dc.get("enabled", True))

    def needs_confirmation(self, intent: Intent) -> bool:
        return intent.action in set(self.dc.get("confirm_actions", []) or [])

    def refusal(self, intent: Intent) -> Optional[str]:
        """Why this intent will not run at all (before asking for confirmation), or None."""
        if intent.action in ALWAYS_ALLOWED:
            return None
        if not self.enabled:
            return "Device control is turned off in my settings."
        if intent.action in POWER_ACTIONS and not self.dc.get("allow_power_actions", False):
            return (f"I won't {SPOKEN[intent.action]} because power actions are switched off. "
                    "Set device_control.allow_power_actions to true if you want that.")
        return None

    def describe(self, intent: Intent) -> str:
        return SPOKEN.get(intent.action, intent.action.replace("_", " "))

    # -- execution ---------------------------------------------------------------------------------
    def execute(self, intent: Intent, confirmed: bool = False) -> ActionResult:
        why = self.refusal(intent)
        if why:
            return ActionResult(False, why)
        if self.needs_confirmation(intent) and not confirmed:
            return ActionResult(False, f"I need your confirmation before I {self.describe(intent)}.")
        handler = getattr(self, f"_do_{intent.action}", None)
        if handler is None:
            return ActionResult(False, f"I don't know how to do {intent.action}.")
        try:
            return handler(intent)
        except Exception as e:  # noqa: BLE001 - the OS layer can fail in ways we cannot enumerate
            return ActionResult(False, f"That didn't work ({type(e).__name__}: {e}).")

    # -- volume / brightness / media -------------------------------------------------------------------
    def _do_volume_up(self, i): return self.backend.volume_change(+int(self.dc.get("volume_step", 10)))
    def _do_volume_down(self, i): return self.backend.volume_change(-int(self.dc.get("volume_step", 10)))
    def _do_volume_set(self, i): return self.backend.volume_set(int(i.args["percent"]))
    def _do_mute(self, i): return self.backend.set_mute(True)
    def _do_unmute(self, i): return self.backend.set_mute(False)
    def _do_brightness_up(self, i): return self.backend.brightness_change(+int(self.dc.get("brightness_step", 10)))
    def _do_brightness_down(self, i): return self.backend.brightness_change(-int(self.dc.get("brightness_step", 10)))
    def _do_brightness_set(self, i): return self.backend.brightness_set(int(i.args["percent"]))
    def _do_media_toggle(self, i): return self.backend.media("toggle")
    def _do_media_next(self, i): return self.backend.media("next")
    def _do_media_previous(self, i): return self.backend.media("previous")
    def _do_media_stop(self, i): return self.backend.media("stop")

    # -- screen / power -----------------------------------------------------------------------------------
    def _do_screenshot(self, i):
        folder = Path(self.cfg.resolve(self.dc.get("screenshot_dir", "~/Pictures/ZivaWhite")))
        folder.mkdir(parents=True, exist_ok=True)
        return self.backend.screenshot(str(folder / f"ziva-{self._now().strftime('%Y%m%d-%H%M%S')}.png"))

    def _do_lock(self, i): return self.backend.lock()
    def _do_sleep(self, i): return self.backend.sleep()
    def _do_restart(self, i): return self.backend.restart()
    def _do_shutdown(self, i): return self.backend.shutdown()

    # -- apps / web --------------------------------------------------------------------------------------------
    def _app_target(self, name: str) -> Optional[str]:
        apps = {str(k).lower(): v for k, v in (self.dc.get("apps", {}) or {}).items()}
        entry = apps.get(name.strip().lower())
        if entry is None:
            return None
        if isinstance(entry, str):
            return entry
        return entry.get(self.backend.os_name) if isinstance(entry, dict) else None

    def _do_open_app(self, i):
        name = str(i.args["name"])
        target = self._app_target(name)
        if target is None:
            known = ", ".join(sorted(str(k) for k in (self.dc.get("apps", {}) or {}))[:8])
            return ActionResult(False, f"I'm only allowed to open apps listed in my settings, and {name} isn't one. "
                                       f"I can open: {known or 'nothing yet'}.")
        return self.backend.open_app(target, name)

    def _do_open_url(self, i):
        url = normalize_url(str(i.args["url"]))
        if url is None:
            return ActionResult(False, "That doesn't look like a website address I can safely open.")
        return self.backend.open_url(url)

    # -- smart home ---------------------------------------------------------------------------------------------
    def _do_smart_home(self, i):
        if self.smart_home is None:
            return ActionResult(False, "Smart home control is switched off in my settings.")
        return self.smart_home.set_state(str(i.args["device"]), i.args["state"] == "on")

    # -- information ------------------------------------------------------------------------------------------------
    def _ps(self):
        if self._psutil is not None:
            return self._psutil
        import psutil

        return psutil

    def _do_time(self, i):
        return ActionResult(True, f"It's {self._now().strftime('%I:%M %p').lstrip('0')}.")

    def _do_date(self, i):
        d = self._now()
        return ActionResult(True, f"Today is {d.strftime('%A')}, {d.strftime('%B')} {d.day}, {d.year}.")

    def _do_battery(self, i):
        b = self._ps().sensors_battery()
        if b is None:
            return ActionResult(True, "This device doesn't report a battery.")
        state = "charging" if b.power_plugged else "on battery"
        return ActionResult(True, f"The battery is at {round(b.percent)} percent, {state}.")

    def _do_system_status(self, i):
        ps = self._ps()
        return ActionResult(True, f"CPU is at {round(ps.cpu_percent(interval=0.3))} percent and memory is "
                                  f"{round(ps.virtual_memory().percent)} percent used.")

    # -- timers ---------------------------------------------------------------------------------------------------------
    def _do_timer_set(self, i):
        if self.timers is None:
            return ActionResult(False, "Timers aren't available right now.")
        secs, label = int(i.args["seconds"]), str(i.args.get("label", ""))
        if not 1 <= secs <= 7 * 24 * 3600:
            return ActionResult(False, "I can set timers from one second up to a week.")
        self.timers.add(secs, label)
        if label:
            return ActionResult(True, f"Okay, I'll remind you to {label} in {format_duration(secs)}.")
        return ActionResult(True, f"Timer set for {format_duration(secs)}.")

    def _do_timer_cancel(self, i):
        n = self.timers.cancel_all() if self.timers else 0
        return ActionResult(True, f"Cancelled {n} timer{'s' if n != 1 else ''}." if n else "You have no timers set.")

    def _do_timer_list(self, i):
        items = self.timers.list() if self.timers else []
        if not items:
            return ActionResult(True, "You have no timers set.")
        parts = [(f"{t.label}: {format_duration(self.timers.remaining(t))} left" if t.label else
                  f"{format_duration(self.timers.remaining(t))} left on your {format_duration(t.seconds)} timer")
                 for t in items[:5]]
        return ActionResult(True, "; ".join(parts) + ".")
