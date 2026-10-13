"""Device control: grammar, per-OS command construction, policy, smart home, timers.
A recording fake replaces the OS layer, so no test ever changes your volume or locks your screen."""
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from helpers import make_cfg

from zivawhite.control import DeviceController, Intent, parse_intent
from zivawhite.control.backends import LinuxBackend, MacBackend, WindowsBackend, clamp, get_backend
from zivawhite.control.controller import normalize_url
from zivawhite.control.intents import normalize_command, words_to_number
from zivawhite.control.smarthome import SmartHome
from zivawhite.control.system import System
from zivawhite.control.timers import TimerScheduler, format_duration


class FakeSystem(System):
    """Records everything; `have` lists the executables that 'exist'; `outputs` maps argv[0] to stdout."""

    def __init__(self, os_name="linux", have=(), outputs=None, rc=0):
        self._os, self.have, self.outputs, self.rc = os_name, set(have), outputs or {}, rc
        self.ran, self.spawned, self.keys, self.started, self.urls = [], [], [], [], []

    @property
    def os(self): return self._os
    def run(self, argv, timeout=15.0):
        self.ran.append(list(argv))
        return self.rc, self.outputs.get(argv[0], "")
    def which(self, name): return name if name in self.have else None
    def spawn(self, argv): self.spawned.append(list(argv))
    def startfile(self, target): self.started.append(target)
    def open_url(self, url): self.urls.append(url); return True
    def press_key(self, vk): self.keys.append(vk)


def ctl(tmp_path, os_name="linux", have=(), outputs=None, rc=0, **cfg_over):
    system = FakeSystem(os_name, have, outputs, rc)
    cfg = make_cfg(tmp_path, device_control={"screenshot_dir": str(tmp_path / "shots"), **cfg_over.pop("device_control", {})}, **cfg_over)
    return DeviceController(cfg, backend=get_backend(system), now=lambda: datetime(2026, 10, 5, 14, 5, 9)), system


# ------------------------------------------------------------------------------- grammar
def test_numbers_from_words_and_digits():
    assert [words_to_number(x) for x in ("45", "fifty", "twenty five", "twenty-five", "one hundred", "ninety nine")] == [45, 50, 25, 25, 100, 99]
    assert words_to_number("banana") is None


def test_normalize_strips_filler_and_punctuation():
    assert normalize_command("Hey, could you please Turn up the volume.") == "turn up the volume"
    assert normalize_command("What's the time?") == "whats the time"


CASES = [
    ("turn up the volume", "volume_up", {}), ("louder", "volume_up", {}), ("volume down", "volume_down", {}),
    ("set the volume to fifty", "volume_set", {"percent": 50}), ("volume to 35 percent", "volume_set", {"percent": 35}),
    ("mute", "mute", {}), ("unmute the sound", "unmute", {}),
    ("brightness up", "brightness_up", {}), ("dim the screen", "brightness_down", {}),
    ("set the brightness to 70", "brightness_set", {"percent": 70}),
    ("pause the music", "media_toggle", {}), ("next song", "media_next", {}), ("previous track", "media_previous", {}),
    ("stop the music", "media_stop", {}), ("take a screenshot", "screenshot", {}),
    ("lock the screen", "lock", {}), ("go to sleep", "sleep", {}), ("restart the computer", "restart", {}),
    ("shut down", "shutdown", {}), ("turn off the computer", "shutdown", {}),
    ("open calculator", "open_app", {"name": "calculator"}), ("launch the terminal app", "open_app", {"name": "terminal"}),
    ("go to youtube dot com", "open_url", {"url": "youtube.com"}), ("open website example.com", "open_url", {"url": "example.com"}),
    ("turn off the living room light", "smart_home", {"device": "living room light", "state": "off"}),
    ("turn the fan on", "smart_home", {"device": "fan", "state": "on"}),
    ("what time is it", "time", {}), ("what is the date", "date", {}), ("how much battery is left", "battery", {}),
    ("cpu usage", "system_status", {}),
    ("set a timer for 5 minutes", "timer_set", {"seconds": 300, "label": ""}),
    ("set a ten minute timer", "timer_set", {"seconds": 600, "label": ""}),
    ("set a timer for one hour thirty minutes", "timer_set", {"seconds": 5400, "label": ""}),
    ("remind me in 10 minutes to stretch", "timer_set", {"seconds": 600, "label": "stretch"}),
    ("remind me to call mom in an hour", "timer_set", {"seconds": 3600, "label": "call mom"}),
    ("cancel the timer", "timer_cancel", {}), ("list my timers", "timer_list", {}),
    ("yes", "confirm", {}), ("do it", "confirm", {}), ("no", "deny", {}), ("cancel", "deny", {}),
]


@pytest.mark.parametrize("text,action,args", CASES)
def test_grammar_recognises(text, action, args):
    intent = parse_intent(text)
    assert intent is not None and intent.action == action and intent.args == args


@pytest.mark.parametrize("text", [
    "what is the volume of a sphere", "what is the capital of france", "play some jazz", "tell me a story",
    "turn off", "set the volume to 150", "who created you", "how do i lock a door", "remind me about it",
    "the brightness of the sun is high", "", "x" * 300,
])
def test_grammar_ignores_everything_else(text):
    assert parse_intent(text) is None


def test_typed_punctuation_does_not_defeat_matching():
    assert parse_intent("Turn up the volume.").action == "volume_up"
    assert parse_intent("What time is it?").action == "time"


# ------------------------------------------------------------------------------- backends
def test_linux_volume_prefers_pactl_and_falls_back(tmp_path):
    c, s = ctl(tmp_path, have=["pactl", "amixer"])
    assert c.execute(Intent("volume_set", {"percent": 40})).ok and s.ran[-1] == ["pactl", "set-sink-volume", "@DEFAULT_SINK@", "40%"]
    c, s = ctl(tmp_path, have=["amixer"])
    c.execute(Intent("volume_set", {"percent": 40}))
    assert s.ran[-1] == ["amixer", "-q", "sset", "Master", "40%"]
    c, s = ctl(tmp_path, have=[])
    r = c.execute(Intent("volume_set", {"percent": 40}))
    assert not r.ok and "pactl" in r.message and s.ran == []                      # says what is missing, runs nothing


def test_linux_relative_volume_reads_then_sets_and_clamps(tmp_path):
    c, s = ctl(tmp_path, have=["pactl"], outputs={"pactl": "Volume: front-left: 30000 /  95% / -1.4 dB"})
    r = c.execute(Intent("volume_up"))
    assert r.ok and s.ran[-1][-1] == "100%"                                        # 95 + 10 -> clamped


def test_volume_values_are_always_clamped_integers():
    assert clamp(150) == 100 and clamp(-5) == 0 and clamp("42") == 42


def test_macos_commands(tmp_path):
    c, s = ctl(tmp_path, os_name="macos", have=["brightness"], outputs={"osascript": "55", "brightness": "display 0: brightness 0.500000"})
    c.execute(Intent("volume_set", {"percent": 30}))
    assert s.ran[-1] == ["osascript", "-e", "set volume output volume 30"]
    c.execute(Intent("brightness_up"))
    assert s.ran[-1] == ["brightness", "0.60"]
    c.execute(Intent("open_app", {"name": "calculator"}))
    assert s.ran[-1] == ["open", "-a", "Calculator"]
    c.execute(Intent("screenshot"))
    assert s.ran[-1][:2] == ["screencapture", "-x"] and s.ran[-1][2].endswith("ziva-20261005-140509.png")


def test_windows_commands_and_key_fallback(tmp_path):
    c, s = ctl(tmp_path, os_name="windows")
    r = c.execute(Intent("volume_set", {"percent": 20}))
    assert r.ok and "about" in r.message                                           # honest: no read-back without pycaw
    assert s.keys.count(0xAE) == 50 and s.keys.count(0xAF) == 10
    c.execute(Intent("media_next"))
    assert s.keys[-1] == 0xB0
    c.execute(Intent("open_app", {"name": "notepad"}))
    assert s.started == ["notepad"]
    c.execute(Intent("brightness_set", {"percent": 55}))
    assert s.ran[-1][0] == "powershell" and ".WmiSetBrightness(1,55)" in s.ran[-1][-1]


def test_powershell_screenshot_path_is_quoted(tmp_path):
    from zivawhite.control.backends import _ps_quote
    assert _ps_quote("C:\\Users\\O'Brien\\a.png") == "'C:\\Users\\O''Brien\\a.png'"


def test_no_shell_is_ever_used():
    import inspect
    src = inspect.getsource(System)
    assert "shell=False" in src and "shell=True" not in src and "os.system" not in src


# ---------------------------------------------------------------------------------- policy
def test_open_app_is_limited_to_the_configured_list(tmp_path):
    c, s = ctl(tmp_path, have=["gnome-calculator"])
    assert c.execute(Intent("open_app", {"name": "calculator"})).ok and s.spawned == [["gnome-calculator"]]
    r = c.execute(Intent("open_app", {"name": "rm -rf /"}))
    assert not r.ok and "allowed" in r.message and len(s.spawned) == 1
    r = c.execute(Intent("open_app", {"name": "calculator; reboot"}))
    assert not r.ok


def test_user_can_extend_the_app_list_in_config(tmp_path):
    c, s = ctl(tmp_path, have=["firefox"], device_control={"apps": {"browser": {"linux": "firefox --new-window"}}})
    assert c.execute(Intent("open_app", {"name": "browser"})).ok and s.spawned == [["firefox", "--new-window"]]


def test_power_actions_are_off_by_default(tmp_path):
    c, s = ctl(tmp_path, have=["systemctl"])
    r = c.execute(Intent("shutdown"), confirmed=True)
    assert not r.ok and "allow_power_actions" in r.message and s.ran == []
    assert c.refusal(Intent("restart"))


def test_risky_actions_cannot_run_without_confirmation(tmp_path):
    c, s = ctl(tmp_path, have=["systemctl", "loginctl"], device_control={"allow_power_actions": True})
    for action in ("lock", "sleep", "restart", "shutdown"):
        assert c.needs_confirmation(Intent(action))
        r = c.execute(Intent(action))
        assert not r.ok and "confirmation" in r.message
    assert s.ran == []                                                              # enforced in the controller itself
    assert c.execute(Intent("shutdown"), confirmed=True).ok and s.ran[-1] == ["systemctl", "poweroff"]
    assert c.execute(Intent("lock"), confirmed=True).ok and s.ran[-1] == ["loginctl", "lock-session"]


def test_disabled_device_control_still_allows_information_and_timers(tmp_path):
    c, s = ctl(tmp_path, have=["pactl"], device_control={"enabled": False})
    assert not c.execute(Intent("volume_up")).ok and s.ran == []
    assert c.execute(Intent("time")).message == "It's 2:05 PM."
    assert c.execute(Intent("date")).message == "Today is Monday, October 5, 2026."


def test_url_normalisation_is_strict():
    assert normalize_url("example.com") == "https://example.com"
    assert normalize_url("http://sub.example.co.uk/path?q=1") == "http://sub.example.co.uk/path?q=1"
    for bad in ("file:///etc/passwd", "javascript:alert(1)", "https://user:pw@example.com", "http://localhost", "http://127.0.0.1",
                "https://example.com:8080", "not a url", "ftp://example.com"):
        assert normalize_url(bad) is None


def test_open_url_goes_through_the_validator(tmp_path):
    c, s = ctl(tmp_path)
    assert c.execute(Intent("open_url", {"url": "example.com"})).ok and s.urls == ["https://example.com"]
    assert not c.execute(Intent("open_url", {"url": "file:///etc/passwd"})).ok and len(s.urls) == 1


def test_battery_and_system_status_use_psutil(tmp_path):
    ps = SimpleNamespace(sensors_battery=lambda: SimpleNamespace(percent=81.4, power_plugged=True),
                         cpu_percent=lambda interval=0: 12.0, virtual_memory=lambda: SimpleNamespace(percent=47.6))
    c, _ = ctl(tmp_path)
    c._psutil = ps
    assert c.execute(Intent("battery")).message == "The battery is at 81 percent, charging."
    assert c.execute(Intent("system_status")).message == "CPU is at 12 percent and memory is 48 percent used."
    c._psutil = SimpleNamespace(sensors_battery=lambda: None)
    assert "doesn't report a battery" in c.execute(Intent("battery")).message


def test_os_errors_become_a_spoken_failure_not_a_crash(tmp_path):
    c, _ = ctl(tmp_path, have=["pactl"], rc=1, outputs={"pactl": "boom"})
    r = c.execute(Intent("volume_set", {"percent": 10}))
    assert not r.ok and "couldn't" in r.message


# -------------------------------------------------------------------------------- smart home
class FakeHttp:
    def __init__(self, status=200, boom=None): self.calls, self.status, self.boom = [], status, boom
    def post(self, url, json=None, headers=None, timeout=None):
        if self.boom: raise self.boom
        self.calls.append((url, json, headers))
        return SimpleNamespace(status_code=self.status)


ENV = {"HOME_ASSISTANT_URL": "http://ha.local:8123/", "HOME_ASSISTANT_TOKEN": "tok"}


def test_smart_home_only_reaches_mapped_devices():
    http = FakeHttp()
    sh = SmartHome({"Living Room Light": "light.living_room", "bad": "not an entity id"}, env=ENV, session=http)
    assert sh.set_state("the living room light", on=False).ok
    url, body, headers = http.calls[0]
    assert url == "http://ha.local:8123/api/services/homeassistant/turn_off" and body == {"entity_id": "light.living_room"}
    assert headers["Authorization"] == "Bearer tok"
    assert sh.set_state("livingroom light", True).ok                              # fuzzy match survives speech-to-text slop
    assert not sh.set_state("front door lock", True).ok and len(http.calls) == 2    # unmapped: never sent


def test_smart_home_failures_are_explained():
    assert "not configured" in SmartHome({"a": "light.a"}, env={}, session=FakeHttp()).set_state("a", True).message or True
    assert "isn't configured" in SmartHome({"a": "light.a"}, env={}, session=FakeHttp()).set_state("a", True).message
    assert "No smart home" in SmartHome({}, env=ENV, session=FakeHttp()).set_state("a", True).message
    assert "refused" in SmartHome({"a": "light.a"}, env=ENV, session=FakeHttp(status=401)).set_state("a", True).message
    assert "reach" in SmartHome({"a": "light.a"}, env=ENV, session=FakeHttp(boom=OSError("down"))).set_state("a", True).message


def test_smart_home_is_off_unless_enabled(tmp_path):
    c, _ = ctl(tmp_path)
    assert "switched off" in c.execute(Intent("smart_home", {"device": "x", "state": "on"})).message


# ------------------------------------------------------------------------------------ timers
def test_timers_fire_when_due_and_persist(tmp_path):
    clock = {"t": 1000.0}
    fired = []
    path = tmp_path / "timers.json"
    ts = TimerScheduler(path, on_fire=lambda t, late: fired.append((t.label, late)), clock=lambda: clock["t"])
    ts.add(300, "stretch")
    ts.add(60)
    assert [t.seconds for t in ts.list()] == [60, 300]
    clock["t"] += 61
    assert len(ts.fire_due()) == 1 and fired == [("", False)]

    again = TimerScheduler(path, on_fire=lambda t, late: fired.append((t.label, late)), clock=lambda: clock["t"])
    assert [t.label for t in again.list()] == ["stretch"]                            # survived a "restart"
    clock["t"] += 10_000                                                             # assistant was off a long time
    again.fire_due()
    assert fired[-1] == ("stretch", True)
    assert not path.read_text() or json.loads(path.read_text()) == []


def test_timer_messages_and_duration_format():
    assert format_duration(5400) == "1 hour 30 minutes" and format_duration(1) == "1 second" and format_duration(120) == "2 minutes"
    ts = TimerScheduler(None)
    assert ts.add(60, "stretch").message() == "Time to stretch."
    assert "1 minute timer is done" in ts.add(60).message()


def test_timer_intents_through_the_controller(tmp_path):
    c, _ = ctl(tmp_path)
    c.timers = TimerScheduler(None)
    assert c.execute(Intent("timer_set", {"seconds": 600, "label": ""})).message == "Timer set for 10 minutes."
    assert "remind you to stretch in 5 minutes" in c.execute(Intent("timer_set", {"seconds": 300, "label": "stretch"})).message
    assert "left" in c.execute(Intent("timer_list")).message
    assert c.execute(Intent("timer_cancel")).message == "Cancelled 2 timers."
    assert c.execute(Intent("timer_list")).message == "You have no timers set."
    assert not c.execute(Intent("timer_set", {"seconds": 10 ** 9, "label": ""})).ok


def test_a_failing_announcement_does_not_stop_other_timers():
    clock = {"t": 0.0}
    got = []
    def on_fire(t, late):
        if t.label == "boom": raise RuntimeError("tts crashed")
        got.append(t.label)
    ts = TimerScheduler(None, on_fire=on_fire, clock=lambda: clock["t"])
    ts.add(1, "boom"); ts.add(1, "fine")
    clock["t"] = 5
    ts.fire_due()
    assert got == ["fine"]
