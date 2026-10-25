"""
control/backends.py - what "turn the volume up" means on each operating system.

Rules every method follows:
  * never a shell: commands are argument lists built here, and the only values ever interpolated
    into them are integers we clamped ourselves or a screenshot path we generated;
  * try the tools a platform commonly has, in order, and say plainly which one is missing when
    none is ("install playerctl") instead of pretending to succeed;
  * report what actually happened. Where the OS gives no way to read the current value (Windows
    volume without pycaw) the message says "about", because we know we asked, not what the result is.
"""
from __future__ import annotations

import re
import shlex
from typing import Dict, List, Optional, Sequence

from .result import ActionResult
from .system import System

VK = {"mute": 0xAD, "down": 0xAE, "up": 0xAF, "next": 0xB0, "previous": 0xB1, "stop": 0xB2, "toggle": 0xB3}


def clamp(n: int, lo: int = 0, hi: int = 100) -> int:
    return max(lo, min(hi, int(n)))


def _ps(script: str) -> List[str]:
    return ["powershell", "-NoProfile", "-NonInteractive", "-Command", script]


def _ps_quote(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


class Backend:
    os_name = "generic"

    def __init__(self, system: Optional[System] = None):
        self.sys = system or System()

    # -- helpers ---------------------------------------------------------------------------
    def _first_available(self, candidates: Sequence[Sequence[str]]) -> Optional[Sequence[str]]:
        for argv in candidates:
            if self.sys.which(argv[0]):
                return argv
        return None

    def _missing(self, what: str, tools: str) -> ActionResult:
        return ActionResult(False, f"I couldn't {what}: none of {tools} is installed.")

    def _ran(self, rc_out, ok_msg: str, fail_msg: str) -> ActionResult:
        rc, out = rc_out
        detail = " ".join(out.split())[:100] or f"exit {rc}"        # one short line: this gets spoken aloud
        return ActionResult(True, ok_msg) if rc == 0 else ActionResult(False, f"{fail_msg} ({detail})")

    # -- interface -------------------------------------------------------------------------------
    def volume_get(self) -> Optional[int]:
        return None

    def volume_set(self, percent: int) -> ActionResult:
        raise NotImplementedError

    def volume_change(self, delta: int) -> ActionResult:
        cur = self.volume_get()
        if cur is None:
            return ActionResult(False, "I can't read the current volume on this system.")
        return self.volume_set(clamp(cur + delta))

    def set_mute(self, muted: bool) -> ActionResult:
        raise NotImplementedError

    def brightness_set(self, percent: int) -> ActionResult:
        raise NotImplementedError

    def brightness_get(self) -> Optional[int]:
        return None

    def brightness_change(self, delta: int) -> ActionResult:
        cur = self.brightness_get()
        if cur is None:
            return ActionResult(False, "I can't read the current brightness on this system.")
        return self.brightness_set(clamp(cur + delta))

    def media(self, action: str) -> ActionResult:
        raise NotImplementedError

    def screenshot(self, path: str) -> ActionResult:
        raise NotImplementedError

    def lock(self) -> ActionResult:
        raise NotImplementedError

    def sleep(self) -> ActionResult:
        raise NotImplementedError

    def restart(self) -> ActionResult:
        raise NotImplementedError

    def shutdown(self) -> ActionResult:
        raise NotImplementedError

    def open_app(self, target: str, name: str) -> ActionResult:
        raise NotImplementedError

    def open_url(self, url: str) -> ActionResult:
        ok = self.sys.open_url(url)
        return ActionResult(ok, f"Opening {url}." if ok else "I couldn't open a browser.")


# =============================================================================== Windows
class WindowsBackend(Backend):
    os_name = "windows"

    def _pycaw(self):
        try:
            from ctypes import POINTER, cast

            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

            dev = AudioUtilities.GetSpeakers()
            iface = dev.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            return cast(iface, POINTER(IAudioEndpointVolume))
        except Exception:  # noqa: BLE001 - pycaw is optional; fall back to key presses
            return None

    def volume_get(self) -> Optional[int]:
        ep = self._pycaw()
        return None if ep is None else round(ep.GetMasterVolumeLevelScalar() * 100)

    def volume_set(self, percent: int) -> ActionResult:
        percent = clamp(percent)
        ep = self._pycaw()
        if ep is not None:
            ep.SetMasterVolumeLevelScalar(percent / 100.0, None)
            return ActionResult(True, f"Volume set to {percent} percent.")
        # No pycaw: each volume key moves the master volume by 2 points, so go to zero and climb.
        for _ in range(50):
            self.sys.press_key(VK["down"])
        for _ in range(round(percent / 2)):
            self.sys.press_key(VK["up"])
        return ActionResult(True, f"Volume set to about {percent} percent. (Install pycaw for exact control.)")

    def volume_change(self, delta: int) -> ActionResult:
        if self.volume_get() is None:                        # keys are relative anyway: use them directly
            key = VK["up"] if delta > 0 else VK["down"]
            for _ in range(max(1, round(abs(delta) / 2))):
                self.sys.press_key(key)
            return ActionResult(True, "Volume up." if delta > 0 else "Volume down.")
        return super().volume_change(delta)

    def set_mute(self, muted: bool) -> ActionResult:
        ep = self._pycaw()
        if ep is not None:
            ep.SetMute(1 if muted else 0, None)
            return ActionResult(True, "Muted." if muted else "Unmuted.")
        self.sys.press_key(VK["mute"])                       # the key only toggles
        return ActionResult(True, "Toggled mute.")

    def brightness_get(self) -> Optional[int]:
        rc, out = self.sys.run(_ps("(Get-CimInstance -Namespace root/WMI -ClassName WmiMonitorBrightness).CurrentBrightness"))
        m = re.search(r"\d+", out) if rc == 0 else None
        return int(m.group()) if m else None

    def brightness_set(self, percent: int) -> ActionResult:
        percent = clamp(percent)
        rc, out = self.sys.run(_ps("(Get-CimInstance -Namespace root/WMI -ClassName WmiMonitorBrightnessMethods)"
                                   f".WmiSetBrightness(1,{percent})"))
        return ActionResult(True, f"Brightness set to {percent} percent.") if rc == 0 else ActionResult(
            False, "I couldn't change the brightness. This works on laptop screens; external monitors need their own controls.")

    def media(self, action: str) -> ActionResult:
        self.sys.press_key(VK[{"toggle": "toggle", "next": "next", "previous": "previous", "stop": "stop"}[action]])
        return ActionResult(True, {"toggle": "Done.", "next": "Next track.", "previous": "Previous track.", "stop": "Stopped."}[action])

    def screenshot(self, path: str) -> ActionResult:
        script = ("Add-Type -AssemblyName System.Windows.Forms,System.Drawing; "
                  "$b=[System.Windows.Forms.SystemInformation]::VirtualScreen; "
                  "$bmp=New-Object System.Drawing.Bitmap $b.Width,$b.Height; "
                  "$g=[System.Drawing.Graphics]::FromImage($bmp); "
                  "$g.CopyFromScreen($b.Location,[System.Drawing.Point]::Empty,$b.Size); "
                  f"$bmp.Save({_ps_quote(path)})")
        return self._ran(self.sys.run(_ps(script), timeout=30), f"Screenshot saved to {path}.", "I couldn't take a screenshot")

    def lock(self) -> ActionResult:
        return self._ran(self.sys.run(["rundll32.exe", "user32.dll,LockWorkStation"]), "Locking the screen.", "I couldn't lock the screen")

    def sleep(self) -> ActionResult:
        return self._ran(self.sys.run(["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"]), "Going to sleep.", "I couldn't sleep")

    def restart(self) -> ActionResult:
        return self._ran(self.sys.run(["shutdown", "/r", "/t", "5"]), "Restarting in five seconds.", "I couldn't restart")

    def shutdown(self) -> ActionResult:
        return self._ran(self.sys.run(["shutdown", "/s", "/t", "5"]), "Shutting down in five seconds.", "I couldn't shut down")

    def open_app(self, target: str, name: str) -> ActionResult:
        try:
            self.sys.startfile(target)
        except OSError as e:
            return ActionResult(False, f"I couldn't open {name}: {e}")
        return ActionResult(True, f"Opening {name}.")


# ================================================================================= macOS
class MacBackend(Backend):
    os_name = "macos"

    def _osa(self, script: str):
        return self.sys.run(["osascript", "-e", script])

    def volume_get(self) -> Optional[int]:
        rc, out = self._osa("output volume of (get volume settings)")
        m = re.search(r"\d+", out) if rc == 0 else None
        return int(m.group()) if m else None

    def volume_set(self, percent: int) -> ActionResult:
        percent = clamp(percent)
        return self._ran(self._osa(f"set volume output volume {percent}"), f"Volume set to {percent} percent.", "I couldn't set the volume")

    def set_mute(self, muted: bool) -> ActionResult:
        return self._ran(self._osa(f"set volume output muted {'true' if muted else 'false'}"),
                         "Muted." if muted else "Unmuted.", "I couldn't change mute")

    def brightness_get(self) -> Optional[int]:
        if not self.sys.which("brightness"):
            return None
        rc, out = self.sys.run(["brightness", "-l"])
        m = re.search(r"brightness\s+([0-9.]+)", out) if rc == 0 else None
        return round(float(m.group(1)) * 100) if m else None

    def brightness_set(self, percent: int) -> ActionResult:
        if not self.sys.which("brightness"):
            return ActionResult(False, "Brightness control on macOS needs the 'brightness' tool: brew install brightness.")
        percent = clamp(percent)
        return self._ran(self.sys.run(["brightness", f"{percent / 100:.2f}"]), f"Brightness set to {percent} percent.",
                         "I couldn't change the brightness")

    def media(self, action: str) -> ActionResult:
        verb = {"toggle": "playpause", "next": "next track", "previous": "previous track", "stop": "pause"}[action]
        script = (f'if application "Spotify" is running then\ntell application "Spotify" to {verb}\n'
                  f'else if application "Music" is running then\ntell application "Music" to {verb}\n'
                  'else\nerror "no player"\nend if')
        rc, out = self._osa(script)
        if rc != 0:
            return ActionResult(False, "No music player (Music or Spotify) is running.")
        return ActionResult(True, {"toggle": "Done.", "next": "Next track.", "previous": "Previous track.", "stop": "Paused."}[action])

    def screenshot(self, path: str) -> ActionResult:
        return self._ran(self.sys.run(["screencapture", "-x", path]), f"Screenshot saved to {path}.", "I couldn't take a screenshot")

    def lock(self) -> ActionResult:
        rc, _ = self._osa('tell application "System Events" to keystroke "q" using {control down, command down}')
        if rc == 0:
            return ActionResult(True, "Locking the screen.")
        return self._ran(self.sys.run(["pmset", "displaysleepnow"]), "Locking the screen.", "I couldn't lock the screen")

    def sleep(self) -> ActionResult:
        return self._ran(self.sys.run(["pmset", "sleepnow"]), "Going to sleep.", "I couldn't sleep")

    def restart(self) -> ActionResult:
        return self._ran(self._osa('tell application "System Events" to restart'), "Restarting.", "I couldn't restart")

    def shutdown(self) -> ActionResult:
        return self._ran(self._osa('tell application "System Events" to shut down'), "Shutting down.", "I couldn't shut down")

    def open_app(self, target: str, name: str) -> ActionResult:
        return self._ran(self.sys.run(["open", "-a", target]), f"Opening {name}.", f"I couldn't open {name}")


# ================================================================================= Linux
class LinuxBackend(Backend):
    os_name = "linux"

    def volume_get(self) -> Optional[int]:
        if self.sys.which("pactl"):
            rc, out = self.sys.run(["pactl", "get-sink-volume", "@DEFAULT_SINK@"])
            m = re.search(r"(\d+)%", out) if rc == 0 else None
            if m:
                return int(m.group(1))
        if self.sys.which("wpctl"):
            rc, out = self.sys.run(["wpctl", "get-volume", "@DEFAULT_AUDIO_SINK@"])
            m = re.search(r"([0-9.]+)", out) if rc == 0 else None
            if m:
                return round(float(m.group(1)) * 100)
        if self.sys.which("amixer"):
            rc, out = self.sys.run(["amixer", "sget", "Master"])
            m = re.search(r"\[(\d+)%\]", out) if rc == 0 else None
            if m:
                return int(m.group(1))
        return None

    def volume_set(self, percent: int) -> ActionResult:
        percent = clamp(percent)
        cmd = self._first_available([["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{percent}%"],
                                     ["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", f"{percent / 100:.2f}"],
                                     ["amixer", "-q", "sset", "Master", f"{percent}%"]])
        if cmd is None:
            return self._missing("set the volume", "pactl, wpctl or amixer")
        return self._ran(self.sys.run(cmd), f"Volume set to {percent} percent.", "I couldn't set the volume")

    def set_mute(self, muted: bool) -> ActionResult:
        cmd = self._first_available([["pactl", "set-sink-mute", "@DEFAULT_SINK@", "1" if muted else "0"],
                                     ["wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@", "1" if muted else "0"],
                                     ["amixer", "-q", "sset", "Master", "mute" if muted else "unmute"]])
        if cmd is None:
            return self._missing("change mute", "pactl, wpctl or amixer")
        return self._ran(self.sys.run(cmd), "Muted." if muted else "Unmuted.", "I couldn't change mute")

    def brightness_get(self) -> Optional[int]:
        if self.sys.which("brightnessctl"):
            rc, out = self.sys.run(["brightnessctl", "-m"])
            m = re.search(r"(\d+)%", out) if rc == 0 else None
            return int(m.group(1)) if m else None
        return None

    def brightness_set(self, percent: int) -> ActionResult:
        percent = clamp(percent)
        cmd = self._first_available([["brightnessctl", "set", f"{percent}%"], ["light", "-S", str(percent)],
                                     ["xbacklight", "-set", str(percent)]])
        if cmd is None:
            return self._missing("change the brightness", "brightnessctl, light or xbacklight")
        return self._ran(self.sys.run(cmd), f"Brightness set to {percent} percent.", "I couldn't change the brightness")

    def brightness_change(self, delta: int) -> ActionResult:
        if self.sys.which("brightnessctl"):                    # relative form needs no read-back
            arg = f"+{abs(delta)}%" if delta > 0 else f"{abs(delta)}%-"
            return self._ran(self.sys.run(["brightnessctl", "set", arg]), "Brighter." if delta > 0 else "Dimmer.",
                             "I couldn't change the brightness")
        return super().brightness_change(delta)

    def media(self, action: str) -> ActionResult:
        if not self.sys.which("playerctl"):
            return self._missing("control media", "playerctl")
        verb = {"toggle": "play-pause", "next": "next", "previous": "previous", "stop": "stop"}[action]
        rc, out = self.sys.run(["playerctl", verb])
        if rc != 0:
            return ActionResult(False, "No media player is running.")
        return ActionResult(True, {"toggle": "Done.", "next": "Next track.", "previous": "Previous track.", "stop": "Stopped."}[action])

    def screenshot(self, path: str) -> ActionResult:
        cmd = self._first_available([["grim", path], ["gnome-screenshot", "-f", path], ["spectacle", "-b", "-n", "-o", path],
                                     ["scrot", path], ["maim", path], ["import", "-window", "root", path]])
        if cmd is None:
            return self._missing("take a screenshot", "grim, gnome-screenshot, spectacle, scrot, maim or imagemagick")
        return self._ran(self.sys.run(cmd, timeout=30), f"Screenshot saved to {path}.", "I couldn't take a screenshot")

    def lock(self) -> ActionResult:
        cmd = self._first_available([["loginctl", "lock-session"], ["xdg-screensaver", "lock"],
                                     ["gnome-screensaver-command", "-l"], ["dm-tool", "lock"]])
        if cmd is None:
            return self._missing("lock the screen", "loginctl, xdg-screensaver or gnome-screensaver-command")
        return self._ran(self.sys.run(cmd), "Locking the screen.", "I couldn't lock the screen")

    def _systemctl(self, verb: str, ok: str, fail: str) -> ActionResult:
        if not self.sys.which("systemctl"):
            return self._missing(fail, "systemctl")
        return self._ran(self.sys.run(["systemctl", verb]), ok, f"I couldn't {fail}")

    def sleep(self) -> ActionResult:
        return self._systemctl("suspend", "Going to sleep.", "sleep")

    def restart(self) -> ActionResult:
        return self._systemctl("reboot", "Restarting.", "restart")

    def shutdown(self) -> ActionResult:
        return self._systemctl("poweroff", "Shutting down.", "shut down")

    def open_app(self, target: str, name: str) -> ActionResult:
        argv = shlex.split(target)
        if not argv or not self.sys.which(argv[0]):
            return ActionResult(False, f"{name} isn't installed here (looked for '{argv[0] if argv else target}').")
        try:
            self.sys.spawn(argv)
        except OSError as e:
            return ActionResult(False, f"I couldn't open {name}: {e}")
        return ActionResult(True, f"Opening {name}.")


def get_backend(system: Optional[System] = None) -> Backend:
    system = system or System()
    return {"windows": WindowsBackend, "macos": MacBackend}.get(system.os, LinuxBackend)(system)
